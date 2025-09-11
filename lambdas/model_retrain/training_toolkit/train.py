import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
# Import centralized configuration
from config import config
from util import (
    get_master_data,
    get_validated_data,
    upload_bytes,
    db_config
)
from transformation import (
    check_class_imbalance,
    handle_class_imbalance,
    safe_roc_auc
)
from pipeline import (
    create_preprocessing_pipeline,
    create_model_pipeline
)
import psycopg2
from psycopg2 import OperationalError
import json
import io
from dotenv import load_dotenv

load_dotenv()



def train_model(imbalance_method='class_weight'):
    """
    Complete training pipeline - trains model from scratch on all available data
    
    IMBALANCED DATASET HANDLING:
    This function explicitly uses CLASS WEIGHTS to handle imbalanced datasets.
    Class weights automatically balance the importance of minority vs majority classes
    during training, ensuring fair model performance across all classes.
    
    METHOD: Class Weights (sklearn's 'balanced' parameter)
    - Calculates inverse frequency weights for each class
    - Gives higher importance to minority class samples
    - No data resampling required (efficient)
    - Works well for mild to moderate imbalance
    """

    data = None 
    try:
        connection = psycopg2.connect(**db_config)
        print("Successfully connected to PostgreSQL")

        with connection:
            with connection.cursor() as cursor:
                master_data = get_master_data(cursor)
                validation_data = get_validated_data(cursor)
                data = pd.concat([master_data, validation_data], ignore_index=True)

    except OperationalError as e:
        print("Training failed - Error while connecting to PostgreSQL:", e)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Connection closed.")


    # Only use rows with a label for retraining
    data = data.dropna(subset=['Depression'])

    #DATA CLEANING
    # Convert 'Depression' to integer (if not already)
    data['Depression'] = data['Depression'].astype(int)
    
    # IMBALANCED DATASET HANDLING - CLASS WEIGHTS METHOD
    # Check for class imbalance early in the pipeline
    y_initial = data['Depression']
    is_imbalanced, minority_ratio, imbalance_info = check_class_imbalance(y_initial)
    
    # Force class weights method for clear demonstration
    imbalance_method = 'class_weight'

    # Dynamically identify categorical columns (non-numeric columns)
    # Exclude target variable and ID from categorical conversion
    exclude_cols = ['Depression', 'id']
    categorical_cols = []
    
    for col in data.columns:
        if col not in exclude_cols:
            # Check if column contains string/object data or has few unique values
            if (data[col].dtype == 'object' or 
                (data[col].dtype in ['int64', 'float64'] and data[col].nunique() <= 10)):
                categorical_cols.append(col)
    
    # Convert identified categorical columns
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')

    # Define a function to extract numeric hours from Sleep Duration column
    def extract_hours(s):
        # Find a number (including decimals)
        match = re.search(r"(\d+(\.\d+)?)", str(s))
        return float(match.group(1)) if match else np.nan

    # Clean Sleep Duration column
    if 'Sleep Duration' in data.columns:
        data['Sleep Duration'] = data['Sleep Duration'].apply(extract_hours)

    # DATA CLEANING: Fix data quality issues
    # Clean Gender column (remove numeric values that shouldn't be there)
    if 'Gender' in data.columns:
        # Keep only 'Male' and 'Female', replace others with mode
        valid_genders = ['Male', 'Female']
        gender_mode = data[data['Gender'].isin(valid_genders)]['Gender'].mode()
        if len(gender_mode) > 0:
            data.loc[~data['Gender'].isin(valid_genders), 'Gender'] = gender_mode[0]
        else:
            data.loc[~data['Gender'].isin(valid_genders), 'Gender'] = 'Unknown'
    
    # Clean Age column (should be numeric)
    if 'Age' in data.columns:
        # Convert Age to numeric, replacing non-numeric with NaN
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
        # Fill missing ages with median
        data['Age'] = data['Age'].fillna(data['Age'].median())    
    # Automatically identify categorical and numerical features AFTER cleaning
    exclude_cols = ['Depression', 'id']
    
    # Identify categorical columns
    categorical_features = []
    numerical_features = []
    
    for col in data.columns:
        if col not in exclude_cols:
            # Check if column is clearly textual/categorical
            if (data[col].dtype == 'object' or 
                data[col].dtype.name == 'category'):
                categorical_features.append(col)
            else:
                # All numeric columns are numerical features
                numerical_features.append(col)

    # Prepare feature matrix (X) and target variable (y)
    X = data.drop(columns=exclude_cols, errors='ignore')
    y = data['Depression']

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)

    # Split Data into Training and Testing Sets using config parameters
    training_params = config.get_training_params()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=training_params['test_size'], 
        random_state=training_params['random_state'], 
        stratify=y
    )

    # HANDLE CLASS IMBALANCE USING CLASS WEIGHTS
    # This demonstrates explicit handling of imbalanced datasets
    X_train_balanced, y_train_balanced, class_weights = handle_class_imbalance(
        X_train, y_train, method=imbalance_method, random_state=42
    )

    # MODELS WITH CLASS WEIGHTS FOR IMBALANCED DATA
    # Create Logistic Regression pipeline using config parameters
    log_model = LogisticRegression(
        max_iter=training_params['logistic_max_iter'], 
        class_weight=class_weights or 'balanced'
    )
    log_pipeline = create_model_pipeline(log_model, preprocessor)
    log_pipeline.fit(X_train_balanced, y_train_balanced)
    
    # Predictions and evaluation for Logistic Regression
    y_pred_log = log_pipeline.predict(X_test)
    y_pred_proba_log = log_pipeline.predict_proba(X_test)[:, 1]  # For ROC-AUC

    # Create Random Forest pipeline using config parameters
    rf_model = RandomForestClassifier(
        n_estimators=training_params['rf_n_estimators'], 
        random_state=training_params['random_state'], 
        class_weight=class_weights or 'balanced'
    )
    rf_pipeline = create_model_pipeline(rf_model, preprocessor)
    rf_pipeline.fit(X_train_balanced, y_train_balanced)
    
    # Predictions and evaluation for Random Forest
    y_pred_rf = rf_pipeline.predict(X_test)
    y_pred_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]  # For ROC-AUC


    # Calculate metrics for both models
    log_accuracy = accuracy_score(y_test, y_pred_log)
    log_f1 = f1_score(y_test, y_pred_log)
    log_roc_auc = safe_roc_auc(y_test, y_pred_proba_log)
    
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_roc_auc = safe_roc_auc(y_test, y_pred_proba_rf)
    
    # CUSTOM THRESHOLD OPTIMIZATION FOR LOGISTIC REGRESSION
    # Apply custom threshold to prioritize recall for depression detection
    custom_threshold = 0.3
    y_pred_log_custom_threshold = (y_pred_proba_log >= custom_threshold).astype(int)
    
    # Calculate metrics with custom threshold
    log_custom_accuracy = accuracy_score(y_test, y_pred_log_custom_threshold)
    log_custom_f1 = f1_score(y_test, y_pred_log_custom_threshold)
    log_custom_roc_auc = safe_roc_auc(y_test, y_pred_proba_log)  # ROC-AUC doesn't change with threshold
    
    # K-FOLD CROSS-VALIDATION FOR ROBUST MODEL EVALUATION
    print("Performing K-Fold Cross-Validation...")
    
    # Setup stratified k-fold to maintain class distribution
    cv_folds = 5
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=training_params['random_state'])
    
    # Cross-validation for Random Forest
    rf_cv_scores = cross_val_score(rf_pipeline, X, y, cv=skf, scoring='f1')
    rf_cv_accuracy = cross_val_score(rf_pipeline, X, y, cv=skf, scoring='accuracy')
    rf_cv_roc_auc = cross_val_score(rf_pipeline, X, y, cv=skf, scoring='roc_auc')
    
    # Cross-validation for Logistic Regression
    log_cv_scores = cross_val_score(log_pipeline, X, y, cv=skf, scoring='f1')
    log_cv_accuracy = cross_val_score(log_pipeline, X, y, cv=skf, scoring='accuracy')
    log_cv_roc_auc = cross_val_score(log_pipeline, X, y, cv=skf, scoring='roc_auc')
    
    print(f"Cross-validation completed with {cv_folds} folds")
    
    # ENSEMBLE METHOD: COMBINE BOTH MODELS FOR BETTER PERFORMANCE
    # Weighted ensemble based on cross-validation F1 scores
    rf_weight = rf_cv_scores.mean()
    log_weight = log_cv_scores.mean()
    total_weight = rf_weight + log_weight
    
    # Normalize weights
    rf_normalized_weight = rf_weight / total_weight
    log_normalized_weight = log_weight / total_weight
    
    # Create ensemble predictions (weighted average of probabilities)
    y_pred_proba_ensemble = (rf_normalized_weight * y_pred_proba_rf + 
                            log_normalized_weight * y_pred_proba_log)
    y_pred_ensemble = (y_pred_proba_ensemble >= 0.5).astype(int)
    
    # Calculate ensemble metrics
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    ensemble_f1 = f1_score(y_test, y_pred_ensemble)
    ensemble_roc_auc = safe_roc_auc(y_test, y_pred_proba_ensemble)
    
    # ADAPTIVE MODEL SELECTION: Choose best model based on CV performance
    if rf_cv_scores.mean() > log_cv_scores.mean():
        best_model = rf_pipeline
        best_model_name = "Random Forest"
        best_cv_f1 = rf_cv_scores.mean()
    else:
        best_model = log_pipeline
        best_model_name = "Logistic Regression"
        best_cv_f1 = log_cv_scores.mean()
    
    print(f"Best performing model: {best_model_name} (CV F1: {best_cv_f1:.4f})")
    print(f"Ensemble F1 Score: {ensemble_f1:.4f}")
    
    # Save comprehensive metrics using config paths
    report = io.StringIO()
    # Random Forest metrics (primary model)
    report.write('=== RANDOM FOREST PIPELINE RESULTS ===\n')
    report.write(classification_report(y_test, y_pred_rf))
    report.write('\nRandom Forest Confusion Matrix:\n')
    report.write(str(confusion_matrix(y_test, y_pred_rf)))
    report.write(f"\nAccuracy: {rf_accuracy:.4f}\n")
    report.write(f"F1 Score: {rf_f1:.4f}\n")
    roc_auc_str = f"{rf_roc_auc:.4f}" if not np.isnan(rf_roc_auc) else "N/A (single class in test set)"
    report.write(f"ROC-AUC: {roc_auc_str}\n")
    
    # Logistic Regression metrics
    report.write('\n=== LOGISTIC REGRESSION PIPELINE RESULTS ===\n')
    report.write(classification_report(y_test, y_pred_log))
    report.write('\nLogistic Regression Confusion Matrix:\n')
    report.write(str(confusion_matrix(y_test, y_pred_log)))
    report.write(f"\nAccuracy: {log_accuracy:.4f}\n")
    report.write(f"F1 Score: {log_f1:.4f}\n")
    log_roc_auc_str = f"{log_roc_auc:.4f}" if not np.isnan(log_roc_auc) else "N/A (single class in test set)"
    report.write(f"ROC-AUC: {log_roc_auc_str}\n")
    
    # CUSTOM THRESHOLD RESULTS FOR LOGISTIC REGRESSION
    report.write(f'\n=== LOGISTIC REGRESSION WITH CUSTOM THRESHOLD ({custom_threshold}) ===\n')
    report.write(classification_report(y_test, y_pred_log_custom_threshold))
    report.write('\nCustom Threshold Confusion Matrix:\n')
    report.write(str(confusion_matrix(y_test, y_pred_log_custom_threshold)))
    report.write(f"\nAccuracy: {log_custom_accuracy:.4f}\n")
    report.write(f"F1 Score: {log_custom_f1:.4f}\n")
    log_custom_roc_auc_str = f"{log_custom_roc_auc:.4f}" if not np.isnan(log_custom_roc_auc) else "N/A (single class in test set)"
    report.write(f"ROC-AUC: {log_custom_roc_auc_str}\n")
    
    # THRESHOLD COMPARISON
    report.write(f"\n=== THRESHOLD COMPARISON FOR LOGISTIC REGRESSION ===\n")
    report.write(f"Default Threshold (0.5):\n")
    report.write(f"  Accuracy: {log_accuracy:.4f}, F1: {log_f1:.4f}\n")
    report.write(f"Custom Threshold ({custom_threshold}):\n")
    report.write(f"  Accuracy: {log_custom_accuracy:.4f}, F1: {log_custom_f1:.4f}\n")
    report.write(f"Improvement in F1: {log_custom_f1 - log_f1:+.4f}\n")
    report.write(f"Change in Accuracy: {log_custom_accuracy - log_accuracy:+.4f}\n")
    
    # K-FOLD CROSS-VALIDATION RESULTS
    report.write(f"\n=== K-FOLD CROSS-VALIDATION RESULTS ({cv_folds} FOLDS) ===\n")
    report.write(f"\nRANDOM FOREST CROSS-VALIDATION:\n")
    report.write(f"F1 Score: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}\n")
    report.write(f"Accuracy: {rf_cv_accuracy.mean():.4f} ± {rf_cv_accuracy.std():.4f}\n")
    report.write(f"ROC-AUC: {rf_cv_roc_auc.mean():.4f} ± {rf_cv_roc_auc.std():.4f}\n")
    report.write(f"Individual F1 scores: {[f'{score:.4f}' for score in rf_cv_scores]}\n")
    
    report.write(f"\nLOGISTIC REGRESSION CROSS-VALIDATION:\n")
    report.write(f"F1 Score: {log_cv_scores.mean():.4f} ± {log_cv_scores.std():.4f}\n")
    report.write(f"Accuracy: {log_cv_accuracy.mean():.4f} ± {log_cv_accuracy.std():.4f}\n")
    report.write(f"ROC-AUC: {log_cv_roc_auc.mean():.4f} ± {log_cv_roc_auc.std():.4f}\n")
    report.write(f"Individual F1 scores: {[f'{score:.4f}' for score in log_cv_scores]}\n")
    
    # Cross-validation vs Single Split Comparison
    report.write(f"\n=== CROSS-VALIDATION vs SINGLE SPLIT COMPARISON ===\n")
    report.write(f"Random Forest:\n")
    report.write(f"  Single Split F1: {rf_f1:.4f}\n")
    report.write(f"  CV Mean F1: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}\n")
    report.write(f"  Difference: {rf_cv_scores.mean() - rf_f1:+.4f}\n")
    
    report.write(f"\nLogistic Regression:\n")
    report.write(f"  Single Split F1: {log_f1:.4f}\n")
    report.write(f"  CV Mean F1: {log_cv_scores.mean():.4f} ± {log_cv_scores.std():.4f}\n")
    report.write(f"  Difference: {log_cv_scores.mean() - log_f1:+.4f}\n")
    
    # ENSEMBLE MODEL RESULTS
    report.write(f"\n=== ENSEMBLE MODEL RESULTS ===\n")
    report.write(f"Weighted Ensemble (RF: {rf_normalized_weight:.3f}, LR: {log_normalized_weight:.3f}):\n")
    report.write(classification_report(y_test, y_pred_ensemble))
    report.write('\nEnsemble Confusion Matrix:\n')
    report.write(str(confusion_matrix(y_test, y_pred_ensemble)))
    report.write(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f}\n")
    report.write(f"Ensemble F1 Score: {ensemble_f1:.4f}\n")
    ensemble_roc_auc_str = f"{ensemble_roc_auc:.4f}" if not np.isnan(ensemble_roc_auc) else "N/A"
    report.write(f"Ensemble ROC-AUC: {ensemble_roc_auc_str}\n")
    
    # MODEL COMPARISON AND SELECTION
    report.write(f"\n=== MODEL PERFORMANCE COMPARISON ===\n")
    report.write(f"Random Forest F1: {rf_f1:.4f} (CV: {rf_cv_scores.mean():.4f})\n")
    report.write(f"Logistic Regression F1: {log_f1:.4f} (CV: {log_cv_scores.mean():.4f})\n")
    report.write(f"Ensemble F1: {ensemble_f1:.4f}\n")
    report.write(f"\nBest Individual Model: {best_model_name}\n")
    
    # Recommendation based on performance
    if ensemble_f1 > max(rf_f1, log_f1):
        report.write(f"RECOMMENDATION: Use Ensemble Model (F1 improvement: +{ensemble_f1 - max(rf_f1, log_f1):.4f})\n")
    else:
        report.write(f"RECOMMENDATION: Use {best_model_name} (Best individual performance)\n")
    
    # Clinical interpretation
    report.write(f"\n=== CLINICAL EVALUATION ===\n")
    report.write(f"ROC-AUC measures diagnostic accuracy (0.5=random, 1.0=perfect)\n")
    report.write(f"F1-Score balances precision/recall for depression detection\n")
    report.write(f"Preferred model: {'Random Forest' if rf_f1 >= log_f1 else 'Logistic Regression'}\n")
    
    # Add class imbalance information
    report.write(f"\n=== Class Imbalance Handling ===\n")
    report.write(f"Method used: {imbalance_method}\n")
    report.write(f"Original class distribution: {imbalance_info['class_counts']}\n")
    report.write(f"Minority class ratio: {imbalance_info['minority_ratio']:.4f}\n")
    report.write(f"Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}:1\n")
    
    if imbalance_method != 'none':
        # Show training data distribution after balancing
        train_dist = pd.Series(y_train_balanced).value_counts().sort_index()
        report.write(f"Training data after balancing: {train_dist.to_dict()}\n")

    metrics_report_bytes = report.getvalue().encode('utf-8')
    upload_success = upload_bytes(os.getenv('METRICS_KEY'), metrics_report_bytes) 

    if not upload_success:
        print("Failed to upload report to S3.")
    else:
        print("Report uploaded successfully.")


    # Save ensemble model information for production use
    ensemble_info = {
        'rf_weight': rf_normalized_weight,
        'log_weight': log_normalized_weight,
        'best_model': best_model_name,
        'ensemble_threshold': 0.5
    }

    # Save all models to s3 for production use using config paths
    rf_buffer = io.BytesIO()
    log_buffer = io.BytesIO()
    ensemble_buffer = io.BytesIO()
    
    joblib.dump(rf_pipeline, rf_buffer)
    joblib.dump(log_pipeline, log_buffer)
    joblib.dump(ensemble_info, ensemble_buffer)

    rf_buffer.seek(0)
    log_buffer.seek(0)
    ensemble_buffer.seek(0)

    rf_upload_success = upload_bytes(os.getenv('RF_MODEL_KEY'), rf_buffer.getvalue())
    log_upload_success = upload_bytes(os.getenv('LOG_MODEL_KEY'), log_buffer.getvalue())
    en_upload_success = upload_bytes(os.getenv('ENSEMBLE_KEY'), ensemble_buffer.getvalue())
    
    if not rf_upload_success or not log_upload_success or not en_upload_success:
        print("Failed to retrain model")
        return None
    else:
        print("Models retrained successfully")

    return rf_pipeline, log_pipeline, ensemble_info  # Return all trained components
