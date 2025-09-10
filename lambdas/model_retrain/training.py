import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from datetime import datetime
# Import centralized configuration
from config import config

"""
STUDENT DEPRESSION PREDICTION - MACHINE LEARNING MODEL TRAINING

IMBALANCED DATASET HANDLING IMPLEMENTATION:
===========================================

This module demonstrates proper handling of imbalanced datasets using CLASS WEIGHTS.

PROBLEM: The depression dataset has class imbalance (41% vs 59% distribution)
SOLUTION: Class Weights method is applied to both Logistic Regression and Random Forest

CLASS WEIGHTS TECHNIQUE:
- Automatically calculates inverse frequency weights for each class
- Gives higher importance to minority class during training  
- Prevents model bias toward majority class
- No data loss or artificial sample generation required
- Computationally efficient and suitable for mild imbalance

IMPLEMENTATION DETAILS:
- Uses sklearn's 'balanced' class_weight parameter
- Applied to both RandomForestClassifier and LogisticRegression
- Weights calculated as: n_samples / (n_classes * np.bincount(y))
- Results in better recall for minority class detection

METHOD SELECTION RATIONALE:
- Class weights chosen for its simplicity and effectiveness
- No external dependencies required (built into scikit-learn)
- Efficient for mild imbalance scenarios like ours (1.41:1 ratio)
- Production-ready and well-established technique

"""

def safe_roc_auc(y_true, y_proba):
    """
    Safe ROC-AUC calculation that handles edge cases.
    Returns NaN if both classes are not present in y_true or if calculation fails.
    """
    try:
        if len(np.unique(y_true)) == 2:
            return roc_auc_score(y_true, y_proba)
        else:
            return float('nan')
    except Exception as e:
        return float('nan')

def handle_class_imbalance(X, y, method='class_weight', random_state=42):
    """
    Handle class imbalance using CLASS WEIGHTS technique.
    
    This function applies class weights to balance the importance of 
    minority vs majority classes during model training.
    
    Returns: X, y, class_weights_dict (or None if single class)
    """
    classes = np.unique(y)
    if method == 'class_weight':
        if len(classes) < 2:
            # Degenerate case: no real weighting possible with single class
            return X, y, None
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return X, y, dict(zip(classes, weights))
    else:
        # Default fallback remains the same
        weights = compute_class_weight('balanced', classes=classes, y=y) if len(classes) >= 2 else None
        return X, y, dict(zip(classes, weights)) if weights is not None else (X, y, None)

def create_preprocessing_pipeline(categorical_features, numerical_features):
    """
    Create a robust preprocessing pipeline using sklearn transformers.
    
    This pipeline handles:
    - Missing value imputation
    - Categorical encoding with unknown value handling
    - Numerical scaling
    
    Returns: ColumnTransformer for preprocessing
    """
    # Numerical preprocessing: impute missing values, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: impute missing values, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def create_model_pipeline(model, preprocessor):
    """
    Create a complete ML pipeline with preprocessing + model.
    
    This ensures preprocessing consistency between training and prediction.
    """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

def check_class_imbalance(y, threshold=0.3):
    """
    Check if dataset has class imbalance.
    Returns: is_imbalanced (bool), minority_ratio (float), imbalance_info (dict)
    """
    class_counts = pd.Series(y).value_counts().sort_index()
    total = len(y)
    class_ratios = class_counts / total
    
    minority_ratio = class_ratios.min()
    is_imbalanced = minority_ratio < threshold
    
    imbalance_info = {
        'class_counts': class_counts.to_dict(),
        'class_ratios': class_ratios.to_dict(),
        'minority_ratio': minority_ratio,
        'imbalance_ratio': class_ratios.max() / minority_ratio
    }
    
    return is_imbalanced, minority_ratio, imbalance_info

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
    # Load dataset using config
    data = pd.read_csv(config.get_data_path())

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

    # Model Performance Tracking: Save metrics to file for dashboard
    from sklearn.metrics import accuracy_score, f1_score
    
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
    
    # Ensure directories exist before writing files
    os.makedirs(config.get_logs_dir(), exist_ok=True)
    os.makedirs(config.get_data_dir(), exist_ok=True)
    
    # Save comprehensive metrics using config paths
    metrics_path = config.get_metrics_path()
    with open(metrics_path, 'w') as f:
        # Random Forest metrics (primary model)
        f.write('=== RANDOM FOREST PIPELINE RESULTS ===\n')
        f.write(classification_report(y_test, y_pred_rf))
        f.write('\nRandom Forest Confusion Matrix:\n')
        f.write(str(confusion_matrix(y_test, y_pred_rf)))
        f.write(f"\nAccuracy: {rf_accuracy:.4f}\n")
        f.write(f"F1 Score: {rf_f1:.4f}\n")
        roc_auc_str = f"{rf_roc_auc:.4f}" if not np.isnan(rf_roc_auc) else "N/A (single class in test set)"
        f.write(f"ROC-AUC: {roc_auc_str}\n")
        
        # Logistic Regression metrics
        f.write('\n=== LOGISTIC REGRESSION PIPELINE RESULTS ===\n')
        f.write(classification_report(y_test, y_pred_log))
        f.write('\nLogistic Regression Confusion Matrix:\n')
        f.write(str(confusion_matrix(y_test, y_pred_log)))
        f.write(f"\nAccuracy: {log_accuracy:.4f}\n")
        f.write(f"F1 Score: {log_f1:.4f}\n")
        log_roc_auc_str = f"{log_roc_auc:.4f}" if not np.isnan(log_roc_auc) else "N/A (single class in test set)"
        f.write(f"ROC-AUC: {log_roc_auc_str}\n")
        
        # CUSTOM THRESHOLD RESULTS FOR LOGISTIC REGRESSION
        f.write(f'\n=== LOGISTIC REGRESSION WITH CUSTOM THRESHOLD ({custom_threshold}) ===\n')
        f.write(classification_report(y_test, y_pred_log_custom_threshold))
        f.write('\nCustom Threshold Confusion Matrix:\n')
        f.write(str(confusion_matrix(y_test, y_pred_log_custom_threshold)))
        f.write(f"\nAccuracy: {log_custom_accuracy:.4f}\n")
        f.write(f"F1 Score: {log_custom_f1:.4f}\n")
        log_custom_roc_auc_str = f"{log_custom_roc_auc:.4f}" if not np.isnan(log_custom_roc_auc) else "N/A (single class in test set)"
        f.write(f"ROC-AUC: {log_custom_roc_auc_str}\n")
        
        # THRESHOLD COMPARISON
        f.write(f"\n=== THRESHOLD COMPARISON FOR LOGISTIC REGRESSION ===\n")
        f.write(f"Default Threshold (0.5):\n")
        f.write(f"  Accuracy: {log_accuracy:.4f}, F1: {log_f1:.4f}\n")
        f.write(f"Custom Threshold ({custom_threshold}):\n")
        f.write(f"  Accuracy: {log_custom_accuracy:.4f}, F1: {log_custom_f1:.4f}\n")
        f.write(f"Improvement in F1: {log_custom_f1 - log_f1:+.4f}\n")
        f.write(f"Change in Accuracy: {log_custom_accuracy - log_accuracy:+.4f}\n")
        
        # K-FOLD CROSS-VALIDATION RESULTS
        f.write(f"\n=== K-FOLD CROSS-VALIDATION RESULTS ({cv_folds} FOLDS) ===\n")
        f.write(f"\nRANDOM FOREST CROSS-VALIDATION:\n")
        f.write(f"F1 Score: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}\n")
        f.write(f"Accuracy: {rf_cv_accuracy.mean():.4f} ± {rf_cv_accuracy.std():.4f}\n")
        f.write(f"ROC-AUC: {rf_cv_roc_auc.mean():.4f} ± {rf_cv_roc_auc.std():.4f}\n")
        f.write(f"Individual F1 scores: {[f'{score:.4f}' for score in rf_cv_scores]}\n")
        
        f.write(f"\nLOGISTIC REGRESSION CROSS-VALIDATION:\n")
        f.write(f"F1 Score: {log_cv_scores.mean():.4f} ± {log_cv_scores.std():.4f}\n")
        f.write(f"Accuracy: {log_cv_accuracy.mean():.4f} ± {log_cv_accuracy.std():.4f}\n")
        f.write(f"ROC-AUC: {log_cv_roc_auc.mean():.4f} ± {log_cv_roc_auc.std():.4f}\n")
        f.write(f"Individual F1 scores: {[f'{score:.4f}' for score in log_cv_scores]}\n")
        
        # Cross-validation vs Single Split Comparison
        f.write(f"\n=== CROSS-VALIDATION vs SINGLE SPLIT COMPARISON ===\n")
        f.write(f"Random Forest:\n")
        f.write(f"  Single Split F1: {rf_f1:.4f}\n")
        f.write(f"  CV Mean F1: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}\n")
        f.write(f"  Difference: {rf_cv_scores.mean() - rf_f1:+.4f}\n")
        
        f.write(f"\nLogistic Regression:\n")
        f.write(f"  Single Split F1: {log_f1:.4f}\n")
        f.write(f"  CV Mean F1: {log_cv_scores.mean():.4f} ± {log_cv_scores.std():.4f}\n")
        f.write(f"  Difference: {log_cv_scores.mean() - log_f1:+.4f}\n")
        
        # ENSEMBLE MODEL RESULTS
        f.write(f"\n=== ENSEMBLE MODEL RESULTS ===\n")
        f.write(f"Weighted Ensemble (RF: {rf_normalized_weight:.3f}, LR: {log_normalized_weight:.3f}):\n")
        f.write(classification_report(y_test, y_pred_ensemble))
        f.write('\nEnsemble Confusion Matrix:\n')
        f.write(str(confusion_matrix(y_test, y_pred_ensemble)))
        f.write(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f}\n")
        f.write(f"Ensemble F1 Score: {ensemble_f1:.4f}\n")
        ensemble_roc_auc_str = f"{ensemble_roc_auc:.4f}" if not np.isnan(ensemble_roc_auc) else "N/A"
        f.write(f"Ensemble ROC-AUC: {ensemble_roc_auc_str}\n")
        
        # MODEL COMPARISON AND SELECTION
        f.write(f"\n=== MODEL PERFORMANCE COMPARISON ===\n")
        f.write(f"Random Forest F1: {rf_f1:.4f} (CV: {rf_cv_scores.mean():.4f})\n")
        f.write(f"Logistic Regression F1: {log_f1:.4f} (CV: {log_cv_scores.mean():.4f})\n")
        f.write(f"Ensemble F1: {ensemble_f1:.4f}\n")
        f.write(f"\nBest Individual Model: {best_model_name}\n")
        
        # Recommendation based on performance
        if ensemble_f1 > max(rf_f1, log_f1):
            f.write(f"RECOMMENDATION: Use Ensemble Model (F1 improvement: +{ensemble_f1 - max(rf_f1, log_f1):.4f})\n")
        else:
            f.write(f"RECOMMENDATION: Use {best_model_name} (Best individual performance)\n")
        
        # Clinical interpretation
        f.write(f"\n=== CLINICAL EVALUATION ===\n")
        f.write(f"ROC-AUC measures diagnostic accuracy (0.5=random, 1.0=perfect)\n")
        f.write(f"F1-Score balances precision/recall for depression detection\n")
        f.write(f"Preferred model: {'Random Forest' if rf_f1 >= log_f1 else 'Logistic Regression'}\n")
        
        # Add class imbalance information
        f.write(f"\n=== Class Imbalance Handling ===\n")
        f.write(f"Method used: {imbalance_method}\n")
        f.write(f"Original class distribution: {imbalance_info['class_counts']}\n")
        f.write(f"Minority class ratio: {imbalance_info['minority_ratio']:.4f}\n")
        f.write(f"Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}:1\n")
        
        if imbalance_method != 'none':
            # Show training data distribution after balancing
            train_dist = pd.Series(y_train_balanced).value_counts().sort_index()
            f.write(f"Training data after balancing: {train_dist.to_dict()}\n")

    # Save all models for production use using config paths
    joblib.dump(rf_pipeline, config.get_model_path())
    joblib.dump(log_pipeline, config.get_logistic_model_path())
    
    # Save ensemble model information for production use
    ensemble_info = {
        'rf_weight': rf_normalized_weight,
        'log_weight': log_normalized_weight,
        'best_model': best_model_name,
        'ensemble_threshold': 0.5
    }
    ensemble_path = os.path.join(config.get_models_dir(), 'ensemble_info.pkl')
    joblib.dump(ensemble_info, ensemble_path)
    
    return rf_pipeline, log_pipeline, ensemble_info  # Return all trained components

def trigger_retrain():
    """
    Trigger function for F1-SCORE BASED performance monitoring.
    
    F1-score is the primary metric because:
    - Better for imbalanced depression data
    - Clinically more relevant than accuracy  
    - Balances precision and recall for depression detection
    """
    if performance_based_retrain_test():
        train_model()
        update_retrain_tracking()
        return True
    else:
        return False

def update_retrain_tracking():
    """
    Update tracking information after retraining
    """
    try:
        master_path = config.get_data_path()
        if os.path.exists(master_path):
            df = pd.read_csv(master_path)
            current_count = df.dropna(subset=['Depression']).shape[0]
            
            config_path = config.get_retrain_config_path()
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                f.write(str(current_count))
    except Exception as e:
        pass

def performance_based_retrain_test(f1_threshold=None, roc_auc_threshold=None):
    """
    PERFORMANCE-BASED RETRAINING (F1-Score + ROC-AUC Focus)
    
    Determines if retraining is needed based on F1-score and ROC-AUC degradation.
    Both metrics are clinically meaningful for depression prediction:
    - F1-score: Balances precision and recall
    - ROC-AUC: Measures diagnostic accuracy across all thresholds
    
    Returns True if F1 score or ROC-AUC drops significantly
    """
    if f1_threshold is None:
        f1_threshold = config.get_performance_threshold()
    if roc_auc_threshold is None:
        roc_auc_threshold = config.get_performance_threshold()
        
    try:
        master_path = config.get_data_path()
        model_path = config.get_model_path()
        if not os.path.exists(master_path) or not os.path.exists(model_path):
            return False
        
        df = pd.read_csv(master_path)
        labeled_df = df.dropna(subset=['Depression'])
        
        # Get last retrain count using config
        config_path = config.get_retrain_config_path()
        last_count = 0
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                last_count = int(f.read().strip())
        
        if len(labeled_df) <= last_count + 10:  # Need minimum new data
            return False
        
        # Get new data since last retrain
        new_data = labeled_df.iloc[last_count:]
        
        if len(new_data) < 10:  # Need minimum samples for reliable test
            return False
        
        # Load current model pipeline using config path
        model_pipeline = joblib.load(config.get_model_path())
        
        # Prepare features for prediction
        X_new = new_data.drop(columns=['id', 'Depression'], errors='ignore')
        y_true = new_data['Depression'].astype(int)
        
        # Make predictions using the pipeline (handles preprocessing automatically)
        try:
            y_pred = model_pipeline.predict(X_new)
            y_pred_proba = model_pipeline.predict_proba(X_new)[:, 1]
        except Exception as e:
            return False
        
        # Calculate performance metrics - Focus on F1-score and ROC-AUC for depression prediction
        from sklearn.metrics import accuracy_score, f1_score
        current_accuracy = accuracy_score(y_true, y_pred)
        current_f1 = f1_score(y_true, y_pred)
        current_roc_auc = safe_roc_auc(y_true, y_pred_proba)
        
        # Get baseline performance from last training
        baseline_accuracy, baseline_f1, baseline_roc_auc = get_baseline_performance()
        
        # Handle case where no baseline exists (first run or error)
        if baseline_accuracy is None or baseline_f1 is None or baseline_roc_auc is None:
            return False  # Don't retrain if we can't compare performance
        
        # Calculate performance drops (handle NaN values safely)
        f1_drop = baseline_f1 - current_f1
        accuracy_drop = baseline_accuracy - current_accuracy
        roc_auc_drop = baseline_roc_auc - current_roc_auc if not (np.isnan(baseline_roc_auc) or np.isnan(current_roc_auc)) else 0
        
        # Primary trigger: F1-score drop (most important for depression detection)
        if f1_drop > f1_threshold:
            return True
        
        # Secondary trigger: ROC-AUC drop (diagnostic accuracy) - only if valid ROC-AUC values
        if not np.isnan(roc_auc_drop) and roc_auc_drop > roc_auc_threshold:
            return True
        
        # Tertiary trigger: Severe accuracy drop (backup check)    
        if accuracy_drop > 0.10:  # 10% accuracy drop as backup
            return True
            
        return False
        
    except Exception as e:
        return False

def get_baseline_performance():
    """Get baseline performance metrics from last training (including ROC-AUC)"""
    try:
        metrics_path = config.get_metrics_path()
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                content = f.read()
                # Extract accuracy, F1, and ROC-AUC from saved metrics
                import re
                # Look for Random Forest metrics (primary model)
                accuracy_match = re.search(r'=== RANDOM FOREST.*?Accuracy: ([\d.]+)', content, re.DOTALL)
                f1_match = re.search(r'=== RANDOM FOREST.*?F1 Score: ([\d.]+)', content, re.DOTALL)
                roc_auc_match = re.search(r'=== RANDOM FOREST.*?ROC-AUC: ([\d.]+)', content, re.DOTALL)
                
                if accuracy_match and f1_match and roc_auc_match:
                    return (float(accuracy_match.group(1)), 
                           float(f1_match.group(1)), 
                           float(roc_auc_match.group(1)))
        
        # If no metrics file exists, calculate baseline by retraining
        train_model()  # This will create the metrics file
        
        # Try to read metrics again after training
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                content = f.read()
                import re
                accuracy_match = re.search(r'=== RANDOM FOREST.*?Accuracy: ([\d.]+)', content, re.DOTALL)
                f1_match = re.search(r'=== RANDOM FOREST.*?F1 Score: ([\d.]+)', content, re.DOTALL)
                roc_auc_match = re.search(r'=== RANDOM FOREST.*?ROC-AUC: ([\d.]+)', content, re.DOTALL)
                
                if accuracy_match and f1_match and roc_auc_match:
                    return (float(accuracy_match.group(1)), 
                           float(f1_match.group(1)), 
                           float(roc_auc_match.group(1)))
        
        # If still no metrics, return None to indicate failure
        return None, None, None
    except Exception as e:
        return None, None, None
