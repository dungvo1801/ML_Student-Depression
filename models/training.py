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

AUTHOR: Student Implementation for Academic Project
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# Class weights are the only method needed for this implementation
from sklearn.utils.class_weight import compute_class_weight
import joblib
import subprocess
import shutil
import os

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
    # Load dataset
    data = pd.read_csv('models/student_depression_dataset.csv')

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

    # Check unique values in some columns to decide cleaning strategy
    # print("Unique values in 'Sleep Duration':", data['Sleep Duration'].unique())
    # print("Unique values in 'Financial Stress':", data['Financial Stress'].unique())

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

    # Split Data into Training and Testing Sets (with stratification to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # HANDLE CLASS IMBALANCE USING CLASS WEIGHTS
    # This demonstrates explicit handling of imbalanced datasets
    X_train_balanced, y_train_balanced, class_weights = handle_class_imbalance(
        X_train, y_train, method=imbalance_method, random_state=42
    )

    # MODELS WITH CLASS WEIGHTS FOR IMBALANCED DATA
    # Create Logistic Regression pipeline
    log_model = LogisticRegression(max_iter=1000, class_weight=class_weights or 'balanced')
    log_pipeline = create_model_pipeline(log_model, preprocessor)
    log_pipeline.fit(X_train_balanced, y_train_balanced)
    
    # Predictions and evaluation for Logistic Regression
    y_pred_log = log_pipeline.predict(X_test)
    y_pred_proba_log = log_pipeline.predict_proba(X_test)[:, 1]  # For ROC-AUC

    # Create Random Forest pipeline
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                    class_weight=class_weights or 'balanced')
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
    
    # Ensure directories exist before writing files
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save comprehensive metrics
    metrics_path = 'logs/model_metrics.txt'
    with open(metrics_path, 'w') as f:
        # Random Forest metrics (primary model)
        f.write('=== RANDOM FOREST PIPELINE RESULTS ===\n')
        f.write(classification_report(y_test, y_pred_rf))
        f.write('\nRandom Forest Confusion Matrix:\n')
        f.write(str(confusion_matrix(y_test, y_pred_rf)))
        f.write(f"\nAccuracy: {rf_accuracy:.4f}\n")
        f.write(f"F1 Score: {rf_f1:.4f}\n")
        f.write(f"ROC-AUC: {rf_roc_auc:.4f if not np.isnan(rf_roc_auc) else 'N/A (single class in test set)'}\n")
        
        # Logistic Regression metrics
        f.write('\n=== LOGISTIC REGRESSION PIPELINE RESULTS ===\n')
        f.write(classification_report(y_test, y_pred_log))
        f.write('\nLogistic Regression Confusion Matrix:\n')
        f.write(str(confusion_matrix(y_test, y_pred_log)))
        f.write(f"\nAccuracy: {log_accuracy:.4f}\n")
        f.write(f"F1 Score: {log_f1:.4f}\n")
        f.write(f"ROC-AUC: {log_roc_auc:.4f if not np.isnan(log_roc_auc) else 'N/A (single class in test set)'}\n")
        
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

    # Save both model pipelines for production use
    joblib.dump(rf_pipeline, 'models/rf_model.pkl')
    joblib.dump(log_pipeline, 'models/log_model.pkl')
    
    return rf_pipeline, log_pipeline  # Return both trained pipelines

def fetch_from_kaggle(dataset="ngocdung/student-depression-dataset", filename="student_depression_dataset.csv", dest_path="models/student_depression_dataset.csv"):
    """
    Fetches the latest dataset from Kaggle using the CLI. Falls back to local backup if Kaggle fetch fails.
    Returns a status message.
    """
    try:
        # Download using Kaggle CLI
        kaggle_cmd = [
            "kaggle", "datasets", "download", "-d", dataset, "-f", filename, "-p", "models", "--force"
        ]
        result = subprocess.run(kaggle_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise Exception(result.stderr)
        # Unzip if needed
        zip_path = os.path.join("models", filename.replace(".csv", ".zip"))
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("models")
            os.remove(zip_path)
        # Check if file exists
        if not os.path.exists(dest_path):
            raise Exception("Kaggle download did not produce the expected file.")
        return "Successfully fetched latest dataset from Kaggle."
    except Exception as e:
        # Fallback: try to restore from latest backup
        version_dir = "models/versions"
        backups = sorted([f for f in os.listdir(version_dir) if f.endswith('.csv')], reverse=True)
        if backups:
            latest_backup = os.path.join(version_dir, backups[0])
            shutil.copy2(latest_backup, dest_path)
            return f"Kaggle fetch failed ({str(e)}). Restored from latest backup: {backups[0]}"
        else:
            return f"Kaggle fetch failed ({str(e)}). No backup available."

def restore_backup(backup_file, dest_path="models/student_depression_dataset.csv"):
    """
    Restores the master dataset from a selected backup file.
    Returns a status message.
    """
    try:
        version_dir = "models/versions"
        backup_path = os.path.join(version_dir, backup_file)
        if not os.path.exists(backup_path):
            return f"Backup file {backup_file} not found."
        shutil.copy2(backup_path, dest_path)
        return f"Successfully restored dataset from backup: {backup_file}"
    except Exception as e:
        return f"Failed to restore backup: {str(e)}"

def should_retrain(threshold=50):
    """
    Check if model should be retrained based on new data availability
    """
    try:
        master_path = 'models/student_depression_dataset.csv'
        if not os.path.exists(master_path):
            return False
        
        df = pd.read_csv(master_path)
        current_count = df.dropna(subset=['Depression']).shape[0]
        
        # Get last retrain count
        config_path = 'logs/retrain_config.txt'
        last_count = 0
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                last_count = int(f.read().strip())
        
        return (current_count - last_count) >= threshold
    except:
        return False

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
        master_path = 'models/student_depression_dataset.csv'
        if os.path.exists(master_path):
            df = pd.read_csv(master_path)
            current_count = df.dropna(subset=['Depression']).shape[0]
            
            config_path = 'logs/retrain_config.txt'
            os.makedirs('logs', exist_ok=True)
            with open(config_path, 'w') as f:
                f.write(str(current_count))
    except Exception as e:
        pass

def performance_based_retrain_test(f1_threshold=0.05, roc_auc_threshold=0.05):
    """
    PERFORMANCE-BASED RETRAINING (F1-Score + ROC-AUC Focus)
    
    Determines if retraining is needed based on F1-score and ROC-AUC degradation.
    Both metrics are clinically meaningful for depression prediction:
    - F1-score: Balances precision and recall
    - ROC-AUC: Measures diagnostic accuracy across all thresholds
    
    Returns True if F1 score or ROC-AUC drops significantly
    """
    try:
        master_path = 'models/student_depression_dataset.csv'
        if not os.path.exists(master_path) or not os.path.exists('models/rf_model.pkl'):
            return False
        
        df = pd.read_csv(master_path)
        labeled_df = df.dropna(subset=['Depression'])
        
        # Get last retrain count
        config_path = 'logs/retrain_config.txt'
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
        
        # Load current model pipeline
        model_pipeline = joblib.load('models/rf_model.pkl')
        
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
        metrics_path = 'logs/model_metrics.txt'
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

def calculate_current_model_performance():
    """
    Alternative approach: Calculate baseline by testing current model 
    on a held-out validation set from existing data.
    This avoids hard-coded defaults completely.
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        
        # Load current dataset
        data = pd.read_csv('models/student_depression_dataset.csv')
        labeled_data = data.dropna(subset=['Depression'])
        
        if len(labeled_data) < 100:  # Need sufficient data
            return None, None
            
        # Prepare features (simplified preprocessing)
        X = labeled_data.drop(['id', 'Depression'], axis=1, errors='ignore')
        y = labeled_data['Depression'].astype(int)
        
        # Simple preprocessing for validation
        for col in X.columns:
            if X[col].dtype == 'object':
                # Convert categorical to numeric (basic approach)
                X[col] = pd.Categorical(X[col]).codes
        
        # Fill any missing values
        X = X.fillna(X.median())
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load current model
        if not os.path.exists('models/rf_model.pkl'):
            return None, None
            
        model = joblib.load('models/rf_model.pkl')
        
        # Test current model on validation set
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        return accuracy, f1
        
    except Exception as e:
        return None, None