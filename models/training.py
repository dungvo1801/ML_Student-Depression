import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import subprocess
import shutil
import os
from scipy.stats import chi2_contingency
from datetime import datetime
# Import centralized configuration - handle path properly
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import config

def validate_and_clean_data(df):
    """
    Validate and clean uploaded data before processing.
    
    Args:
        df: pandas DataFrame from uploaded CSV
        
    Returns:
        cleaned DataFrame
        
    Raises:
        ValueError: if data has critical issues
    """
    try:
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Basic validation
        if cleaned_df.empty:
            raise ValueError("Uploaded file is empty")
        
        # Remove completely empty rows and columns
        cleaned_df = cleaned_df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle numeric columns - convert strings to numbers where possible
        for col in cleaned_df.columns:
            if col not in ['id']:  # Skip ID column
                try:
                    # Try to convert to numeric, keep as object if not possible
                    numeric_series = pd.to_numeric(cleaned_df[col], errors='coerce')
                    # Only convert if most values are successfully converted
                    if numeric_series.notna().sum() / len(cleaned_df) > 0.7:
                        cleaned_df[col] = numeric_series
                except:
                    pass
        
        # Basic range validation for numeric columns using config limits
        validation_limits = config.get_validation_limits()
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            if col not in ['id']:
                # Remove obvious outliers using configured ranges
                if 'age' in col.lower():
                    age_min, age_max = validation_limits['age']
                    cleaned_df[col] = cleaned_df[col].clip(age_min, age_max)
                elif 'sleep' in col.lower():
                    sleep_min, sleep_max = validation_limits['sleep']
                    cleaned_df[col] = cleaned_df[col].clip(sleep_min, sleep_max)
                elif 'stress' in col.lower():
                    stress_min, stress_max = validation_limits['stress']
                    cleaned_df[col] = cleaned_df[col].clip(stress_min, stress_max)
        
        return cleaned_df
        
    except Exception as e:
        # If validation fails, return original data with warning
        print(f"Data validation warning: {e}")
        return df

def handle_class_imbalance(X, y, method='class_weight', random_state=42):
    """
    Handle class imbalance using class weights technique.
    
    This function applies class weights to balance the importance of 
    minority vs majority classes during model training.
    
    Returns: X, y, class_weights_dict (or None if single class)
    """
    classes = np.unique(y)
    if len(classes) < 2:
        # Degenerate case: no real weighting possible with single class
        return X, y, None
    
    if method == 'class_weight':
        # Calculate class weights for balanced training
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, weights))
        return X, y, class_weight_dict
    
    elif method == 'none':
        return X, y, None
    
    else:
        # Default to class weights for any other method
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, weights))
        return X, y, class_weight_dict

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

def train_model(imbalance_method='auto'):
    """
    Complete training pipeline - trains model from scratch on all available data
    Always handles class imbalance automatically for better model performance.
    ONLY trains on human-verified labels, never on model predictions.
    """
    # Load dataset using config
    data = pd.read_csv(config.get_data_path())

    # CRITICAL: Only use rows with REAL human-verified labels (not model predictions)
    # Never train on model's own predictions to avoid feedback loops
    verified_data = data.dropna(subset=['Depression'])
    
    # Additional safety: exclude any rows that might have been auto-labeled
    # If Depression_Pred exists, it means these were model predictions
    if 'Depression_Pred' in data.columns:
        # Only use rows where Depression was NOT predicted (i.e., real ground truth)
        verified_data = verified_data[verified_data['Depression_Pred'].isna() | 
                                    (verified_data['Depression'] != verified_data['Depression_Pred'])]
    
    min_samples = config.get_training_params()['min_training_samples']
    if len(verified_data) < min_samples:
        raise ValueError(f"Insufficient verified training data. Need at least {min_samples} human-labeled samples.")

    #DATA CLEANING
    # Use verified data instead of all data
    data = verified_data.copy()
    
    # Convert 'Depression' to integer (if not already)
    data['Depression'] = data['Depression'].astype(int)
    
    # Check for class imbalance early in the pipeline
    y_initial = data['Depression']
    is_imbalanced, minority_ratio, imbalance_info = check_class_imbalance(y_initial)
    
    # Auto-select imbalance handling method using config
    training_params = config.get_training_params()
    if imbalance_method == 'auto':
        # Use configured default method
        imbalance_method = training_params['imbalance_method']

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

    data['Sleep Duration'] = data['Sleep Duration'].apply(extract_hours)

    # Convert Financial Stress to categorical if it represents levels (e.g., Low, Medium, High)
    data['Financial Stress'] = data['Financial Stress'].astype('category')


     #Display missing values per column
    missing_values = data.isnull().sum()
    
    # Dynamically impute missing values for numerical columns with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Depression', 'id'] and data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())

    # Feature Engineering
    # Dynamically identify categorical columns for encoding
    # Get all categorical columns except target variable
    cat_features = [col for col in categorical_cols if col in data.columns and col != 'Depression']

    # Use one-hot encoding on identified categorical features
    data_encoded = pd.get_dummies(data, columns=cat_features, drop_first=True)

    #----------------------------------
    #Model Building

    #----------------------------------
    # Feature Selection
    #---------------------------------- 
    # Drop unwanted columns from the original dataframe
    # 'id' is excluded because it's not a predictive feature (just an identifier)
    # 'Depression' is the target variable
    # Other categorical columns will be encoded separately
    # Dynamically prepare feature matrix by dropping non-feature columns
    non_feature_cols = ['id', 'Depression']
    
    # Prepare feature matrix (X) and target variable (y)
    X = data_encoded.drop(columns=non_feature_cols, errors='ignore')
    y = data['Depression']

    # Dynamically standardize numerical features
    scaler = StandardScaler()
    # Identify numerical columns in the feature matrix (excluding encoded categorical features)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude dummy/encoded categorical columns (they typically have _suffix)
    numeric_features = [col for col in numeric_features if not any(suffix in col for suffix in ['_Yes', '_No', '_Male', '_Female', '_True', '_False'])]
    
    if numeric_features:
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

    #---------------------------------- 
    # Split Data into Training and Testing Sets using config parameters
    training_params = config.get_training_params()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=training_params['test_size'], 
        random_state=training_params['random_state']
    )

    # Handle class imbalance on training data
    X_train_balanced, y_train_balanced, class_weights = handle_class_imbalance(
        X_train, y_train, method=imbalance_method, random_state=42
    )

    # Logistic Regression with class weights if applicable
    if class_weights is not None:
        log_model = LogisticRegression(
            max_iter=training_params['logistic_max_iter'], 
            class_weight=class_weights
        )
    else:
        log_model = LogisticRegression(max_iter=training_params['logistic_max_iter'])
    
    log_model.fit(X_train_balanced, y_train_balanced)
    # Predictions and evaluation
    y_pred_log = log_model.predict(X_test)

    # Random Forest Classifier with class weights if applicable
    if class_weights is not None:
        rf_model = RandomForestClassifier(
            n_estimators=training_params['rf_n_estimators'], 
            random_state=training_params['random_state'], 
            class_weight=class_weights
        )
    else:
        rf_model = RandomForestClassifier(
            n_estimators=training_params['rf_n_estimators'], 
            random_state=training_params['random_state']
        )
    
    rf_model.fit(X_train_balanced, y_train_balanced)
    # Predictions and evaluation
    y_pred_rf = rf_model.predict(X_test)

    # Model Performance Tracking: Save metrics using config paths
    from sklearn.metrics import accuracy_score, f1_score
    metrics_path = config.get_metrics_path()
    with open(metrics_path, 'w') as f:
        f.write('Random Forest Classification Report:\n')
        f.write(classification_report(y_test, y_pred_rf))
        f.write('\nRandom Forest Confusion Matrix:\n')
        f.write(str(confusion_matrix(y_test, y_pred_rf)))
        f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred_rf):.4f}\n")
        f.write(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}\n")
        
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

    # Save the trained model and preprocessing info using config paths
    joblib.dump(rf_model, config.get_model_path())
    
    # Calculate performance metrics for auto-tuning
    accuracy = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf)
    
    # Save preprocessing information AND training metadata for prediction pipeline
    training_metadata = {
        'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_data_rows': len(data),
        'last_training_row_id': data['id'].max() if 'id' in data.columns else len(data),
        'feature_columns_used': list(X.columns),
        'verified_labels_only': True,  # Flag to indicate we only used verified labels
        'model_version': '1.0',
        'performance': {'accuracy': accuracy, 'f1_score': f1}
    }
    
    preprocessing_info = {
        'scaler': scaler,
        'categorical_cols': categorical_cols,
        'numeric_features': numeric_features,
        'feature_columns': list(X.columns),
        'class_weights': class_weights,
        'training_metadata': training_metadata
    }
    joblib.dump(preprocessing_info, config.get_preprocessing_path())
    
    # Record performance for auto-tuning
    config.record_performance(
        accuracy=accuracy,
        f1_score=f1,
        retrain_triggered=True,  # This training was triggered
        samples_added=len(verified_data),
        method_used=imbalance_method
    )
    
    return rf_model  # Return the trained model if needed

def should_retrain(threshold=None):
    """
    Check if model should be retrained based on new VERIFIED data availability
    Only counts human-verified labels, never model predictions
    """
    if threshold is None:
        threshold = config.get_retrain_threshold()
    
    try:
        master_path = config.get_data_path()
        if not os.path.exists(master_path):
            return False
        
        df = pd.read_csv(master_path)
        
        # CRITICAL: Only count rows with VERIFIED labels (not model predictions)
        verified_df = df.dropna(subset=['Depression'])
        
        # Additional safety: exclude rows that were auto-labeled by our model
        if 'Depression_Pred' in df.columns:
            # Only count rows where Depression was NOT predicted (i.e., real ground truth)
            verified_df = verified_df[verified_df['Depression_Pred'].isna() | 
                                    (verified_df['Depression'] != verified_df['Depression_Pred'])]
        
        current_verified_count = len(verified_df)
        
        # Get last retrain count from training metadata
        try:
            preprocessing_info = joblib.load(config.get_preprocessing_path())
            training_metadata = preprocessing_info.get('training_metadata', {})
            last_count = training_metadata.get('training_data_rows', 0)
        except:
            # Fallback to old config method
            config_path = config.get_retrain_config_path()
            last_count = 0
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    last_count = int(f.read().strip())
        
        return (current_verified_count - last_count) >= threshold
    except:
        return False

def trigger_retrain():
    """
    Trigger function that decides when to retrain using performance monitoring
    """
    try:
    # if should_retrain_smart(method='performance'):
        train_model()
        update_retrain_tracking()
        return True
    except Exception as e:
        print(f"Retraining error: {e}")
        return False
    # else:
    #     return False
def update_retrain_tracking():
    """
    Update tracking information after retraining
    Uses training metadata instead of simple count
    """
    try:
        # The tracking is now done in training metadata saved with preprocessing_info
        # This function is kept for backward compatibility
        master_path = config.get_data_path()
        if os.path.exists(master_path):
            df = pd.read_csv(master_path)
            
            # Count only verified labels
            verified_df = df.dropna(subset=['Depression'])
            if 'Depression_Pred' in df.columns:
                verified_df = verified_df[verified_df['Depression_Pred'].isna() | 
                                        (verified_df['Depression'] != verified_df['Depression_Pred'])]
            
            current_verified_count = len(verified_df)
            
            config_path = config.get_retrain_config_path()
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                f.write(str(current_verified_count))
    except Exception as e:
        pass

# def chi_square_retrain_test(p_threshold=None):
#     """
#     Use chi-square test to determine if new data distribution differs significantly 
#     from training data, indicating need for retraining
#     """
#     if p_threshold is None:
#         p_threshold, _ = config.get_drift_thresholds()
        
#     try:
#         master_path = config.get_data_path()
#         if not os.path.exists(master_path):
#             return False
        
#         df = pd.read_csv(master_path)
#         labeled_df = df.dropna(subset=['Depression'])
        
#         # Get last retrain count to identify "new" vs "old" data
#         config_path = config.get_retrain_config_path()
#         last_count = 0
#         if os.path.exists(config_path):
#             with open(config_path, 'r') as f:
#                 last_count = int(f.read().strip())
        
#         if len(labeled_df) <= last_count + 10:  # Need minimum new data
#             return False
            
#         # Split into old training data vs new data
#         old_data = labeled_df.iloc[:last_count]
#         new_data = labeled_df.iloc[last_count:]
        
#         # Test on categorical features
#         categorical_cols = []
#         for col in df.columns:
#             if col not in ['Depression', 'id'] and df[col].dtype == 'object':
#                 categorical_cols.append(col)
        
#         significant_changes = 0
#         total_tests = 0
        
#         for col in categorical_cols:
#             if col in old_data.columns and col in new_data.columns:
#                 # Create contingency table
#                 old_counts = old_data[col].value_counts()
#                 new_counts = new_data[col].value_counts()
                
#                 # Align indices (same categories)
#                 all_categories = set(old_counts.index) | set(new_counts.index)
#                 old_aligned = [old_counts.get(cat, 0) for cat in all_categories]
#                 new_aligned = [new_counts.get(cat, 0) for cat in all_categories]
                
#                 # Skip if too few samples
#                 if sum(old_aligned) < 5 or sum(new_aligned) < 5:
#                     continue
                    
#                 # Chi-square test
#                 contingency_table = np.array([old_aligned, new_aligned])
#                 if contingency_table.sum() > 0:
#                     chi2, p_value, dof, expected = chi2_contingency(contingency_table)
#                     total_tests += 1
                    
#                     if p_value < p_threshold:
#                         significant_changes += 1
        
#         # Retrain if significant portion of features show distribution shift
#         if total_tests > 0 and (significant_changes / total_tests) >= 0.3:
#             return True
            
#         return False
        
#     except Exception as e:
#         # Fallback to count-based method
#         return should_retrain()

# def check_scipy_dependency():
#     """Check if scipy is available for chi-square test"""
#     try:
#         from scipy.stats import chi2_contingency
#         return True
#     except ImportError:
#         return False

# def should_retrain_smart(method='hybrid', count_threshold=None):
#     """
#     Smart retraining decision with multiple methods:
#     - 'distribution': Chi-square test on data distribution changes
#     - 'performance': Model performance degradation (accuracy/F1 drop)
#     - 'hybrid': Both distribution and performance checks
#     - 'count': Traditional count-based threshold
#     """
#     if count_threshold is None:
#         count_threshold = config.get_retrain_threshold()
        
#     if not check_scipy_dependency():
#         return should_retrain(count_threshold)
    
#     if method == 'distribution':
#         return chi_square_retrain_test() or should_retrain(count_threshold)
#     elif method == 'performance':
#         return performance_based_retrain_test() or should_retrain(count_threshold)
#     elif method == 'hybrid':
#         return should_retrain_hybrid()
#     else:  # count method
#         return should_retrain(count_threshold)

# def performance_based_retrain_test(performance_threshold=None):
#     """
#     Use model performance degradation to determine if retraining is needed
#     Compare current predictions vs VERIFIED labels only (never model predictions)
#     """
#     if performance_threshold is None:
#         performance_threshold = config.get_performance_threshold()
        
#     try:
#         master_path = config.get_data_path()
#         model_path = config.get_model_path()
#         if not os.path.exists(master_path) or not os.path.exists(model_path):
#             return False
        
#         df = pd.read_csv(master_path)
        
#         # CRITICAL: Only use rows with VERIFIED labels (never model predictions)
#         verified_df = df.dropna(subset=['Depression'])
        
#         # Additional safety: exclude rows that were auto-labeled by our model
#         if 'Depression_Pred' in df.columns:
#             # Only use rows where Depression was NOT predicted (i.e., real ground truth)
#             verified_df = verified_df[verified_df['Depression_Pred'].isna() | 
#                                     (verified_df['Depression'] != verified_df['Depression_Pred'])]
        
#         # Get training metadata to know what was used for last training
#         try:
#             preprocessing_info = joblib.load(config.get_preprocessing_path())
#             training_metadata = preprocessing_info.get('training_metadata', {})
#             last_training_row_id = training_metadata.get('last_training_row_id', 0)
#         except:
#             # Fallback: use retrain config
#             config_path = config.get_retrain_config_path()
#             last_training_row_id = 0
#             if os.path.exists(config_path):
#                 with open(config_path, 'r') as f:
#                     last_training_row_id = int(f.read().strip())
        
#         if len(verified_df) <= last_training_row_id + 10:  # Need minimum new verified data
#             return False
        
#         # Get new VERIFIED data since last retrain
#         if 'id' in verified_df.columns:
#             new_verified_data = verified_df[verified_df['id'] > last_training_row_id]
#         else:
#             new_verified_data = verified_df.iloc[last_training_row_id:]
        
#         if len(new_verified_data) < 10:  # Need minimum samples for reliable test
#             return False
        
#         # Load current model
#         model = joblib.load(config.get_model_path())
        
#         # Prepare features (minimal preprocessing for performance test)
#         # Note: This is a simplified version - in production you'd want full preprocessing
#         X_new = new_verified_data.drop(columns=['id', 'Depression'], errors='ignore')
        
#         # Handle categorical columns - convert to same format as training
#         for col in X_new.columns:
#             if X_new[col].dtype == 'object':
#                 X_new[col] = X_new[col].astype('category')
        
#         # Simple encoding for categorical columns (basic version)
#         X_new_encoded = pd.get_dummies(X_new, drop_first=True)
        
#         y_true = new_verified_data['Depression'].astype(int)
        
#         # Make predictions
#         try:
#             y_pred = model.predict(X_new_encoded)
#         except Exception:
#             # If prediction fails due to feature mismatch, fallback to count-based
#             return False
        
#         # Calculate performance metrics
#         from sklearn.metrics import accuracy_score, f1_score
#         current_accuracy = accuracy_score(y_true, y_pred)
#         current_f1 = f1_score(y_true, y_pred)
        
#         # Get baseline performance from last training
#         baseline_accuracy, baseline_f1 = get_baseline_performance()
        
#         # Check if performance dropped significantly
#         accuracy_drop = baseline_accuracy - current_accuracy
#         f1_drop = baseline_f1 - current_f1
        
#         if accuracy_drop > performance_threshold or f1_drop > performance_threshold:
#             return True
            
#         return False
        
#     except Exception as e:
#         return False

# def get_baseline_performance():
#     """Get baseline performance metrics from last training"""
#     try:
#         metrics_path = config.get_metrics_path()
#         if os.path.exists(metrics_path):
#             with open(metrics_path, 'r') as f:
#                 content = f.read()
#                 # Extract accuracy and F1 from saved metrics
#                 import re
#                 accuracy_match = re.search(r'Accuracy: ([\d.]+)', content)
#                 f1_match = re.search(r'F1 Score: ([\d.]+)', content)
                
#                 if accuracy_match and f1_match:
#                     return float(accuracy_match.group(1)), float(f1_match.group(1))
        
#         # Default baseline if no metrics available
#         return 0.8, 0.8
#     except:
#         return 0.8, 0.8

# def should_retrain_hybrid(use_performance=True, use_distribution=True):
#     """
#     Hybrid approach: retrain based on both performance drop AND distribution changes
#     """
#     performance_trigger = False
#     distribution_trigger = False
    
#     if use_performance and check_scipy_dependency():
#         performance_trigger = performance_based_retrain_test()
    
#     if use_distribution and check_scipy_dependency():
#         distribution_trigger = chi_square_retrain_test()
    
#     # Retrain if either trigger fires
#     return performance_trigger or distribution_trigger or should_retrain(50)

# def predict_depression(data_df):
#     """
#     Make predictions on new data using the trained model.
    
#     Args:
#         data_df: pandas DataFrame with features (excluding Depression column)
        
#     Returns:
#         predictions: list of 0/1 predictions
#         probabilities: list of probability scores
#     """
#     try:
#         # Load the model and preprocessing info
#         model = joblib.load(config.get_model_path())
#         preprocessing_info = joblib.load(config.get_preprocessing_path())
        
#         # Make a copy of input data
#         df = data_df.copy()
        
#         # Remove ID column if present
#         df = df.drop(columns=['id'], errors='ignore')
        
#         # Apply the same preprocessing as training
#         scaler = preprocessing_info['scaler']
#         categorical_cols = preprocessing_info['categorical_cols']
#         numeric_features = preprocessing_info['numeric_features']
        
#         # Handle missing values and data types
#         for col in df.columns:
#             if col in categorical_cols:
#                 df[col] = df[col].astype('category')
        
#         # Extract numeric hours from Sleep Duration if present
#         if 'Sleep Duration' in df.columns:
#             def extract_hours(s):
#                 match = re.search(r"(\d+(\.\d+)?)", str(s))
#                 return float(match.group(1)) if match else np.nan
#             df['Sleep Duration'] = df['Sleep Duration'].apply(extract_hours)
        
#         # Convert Financial Stress to categorical if present
#         if 'Financial Stress' in df.columns:
#             df['Financial Stress'] = df['Financial Stress'].astype('category')
        
#         # Impute missing values for numerical columns
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         for col in numeric_cols:
#             if df[col].isnull().sum() > 0:
#                 df[col] = df[col].fillna(df[col].median())
        
#         # Get categorical features for encoding
#         cat_features = [col for col in categorical_cols if col in df.columns]
        
#         # One-hot encode categorical features
#         df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)
        
#         # Standardize numerical features
#         if numeric_features:
#             # Only scale features that exist in both training and prediction data
#             features_to_scale = [col for col in numeric_features if col in df_encoded.columns]
#             if features_to_scale:
#                 df_encoded[features_to_scale] = scaler.transform(df_encoded[features_to_scale])
        
#         # Ensure all training features are present
#         training_features = preprocessing_info['feature_columns']
#         missing_features = [col for col in training_features if col not in df_encoded.columns]
        
#         # Add missing columns efficiently using pd.concat instead of loop
#         if missing_features:
#             missing_df = pd.DataFrame(0, index=df_encoded.index, columns=missing_features)
#             df_encoded = pd.concat([df_encoded, missing_df], axis=1)
        
#         # Select only training features in correct order
#         df_final = df_encoded[training_features]
        
#         # Make predictions
#         predictions = model.predict(df_final)
#         probabilities = model.predict_proba(df_final)[:, 1]  # Probability of class 1 (Depression)
        
#         return predictions.tolist(), probabilities.tolist()
        
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         # Fallback to simple rule-based prediction
#         predictions = [0] * len(data_df)  # Default to no depression
#         probabilities = [0.5] * len(data_df)  # Neutral probability
#         return predictions, probabilities