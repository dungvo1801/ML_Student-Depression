import pandas as pd
import numpy as np
import re
import joblib
import os
from config import config
from util import (
    download_bytes
)

from train import train_model
from transformation import safe_roc_auc

from dotenv import load_dotenv

load_dotenv()

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
        try:
            metrics_bytes = download_bytes()  
            if metrics_bytes is None:
                print("Failed to download metrics report from S3.")
            content = metrics_bytes.decode('utf-8')
            # Use regex to extract Random Forest metrics
            accuracy_match = re.search(r'=== RANDOM FOREST.*?Accuracy: ([\d.]+)', content, re.DOTALL)
            f1_match = re.search(r'=== RANDOM FOREST.*?F1 Score: ([\d.]+)', content, re.DOTALL)
            roc_auc_match = re.search(r'=== RANDOM FOREST.*?ROC-AUC: ([\d.]+)', content, re.DOTALL)

            if accuracy_match and f1_match and roc_auc_match:
                return (
                    float(accuracy_match.group(1)),
                    float(f1_match.group(1)),
                    float(roc_auc_match.group(1))
                )
        except UnicodeDecodeError:
            print("Failed to decode metrics report as UTF-8.")
        
        # If no metrics file exists, calculate baseline by retraining
        train_success = train_model()  # This will create the metrics file
        if not train_success:
            print("Failed to download metrics report from S3 after retrain.")
            return None, None, None
        
        try:
            metrics_bytes = download_bytes()  
            if metrics_bytes is None:
                print("Failed to download metrics report from S3 after retrain.")
            content = metrics_bytes.decode('utf-8')
            # Use regex to extract Random Forest metrics
            accuracy_match = re.search(r'=== RANDOM FOREST.*?Accuracy: ([\d.]+)', content, re.DOTALL)
            f1_match = re.search(r'=== RANDOM FOREST.*?F1 Score: ([\d.]+)', content, re.DOTALL)
            roc_auc_match = re.search(r'=== RANDOM FOREST.*?ROC-AUC: ([\d.]+)', content, re.DOTALL)

            if accuracy_match and f1_match and roc_auc_match:
                return (
                    float(accuracy_match.group(1)),
                    float(f1_match.group(1)),
                    float(roc_auc_match.group(1))
                )
        except UnicodeDecodeError:
            print("Failed to decode metrics report as UTF-8 after retrain.")
        
        # If still no metrics, return None to indicate failure
        return None, None, None
    except Exception as e:
        print(f"Failed to download metrics report from S3 after retrain: {e}")
        return None, None, None