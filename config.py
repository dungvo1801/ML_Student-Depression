"""
Configuration management for Student Depression Prediction System
Centralizes all paths, thresholds, and parameters with validation and auto-tuning capabilities
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import joblib

class Config:
    """
    Centralized configuration management with dynamic threshold tuning
    """
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration with sensible starting values"""
        return {
            # File Paths
            "paths": {
                "data_dir": "models",
                "logs_dir": "logs", 
                "uploads_dir": "uploads",
                "versions_dir": "models/versions",
                "dataset_file": "student_depression_dataset.csv",
                "model_file": "rf_model.pkl",
                "preprocessing_file": "preprocessing_info.pkl",
                "logistic_model_file": "log_model.pkl",
                "metrics_file": "model_metrics.txt",
                "retrain_config_file": "retrain_config.txt",
                "upload_history_file": "upload_history.csv",
                "login_history_file": "login_history.csv"
            },
            
            # Kaggle Configuration
            "kaggle": {
                "dataset": "ngocdung/student-depression-dataset",
                "filename": "student_depression_dataset.csv"
            },
            
            # Model Training Parameters
            "training": {
                "test_size": 0.2,
                "random_state": 42,
                "rf_n_estimators": 100,
                "logistic_max_iter": 1000,
                "min_training_samples": 10,
                "imbalance_method": "class_weight"
            },
            
            # Data Validation Thresholds
            "validation": {
                "numeric_conversion_threshold": 0.7,  # 70% of values must convert to numeric
                "age_max": 120,
                "sleep_max": 24,
                "stress_max": 10,
                "age_min": 0,
                "sleep_min": 0,
                "stress_min": 0
            },
            
            # Retraining Thresholds (ADAPTIVE - will be auto-tuned)
            "retraining": {
                "count_threshold": 50,           # Start conservative, will adapt
                "performance_threshold": 0.05,   # Start conservative, will adapt  
                "drift_p_threshold": 0.05,       # Statistical significance
                "drift_feature_ratio": 0.3,      # 30% of features show drift
                "min_new_samples": 10,           # Minimum for reliable evaluation
                "auto_tune": True,               # Enable auto-tuning
                "performance_history_limit": 10  # Track last N performance measurements
            },
            
            # Auto-tuning Configuration
            "auto_tune": {
                "enabled": True,
                "performance_window": 5,         # Consider last 5 retraining cycles
                "count_threshold_range": [20, 100],      # Auto-adjust between these values
                "performance_threshold_range": [0.02, 0.10],  # Auto-adjust between these values
                "adaptation_rate": 0.1,          # How fast to adapt (10% adjustment per cycle)
                "min_accuracy_target": 0.80,     # Don't retrain if accuracy is still above this
                "stability_threshold": 3         # Need 3 consecutive stable cycles to increase thresholds
            },
            
            # Performance History (for auto-tuning)
            "performance_history": [],
            
            # Metadata
            "metadata": {
                "config_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "auto_tuned_count": 0
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults to handle new keys
                default_config = self._get_default_config()
                return self._merge_configs(default_config, config)
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}. Using defaults.")
                return self._get_default_config()
        else:
            # Create default config file
            default_config = self._get_default_config()
            self.save_config(default_config)
            return default_config
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge loaded config with defaults"""
        for key, value in default.items():
            if key not in loaded:
                loaded[key] = value
            elif isinstance(value, dict) and isinstance(loaded[key], dict):
                loaded[key] = self._merge_configs(value, loaded[key])
        return loaded
    
    def save_config(self, config: Optional[Dict] = None):
        """Save current configuration to file"""
        config_to_save = config or self.config
        config_to_save["metadata"]["last_updated"] = datetime.now().isoformat()
        
        try:
            os.makedirs(os.path.dirname(self.config_file) if os.path.dirname(self.config_file) else '.', exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save config: {e}")
    
    # Path getters (with automatic directory creation)
    def get_data_path(self) -> str:
        """Get full path to dataset file"""
        path = os.path.join(self.config["paths"]["data_dir"], self.config["paths"]["dataset_file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    def get_model_path(self) -> str:
        """Get full path to model file"""
        path = os.path.join(self.config["paths"]["data_dir"], self.config["paths"]["model_file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    def get_preprocessing_path(self) -> str:
        """Get full path to preprocessing file"""
        path = os.path.join(self.config["paths"]["data_dir"], self.config["paths"]["preprocessing_file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    def get_metrics_path(self) -> str:
        """Get full path to metrics file"""
        path = os.path.join(self.config["paths"]["logs_dir"], self.config["paths"]["metrics_file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    def get_retrain_config_path(self) -> str:
        """Get full path to retrain config file"""
        path = os.path.join(self.config["paths"]["logs_dir"], self.config["paths"]["retrain_config_file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    def get_versions_dir(self) -> str:
        """Get versions directory path"""
        path = self.config["paths"]["versions_dir"]
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_upload_history_path(self) -> str:
        """Get upload history file path"""
        path = os.path.join(self.config["paths"]["logs_dir"], self.config["paths"]["upload_history_file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    def get_login_history_path(self) -> str:
        """Get login history file path"""
        path = os.path.join(self.config["paths"]["logs_dir"], self.config["paths"]["login_history_file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    # Threshold getters
    def get_retrain_threshold(self) -> int:
        """Get current retrain count threshold (auto-tuned)"""
        return self.config["retraining"]["count_threshold"]
    
    def get_performance_threshold(self) -> float:
        """Get current performance threshold (auto-tuned)"""
        return self.config["retraining"]["performance_threshold"]
    
    def get_drift_thresholds(self) -> tuple:
        """Get drift detection thresholds"""
        return (
            self.config["retraining"]["drift_p_threshold"],
            self.config["retraining"]["drift_feature_ratio"]
        )
    
    # Auto-tuning methods
    def record_performance(self, accuracy: float, f1_score: float, retrain_triggered: bool, 
                          samples_added: int, method_used: str):
        """Record performance metrics for auto-tuning"""
        if not self.config["auto_tune"]["enabled"]:
            return
            
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "f1_score": f1_score,
            "retrain_triggered": retrain_triggered,
            "samples_added": samples_added,
            "method_used": method_used,
            "count_threshold_used": self.get_retrain_threshold(),
            "performance_threshold_used": self.get_performance_threshold()
        }
        
        # Add to history
        self.config["performance_history"].append(performance_record)
        
        # Keep only recent history
        max_history = self.config["auto_tune"]["performance_window"] * 2
        if len(self.config["performance_history"]) > max_history:
            self.config["performance_history"] = self.config["performance_history"][-max_history:]
        
        # Auto-tune thresholds
        self._auto_tune_thresholds()
        self.save_config()
    
    def _auto_tune_thresholds(self):
        """Automatically tune retraining thresholds based on performance history"""
        if not self.config["auto_tune"]["enabled"]:
            return
            
        history = self.config["performance_history"]
        if len(history) < 3:  # Need minimum history
            return
        
        recent_history = history[-self.config["auto_tune"]["performance_window"]:]
        
        # Analyze recent performance
        avg_accuracy = sum(r["accuracy"] for r in recent_history) / len(recent_history)
        retrain_frequency = sum(1 for r in recent_history if r["retrain_triggered"]) / len(recent_history)
        
        adaptation_rate = self.config["auto_tune"]["adaptation_rate"]
        count_range = self.config["auto_tune"]["count_threshold_range"]
        perf_range = self.config["auto_tune"]["performance_threshold_range"]
        min_accuracy = self.config["auto_tune"]["min_accuracy_target"]
        
        # Adaptive logic
        current_count_threshold = self.config["retraining"]["count_threshold"]
        current_perf_threshold = self.config["retraining"]["performance_threshold"]
        
        # If accuracy is high and retraining too frequent, increase thresholds
        if avg_accuracy > min_accuracy + 0.05 and retrain_frequency > 0.6:
            new_count_threshold = min(count_range[1], current_count_threshold * (1 + adaptation_rate))
            new_perf_threshold = min(perf_range[1], current_perf_threshold * (1 + adaptation_rate))
            
        # If accuracy is dropping, decrease thresholds (retrain more aggressively)
        elif avg_accuracy < min_accuracy:
            new_count_threshold = max(count_range[0], current_count_threshold * (1 - adaptation_rate))
            new_perf_threshold = max(perf_range[0], current_perf_threshold * (1 - adaptation_rate))
            
        # If retraining too rarely and accuracy is not great, decrease count threshold
        elif retrain_frequency < 0.2 and avg_accuracy < min_accuracy + 0.02:
            new_count_threshold = max(count_range[0], current_count_threshold * (1 - adaptation_rate))
            new_perf_threshold = current_perf_threshold  # Keep same
            
        else:
            return  # No adjustment needed
        
        # Apply changes
        if abs(new_count_threshold - current_count_threshold) > 1:  # Significant change
            self.config["retraining"]["count_threshold"] = int(new_count_threshold)
            self.config["metadata"]["auto_tuned_count"] += 1
            print(f"Auto-tuned count threshold: {current_count_threshold} → {int(new_count_threshold)}")
        
        if abs(new_perf_threshold - current_perf_threshold) > 0.005:  # Significant change
            self.config["retraining"]["performance_threshold"] = round(new_perf_threshold, 4)
            self.config["metadata"]["auto_tuned_count"] += 1
            print(f"Auto-tuned performance threshold: {current_perf_threshold} → {round(new_perf_threshold, 4)}")
    
    def get_kaggle_config(self) -> tuple:
        """Get Kaggle dataset configuration"""
        return (
            self.config["kaggle"]["dataset"],
            self.config["kaggle"]["filename"]
        )
    
    def get_validation_limits(self) -> Dict[str, tuple]:
        """Get data validation limits"""
        val = self.config["validation"]
        return {
            "age": (val["age_min"], val["age_max"]),
            "sleep": (val["sleep_min"], val["sleep_max"]),
            "stress": (val["stress_min"], val["stress_max"])
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters"""
        return self.config["training"]
    
    def update_config(self, section: str, key: str, value: Any):
        """Update a specific configuration value"""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            self.save_config()
        else:
            raise KeyError(f"Configuration key {section}.{key} not found")

# Global configuration instance
config = Config()

# Convenience functions for backward compatibility
def get_data_path() -> str:
    return config.get_data_path()

def get_model_path() -> str:
    return config.get_model_path()

def get_preprocessing_path() -> str:
    return config.get_preprocessing_path()

def get_retrain_threshold() -> int:
    return config.get_retrain_threshold()

def get_performance_threshold() -> float:
    return config.get_performance_threshold()
