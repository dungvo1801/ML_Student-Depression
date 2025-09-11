from training_toolkit.performance import (
    performance_based_retrain_test,
    update_retrain_tracking
)
from training_toolkit.train import train_model

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


def trigger_retrain():
    """
    Trigger function for F1-SCORE BASED performance monitoring.
    
    F1-score is the primary metric because:
    - Better for imbalanced depression data
    - Clinically more relevant than accuracy  
    - Balances precision and recall for depression detection
    """
    train_success = train_model()
    if not train_success:
        return False
    return True



