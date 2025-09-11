import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


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
