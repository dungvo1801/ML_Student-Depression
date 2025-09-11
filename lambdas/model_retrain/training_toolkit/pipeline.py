
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


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
