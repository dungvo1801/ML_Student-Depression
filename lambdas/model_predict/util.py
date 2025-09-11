from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import re
import boto3
import io
# import pickle  
import joblib

# Load environment variables from .env file
load_dotenv()


def download_bytes(s3_key) -> bytes:
    try:
        # session = boto3.Session(
        #     aws_access_key_id=os.getenv('AWS_KEY'),
        #     aws_secret_access_key=os.getenv('AWS_SECRET'),
        #     region_name=os.getenv('REGION_NAME')
        # )

        s3 = boto3.client('s3')
        bucket_name=os.getenv('BUCKET_NAME')

        file_obj = io.BytesIO()
        s3.download_fileobj(bucket_name, s3_key, file_obj)
        file_obj.seek(0)  # Reset to start of buffer

        print(f"Downloaded s3://{bucket_name}/{s3_key} to bytes")
        return file_obj.read()
    except Exception as e:
        print(f"Download failed: {e}")
        return None

# Load environment variables from .env file
load_dotenv()

config = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'dbname': os.getenv('DB_NAME')
}


# def insert_prediction(cursor, df):
#     print(f"Inserting new prediction record: {df}")
#     cursor.execute("""
#         INSERT INTO predictions (user, filename, input_data, prediction, timestamp) VALUES (%s,%s,%s,%s,%s);
#     """, (df,))
#     print("Record inserted.")
def insert_prediction(cursor, df):
    print(f"Inserting {len(df)} prediction record(s)...")

    # Define column order (excluding 'id' and 'timestamp')
    columns = [
        "Gender", "Age", "City", "Profession", "Academic Pressure",
        "Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction",
        "Sleep Duration", "Dietary Habits", "Degree",
        "Have you ever had suicidal thoughts ?", "Work/Study Hours",
        "Financial Stress", "Family History of Mental Illness", "Depression"
    ]

    # Quote columns with spaces or special characters
    
    quoted_columns = [f'"{col}"' for col in columns]
    quoted_columns.append('"Validation"')

    # Build SQL placeholders for each row
    placeholders = ', '.join(['%s'] * len(quoted_columns))

    # Convert DataFrame to list of tuples, appending `None` for "Validation"
    values = []
    for _, row in df.iterrows():
        row_values = [row.get(col) for col in columns] + [None]
        values.append(tuple(row_values))

    # Build the SQL query
    sql = f"""
        INSERT INTO predictions ({', '.join(quoted_columns)})
        VALUES ({placeholders});
    """

    # Execute many rows at once
    cursor.executemany(sql, values)
    print(f"Inserted {len(values)} record(s) with Validation = None.")


# def predict_depression(X_new, use_ensemble=True):
#     """
#     Make predictions using ensemble method or best individual model.
    
#     Args:
#         X_new: New data to predict
#         use_ensemble: If True, uses weighted ensemble; if False, uses best individual model
    
#     Returns:
#         predictions, probabilities, method_used
#     """
#     try:

#         # # Download model and preprocessing info from S3
#         # model_bytes = download_bytes(os.getenv('RF_MODEL_KEY'))
#         # preprocessing_bytes = download_bytes(os.getenv('PREPROCESSING_MODEL_KEY'))

#         # if model_bytes is None or preprocessing_bytes is None:
#         #     raise Exception("Failed to download model or preprocessing files from S3")

#         # # Deserialize the downloaded bytes into Python objects
#         # model = pickle.loads(model_bytes)
#         # preprocessing_info = pickle.loads(preprocessing_bytes)
        
#         model_bytes = download_bytes(os.getenv('RF_MODEL_KEY'))
#         log_model_bytes =  download_bytes(os.getenv('LOG_MODEL_KEY'))
#         ensemble_model_bytes =  download_bytes(os.getenv('ENSEMBLE_KEY'))
        
#         rf_model = joblib.load(io.BytesIO(model_bytes))
#         log_model = joblib.load(io.BytesIO(log_model_bytes))
#         ensemble_info = joblib.load(io.BytesIO(ensemble_model_bytes))

#         if use_ensemble:
#             # Get predictions from both models
#             rf_proba = rf_model.predict_proba(X_new)[:, 1]
#             log_proba = log_model.predict_proba(X_new)[:, 1]
            
#             # Weighted ensemble
#             ensemble_proba = (ensemble_info['rf_weight'] * rf_proba + 
#                             ensemble_info['log_weight'] * log_proba)
#             ensemble_pred = (ensemble_proba >= ensemble_info['ensemble_threshold']).astype(int)
            
#             return ensemble_pred, ensemble_proba, "Weighted Ensemble"
#         else:
#             # Use best individual model
#             if ensemble_info['best_model'] == 'Random Forest':
#                 pred = rf_model.predict(X_new)
#                 proba = rf_model.predict_proba(X_new)[:, 1]
#                 return pred, proba, "Random Forest"
#             else:
#                 pred = log_model.predict(X_new)
#                 proba = log_model.predict_proba(X_new)[:, 1]
#                 return pred, proba, "Logistic Regression"
                
#     except Exception as e:
#         print(f"Error in ensemble prediction: {e}")
#         return None, None, "Error"


def predict_with_ensemble(X_test, custom_threshold=0.3):
    """
    Make predictions using ensemble of models with fallback mechanisms
    """
    model_bytes = download_bytes(os.getenv('RF_MODEL_KEY'))
    log_model_bytes =  download_bytes(os.getenv('LOG_MODEL_KEY'))
    ensemble_model_bytes =  download_bytes(os.getenv('ENSEMBLE_KEY'))
    
    rf_model = joblib.load(io.BytesIO(model_bytes))
    log_model = joblib.load(io.BytesIO(log_model_bytes))
    ensemble_info = joblib.load(io.BytesIO(ensemble_model_bytes))

    rf_pred = None
    rf_proba = None
    log_pred = None
    log_proba = None
    
    # Get predictions from Random Forest (with fallback)
    if rf_model is not None:
        try:
            rf_pred = rf_model.predict(X_test)
            rf_proba = rf_model.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"Random Forest prediction failed: {e}")
            rf_pred = None
            rf_proba = None
    
    # Get predictions from Logistic Regression (with fallback)
    if log_model is not None:
        try:
            log_pred = log_model.predict(X_test)
            log_proba = log_model.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"Logistic Regression prediction failed: {e}")
            log_pred = None
            log_proba = None
    
    # Determine prediction strategy based on available models
    if rf_proba is not None and log_proba is not None:
        # Both models available - use ensemble
        ensemble_proba = (ensemble_info['rf_weight'] * rf_proba + 
                         ensemble_info['log_weight'] * log_proba)
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        ensemble_pred_custom = (ensemble_proba >= custom_threshold).astype(int)
        prediction_method = "Ensemble (Both Models)"
        
    elif rf_proba is not None:
        # Only Random Forest available
        ensemble_proba = rf_proba
        ensemble_pred = rf_pred
        ensemble_pred_custom = (rf_proba >= custom_threshold).astype(int)
        prediction_method = "Random Forest Only (Fallback)"
        
    elif log_proba is not None:
        # Only Logistic Regression available
        ensemble_proba = log_proba
        ensemble_pred = log_pred
        ensemble_pred_custom = (log_proba >= custom_threshold).astype(int)
        prediction_method = "Logistic Regression Only (Fallback)"
        
    else:
        # No models available
        return [], [], [], "No Models Available"
    
    return ensemble_pred.tolist(), ensemble_proba.tolist(), ensemble_pred_custom.tolist(), prediction_method







