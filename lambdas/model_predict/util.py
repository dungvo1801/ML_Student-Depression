from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import re
import boto3
import io
import pickle  

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


def insert_prediction(cursor, df):
    print(f"Inserting new prediction record: {df}")
    cursor.execute("""
        INSERT INTO predictions (user, filename, input_data, prediction, timestamp) VALUES (%s,%s,%s,%s,%s);
    """, (df,))
    print("Record inserted.")
    


def predict_depression(data_df):
    """
    Make predictions on new data using the trained model.
    
    Args:
        data_df: pandas DataFrame with features (excluding Depression column)
        
    Returns:
        predictions: list of 0/1 predictions
        probabilities: list of probability scores
    """
    try:

        # Download model and preprocessing info from S3
        model_bytes = download_bytes(os.getenv('RF_MODEL_KEY'))
        preprocessing_bytes = download_bytes(os.getenv('PREPROCESSING_MODEL_KEY'))

        if model_bytes is None or preprocessing_bytes is None:
            raise Exception("Failed to download model or preprocessing files from S3")

        # Deserialize the downloaded bytes into Python objects
        model = pickle.loads(model_bytes)
        preprocessing_info = pickle.loads(preprocessing_bytes)

        # Make a copy of input data
        df = data_df.copy()
        
        # Remove ID column if present
        df = df.drop(columns=['id'], errors='ignore')
        
        # Apply the same preprocessing as training
        scaler = preprocessing_info['scaler']
        categorical_cols = preprocessing_info['categorical_cols']
        numeric_features = preprocessing_info['numeric_features']
        
        # Handle missing values and data types
        for col in df.columns:
            if col in categorical_cols:
                df[col] = df[col].astype('category')
        
        # Extract numeric hours from Sleep Duration if present
        if 'Sleep Duration' in df.columns:
            def extract_hours(s):
                match = re.search(r"(\d+(\.\d+)?)", str(s))
                return float(match.group(1)) if match else np.nan
            df['Sleep Duration'] = df['Sleep Duration'].apply(extract_hours)
        
        # Convert Financial Stress to categorical if present
        if 'Financial Stress' in df.columns:
            df['Financial Stress'] = df['Financial Stress'].astype('category')
        
        # Impute missing values for numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Get categorical features for encoding
        cat_features = [col for col in categorical_cols if col in df.columns]
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)
        
        # Standardize numerical features
        if numeric_features:
            # Only scale features that exist in both training and prediction data
            features_to_scale = [col for col in numeric_features if col in df_encoded.columns]
            if features_to_scale:
                df_encoded[features_to_scale] = scaler.transform(df_encoded[features_to_scale])
        
        # Ensure all training features are present
        training_features = preprocessing_info['feature_columns']
        missing_features = [col for col in training_features if col not in df_encoded.columns]
        
        # Add missing columns efficiently using pd.concat instead of loop
        if missing_features:
            missing_df = pd.DataFrame(0, index=df_encoded.index, columns=missing_features)
            df_encoded = pd.concat([df_encoded, missing_df], axis=1)
        
        # Select only training features in correct order
        df_final = df_encoded[training_features]
        
        # Make predictions
        predictions = model.predict(df_final)
        probabilities = model.predict_proba(df_final)[:, 1]  # Probability of class 1 (Depression)
        
        return predictions.tolist(), probabilities.tolist()
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Fallback to simple rule-based prediction
        predictions = [0] * len(data_df)  # Default to no depression
        probabilities = [0.5] * len(data_df)  # Neutral probability
        return predictions, probabilities