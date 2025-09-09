import pandas as pd
import numpy as np
import os
import sys
import boto3
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

config = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'dbname': os.getenv('DB_NAME')
}

validation_config = {
    "numeric_conversion_threshold": 0.7,  # 70% of values must convert to numeric
    "age_max": 120,
    "sleep_max": 24,
    "stress_max": 10,
    "age_min": 0,
    "sleep_min": 0,
    "stress_min": 0
}

def upload_bytes(s3_key, file_bytes: bytes):
    # session = boto3.Session(
    #     aws_access_key_id=os.getenv('AWS_KEY'),
    #     aws_secret_access_key=os.getenv('AWS_SECRET'),
    #     region_name=os.getenv('REGION_NAME')
    # )
    s3 = boto3.client('s3')
    bucket_name=os.getenv('BUCKET_NAME')
    folder=os.getenv('FOLDER', "")
    try:
        file_obj = io.BytesIO(file_bytes)
        s3.upload_fileobj(file_obj, bucket_name, f"{folder}/{s3_key}")
        print(f"Uploaded bytes to s3://{bucket_name}/{folder}/{s3_key}")
    except Exception as e:
        print(f"Upload failed: {e}")

def insert_upload_log(cursor, log):
    print(f"Inserting new log record for user: {log}")
    cursor.execute("""
        INSERT INTO upload_log ("user", "filename") VALUES (%s,%s);
    """, log)
    print("Record inserted.")

def get_master_data(cursor):
    print("Fetching user logs...")
    cursor.execute("""
        SELECT * FROM student_depression_master;
    """)
    # Convert to DataFrame
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    
    print(f"Retrieved {len(df)} rows from 'student_depression_master'.")
    return df
    
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
        validation_limits = {
            "age": (validation_config["age_min"], validation_config["age_max"]),
            "sleep": (validation_config["sleep_min"], validation_config["sleep_max"]),
            "stress": (validation_config["stress_min"], validation_config["stress_max"])
        }
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