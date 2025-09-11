from dotenv import load_dotenv
import os
import pandas as pd
import boto3
import io

# Load environment variables from .env file
load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'dbname': os.getenv('DB_NAME')
}

def download_bytes() -> bytes:
    try:
        # session = boto3.Session(
        #     aws_access_key_id=os.getenv('AWS_KEY'),
        #     aws_secret_access_key=os.getenv('AWS_SECRET'),
        #     region_name=os.getenv('REGION_NAME')
        # )
        s3 = boto3.client('s3')
        bucket_name=os.getenv('BUCKET_NAME')
        s3_key=os.getenv('METRICS_KEY')

        file_obj = io.BytesIO()
        s3.download_fileobj(bucket_name, s3_key, file_obj)
        file_obj.seek(0)  # Reset to start of buffer

        print(f"Downloaded s3://{bucket_name}/{s3_key} to bytes")
        return file_obj.read()
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def upload_bytes(s3_key, file_bytes: bytes):
    # session = boto3.Session(
    #     aws_access_key_id=os.getenv('AWS_KEY'),
    #     aws_secret_access_key=os.getenv('AWS_SECRET'),
    #     region_name=os.getenv('REGION_NAME')
    # )
    s3 = boto3.client('s3')
    bucket_name=os.getenv('BUCKET_NAME')
    try:
        file_obj = io.BytesIO(file_bytes)
        s3.upload_fileobj(file_obj, bucket_name, s3_key)
        print(f"Uploaded bytes to s3://{bucket_name}/{s3_key}")
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def get_validated_data(cursor):
    print("Fetching predictions...")

    cursor.execute('SELECT * FROM predictions WHERE "Validation" IS NOT NULL;')
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    print(f"Retrieved {len(rows)} rows from predictions")

    df = pd.DataFrame(rows, columns=columns)
    df.drop(columns=[col for col in ["timestamp", "Depression"] if col in df.columns], inplace=True)
    if "Validation" in df.columns:
        df.rename(columns={"Validation": "Depression"}, inplace=True)

    return df

def get_master_data(cursor):
    print("Fetching master data...")
    cursor.execute("SELECT * FROM student_depression_master limit 4;")
    
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    
    print(f"Retrieved {len(rows)} rows from master data")
    
    df = pd.DataFrame(rows, columns=columns)
    return df

