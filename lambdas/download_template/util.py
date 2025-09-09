import boto3
import io
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


def download_bytes() -> bytes:
    try:
        # session = boto3.Session(
        #     aws_access_key_id=os.getenv('AWS_KEY'),
        #     aws_secret_access_key=os.getenv('AWS_SECRET'),
        #     region_name=os.getenv('REGION_NAME')
        # )
        s3 = boto3.client('s3')
        bucket_name=os.getenv('BUCKET_NAME')
        s3_key=os.getenv('TEMPLATE_NAME')

        file_obj = io.BytesIO()
        s3.download_fileobj(bucket_name, s3_key, file_obj)
        file_obj.seek(0)  # Reset to start of buffer

        print(f"Downloaded s3://{bucket_name}/{s3_key} to bytes")
        return file_obj.read()
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def upload_bytes(s3_key, file_bytes: bytes):
    s3 = boto3.client('s3')
    bucket_name=os.getenv('BUCKET_NAME')
    try:
        file_obj = io.BytesIO(file_bytes)
        s3.upload_fileobj(file_obj, bucket_name, s3_key)
        print(f"Uploaded bytes to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Upload failed: {e}")



