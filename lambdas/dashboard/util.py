import boto3
import io
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2 import OperationalError
from botocore.exceptions import ClientError
 
# Load environment variables from .env file
load_dotenv()


def get_total_records(cursor, table):
    print("Fetching master data...")
    cursor.execute(f"SELECT * FROM {table};")
   
    rows = cursor.fetchall()
    return rows

def get_upload_history(cursor, table):
    print("Fetching upload history...")

    cursor.execute(f"SELECT id, \"user\", \"filename\", time FROM {table} ORDER BY time DESC LIMIT 100;")
   
    rows = cursor.fetchall()
    return rows

def get_login_history(cursor, table):
    # Get last 20 login records
    print("Fetching login history...")

    cursor.execute(f"SELECT id, \"user\", time FROM {table} ORDER BY time DESC LIMIT 20;")

    rows = cursor.fetchall()
    return rows

def get_last_modified(s3_key):
 
    # session = boto3.Session(
    #     aws_access_key_id=os.getenv('AWS_KEY'),
    #     aws_secret_access_key=os.getenv('AWS_SECRET'),
    #     region_name=os.getenv('REGION_NAME')
    # )
 
    s3 = boto3.client('s3')
    bucket_name = os.getenv('BUCKET_NAME')
 
    try:
        response = s3.head_object(Bucket=bucket_name, Key=s3_key)
        # This is a timezone-aware datetime in UTC
        return response['LastModified']
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code')
        if code in ('404', 'NoSuchKey'):
            print(f"Object not found: s3://{bucket_name}/{s3_key}")
            return None
        raise  # bubble up other errors
    
config = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'dbname': os.getenv('DB_NAME')
}
 
def _download_file_from_s3(s3_key: str) -> bytes:
    """Internal helper to download file from S3"""
    try:
        s3 = boto3.client('s3')
        bucket_name = os.getenv('BUCKET_NAME')
        file_obj = io.BytesIO()
        s3.download_fileobj(bucket_name, s3_key, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        print(f"Download failed for {s3_key}: {e}")
        return None
 
def get_dashboard_data():
    """Get dashboard data from S3 files"""
    try:
 
        # Get total records, upload history, login history from PostgreSQL
        total_records = 0
        upload_history = []
        login_history = []
        try:
            connection = psycopg2.connect(**config)
           
            with connection:
                with connection.cursor() as cursor:
                    # Count total records in student_depression_master table
                    total_records =len(get_total_records(cursor, "student_depression_master"))

                    #upload history: get last 100 upload records
                    upload_history_rows = get_upload_history(cursor, "upload_log")
                    # Convert to list of dictionaries similar to CSV format
                    for row in upload_history_rows:
                        upload_history.append({
                            'id': row[0],
                            'user': row[1],
                            'filename': row[2],
                            'time': row[3].strftime('%Y-%m-%d %H:%M:%S') if row[3] else 'N/A'
                        })
                   
                   # Get last 20 login records
                    login_history_rows = get_login_history(cursor, "user_log")

                    # Convert to list of dictionaries similar to CSV format
                    for row in login_history_rows:
                        login_history.append({
                            'id': row[0],
                            'user': row[1],
                            'time': row[2].strftime('%Y-%m-%d %H:%M:%S') if row[2] else 'N/A'
                        })

        except (OperationalError, Exception) as e:
            print(f"Error fetching total records from database: {e}")
            # Fallback to 0 if database is unavailable
            total_records = 0
        finally:
            if 'connection' in locals() and connection:
                connection.close()
 
        # Get last retrain time from model file metadata
        last_retrain = 'N/A'
        try:
            last_retrain_rf = get_last_modified(os.getenv('RF_MODEL_KEY')).strftime('%Y-%m-%d %H:%M:%S')
            last_retrain_log = get_last_modified(os.getenv('LOG_MODEL_KEY')).strftime('%Y-%m-%d %H:%M:%S')
            last_retrain_en = get_last_modified(os.getenv('ENSEMBLE_KEY')).strftime('%Y-%m-%d %H:%M:%S')

            last_retrain = f"Random Forest Model: {last_retrain_rf} - Logistic Regression Model:{last_retrain_log} - Ensemble Model: {last_retrain_en}"
        except Exception:
            pass
 
        # Get model metrics
        model_metrics = None
        try:
            metrics_data = _download_file_from_s3('model_metrics.txt')  # Updated S3 key
            if metrics_data:
                model_metrics = metrics_data.decode('utf-8')
        except Exception:
            pass
 
        return {
            'total_records': total_records,
            'last_retrain': last_retrain,
            'upload_history': upload_history,
            'model_metrics': model_metrics,
            'login_history': login_history
        }
 
    except Exception as e:
        print(f"Error getting dashboard data: {e}")
        return {
            'total_records': 0,
            'last_retrain': 'N/A',
            'upload_history': [],
            'model_metrics': None,
            'login_history': []
        }
 
