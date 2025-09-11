import json
import pandas as pd
import base64
from datetime import datetime
import io
import psycopg2
from psycopg2 import OperationalError
from lambdas.upload.util import (
    get_master_data,
    insert_upload_log,
    upload_bytes,
    validate_and_clean_data,
    config
)


def lambda_handler(event, context):
    body = json.loads(event['body']) 
    file_bytes = body.get('file_bytes')
    filename = body.get('filename')
    user = body.get('user')

    try:

        file_bytes = base64.b64decode(file_bytes)
        file_bytes64 = io.BytesIO(file_bytes)

        # Validate columns (exclude 'Depression')
        new_df = pd.read_csv(file_bytes64)
        
        # APPLY DATA VALIDATION BEFORE PROCESSING
        try:
            
            new_df = validate_and_clean_data(new_df)
        except Exception as validation_error:
            print(f'Data validation warning: {str(validation_error)}. Continuing with original data.')
            # Continue with original data if validation fails

        try:
            connection = psycopg2.connect(**config)
            print("Successfully connected to PostgreSQL")

            with connection:
                with connection.cursor() as cursor:
                    master_df = get_master_data(cursor)
                    # Exclude all prediction-related columns when comparing with uploaded file
                    prediction_cols = ['Depression', "Id"]
                    new_df = new_df.drop(columns=['id'])

                    master_cols = [col.lower() for col in master_df.columns if col.lower() not in [c.lower() for c in prediction_cols]]
                    new_cols = [col.lower() for col in new_df.columns]
                    
                    if set(new_cols) != set(master_cols):
                        print('Uploaded file columns do not match required features. Please use the correct template.')
                        return {
                            "statusCode": 400,
                            "body": "Uploaded file columns do not match required features. Please use the correct template."
                        }
                    # Add empty 'Depression' column for new records
                    tmp_file = f"tmp{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                    upload_bytes(s3_key=tmp_file, file_bytes=file_bytes)
                    insert_upload_log(cursor, (user, filename))
                    return {
                        "statusCode": 200,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps({
                            "message": "File uploaded and validated successfully.",
                            "tmp_file": tmp_file
                        }),
                        "isBase64Encoded": False
                    }

        except OperationalError as e:
            print("Error while connecting to PostgreSQL:", e)
        finally:
            if 'connection' in locals() and connection:
                connection.close()
                print("Connection closed.")

    except Exception as e:
        print(f'Error processing file: {str(e)}. Please check your data format and try again.')
        return {
            "statusCode": 400,
            "body": f"Error processing file: {str(e)}. Please check your data format and try again."
        }




