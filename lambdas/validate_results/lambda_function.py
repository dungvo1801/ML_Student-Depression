import json
import pandas as pd
import base64
from datetime import datetime
import io
import psycopg2
from psycopg2 import OperationalError
from util import (
    insert_validation,
    validate_and_clean_data,
    config
)


def lambda_handler(event, context):
    body = json.loads(event['body']) 
    file_bytes = body.get('file_bytes')
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
                    success = insert_validation(cursor, new_df)
                    if success:
                        return {
                            "statusCode": 200,
                            "headers": {"Content-Type": "application/json"},
                            "body": "Successfully validation prediction data",
                            "isBase64Encoded": False
                        }
                    else:
                        return {
                            "statusCode": 400,
                            "headers": {"Content-Type": "application/json"},
                            "body": f"Error processing file: {str(e)}. Please check your data format and try again.",
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




