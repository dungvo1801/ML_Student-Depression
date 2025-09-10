import psycopg2
from psycopg2 import OperationalError
from util import (
    get_prediction_results,
    config
)
import json
from datetime import datetime

def lambda_handler(event, context):
    try:
        connection = psycopg2.connect(**config)
        print("Successfully connected to PostgreSQL")

        with connection:
            with connection.cursor() as cursor:
                rows, columns = get_prediction_results(cursor)

                result = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    filtered_row = {k: v for k, v in row_dict.items() if not isinstance(v, datetime)}
                    
                    result.append(filtered_row)

                return {
                    "statusCode": 200,
                    "body": json.dumps(result)
                }

    except OperationalError as e:
        print("Error while connecting to PostgreSQL:", e)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Connection closed.")
