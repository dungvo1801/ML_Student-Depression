import json
import pandas as pd
import psycopg2
from psycopg2 import OperationalError
import io

def lambda_handler(event, context):
    body = json.loads(event['body']) 
    filename = f"tmp/{body.get('filename')}"


    # return {
    #     "statusCode": 200,
    #     "headers": {"Content-Type": "application/json"},
    #     "body": json.dumps({
    #         "results": result_rows,
    #         "prediction_dist_data": prediction_dist_data,
    #         "columns": list(df_with_predictions.columns),
    #         "depressed_count": pred_counts.get('Depressed', 0),
    #         "not_depressed_count": pred_counts.get('Not Depressed', 0)
    #     }),
    #     "isBase64Encoded": False
    # }
    
    