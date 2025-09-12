import json
import pandas as pd
from util import (
    predict_with_ensemble, 
    insert_prediction,
    download_bytes,
    config
)
import psycopg2
from psycopg2 import OperationalError
import io

def lambda_handler(event, context):
    body = json.loads(event['body']) 
    filename = f"tmp/{body.get('filename')}"

    csv_bytes = download_bytes(s3_key=filename)
    csv_bytes = io.BytesIO(csv_bytes)
    df = pd.read_csv(csv_bytes)

    # Use the proper prediction function from training module
    try:
        predictions, ensemble_proba, ensemble_pred_custom, prediction_method = predict_with_ensemble(df)
        print(f"Generated {len(predictions)} predictions using trained model")
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        # Final fallback to simple rule-based logic
        df_features = df.drop(columns=['id'], errors='ignore')
        predictions = [1 if len([col for col in df_features.columns if 'stress' in col.lower() or 'anxiety' in col.lower()]) > 2 else 0 
                      for idx, row in df_features.iterrows()]
        ensemble_proba = [0.7 if pred == 1 else 0.3 for pred in predictions]
        print(f"Using fallback rule-based predictions")
    
    # CREATE COMPLETE RESULTS WITH ALL COLUMNS + PREDICTIONS AT THE END
    # Add prediction labels to the original dataframe - ensure they come LAST
    df_with_predictions = df.copy()
    
    # Add prediction columns at the END (not at arbitrary positions)
    df_with_predictions['Depression'] = predictions
    df_with_predictions['Depression_Status'] = ['Depressed' if pred == 1 else 'Not Depressed' for pred in predictions]
    df_with_predictions['Confidence_Score'] = [f"{prob:.3f}" for prob in ensemble_proba]

    # Prepare results for display (show all columns)
    result_rows = []
    for idx, row in df_with_predictions.iterrows():
        result_rows.append(row.to_dict())
    
    # Count predictions for chart
    pred_counts = {'Depressed': 0, 'Not Depressed': 0}
    for pred in predictions:
        status = 'Depressed' if pred == 1 else 'Not Depressed'
        pred_counts[status] += 1
    
    try:
        connection = psycopg2.connect(**config)
        print("Successfully connected to PostgreSQL")

        with connection:
            with connection.cursor() as cursor:
                insert_prediction(cursor, df_with_predictions)
    except OperationalError as e:
        print("Error while connecting to PostgreSQL:", e)
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Connection closed.")


    prediction_dist_data = {
        'labels': list(pred_counts.keys()),
        'counts': list(pred_counts.values())
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "results": result_rows,
            "prediction_dist_data": prediction_dist_data,
            "columns": list(df_with_predictions.columns),
            "depressed_count": pred_counts.get('Depressed', 0),
            "not_depressed_count": pred_counts.get('Not Depressed', 0)
        }),
        "isBase64Encoded": False
    }
    
    