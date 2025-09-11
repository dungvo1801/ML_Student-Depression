from retrain import trigger_retrain
import json

def lambda_handler(event, context):

    success = trigger_retrain()
    if success:
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "message": "Retrained successfully",
                "result": True
            }),
            "isBase64Encoded": False
        }
    else:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "message": "Retraining failed",
                "result": False
            }),
            "isBase64Encoded": False
        }
    
    