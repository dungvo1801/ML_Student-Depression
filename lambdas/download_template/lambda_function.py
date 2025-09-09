from util import download_bytes
import base64

def lambda_handler(event, context):
    template = download_bytes()

    if template:
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/octet-stream"},
            "body": base64.b64encode(template).decode('utf-8'),
            "isBase64Encoded": True
        }
    else:
        return {
            "statusCode": 500,
            "body": "Failed to download template"
        }

    
    