import requests
from dotenv import load_dotenv
import os
import base64

load_dotenv()

def upload_validation_api(file_bytes: bytes, filename: str, user: str) -> requests.Response:
    url = f"{os.getenv('LAMBDA_ENDPOINT')}/validate"

    file_b64 = base64.b64encode(file_bytes).decode('utf-8')

    payload = {
        "file_bytes": file_b64,
        "filename": filename,
        "user": user
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    
    return response
