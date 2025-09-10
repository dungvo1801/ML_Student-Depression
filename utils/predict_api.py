import requests
from dotenv import load_dotenv
import os

load_dotenv()

def predict_api(filename: str) -> requests.Response:
    url = f"{os.getenv('LAMBDA_ENDPOINT')}/predict"

    payload = {
        "filename": filename
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    
    return response
