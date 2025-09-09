import requests
from dotenv import load_dotenv
import os
from typing import Dict

load_dotenv()

def predict_api(df: Dict) -> requests.Response:
    url = f"{os.getenv('LAMBDA_ENDPOINT')}/predict"

    payload = {
        "df": df
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    
    return response
