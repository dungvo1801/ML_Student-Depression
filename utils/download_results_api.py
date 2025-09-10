import requests
from dotenv import load_dotenv
import os

load_dotenv()

def download_results_api() -> requests.Response:
    url = f"{os.getenv('LAMBDA_ENDPOINT')}/download_results"

    payload = {
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    
    return response
