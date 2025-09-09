import requests
from dotenv import load_dotenv
import os

load_dotenv()

def download_template_api() -> requests.Response:
    url = f"{os.getenv('LAMBDA_ENDPOINT')}/download_template"

    payload = {
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    
    return response
