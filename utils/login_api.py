import requests
from dotenv import load_dotenv
import os

load_dotenv()

def login_api(username: str, password: str) -> requests.Response:
    url = f"{os.getenv('LAMBDA_ENDPOINT')}/login"

    payload = {
        "username": username,
        "password": password
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    
    return response
