import requests
import os
from dotenv import load_dotenv

load_dotenv()


def dashboard_api() -> requests.Response:
    """
    Get dashboard data from Lambda API
        
    Returns:
        Dashboard data as dictionary
    """
    try:
        # Get API endpoint from config or environment
        url = f"{os.getenv('LAMBDA_ENDPOINT')}/dashboard"

        payload = {
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            return response
        else:
            print(f"Dashboard API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Dashboard API error: {str(e)}")
        return None
