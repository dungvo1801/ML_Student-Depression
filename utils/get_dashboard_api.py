import requests
import os
 
def get_dashboard_api(username: str) -> requests.Response:
    """
    Get dashboard data from Lambda API
       
    Returns:
        Dashboard data as dictionary
    """
    # Get API endpoint from config or environment
    url = f"{os.getenv('LAMBDA_ENDPOINT')}/dashboard"

    payload = {
        "username": username
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response

