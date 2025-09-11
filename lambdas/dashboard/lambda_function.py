from util import get_dashboard_data
import json
 
def lambda_handler(event, context):

    try:
        dashboard_data = get_dashboard_data()

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                # "message": "Dashboard data retrieved successfully.",
                "dashboard_data": dashboard_data 
            })
        }

    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "message": f"Exception: {e}"
            })
        }