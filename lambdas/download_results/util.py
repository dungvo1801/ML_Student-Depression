from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

config = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'dbname': os.getenv('DB_NAME')
}


def get_prediction_results(cursor):
    print("Fetching predictions...")
    cursor.execute("SELECT * FROM predictions;")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    print(f"Retrieved {len(rows)} rows.")
    return rows, columns