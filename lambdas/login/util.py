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

# Static user credentials
users = {
    'admin': 'password',
    'user': 'userpass'
}


def create_table(cursor):
    print("Creating example table 'user_log' (if not exists)...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_log (
            id SERIAL PRIMARY KEY,
            "user" TEXT,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("Table 'user_log' is ready.")

def insert_log(cursor, user):
    print(f"Inserting new log record for user: {user}")
    cursor.execute("""
        INSERT INTO user_log ("user") VALUES (%s);
    """, (user,))
    print("Record inserted.")
