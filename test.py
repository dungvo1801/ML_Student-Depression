import psycopg2
from psycopg2 import OperationalError, sql

config = {
    'host': 'st-instance.cfi4yu2gqyci.ap-southeast-1.rds.amazonaws.com',
    'port': 5432,
    'user': 'postgres',
    'password': 'Rmit123$%^',
    'dbname': 'student_clinic'
}

def show_tables(cursor):
    print("\n📋 Existing tables in the 'public' schema:")
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    if not tables:
        print("⚠️ No tables found.")
    else:
        for table in tables:
            print(f" - {table[0]}")

def create_example_table(cursor):
    print("\n🛠️ Creating example table 'students' (if not exists)...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            "user" TEXT,
            filename TEXT,
            input_data TEXT,
            prediction TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("✅ Table 'students' is ready.")

def get_user_logs(cursor):
    print("Fetching user logs...")
    cursor.execute("""
        SELECT * FROM predictions;
    """)
    rows = cursor.fetchall()
    print(f"Retrieved {rows} .")
    return rows


try:
    connection = psycopg2.connect(**config)
    print("✅ Successfully connected to PostgreSQL")

    with connection:
        with connection.cursor() as cursor:
            show_tables(cursor)
            get_user_logs(cursor)

except OperationalError as e:
    print("❌ Error while connecting to PostgreSQL:", e)

finally:
    if 'connection' in locals() and connection:
        connection.close()
        print("\n🔒 Connection closed.")


import requests

# url = "https://81it9chkja.execute-api.ap-southeast-1.amazonaws.com/v1/login" 

# payload = {
#     "username": "admin",
#     "password": "password"
# }

# headers = {
#     "Content-Type": "application/json"
# }

# response = requests.post(url, json=payload, headers=headers)

# print(response.status_code)
# print(response.text)


# url = "https://81it9chkja.execute-api.ap-southeast-1.amazonaws.com/v1/download_template" 

# payload = {
# }

# headers = {
#     "Content-Type": "application/json"
# }

# response = requests.post(url, headers=headers)

# print(response.status_code)
# print(response.text)
