# import psycopg2
# from psycopg2 import OperationalError, sql

# config = {
#     'host': 'st-instance.cfi4yu2gqyci.ap-southeast-1.rds.amazonaws.com',
#     'port': 5432,
#     'user': 'postgres',
#     'password': 'Rmit123$%^',
#     'dbname': 'student_clinic'
# }

# def show_tables(cursor):
#     print("\n📋 Existing tables in the 'public' schema:")
#     cursor.execute("""
#         SELECT table_name
#         FROM information_schema.tables
#         WHERE table_schema = 'public'
#         ORDER BY table_name;
#     """)
#     tables = cursor.fetchall()
#     if not tables:
#         print("⚠️ No tables found.")
#     else:
#         for table in tables:
#             print(f" - {table[0]}")

# def create_example_table(cursor):
#     print("\n🛠️ Creating example table 'students' (if not exists)...")
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS upload_log (
#             id SERIAL PRIMARY KEY,
#             "user" TEXT,
#             "filename" TEXT,
#             time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#     """)
#     print("✅ Table 'students' is ready.")

# def insert_prediction(cursor, df):
#     print(f"Inserting new prediction record: {df}")
#     cursor.execute("""
#         INSERT INTO predictions ("Gender", "Age", "City", "Profession", "Academic Pressure","Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction","Sleep Duration", "Dietary Habits", "Degree","Have you ever had suicidal thoughts ?", "Work/Study Hours","Financial Stress", "Family History of Mental Illness", "Depression") VALUES ("M",12);
#     """)
#     print("Record inserted.")
    

# def get_user_logs(cursor):
#     print("Fetching user logs...")
#     cursor.execute("""
#         SELECT * FROM predictions;
#     """)
#     rows = cursor.fetchall()
#     print(f"Retrieved {rows} .")
#     return rows

# def get_prediction_logs(cursor):
#     print("Fetching user logs...")
    
#     cursor.execute("SELECT * FROM student_depression_master;")
    
#     # Get column names from cursor description
#     column_names = [desc[0] for desc in cursor.description]
#     print("Columns:", column_names)

#     # Fetch all rows
#     rows = cursor.fetchall()
#     print(f"Retrieved {len(rows)} rows.")
    
#     # Optionally, return rows as list of dicts
#     logs = [dict(zip(column_names, row)) for row in rows]
    
#     return logs


# try:
#     connection = psycopg2.connect(**config)
#     print("✅ Successfully connected to PostgreSQL")

#     with connection:
#         with connection.cursor() as cursor:
#             # show_tables(cursor)
#             # insert_prediction(cursor, "")
#             print(get_prediction_logs(cursor))
#             # get_user_logs(cursor)

# except OperationalError as e:
#     print("❌ Error while connecting to PostgreSQL:", e)

# finally:
#     if 'connection' in locals() and connection:
#         connection.close()
#         print("\n🔒 Connection closed.")


import requests
import base64

url = "https://81it9chkja.execute-api.ap-southeast-1.amazonaws.com/v1/predict" 

filename = "C:/Users/Admin/Desktop/AI/ML_Student-Depression/models/student_depression_dataset.csv"

# Open and read file bytes
with open(filename, "rb") as f:
    file_bytes = f.read()

# Base64 encode bytes so they can be safely included in JSON
file_b64 = base64.b64encode(file_bytes).decode('utf-8')


payload = {
    "filename": "tmp20250910150412.csv"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.status_code)
print(response.text)


# url = "https://81it9chkja.execute-api.ap-southeast-1.amazonaws.com/v1/download_template" 

# payload = {
# }

# headers = {
#     "Content-Type": "application/json"
# }

# response = requests.post(url, headers=headers)

# print(response.status_code)
# print(response.text)

# from datetime import datetime
# from lambdas.model_predict.util import download_bytes
# import base64
# import io
# import pandas as pd
# filename = "C:/Users/Admin/Desktop/AI/ML_Student-Depression/models/student_depression_dataset.csv"

# # Open and read file bytes
# with open(filename, "rb") as f:
#     file_bytes = f.read()

# # Base64 encode bytes so they can be safely included in JSON
# file_b64 = base64.b64encode(file_bytes).decode('utf-8')


# csv_bytes = download_bytes(s3_key='tmp20250909205309.csv')
# print(csv_bytes)
# csv_bytes = io.BytesIO(csv_bytes)
# df = pd.read_csv(csv_bytes)
# print(df)


# from lambdas.model_predict.lambda_function import lambda_handler

# lambda_handler(event="", context="")





