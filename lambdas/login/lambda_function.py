import json
import logging
from datetime import datetime
import psycopg2
from psycopg2 import OperationalError
from util import (
    create_table,
    insert_log,
    users,
    config
)


def lambda_handler(event, context):
    body = json.loads(event['body']) 
    username = body.get('username')
    password = body.get('password')

    try:
        connection = psycopg2.connect(**config)
        print("Successfully connected to PostgreSQL")

        with connection:
            with connection.cursor() as cursor:
                create_table(cursor)
    except OperationalError as e:
        print("Error while connecting to PostgreSQL:", e)
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Connection closed.")

    # Check if user exists and password matches
    if username in users and users[username] == password:   
        try:
            connection = psycopg2.connect(**config)
            print("Successfully connected to PostgreSQL")
            with connection:
                with connection.cursor() as cursor:
                    insert_log(cursor, username)
        except OperationalError as e:
            print("Error while connecting to PostgreSQL:", e)
            return False
        finally:
            if 'connection' in locals() and connection:
                connection.close()
                print("Connection closed.")
        return True
    else:
        return False

    
    

