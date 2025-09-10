import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

config = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'dbname': os.getenv('DB_NAME')
}

validation_config = {
    "numeric_conversion_threshold": 0.7,  # 70% of values must convert to numeric
    "age_max": 120,
    "sleep_max": 24,
    "stress_max": 10,
    "age_min": 0,
    "sleep_min": 0,
    "stress_min": 0
}

def insert_validation(cursor, df):
    print(f"Inserting {len(df)} validation record(s)...")

    columns = [
        "id", "Gender", "Age", "City", "Profession", "Academic Pressure",
        "Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction",
        "Sleep Duration", "Dietary Habits", "Degree",
        "Have you ever had suicidal thoughts ?", "Work/Study Hours",
        "Financial Stress", "Family History of Mental Illness", "Depression", "Validation"
    ]

    quoted_columns = [f'"{col}"' for col in columns]
    placeholders = ', '.join(['%s'] * len(quoted_columns))

    values = []
    skipped = 0

    for _, row in df.iterrows():
        validation_value = row.get("Validation")

        # Check if Validation is an int AND is either 0 or 1
        if not isinstance(validation_value, int) or validation_value not in (0, 1):
            skipped += 1
            continue  # skip invalid rows

        row_values = [row.get(col) for col in columns]
        values.append(tuple(row_values))

    if not values:
        print("No valid records to insert (all skipped).")
        return False

    update_columns = [col for col in quoted_columns if col != '"id"']
    update_stmt = ', '.join([f'{col}=EXCLUDED.{col}' for col in update_columns])

    sql = f"""
        INSERT INTO predictions ({', '.join(quoted_columns)})
        VALUES ({placeholders})
        ON CONFLICT (id) DO UPDATE SET
        {update_stmt};
    """

    cursor.executemany(sql, values)
    print(f"Insert/update complete. Inserted {len(values)} record(s), skipped {skipped}.")
    return True


    
def validate_and_clean_data(df):
    """
    Validate and clean uploaded data before processing.
    
    Args:
        df: pandas DataFrame from uploaded CSV
        
    Returns:
        cleaned DataFrame
        
    Raises:
        ValueError: if data has critical issues
    """
    try:
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Basic validation
        if cleaned_df.empty:
            raise ValueError("Uploaded file is empty")
        
        # Remove completely empty rows and columns
        cleaned_df = cleaned_df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle numeric columns - convert strings to numbers where possible
        for col in cleaned_df.columns:
            if col not in ['id']:  # Skip ID column
                try:
                    # Try to convert to numeric, keep as object if not possible
                    numeric_series = pd.to_numeric(cleaned_df[col], errors='coerce')
                    # Only convert if most values are successfully converted
                    if numeric_series.notna().sum() / len(cleaned_df) > 0.7:
                        cleaned_df[col] = numeric_series
                except:
                    pass
        
        # Basic range validation for numeric columns using config limits
        validation_limits = {
            "age": (validation_config["age_min"], validation_config["age_max"]),
            "sleep": (validation_config["sleep_min"], validation_config["sleep_max"]),
            "stress": (validation_config["stress_min"], validation_config["stress_max"])
        }
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            if col not in ['id']:
                # Remove obvious outliers using configured ranges
                if 'age' in col.lower():
                    age_min, age_max = validation_limits['age']
                    cleaned_df[col] = cleaned_df[col].clip(age_min, age_max)
                elif 'sleep' in col.lower():
                    sleep_min, sleep_max = validation_limits['sleep']
                    cleaned_df[col] = cleaned_df[col].clip(sleep_min, sleep_max)
                elif 'stress' in col.lower():
                    stress_min, stress_max = validation_limits['stress']
                    cleaned_df[col] = cleaned_df[col].clip(stress_min, stress_max)
        
        return cleaned_df
        
    except Exception as e:
        # If validation fails, return original data with warning
        print(f"Data validation warning: {e}")
        return df