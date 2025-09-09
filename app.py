from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, send_from_directory, session
import os
from datetime import datetime
import sqlite3
import pandas as pd
import joblib
import shutil
import time
import csv
from datetime import datetime
from models.training import trigger_retrain
from collections import Counter
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Add a secret key for session management and flash messages

# Simple user database
users = {
    'admin': 'password',
    'user': 'userpass'
}

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# BE
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            filename TEXT,
            input_data TEXT,
            prediction TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Call this at app startup
init_db()
#FE
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')
#BE + FE
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    # Check if user exists and password matches
    if username in users and users[username] == password:
        session['username'] = username
        # Log login activity
        login_log_path = 'logs/login_history.csv'

        log_exists = os.path.exists(login_log_path)
        with open(login_log_path, 'a', newline='') as logfile:
            writer = csv.writer(logfile)
            if not log_exists:
                writer.writerow(['user', 'time'])
            writer.writerow([username, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        flash('Login successful! Welcome to the system.', 'success')
        return redirect(url_for('upload_file'))
    else:
        flash('Invalid username or password. Please try again.', 'error')
        return redirect(url_for('hello_world'))

#BE + FE
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Validate columns (exclude 'Depression')
            new_df = pd.read_csv(filepath)
            master_path = 'models/student_depression_dataset.csv'
            if os.path.exists(master_path):
                master_df = pd.read_csv(master_path)
                required_cols = [col for col in master_df.columns if col != 'Depression']
                if list(new_df.columns) != required_cols:
                    flash('Uploaded file columns do not match required features. Please use the correct template.')
                    os.remove(filepath)
                    return redirect(request.url)
                # Add empty 'Depression' column for new records
                new_df['Depression'] = pd.NA
                
                # Fix ID conflicts: Generate continuous IDs
                if 'id' in new_df.columns:
                    # Get the max ID from master dataset and continue from there
                    max_id = master_df['id'].max() if 'id' in master_df.columns and not master_df.empty else 0
                    new_df['id'] = range(max_id + 1, max_id + 1 + len(new_df))
                
                combined_df = pd.concat([master_df, new_df], ignore_index=True)
            else:
                # If no master, create with empty Depression column
                new_df['Depression'] = pd.NA
                
                # Initialize IDs starting from 1 if no master exists
                if 'id' in new_df.columns:
                    new_df['id'] = range(1, len(new_df) + 1)
                
                combined_df = new_df
            # Data versioning: backup current master before overwrite
            if os.path.exists(master_path):
                version_dir = 'models/versions'
                os.makedirs(version_dir, exist_ok=True)
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(version_dir, f'student_depression_dataset_{timestamp}.csv')
                shutil.copy2(master_path, backup_path)
            combined_df.to_csv(master_path, index=False)
            # Duplicate detection: warn if duplicates in uploaded data
            duplicate_rows = new_df.duplicated().sum()
            if duplicate_rows > 0:
                flash(f'Warning: {duplicate_rows} duplicate records detected in your upload.')
            # Log upload history
            log_path = 'logs/upload_history.csv'
            log_exists = os.path.exists(log_path)
            with open(log_path, 'a', newline='') as logfile:
                writer = csv.writer(logfile)
                if not log_exists:
                    writer.writerow(['user', 'filename', 'time'])
                writer.writerow([session.get('username', 'unknown'), file.filename, current_time])
            
            # Check if we should retrain (batch-based)

            trigger_retrain()  # This will only retrain if needed
                
            return redirect(url_for('predict', filename=file.filename))
    return render_template('upload.html', current_time=current_time)

# For UIUX, chart
@app.route('/predict')
def predict():
    filename = request.args.get('filename')
    if not filename:
        flash('No filename provided for prediction.', 'error')
        return redirect(url_for('upload_file'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists before trying to read it
    if not os.path.exists(filepath):
        flash(f'File "{filename}" not found. It may have been deleted or moved. Please upload a new file.', 'error')
        return redirect(url_for('upload_file'))
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        flash(f'Error reading file "{filename}": {str(e)}. Please check the file format and try again.', 'error')
        return redirect(url_for('upload_file'))
    
    # Load the trained model pipeline for real predictions
    try:
        # Try Random Forest pipeline first (primary model)
        model_pipeline = joblib.load('models/rf_model.pkl')
        print("Loaded Random Forest pipeline for prediction")
        
        # Remove ID column from features before prediction (ID is not a predictive feature)
        df_features = df.drop(columns=['id'], errors='ignore')
        
        # Use pipeline for prediction (handles preprocessing automatically)
        predictions = model_pipeline.predict(df_features).tolist()
        prediction_probabilities = model_pipeline.predict_proba(df_features)[:, 1]
        
        print(f"✅ Generated {len(predictions)} predictions using trained pipeline")
        
    except Exception as e:
        print(f"⚠️ Pipeline prediction failed: {e}")
        # Fallback to logistic regression pipeline
        try:
            model_pipeline = joblib.load('models/log_model.pkl')
            print("✅ Loaded Logistic Regression pipeline as fallback")
            
            df_features = df.drop(columns=['id'], errors='ignore')
            predictions = model_pipeline.predict(df_features).tolist()
            prediction_probabilities = model_pipeline.predict_proba(df_features)[:, 1]
            
            print(f"Generated {len(predictions)} predictions using fallback pipeline")
            
        except Exception as e2:
            print(f"Both pipelines failed: {e2}")
            # Final fallback to simple rule-based logic
            df_features = df.drop(columns=['id'], errors='ignore')
            predictions = [1 if len([col for col in df_features.columns if 'stress' in col.lower() or 'anxiety' in col.lower()]) > 2 else 0 
                          for idx, row in df_features.iterrows()]
            prediction_probabilities = [0.7 if pred == 1 else 0.3 for pred in predictions]
            print(f"Using fallback rule-based predictions")
    
    # CREATE COMPLETE RESULTS WITH ALL COLUMNS + PREDICTIONS
    # Add prediction labels to the original dataframe
    df_with_predictions = df.copy()
    df_with_predictions['Depression_Prediction'] = predictions
    df_with_predictions['Depression_Status'] = ['Depressed' if pred == 1 else 'Not Depressed' for pred in predictions]
    df_with_predictions['Confidence_Score'] = [f"{prob:.3f}" for prob in prediction_probabilities]
    
    # Save results file for download
    results_filename = f"results_{filename.replace('.csv', '')}_predictions.csv"
    results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
    df_with_predictions.to_csv(results_filepath, index=False)
    
    # Prepare results for display (show all columns)
    result_rows = []
    for idx, row in df_with_predictions.iterrows():
        result_rows.append(row.to_dict())
    
    # Count predictions for chart
    pred_counts = {'Depressed': 0, 'Not Depressed': 0}
    for pred in predictions:
        status = 'Depressed' if pred == 1 else 'Not Depressed'
        pred_counts[status] += 1
    
    # Save predictions back to master dataset
    master_path = 'models/student_depression_dataset.csv'
    if os.path.exists(master_path):
        master_df = pd.read_csv(master_path)
        # Find the most recently added records with NULL Depression labels
        null_rows = master_df[master_df['Depression'].isna()]
        if len(null_rows) >= len(predictions):
            # Update the most recent NULL rows with predictions
            recent_null_indices = null_rows.tail(len(predictions)).index
            master_df.loc[recent_null_indices, 'Depression'] = predictions
            master_df.to_csv(master_path, index=False)
    
    # Store predictions in database
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute(
            'INSERT INTO predictions (user, filename, input_data, prediction, timestamp) VALUES (?, ?, ?, ?, ?)',
            (
                session.get('username', 'unknown'),
                filename,
                df.to_json(),  # Store the input data as JSON
                str(predictions),  # Store predictions as string
                current_time
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        pass
    
    prediction_dist_data = {
        'labels': list(pred_counts.keys()),
        'counts': list(pred_counts.values())
    }
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('result.html', 
                         results=result_rows, 
                         current_time=current_time, 
                         prediction_dist_data=prediction_dist_data,
                         download_filename=results_filename,
                         columns=list(df_with_predictions.columns))
    
    
#For prediction result, json api
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # Save uploaded file temporarily
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(temp_path)
    
    # Load data with error handling
    try:
        df = pd.read_csv(temp_path)
    except Exception as e:
        return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
    
    # Load model with error handling
    try:
        model = joblib.load('models/rf_model.pkl')
    except Exception as e:
        return jsonify({'error': f'Model not found or corrupted: {str(e)}'}), 500
    # Preprocess input (assume same as training)
    # --- You may need to adjust this block to match your training pipeline ---
    # Example: drop or fill missing columns, encode categoricals, scale numerics, etc.
    # For demo, just use all columns except 'id' if present
    if 'id' in df.columns:
        X_pred = df.drop(columns=['id'])
    else:
        X_pred = df
    # Predict
    preds = model.predict(X_pred)
    # Return results as JSON
    return jsonify({'predictions': preds.tolist()})

#BE
@app.route('/student_depression_template.csv')
def download_template():
    template_path = 'student_depression_template.csv'
    if not os.path.exists(template_path):
        flash('Template file not found. Please contact the administrator.', 'error')
        return redirect(url_for('upload_file'))
    
    try:
        return send_from_directory(directory='.', path=template_path, as_attachment=True)
    except Exception as e:
        flash(f'Error downloading template: {str(e)}', 'error')
        return redirect(url_for('upload_file'))

#FE + BE
@app.route('/dashboard')
def dashboard():
    # Only allow admin
    if session.get('username') != 'admin':
        flash('Access denied: Admins only.')
        return redirect(url_for('upload_file'))
    # Example: Read log file and show last 10 events
    log_path = 'logs/app.log'
    logs = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            logs = f.readlines()[-10:]
    # Stats
    master_path = 'models/student_depression_dataset.csv'
    total_records = 0
    doctor_uploads = 0
    if os.path.exists(master_path):
        df = pd.read_csv(master_path)
        total_records = df.shape[0]
        # Convert id to numeric for comparison
        if 'id' in df.columns:
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
            doctor_uploads = df[df['id'] > 10000].shape[0]
    # Last retrain time
    model_path = 'models/rf_model.pkl'
    last_retrain = 'N/A'
    if os.path.exists(model_path):
        last_retrain = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
    # Read upload history
    upload_history = []
    upload_log_path = 'logs/upload_history.csv'
    if os.path.exists(upload_log_path):
        with open(upload_log_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            upload_history = list(reader)[-100:]
    # Prepare upload trends data for chart

    date_counts = Counter()
    for entry in upload_history:
        try:
            date = entry['time'][:10]  # YYYY-MM-DD
            date_counts[date] += 1
        except Exception:
            continue
    sorted_dates = sorted(date_counts.keys())
    upload_trends_data = {
        'labels': sorted_dates,
        'counts': [date_counts[d] for d in sorted_dates]
    }
    # Read model metrics
    model_metrics = None
    imbalance_method = "Class Weights (Auto)"
    class_distribution = "Not Available"
    imbalance_ratio = "Not Available"
    
    metrics_path = 'logs/model_metrics.txt'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            model_metrics = f.read()
            
        # Extract imbalance handling information
        if '=== Class Imbalance Handling ===' in model_metrics:
            imbalance_section = model_metrics.split('=== Class Imbalance Handling ===')[1]
            lines = imbalance_section.split('\n')
            
            for line in lines:
                if 'Method used:' in line:
                    method = line.split('Method used:')[1].strip()
                    if method == 'class_weight':
                        imbalance_method = "Class Weights (Balanced)"
                    elif method == 'smote':
                        imbalance_method = "SMOTE Oversampling"
                    elif method == 'undersample':
                        imbalance_method = "Random Undersampling"
                    else:
                        imbalance_method = method.title()
                elif 'Original class distribution:' in line:
                    class_distribution = line.split('Original class distribution:')[1].strip()
                elif 'Imbalance ratio:' in line:
                    imbalance_ratio = line.split('Imbalance ratio:')[1].strip()
    # Prepare retrain frequency data for chart
    retrain_dir = 'models/versions'
    retrain_dates = []
    if os.path.exists(retrain_dir):
        for fname in os.listdir(retrain_dir):
            if fname.startswith('student_depression_dataset_') and fname.endswith('.csv'):
                try:
                    date_str = fname.split('_')[-2]  # e.g. 20240906 from 20240906_153000
                    date_fmt = fname.split('_')[-2] + '_' + fname.split('_')[-1].replace('.csv','')
                    dt = datetime.strptime(date_fmt, '%Y%m%d_%H%M%S')
                    retrain_dates.append(dt.strftime('%Y-%m-%d'))
                except Exception:
                    continue
    retrain_counts = Counter(retrain_dates)
    sorted_retrain_dates = sorted(retrain_counts.keys())
    retrain_freq_data = {
        'labels': sorted_retrain_dates,
        'counts': [retrain_counts[d] for d in sorted_retrain_dates]
    }
    # Find the most recent uploaded file for this session (admin)
    last_result_url = None
    if session.get('username') == 'admin' and upload_history:
        last_file = upload_history[-1]['filename']
        last_result_url = url_for('predict', filename=last_file)
    # Read login history
    login_history = []
    login_log_path = 'logs/login_history.csv'
    if os.path.exists(login_log_path):
        with open(login_log_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            login_history = list(reader)[-20:]  # Show last 20 logins
    # Backup versions for restore dropdown
    version_dir = 'models/versions'
    backup_versions = []
    if os.path.exists(version_dir):
        backup_versions = sorted([f for f in os.listdir(version_dir) if f.endswith('.csv')], reverse=True)
    # Status message from query param
    status_message = request.args.get('status_message')
    return render_template('dashboard.html', 
                         logs=logs, 
                         total_records=total_records, 
                         total_uploads=doctor_uploads, 
                         last_retrain=last_retrain, 
                         upload_history=upload_history, 
                         model_metrics=model_metrics, 
                         upload_trends_data=upload_trends_data, 
                         retrain_freq_data=retrain_freq_data, 
                         last_result_url=last_result_url, 
                         login_history=login_history, 
                         backup_versions=backup_versions, 
                         status_message=status_message,
                         imbalance_method=imbalance_method,
                         class_distribution=class_distribution,
                         imbalance_ratio=imbalance_ratio)

#FE
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('hello_world'))


#BE
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get prediction history from database"""
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100')
        rows = c.fetchall()
        conn.close()
        
        # Convert to list of dicts for JSON
        results = [
            {
                'id': row[0], 
                'user': row[1], 
                'filename': row[2], 
                'input_data': row[3], 
                'prediction': row[4], 
                'timestamp': row[5]
            }
            for row in rows
        ]
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#FE
@app.route('/download_results/<filename>')
def download_results(filename):
    """Download results file with predictions"""
    try:
        return send_from_directory(
            directory=app.config['UPLOAD_FOLDER'], 
            path=filename, 
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        flash('Results file not found.')
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)