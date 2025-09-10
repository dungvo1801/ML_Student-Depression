from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, send_from_directory, session, send_file
import os
import pandas as pd
import joblib
from datetime import datetime
import sqlite3
import numpy as np
from utils.login_api import login_api
from utils.download_template_api import download_template_api
from utils.predict_api import predict_api
from utils.upload_api import upload_api
import io
import json

# Import centralized configuration
from utils.config import config

app = Flask(__name__)
app.secret_key = 'secret_key_here'  # Add a secret key for session management and flash messages

# Use config for upload folder
app.config['UPLOAD_FOLDER'] = config.config["paths"]["uploads_dir"]
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#FE
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

#BE + FE
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    authentication = login_api(username=username, password=password)
    # Check if user exists and password matches
    if authentication:
        return redirect(url_for('upload_file'))
    else:
        # Flash error message and redirect back to login page
        flash('Invalid username or password. Please try again.', 'error')
        return redirect(url_for('hello_world'))

# BE + FE
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
            file_bytes = file.read()
            response = upload_api(file_bytes=file_bytes, user=session.get('username'), filename=file.filename)
            print(response.text)
            if response.status_code == 200:
                response_json = response.json()
                tmp_file = response_json['tmp_file']
                return redirect(url_for('predict', filename=tmp_file))
            else: 
                return redirect(request.url)
         
    return render_template('upload.html', current_time=current_time)

@app.route('/predict')
def predict():
    filename = request.args.get('filename')
    
    # Use the proper prediction function from training module
    try:
        response = predict_api(filename)
        
        if response.status_code != 200:
            flash(f"Failed to fetch CSV from Lambda: Status = {response.status_code}")
            return redirect(url_for('upload_file'))
        else:
            print("done")
            content = response.json()
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return render_template('result.html', 
                        results=content["results"], 
                        current_time=current_time, 
                        prediction_dist_data=content["prediction_dist_data"],
                        download_filename=filename,
                        columns=content["columns"],
                        depressed_count=content["depressed_count"],
                        not_depressed_count=content["not_depressed_count"])

            
    except Exception as e:
        flash(f"Prediction failed: {e}")
        return redirect(url_for('upload_file'))

@app.route('/retrain_model', methods=['POST'])
def retrain_model():

    # """Endpoint to handle model retraining"""
    # if session.get('username') != 'admin':
    #     return jsonify({'success': False, 'message': 'Access denied: Admins only.'})
    
    # try:
    #     from models.training import trigger_retrain
    #     trigger_retrain()  # This will only retrain if needed
    # except Exception as retrain_error:
    #     # Log the error but don't fail the upload
    #     flash(f'Note: Retraining failed: {str(retrain_error)}')
    
    try:
        from models.training import trigger_retrain
        trigger_retrain()
        return jsonify({
            'success': True, 
            'message': 'Model retrained successfully! Dashboard will refresh to show updated metrics.'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Retraining failed: {str(e)}'})


#BE
@app.route('/student_depression_template.csv')
def download_template():
    response = download_template_api()

    if response.status_code != 200:
        return "Failed to fetch CSV from Lambda", 500

    file_bytes = response.content 

    # Wrap bytes in a file-like object
    file_obj = io.BytesIO(file_bytes)

    # Send file as attachment
    return send_file(
        file_obj,
        mimetype='text/csv',
        as_attachment=True,
        download_name='student_depression_template.csv'
    )
    
@app.route('/dashboard')
def dashboard():
    # Only allow admin
    if session.get('username') != 'admin':
        flash('Access denied: Admins only.')
        return redirect(url_for('upload_file'))
    # Example: Read log file and show last 10 events
    log_path = os.path.join(config.config["paths"]["logs_dir"], 'app.log')
    logs = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            logs = f.readlines()[-10:]
    # Stats
    master_path = config.get_data_path()
    total_records = 0
    model_path = config.get_model_path()
    last_retrain = 'N/A'
    if os.path.exists(model_path):
        last_retrain = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
    # Read upload history
    upload_history = []
    upload_log_path = config.get_upload_history_path()
    if os.path.exists(upload_log_path):
        import csv
        with open(upload_log_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            upload_history = list(reader)[-100:]
    # Prepare upload trends data for chart
    from collections import Counter
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
   
    metrics_path = config.get_metrics_path()
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
    # Prepare performance history data for chart
    performance_history_data = {
        'labels': [],
        'accuracy': [],
        'f1_score': []
    }
   
    # Extract performance metrics from model metrics file
    if model_metrics:
        try:
            lines = model_metrics.split('\n')
            training_events = []
            current_event = {}
           
            for line in lines:
                if 'Training completed at:' in line:
                    if current_event:
                        training_events.append(current_event)
                    current_event = {'timestamp': line.split('Training completed at:')[1].strip()}
                elif 'Accuracy:' in line and current_event:
                    try:
                        acc = float(line.split('Accuracy:')[1].strip()) * 100
                        current_event['accuracy'] = acc
                    except:
                        pass
                elif 'F1-score:' in line and current_event:
                    try:
                        f1 = float(line.split('F1-score:')[1].strip()) * 100
                        current_event['f1_score'] = f1
                    except:
                        pass
           
            if current_event:
                training_events.append(current_event)
           
            # Prepare chart data from training events
            for i, event in enumerate(training_events[-10:]):  # Show last 10 training events
                if 'accuracy' in event and 'f1_score' in event:
                    performance_history_data['labels'].append(f'Training {i+1}')
                    performance_history_data['accuracy'].append(round(event['accuracy'], 2))
                    performance_history_data['f1_score'].append(round(event['f1_score'], 2))
        except Exception:
            # If parsing fails, show placeholder data
            performance_history_data = {
                'labels': ['Initial Training'],
                'accuracy': [85.0],
                'f1_score': [82.0]
            }
    # Find the most recent uploaded file for this session (admin)
    last_result_url = None
    if session.get('username') == 'admin' and upload_history:
        last_file = upload_history[-1]['filename']
        last_result_url = url_for('predict', filename=last_file)
    # Read login history
    login_history = []
    login_log_path = config.get_login_history_path()
    if os.path.exists(login_log_path):
        import csv
        with open(login_log_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            login_history = list(reader)[-20:]  # Show last 20 logins
    # Status message from query param
    status_message = request.args.get('status_message')
    return render_template('dashboard.html',
                         logs=logs,
                         total_records=total_records,
                         last_retrain=last_retrain,
                         upload_history=upload_history,
                         model_metrics=model_metrics,
                         upload_trends_data=upload_trends_data,
                         performance_history_data=performance_history_data,
                         last_result_url=last_result_url,
                         login_history=login_history,
                         status_message=status_message,
                         imbalance_method=imbalance_method,
                         class_distribution=class_distribution,
                         imbalance_ratio=imbalance_ratio)
#FE
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('hello_world'))

# #BE
# @app.route('/api/predictions', methods=['GET'])
# def get_predictions():
#     """Get prediction history from database"""
#     try:
#         conn = sqlite3.connect('predictions.db')
#         c = conn.cursor()
#         c.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100')
#         rows = c.fetchall()
#         conn.close()
        
#         # Convert to list of dicts for JSON
#         results = [
#             {
#                 'id': row[0], 
#                 'user': row[1], 
#                 'filename': row[2], 
#                 'input_data': row[3], 
#                 'prediction': row[4], 
#                 'timestamp': row[5]
#             }
#             for row in rows
#         ]
#         return jsonify(results)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

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
    import os
    # Use environment PORT variable for cloud deployment, fallback to 5001
    port = int(os.environ.get('PORT', 5001))
    # For production: bind to all interfaces and disable debug
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)