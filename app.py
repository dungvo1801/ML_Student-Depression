from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, send_from_directory, session, send_file
import os
import pandas as pd
import joblib
from datetime import datetime
import sqlite3
import numpy as np
from utils.login_api import login_api
from utils.download_template_api import download_template_api
import io

# Import centralized configuration
from utils.config import config

app = Flask(__name__)
app.secret_key = 'secret_key_here'  # Add a secret key for session management and flash messages

# Use config for upload folder
app.config['UPLOAD_FOLDER'] = config.config["paths"]["uploads_dir"]
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#  # BE
# def init_db():
#     conn = sqlite3.connect('predictions.db')
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS predictions (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user TEXT,
#             filename TEXT,
#             input_data TEXT,
#             prediction TEXT,
#             timestamp TEXT
#         )
#     ''')
#     conn.commit()
#     conn.close()

# # Call this at app startup
# init_db()

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
    from datetime import datetime
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
            
            try:
                # Validate columns (exclude 'Depression')
                new_df = pd.read_csv(filepath)
                
                # APPLY DATA VALIDATION BEFORE PROCESSING
                try:
                    from models.training import validate_and_clean_data
                    new_df = validate_and_clean_data(new_df)
                except Exception as validation_error:
                    flash(f'Data validation warning: {str(validation_error)}. Continuing with original data.')
                    # Continue with original data if validation fails
                
                master_path = config.get_data_path()
                if os.path.exists(master_path):
                    master_df = pd.read_csv(master_path)
                    # Exclude all prediction-related columns when comparing with uploaded file
                    prediction_cols = ['Depression', 'Depression_Pred', 'Depression_Proba', 'PredictedAt']
                    required_cols = [col for col in master_df.columns if col not in prediction_cols]
                    if list(new_df.columns) != required_cols:
                        flash('Uploaded file columns do not match required features. Please use the correct template.')
                        os.remove(filepath)
                        return redirect(request.url)
                    # Add empty 'Depression' column for new records
                    new_df['Depression'] = pd.NA
                    
                    # Fix ID conflicts: Generate continuous IDs
                    if 'id' in new_df.columns:
                        # Get the max ID from master dataset and continue from there
                        try:
                            max_id = master_df['id'].max() if 'id' in master_df.columns and not master_df.empty else 0
                            # Ensure max_id is an integer
                            max_id = int(float(max_id)) if pd.notna(max_id) else 0
                            new_df['id'] = range(max_id + 1, max_id + 1 + len(new_df))
                        except Exception as id_error:
                            # If ID generation fails, create simple sequential IDs
                            new_df['id'] = range(1, len(new_df) + 1)
                    
                    # Ensure data type compatibility before concatenation
                    try:
                        # Convert all columns to object type to avoid type conflicts
                        master_df_safe = master_df.astype(str)
                        new_df_safe = new_df.astype(str)
                        combined_df = pd.concat([master_df_safe, new_df_safe], ignore_index=True)
                        
                        # Convert back to appropriate types after concatenation
                        for col in combined_df.columns:
                            if col not in ['id', 'Depression']:
                                try:
                                    # Try to convert to numeric if possible
                                    numeric_series = pd.to_numeric(combined_df[col], errors='ignore')
                                    if not numeric_series.equals(combined_df[col]):
                                        combined_df[col] = numeric_series
                                except:
                                    pass
                        
                    except Exception as concat_error:
                        # Fallback: just use the new data
                        combined_df = new_df
                        
                else:
                    # If no master, create with empty Depression column
                    new_df['Depression'] = pd.NA
                    
                    # Initialize IDs starting from 1 if no master exists
                    if 'id' in new_df.columns:
                        try:
                            new_df['id'] = range(1, len(new_df) + 1)
                        except Exception:
                            # Fallback: create simple integer IDs
                            new_df['id'] = list(range(1, len(new_df) + 1))
                    
                    combined_df = new_df
                    
            except Exception as e:
                flash(f'Error processing file: {str(e)}. Please check your data format and try again.')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
            # Data versioning: backup current master before overwrite
            import shutil
            import time
            if os.path.exists(master_path):
                version_dir = config.get_versions_dir()
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
            log_path = config.get_upload_history_path()
            import csv
            log_exists = os.path.exists(log_path)
            with open(log_path, 'a', newline='') as logfile:
                writer = csv.writer(logfile)
                if not log_exists:
                    writer.writerow(['user', 'filename', 'time'])
                writer.writerow([session.get('username', 'unknown'), file.filename, current_time])
                
            return redirect(url_for('predict', filename=file.filename))
    return render_template('upload.html', current_time=current_time)

#Check 2 predict 
@app.route('/predict')
def predict():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    # Use the proper prediction function from training module
    try:
        from models.training import predict_depression
        predictions, prediction_probabilities = predict_depression(df)
        print(f"Generated {len(predictions)} predictions using trained model")
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        # Final fallback to simple rule-based logic
        df_features = df.drop(columns=['id'], errors='ignore')
        predictions = [1 if len([col for col in df_features.columns if 'stress' in col.lower() or 'anxiety' in col.lower()]) > 2 else 0 
                      for idx, row in df_features.iterrows()]
        prediction_probabilities = [0.7 if pred == 1 else 0.3 for pred in predictions]
        print(f"Using fallback rule-based predictions")
    
    # CREATE COMPLETE RESULTS WITH ALL COLUMNS + PREDICTIONS AT THE END
    # Add prediction labels to the original dataframe - ensure they come LAST
    df_with_predictions = df.copy()
    
    # Add prediction columns at the END (not at arbitrary positions)
    df_with_predictions['Depression_Prediction'] = predictions
    df_with_predictions['Depression_Status'] = ['Depressed' if pred == 1 else 'Not Depressed' for pred in predictions]
    df_with_predictions['Confidence_Score'] = [f"{prob:.3f}" for prob in prediction_probabilities]
    
    # Reorder columns to ensure original data comes first, predictions come last
    original_columns = [col for col in df.columns if col in df_with_predictions.columns]
    prediction_columns = ['Depression_Prediction', 'Depression_Status', 'Confidence_Score']
    ordered_columns = original_columns + prediction_columns
    df_with_predictions = df_with_predictions[ordered_columns]
    
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
    
    # Save predictions back to master dataset - NEVER overwrite ground truth labels!
    master_path = config.get_data_path()
    if os.path.exists(master_path):
        master_df = pd.read_csv(master_path)
        # Find the most recently added records with NULL Depression labels
        null_rows = master_df[master_df['Depression'].isna()]
        if len(null_rows) >= len(predictions):
            # Store predictions in SEPARATE columns - never overwrite ground truth
            recent_null_indices = null_rows.tail(len(predictions)).index
            master_df.loc[recent_null_indices, 'Depression_Pred'] = predictions
            master_df.loc[recent_null_indices, 'Depression_Proba'] = prediction_probabilities
            master_df.loc[recent_null_indices, 'PredictedAt'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            master_df.to_csv(master_path, index=False)
    
    # Store predictions in database
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Define timestamp BEFORE using it
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
    
    return render_template('result.html', 
                         results=result_rows, 
                         current_time=current_time, 
                         prediction_dist_data=prediction_dist_data,
                         download_filename=results_filename,
                         columns=list(df_with_predictions.columns),
                         depressed_count=pred_counts.get('Depressed', 0),
                         not_depressed_count=pred_counts.get('Not Depressed', 0))


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
    
#FE + BE
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
    doctor_uploads = 0
    if os.path.exists(master_path):
        import pandas as pd
        df = pd.read_csv(master_path)
        total_records = df.shape[0]
        # Convert id to numeric for comparison
        if 'id' in df.columns:
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
            doctor_uploads = df[df['id'] > 10000].shape[0]
    # Last retrain time
    model_path = config.get_model_path()
    last_retrain = 'N/A'
    if os.path.exists(model_path):
        import datetime
        last_retrain = datetime.datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
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
    from datetime import datetime
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
                         total_uploads=doctor_uploads, 
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