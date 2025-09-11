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
from utils.download_results_api import download_results_api  
from utils.upload_validation_api import upload_validation_api
from utils.retrain_api import retrain_api
from utils.get_dashboard_api import get_dashboard_api
import io
import json
import csv

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
    response = login_api(username=username, password=password)
    # Check if user exists and password matches
    if response.json():
        session['username'] = username
        session['password'] = password
        return redirect(url_for('upload_file'))
    else:
        # Flash error message and redirect back to login page
        flash('Invalid username or password. Please try again.', 'error')
        return redirect(url_for('hello_world'))

@app.route('/upload', methods=['GET', 'POST']) 
def upload_file():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file:
            file_bytes = file.read()
            response = upload_api(file_bytes=file_bytes, user=session.get('username'), filename=file.filename)
            if response.status_code == 200:
                response_json = response.json()
                tmp_file = response_json['tmp_file']
                return redirect(url_for('predict', filename=tmp_file))
            else: 
                return redirect(request.url)
         
    return render_template('upload.html', current_time=current_time)

@app.route('/upload_validation', methods=['GET', 'POST']) 
def upload_validation():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if request.method == 'POST':
        if 'validated_file' not in request.files:
            flash('No file part','error')
            return redirect(url_for('upload_file')) 
        file = request.files['validated_file']
        if file.filename == '':
            flash('No selected file','error')
            return redirect(url_for('upload_file')) 
        if file:
            file_bytes = file.read()
            response = upload_validation_api(file_bytes=file_bytes, user=session.get('username'), filename=file.filename)
            print(response.text)
            if response.status_code == 200:
                flash('Insert validation results successfully')
            else:
                flash("Wrong data format. Please check your data format and try again.", 'validated_upload')
                return redirect(url_for('upload_file'))
    return redirect(url_for('upload_file'))



@app.route('/predict')
def predict():
    filename = request.args.get('filename')
    
    # Use the proper prediction function from training module
    try:
        response = predict_api(filename)
        print(response.json())
        
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

    response = retrain_api()
    if response.status_code == 200:
        pass
    else: 
        pass


#BE
@app.route('/student_depression_template.csv')
def download_template():
    response = download_template_api()

    if response.status_code != 200:
        pass

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

    response = get_dashboard_api()
 
    if response.status_code == 200:
        dashboard_data = response.json()['dashboard_data']
        return render_template('dashboard.html',
            total_records=dashboard_data['total_records'],
            last_retrain=dashboard_data['last_retrain'],
            upload_history=dashboard_data['upload_history'],
            model_metrics=dashboard_data['model_metrics'],
            login_history=dashboard_data['login_history']
        )
     
    else:
        dashboard_data = response.json()['dashboard_data']
        return render_template('dashboard.html',
            total_records=0,
            last_retrain='N/A',
            upload_history=[],
            model_metrics=None,
            login_history=[]
        )

#FE
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('hello_world'))


#FE
@app.route('/download_results/<filename>')
def download_results(filename):
    """Download results file with predictions"""

    response = download_results_api()
    if response.status_code != 200:
        flash('Results file not found.')
        return redirect(url_for('upload_file'))

    # Parse JSON string from response
    data = response.json()
    if isinstance(data, dict) and "body" in data:
        data = json.loads(data["body"])  # this is your list of dicts

    if not data:
        flash('No data to download.')
        return redirect(url_for('upload_file'))

    # Convert to CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

    # Convert to BytesIO for download
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)

    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8080)