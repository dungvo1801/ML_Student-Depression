# Frontend Routes

### 1. GET /
**Description:** Render the landing/login page.

**Payload:** None

### 2. POST /login

**Description:** Authenticate via backend helper, then set session and redirect to /upload. On failure, get back to login page.

**Payload:** form-data (username, password)

### 3. GET | POST /upload

**Description:** Upload CSV for prediction (calls backend upload API)

**Payload:** multipart/form-data (file=@students.csv)

**Behavior:** On success, redirect to /predict ; otherwise flash messages returns an error if wrong format file

### 4. GET | POST /upload_validation

**Description:** Upload a validated CSV (calls backend validation insert API).

**Payload:**  multipart/form-data (validated_file=@file.csv)

**Behavior:** Flash messages success on 200; otherwise flash messages returns an error if wrong format file

### 5. GET /predict

**Description:** Render results (calls backend predict API).

**Payload:**  query (filename=<tmp_file_from_upload>)

**Behavior:** Redirect to result page if success; otherwise flash messages returns an error and redirect to /upload

### 6. POST /retrain_model (Admin only via UI control)

**Description:**  Trigger retrain (calls backend retrain API).

**Payload:** None

**Behavior:**  The output messages correspond to success and failure.

### 7. GET /dashboard (Admin only)

**Description:**  Render operational dashboard (reads aggregated metrics from backend).

**Payload:** None

**Behavior:**  Dashboard data is displayed correctly if success; otherwise dashboard data will be None or 0.

### 8. GET /student_depression_template.csv

**Description:** Download the standard CSV template (proxied via backend).

**Payload:** None

**Returns:** CSV file stream as an attachment

### 9. GET /download_results/<filename>

**Description:** Download prediction results as CSV (fetches JSON, converts to CSV on the fly).

**Payload:** None

**Path param:** : filename – suggested download name (not the S3 key).

**Behavior:**  CSV file is downloaded correctly if success; otherwise flash messages return error.
