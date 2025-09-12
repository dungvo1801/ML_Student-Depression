# Frontend Routes

### 1. GET /
**Description:** Render the landing/login page.

**Payload:** None

### 2. POST /login

**Description:** Authenticate via backend helper, then set session and redirect to upload.

**Payload:** form-data (username, password)

### 3. GET | POST /upload

**Description:** Upload CSV for prediction (calls backend upload API).

**Payload:** multipart/form-data (file=@students.csv)

### 4. GET | POST /upload_validation

**Description:** Upload a validated CSV (calls backend validation insert API).

**Payload:**  multipart/form-data (validated_file=@file.csv)

Behavior: Flashes success on 200; otherwise flashes a format error and returns to /upload.

### 5. GET /predict

**Description:** Render results (calls backend predict API and hydrates template).

**Payload:**  query (filename=<tmp_file_from_upload>)

**Return:** : results, prediction_dist_data, columns, depressed_count, not_depressed_count.

### 6. POST /retrain_model (Admin only via UI control)

**Description:**  Trigger retrain (calls backend retrain API).

**Payload:** None

### 7. GET /dashboard (Admin only)

**Description:**  Render operational dashboard (reads aggregated metrics from backend).

**Payload:** None

**Auth:** Redirects non-admin users back to /upload.

### 8. GET /student_depression_template.csv

**Description:** Download the canonical CSV template (proxied via backend).

**Payload:** None

**Returns:** CSV file stream as an attachment.

### 9. GET /download_results/<filename>

**Description:** Download prediction results as CSV (fetches JSON, converts to CSV on the fly).

**Payload:** None

**Path param:** : filename – suggested download name (not the S3 key).
