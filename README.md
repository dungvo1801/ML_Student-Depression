# API Endpoints

## 1. POST /login
##### Description: Authenticate and get a short-lived token.

**Payload:**
```json
{"username": "admin",
"password": "••••••"}
```
#### Returns:
On success, returns a JSON object containing a JWT token, the user’s role (Admin/User), and an expiry in seconds; on failure, returns 401 Unauthorized.

## 2. POST /upload
### Description: Upload a CSV to S3 and index it for processing. 
#### Payload: None
#### Returns:
On acceptance, confirms receipt with an upload_id and status; invalid or missing files result in 400 Bad Request.

## 3. POST /validate
#### Description: Validate schema/content of a previously uploaded CSV. 
#### Payload:
```json
  {"upload_id":"upl_20250912_00123"}
```
#### Returns: 
On success, indicates whether the upload is valid and lists any issues.

## 4. POST /predict
#### Description: Run inference using the active model on a validated upload. 
#### Payload: 
```json
  {"upload_id":"upl_20250912_00123"}
```
#### Returns: 
On success, returns a job identifier, completion status, model version, and either a result key or a signed download URL.

## 5. GET /download_template
#### Description: Download the canonical CSV template. 
#### Payload none
#### Returns: 
On success, streams the template CSV file.

## 6. GET /download_results
#### Description: Download results for a completed prediction job. 
#### Payload: none
#### Returns: 
On success, streams the results CSV (or returns a JSON object with a signed URL for large files).

## 7. POST /retrain (Admin)
#### Description: Train a new model from labeled data; log metrics; version artifact. 
#### Payload: 
```json
{"threshold_new_labels":50}
```
#### Returns: 
On acceptance, confirms the new model version and summary metrics and indicates readiness once completed.

## 8. GET /dashboard (Admin)
#### Description: Operational metrics & recent activity.
#### Payload: none
#### Returns:
On success, returns a JSON object with totals, last retrain time, recent upload/login history.
