# Student Depression Prediction System

## Setup Instructions

1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate environment: `source .venv/bin/activate` (macOS/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python app.py`
6. Open browser to: `http://localhost:5000`

## Usage

1. Upload CSV file with student data
2. Get depression risk predictions
3. View detailed analytics dashboard

## Technical Details

- Uses Random Forest and Logistic Regression
- Handles class imbalance with weighted training
- Includes performance monitoring and retraining
