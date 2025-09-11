# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed, e.g., psycopg2, gcc)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set Flask environment variables
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_RUN_PORT=8080
# ENV FLASK_ENV=production

# AWS App Runner expects the app to run on PORT=8080
# ENV PORT=8080

# Expose the Flask port
EXPOSE 8080

# Start the Flask app
CMD ["python", "app.py"]

