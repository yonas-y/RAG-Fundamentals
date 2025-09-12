# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app/ ./app
COPY data/ ./data

# Expose port (if using FastAPI or similar)
EXPOSE 8000

# Command to run the application (adjust as needed)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python", "app.py"]  # Uncomment if running a simple Python script