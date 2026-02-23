FROM python:3.11-slim

WORKDIR /app

# Install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and dataset.
COPY . .
COPY ../navi_mumbai_real_estate_uncleaned_2500_cleaned.csv ./data/

# Train the model at build time.
RUN python train_model.py

# Expose port.
EXPOSE 8000

# Run the application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
