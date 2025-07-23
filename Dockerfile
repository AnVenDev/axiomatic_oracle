FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "scripts.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]