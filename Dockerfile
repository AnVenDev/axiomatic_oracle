FROM python:3.10-slim

WORKDIR /app

# Copia prima solo requirements.txt per cache migliore
COPY requirements.txt .

# Installa le dipendenze con output verboso per debug
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice
COPY . .

# Esponi la porta
EXPOSE 8000

# Usa CMD invece di ENTRYPOINT per Fargate
CMD ["uvicorn", "scripts.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]