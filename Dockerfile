FROM python:3.10-slim

WORKDIR /app

# Crée le répertoire de logs
RUN mkdir -p /app/logs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]