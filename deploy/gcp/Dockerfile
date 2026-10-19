FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY frontend ./frontend

COPY LAWdata_Corpus_2024 ./LAWdata_Corpus_2024

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}