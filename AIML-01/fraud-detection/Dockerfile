FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by XGBoost / SciPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, trained artifacts, and drift reports
COPY app/ ./app/
COPY dashboard/ ./dashboard/
COPY monitoring/ ./monitoring/
COPY streaming/ ./streaming/
COPY artifacts/ ./artifacts/
COPY reports/ ./reports/

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]