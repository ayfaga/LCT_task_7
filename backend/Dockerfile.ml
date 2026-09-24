FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XFORMERS_DISABLED=1

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "app.ml.api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
