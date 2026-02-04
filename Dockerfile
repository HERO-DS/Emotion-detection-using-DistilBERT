FROM python:3.10-slim

WORKDIR /app

COPY . .

# ULTRA MINIMAL - NO execstack needed
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir Flask==2.1.1 gunicorn==20.1.0 && \
    pip install --no-cache-dir transformers==4.21.1

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "--timeout", "120", "app:app"]
