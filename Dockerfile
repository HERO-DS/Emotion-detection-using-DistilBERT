FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY . .

# Install PyTorch CPU (exact version that works)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Render port
ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers=1", "--threads=4", "--timeout=300"]
