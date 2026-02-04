FROM python:3.10-slim

WORKDIR /app

# Install system dependencies FIRST (GCC needed for transformers)
RUN apt-get update && apt-get install -y \
    gcc g++ libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY . .

# CRITICAL: Install PyTorch FIRST with compatible CPU version
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install ALL other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Fix Gunicorn worker for PyTorch compatibility
ENV GUNICORN_CMD_ARGS="--workers=1 --threads=2 --worker-class=gthread --timeout=180 --preload"

EXPOSE 8080

CMD ["gunicorn", "app:app"]
