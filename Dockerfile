FROM python:3.10-slim

WORKDIR /app

COPY . .

# Install system dependencies FIRST
RUN apt-get update && apt-get install -y \
    gcc g++ libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install PyTorch CPU FIRST (before other deps)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install YOUR FULL requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "--timeout", "180", "app:app"]
