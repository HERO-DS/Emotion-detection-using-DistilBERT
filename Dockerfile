FROM python:3.10-slim

WORKDIR /app

COPY . .

# Install system deps minimally
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install deps (use CPU-only PyTorch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
