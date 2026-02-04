FROM python:3.10-slim

WORKDIR /app

# Install execstack to fix libtorch_cpu.so
RUN apt-get update && apt-get install -y \
    gcc g++ libglib2.0-0 \
    execstack \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install PyTorch CPU with noexecstack fix
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Fix libtorch executable stack issue
RUN find /usr/local/lib/python3.10/site-packages/torch -name "libtorch_cpu.so" -exec execstack -c {} \; || true

EXPOSE 8080
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "--timeout", "120", "app:app"]
