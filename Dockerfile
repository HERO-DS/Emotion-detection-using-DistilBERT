FROM python:3.10-slim

WORKDIR /app

# Copy files
COPY . .

# Install PyTorch CPU first
RUN pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "--timeout", "120", "app:app"]
