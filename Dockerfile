# Use a standard Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy all files to the working directory
COPY . .

# Install system dependencies needed for PyTorch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    clang && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the application will run on
EXPOSE 8080

# Use gunicorn to serve the application
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
