# Use an official Python image with a stable PyTorch environment
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies using pip
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose the required port
EXPOSE 8080

# Start the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
