# Start from an official PyTorch image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8080, as it's standard for cloud deployments like Render
EXPOSE 8080

# Run app.py using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
