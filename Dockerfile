FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app

COPY . .

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

# Install Miniconda and Python 3.10
RUN wget -O Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda.sh -b -p /opt/conda && \
    rm Miniconda.sh && \
    /opt/conda/bin/conda install python=3.10 && \
    /opt/conda/bin/conda clean -ya

# Make RUN commands use Miniconda
ENV PATH=/opt/conda/bin:$PATH

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
