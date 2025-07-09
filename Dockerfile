# ✅ Dockerfile for TensorFlow + Flask on Render
FROM python:3.10-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies required by TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libfreetype6-dev \
    libpng-dev \
    libzmq3-dev \
    pkg-config \
    git \
    unzip \
    liblapack-dev \
    libblas-dev \
    gfortran \
    libhdf5-serial-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy all files
COPY . .

# Upgrade pip & install requirements
RUN pip install --upgrade pip
RUN pip install tensorflow==2.13.0 keras==2.13.1
RUN pip install -r requirements.txt

# Expose the port Render uses
EXPOSE 10000

# Start the Flask app via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
