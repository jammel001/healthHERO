# ✅ Use a full Debian-based Python image for TensorFlow compatibility
FROM python:3.10-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies required by TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    libzmq3-dev \
    unzip \
    curl \
    git \
    liblapack-dev \
    libblas-dev \
    gfortran \
    libhdf5-serial-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy all files from project folder into the container
COPY . .

# Upgrade pip & install TensorFlow first
RUN pip install --upgrade pip
RUN pip install tensorflow==2.13.0 keras==2.13.1

# Now install the rest of the Python dependencies
RUN pip install -r requirements.txt

# Expose the port expected by Render
EXPOSE 10000

# Run the Flask app using gunicorn on port 10000
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
