# Use a specific version of the Python image based on Debian slim for stability
FROM python:3.10-slim

# Set work directory
WORKDIR /eastwood

# Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    python3-dev \
    protobuf-compiler \
    libprotobuf-dev \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    neovim \
    poppler-utils \
    tesseract-ocr \
    curl \
    postgresql \
    cmake \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies manager
RUN pip install --no-cache-dir pipenv

# Copy the Pipfile and Pipfile.lock to cache the dependencies
COPY Pipfile /eastwood/

# Install Python dependencies
RUN pipenv install

# Copy the rest of the application
COPY . /eastwood

# Expose ports 7860 and 8080
EXPOSE 7860 8080 8000

# Optional: Set a default command or entry point
CMD ["sleep", "infinity"]
