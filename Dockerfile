# Change the base image to Python 3.10
FROM python:3.10-slim

# Install system dependencies
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y \
        git \
        gcc \
        g++ \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create necessary directories
RUN mkdir -p /app/Reference/Publishable \
    /app/Reference/Non-Publishable \
    /app/papers \
    /app/output

# Copy the application code
COPY . .

# Command to run the application
CMD ["python", "paper_analysis.py"]