# Use the official Python slim-buster image as a parent image
FROM python:3.8.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy source code to working directory
COPY . app.py /app/

# Install packages from requirements.txt
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Uvicorn and Gunicorn
RUN pip install uvicorn gunicorn

# Expose the port
EXPOSE 8080

# Run the application
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8080"]
