# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY run_reservoir.py .
COPY templates/ ./templates

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir reservoirpy numpy matplotlib scikit-learn flask

# Expose port for Flask app
EXPOSE 5000

# Run Flask app
CMD ["python", "run_reservoir.py"]
