FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . && pip install uvicorn[standard]

# Copy application code
COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/

# Create data directories if they don't exist
RUN mkdir -p /app/data/logs /app/data/chroma_db

EXPOSE 8000

CMD ["uvicorn", "food_cooker.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
