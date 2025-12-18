FROM apache/airflow:2.10.0-python3.11

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements
COPY pyproject.toml /tmp/pyproject.toml

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[dev]" -f /tmp/pyproject.toml || \
    pip install --no-cache-dir \
    langchain>=0.3.0 \
    langchain-community>=0.3.0 \
    langchain-core>=0.3.0 \
    beautifulsoup4>=4.12.0 \
    lxml>=5.0.0 \
    httpx>=0.27.0 \
    qdrant-client>=1.11.0 \
    sentence-transformers>=3.0.0 \
    psycopg2-binary>=2.9.0 \
    sqlalchemy>=2.0.0

# Copy shared modules
COPY --chown=airflow:root shared/ /opt/airflow/shared/

WORKDIR /opt/airflow
