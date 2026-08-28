FROM python:3.11-slim

# System dependencies for pdfplumber, docx, etc.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and configure generous network timeouts
RUN pip install --no-cache-dir --upgrade pip

# Pre-install lightweight CPU-only PyTorch (avoids downloading 1GB+ CUDA wheels and prevents timeouts)
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies with retry resilience
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

# Copy application source code
COPY . .

# Ensure persistent data directory structure exists
RUN mkdir -p data/uploads data/vector_db data/memory

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
