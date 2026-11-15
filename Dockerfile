# ── TALASH — Production Dockerfile ──────────────────────────────────────────
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data directories exist
RUN mkdir -p data/analysis/jd_matches \
             data/uploads \
             data/extracted \
             data/logs \
             data/cvs \
             frontend

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 talash && chown -R talash:talash /app
USER talash

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3   CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/api/v1/health')" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
