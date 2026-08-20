# ── TALASH — Production Dockerfile ──────────────────────────────────────────
# Uses slim Python image; non-root user for security.
# Railway mounts a persistent volume at /app/data so CSVs & uploads survive
# between deploys / restarts.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps for pdfplumber / pypdf (poppler utilities not needed for text extract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories so the app never errors on first boot
RUN mkdir -p data/analysis/jd_matches \
             data/uploads \
             data/extracted \
             data/logs \
             data/cvs \
             frontend

# Non-root user
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# Railway injects PORT env var; default 8080 for local Docker
ENV PORT=8080
EXPOSE 8080

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 2
