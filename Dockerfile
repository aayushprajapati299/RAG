# ==============================================================================
# Multi-Stage Dockerfile for Vectorless RAG
# Stage 1: Build the Vite React TypeScript frontend
# Stage 2: Production Python runtime serving FastAPI and static frontend assets
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Frontend Build Stage
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package manifests first for optimal Docker layer caching
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy frontend source code and compile production assets
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Production Python Runtime Stage
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runner

WORKDIR /app

# Install system dependencies:
# - curl: container healthcheck
# - tesseract-ocr: scanned PDF fallback extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source, database migrations, and assets
COPY backend/ ./backend/
COPY alembic.ini .

# Copy compiled frontend production assets from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create uploads storage directory
RUN mkdir -p /app/uploads

# Configure runtime environment
ENV PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production \
    DEBUG=false \
    UPLOAD_DIR=/app/uploads

EXPOSE 5000

# Container healthcheck querying the /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Non-debug production Uvicorn execution command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "2", "--no-access-log"]
