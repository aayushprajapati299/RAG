# Vectorless RAG Application

An end-to-end, production-ready Retrieval-Augmented Generation (RAG) platform packaged as a single understandable application backed by a single PostgreSQL database:

- **Backend**: FastAPI with async route handlers, server-sent events (SSE) streaming, SQLAlchemy 2.0 ORM, and PostgreSQL with `pgvector`.
- **Hybrid Retrieval**: Combines BM25 Okapi lexical scoring with dense cosine similarity vector search using a lightweight local `sentence-transformers` model (`all-MiniLM-L6-v2`), with automated graceful fallback to pure BM25 if embeddings are unavailable.
- **Frontend**: A modern Vite + React 18 + TypeScript single-page application styled with Tailwind CSS, supporting dark/light mode, real-time citation viewing, drag-and-drop file upload with progress tracking, accessible command palette (`Ctrl+K`), and streaming chat answers.
- **Security**: Double-submit CSRF protection, HttpOnly SameSite=Lax session cookies, bcrypt password hashing, and strict multi-tenant authorization boundaries ensuring users can only access their own documents and conversations.
- **Single Database & Application**: All user profiles, documents, chunks, embeddings, conversations, and messages are stored in a single PostgreSQL database with `pgvector`. No separate vector database, Redis, Celery, or external queues are required.

---

## Architecture Overview

```
               ┌────────────────────────────────────────────────────────┐
               │              Web Browser / Client                      │
               │  - React 18 + TypeScript + Tailwind CSS UI             │
               │  - Dark/Light Theme & Command Palette (Ctrl+K)         │
               │  - SSE Streaming Chat & Accessible Citation Inspector   │
               └───────────────────────────┬────────────────────────────┘
                                           │ HTTP / SSE (/api/*, /health)
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │           FastAPI Application (backend.main:app)       │
               │  - Production Uvicorn (non-debug, multi-worker)        │
               │  - Static SPA Hosting (/ -> frontend/dist)             │
               │  - Health Check Endpoints: /health & /api/health       │
               │  - Auth & CSRF Middleware (HttpOnly Cookies)           │
               └──────────────┬──────────────────────────┬──────────────┘
                              │                          │
           Ingestion & Hybrid │                          │ LLM Generation
           Retrieval          │                          │ (Streaming SSE)
                              ▼                          ▼
               ┌───────────────────────────┐  ┌─────────────────────────┐
               │ PostgreSQL 16 + pgvector  │  │ Groq API                │
               │ - Chunk text & BM25 index │  │ - openai/gpt-oss-20b    │
               │ - 384-dim dense vectors   │  │ - 30s timeout fallback  │
               │ - Strict tenant isolation │  │ - Safe structured errors│
               └───────────────────────────┘  └─────────────────────────┘
```

---

## Key Features & Verified Capabilities

1. **Local Embeddings (Zero Paid Embedding Costs)**:
   - Dense embeddings are generated locally on CPU using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) via `fastembed` / `sentence-transformers`.
   - If embedding generation is unavailable or fails, queries automatically degrade gracefully to pure BM25 lexical ranking without throwing 500 errors.

2. **Document Ingestion Pipeline**:
   - Supports `.pdf` (PyMuPDF text extraction with optional Tesseract OCR fallback for scanned pages) and `.txt` files.
   - Client-side and server-side validation for file types and configurable size limits (default 25 MB).
   - In-process background ingestion tracking with polling status endpoints (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`).

3. **Grounded Generation & Citation Enforcement**:
   - Strict system prompting treats document text as untrusted reference context (`<untrusted_document_context>`), instructs the LLM to answer **only** from retrieved reference context, and requires it to explicitly state when evidence is missing.
   - Citations return source chunk IDs and page numbers; clicking a citation opens an accessible source panel verifying chunk text via an owned chunk endpoint.

4. **Single-Container Deployment**:
   - Multi-stage `Dockerfile` (Node 20 build stage + `python:3.12-slim` runner stage) compiles the Vite React application and embeds `frontend/dist` directly into the FastAPI container.
   - FastAPI serves the React SPA at `/` while serving API endpoints under `/api` and health checks at `/health`. No Nginx or external reverse proxy is required.

---

## Directory Structure

```
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI (free features, no secrets required)
├── backend/
│   ├── main.py              # Canonical FastAPI application entry point
│   ├── config.py            # Pydantic Settings & environment validation
│   ├── database.py          # SQLAlchemy 2.0 engine, SessionLocal, Base
│   ├── models.py            # User, Document, IngestionJob, Chunk, Conversation, Message
│   ├── security.py          # Bcrypt hashing, session cookies, CSRF tokens
│   ├── migrations/          # Alembic migrations (versions/ & env.py)
│   ├── routers/             # auth, chat, documents, health
│   ├── schemas/             # Pydantic schemas (auth, chat, documents)
│   └── services/            # document_ingestion, retrieval, chat, embeddings
├── frontend/                # React 18 + Vite + TypeScript + Tailwind CSS application
│   ├── src/                 # Components, hooks, pages, lib/api.ts
│   └── dist/                # Production build output mounted by FastAPI
├── tests/                   # Pytest suite with dedicated test database & mocks
├── Dockerfile               # Multi-stage production container (Node 20 + Python 3.12-slim)
├── compose.yaml             # Docker Compose for local PostgreSQL + API orchestration
├── render.yaml              # Render Blueprint specification for Docker deployment
├── alembic.ini              # Alembic configuration pointing to backend/migrations
└── requirements.txt         # Production Python dependencies
```

---

## Prerequisites

- **Docker & Docker Compose** (recommended for quickest setup), OR:
- **Python 3.11+ / 3.12+**
- **Node.js 20+ & npm**
- **PostgreSQL 16** with the `pgvector` extension installed
- **Groq API Key** (obtain free from [console.groq.com](https://console.groq.com))

---

## Quick Start with Docker Compose

Docker Compose runs both the PostgreSQL database (with `pgvector`) and the web application container locally:

1. **Clone the repository and copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Configure your Groq API Key** in `.env`:
   ```dotenv
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```

3. **Start the application**:
   ```bash
   docker compose up --build
   ```

4. **Access the application**:
   - Web Application: [http://localhost:5000](http://localhost:5000)
   - Healthcheck Endpoint: [http://localhost:5000/health](http://localhost:5000/health)

---

## Local Development Setup (Without Docker)

### 1. Database Setup
Ensure PostgreSQL is running locally on port 5432 and create the application and test databases:
```sql
CREATE DATABASE rag_db;
CREATE DATABASE rag_test_db;

\c rag_db;
CREATE EXTENSION IF NOT EXISTS vector;

\c rag_test_db;
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start backend development server
uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload
```

### 3. Frontend Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server (proxies /api requests to http://localhost:5000)
npm run dev -- --port 3000
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Database Migrations

Database schema migrations are managed with **Alembic** (`backend/migrations/`):
- `9aaf509f8bd2_initial_schema.py`: Creates tables for `users`, `conversations`, `documents`, `ingestion_jobs`, `chunks`, and `messages`.
- `b2c3d4e5f6a7_add_vector_column_to_chunk.py`: Enables PostgreSQL `vector` extension and adds the 384-dimensional `embedding` column to `chunks`.

To run migrations manually:
```bash
alembic upgrade head
```
*(Note: FastAPI lifespan also executes `Base.metadata.create_all(bind=engine)` automatically on startup if tables do not exist).*

---

## Production Build & Serving Strategy

The application uses a simple, robust production deployment strategy:
1. **Frontend Compilation**:
   `npm run build` runs `tsc -b` and Vite build, emitting optimized static assets into `frontend/dist`.
2. **FastAPI Static Mounting**:
   `backend/main.py` conditionally mounts `frontend/dist` via Starlette's `StaticFiles(directory="frontend/dist", html=True)`.
3. **Execution Command**:
   Production runtime uses a non-debug, multi-worker Uvicorn process:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 5000 --workers 2 --no-access-log
   ```

---

## Running the Automated Test Suite

The test suite contains 32 comprehensive tests spanning authentication, ingestion, hybrid retrieval, Groq timeout/error handling, and user isolation. External APIs (Groq and sentence-transformers) are fully mocked:

```bash
# Set test environment and run pytest
pytest tests/ -v
```

To run frontend typecheck and production build:
```bash
cd frontend
npm run build
```

---

## Continuous Integration (GitHub Actions)

A GitHub Actions workflow is configured under `.github/workflows/ci.yml`. On every push and pull request to `main`, `master`, and `modernization`:
- Launches an ephemeral `pgvector/pgvector:pg16` service container on port 5432.
- Installs Python 3.11, Tesseract OCR, and project dependencies.
- Executes `pytest tests/ -v` using dummy test credentials and mocks (no repository secrets required).
- Installs Node.js 20 dependencies and runs `npm run build` to verify frontend compilation.

---

## Cloud Deployment (Render)

A `render.yaml` Blueprint specification is provided for one-click deployment to Render using Docker:
1. Connect your repository to Render.
2. Render detects `render.yaml` and deploys the multi-stage `Dockerfile`.
3. Set the following environment variables in the Render Dashboard:
   - `GROQ_API_KEY`: Your Groq API key.
   - `DATABASE_URL`: Connection string to your managed PostgreSQL database with `pgvector`.
   - `SECRET_KEY`: A secure random 32+ character string.
4. Render monitors application health via `GET /health`.

---

## Environment Configuration Reference

All settings are managed via Pydantic Settings in `backend/config.py`:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Environment mode (`development`, `testing`, `production`) |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/rag_db` | PostgreSQL connection URL |
| `TEST_DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/rag_test_db` | Pytest suite database URL |
| `GROQ_API_KEY` | `""` | Groq API Key (required in production) |
| `SECRET_KEY` | `dev-secret-key-do-not-use-in-production` | Secret key for signing sessions and CSRF tokens |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | LLM generation model identifier |
| `GROQ_API_URL` | `https://api.groq.com/openai/v1/chat/completions` | Groq chat completions endpoint |
| `GROQ_TEMPERATURE` | `0.2` | Generation temperature |
| `GROQ_MAX_TOKENS` | `1024` | Maximum completion tokens |
| `GROQ_TIMEOUT_SECONDS`| `30.0` | Upstream Groq request timeout in seconds |
| `MAX_CONTEXT_CHUNKS` | `5` | Maximum number of chunks supplied to prompt context |
| `MAX_CONTEXT_CHARS` | `4000` | Maximum character length of reference context |
| `MAX_HISTORY_TURNS` | `5` | Prior conversation turns retained in prompt |
| `CHUNK_SIZE` | `500` | Document chunk character target size |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `HYBRID_BM25_WEIGHT` | `0.5` | Weight assigned to normalized BM25 keyword score |
| `HYBRID_VECTOR_WEIGHT`| `0.5` | Weight assigned to cosine vector similarity score |
| `UPLOAD_DIR` | `uploads` | Directory for uploaded file storage |
| `MAX_UPLOAD_SIZE_BYTES`| `26214400` (25MB) | Maximum file upload size |
| `TESSERACT_CMD` | `None` | Path to Tesseract binary (falls back to PATH) |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `5000` | Server bind port |
| `DEBUG` | `false` | Enable/disable debug logging |
