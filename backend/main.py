import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import Base, engine
import backend.models  # Ensure models are registered with Base metadata
from backend.routers import auth, chat, documents, health

logger = logging.getLogger("backend.main")

# Never enable debug mode in production
debug_mode = settings.DEBUG if settings.is_development else False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager: ensures tables exist on startup and logs diagnostics."""
    logger.info("Application starting up with active Groq model: %s", settings.GROQ_MODEL)
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:
        # If DB connection isn't immediately ready, let migration or retry handle it
        pass
    yield


app = FastAPI(
    title="Vectorless RAG API",
    version="1.0.0",
    debug=debug_mode,
    lifespan=lifespan,
)

# Mount API routers first
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(documents.chunks_router)
app.include_router(chat.router)
app.include_router(chat.conversations_router)

# Mount production frontend static files if dist directory is present
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_development and settings.DEBUG,
    )
