from fastapi import APIRouter
from backend.config import settings

router = APIRouter(tags=["health"])


@router.get("/api/health")
@router.get("/health")
def health_check():
    """Return health status and environment info."""
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
    }
