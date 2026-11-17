import os
from functools import lru_cache
from typing import Optional

# Attempt to load local .env file if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _validate_production_settings(
    environment: str,
    database_url: Optional[str],
    groq_api_key: Optional[str],
    secret_key: Optional[str],
) -> None:
    """Validate that mandatory settings are provided when running in production."""
    is_prod = environment.strip().lower() in ("production", "prod")
    if not is_prod:
        return

    missing_or_invalid = []
    if not groq_api_key or not groq_api_key.strip():
        missing_or_invalid.append("GROQ_API_KEY (must be set in production)")

    if not database_url or not database_url.strip():
        missing_or_invalid.append("DATABASE_URL (must be set in production)")
    elif database_url.strip().lower().startswith("sqlite"):
        missing_or_invalid.append("DATABASE_URL (production requires PostgreSQL, not SQLite)")

    if not secret_key or not secret_key.strip():
        missing_or_invalid.append("SECRET_KEY (must be set in production)")
    elif secret_key.strip() == "dev-secret-key-do-not-use-in-production":
        missing_or_invalid.append("SECRET_KEY (cannot use default development key in production)")

    if missing_or_invalid:
        raise RuntimeError(
            "Configuration error: Missing or invalid required production setting(s):\n- "
            + "\n- ".join(missing_or_invalid)
        )


try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import model_validator

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

        ENVIRONMENT: str = "development"
        DATABASE_URL: str = "sqlite:///./dev.db"
        GROQ_API_KEY: str = ""
        SECRET_KEY: str = "dev-secret-key-do-not-use-in-production"
        GROQ_MODEL: str = "openai/gpt-oss-20b"
        GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
        GROQ_TEMPERATURE: float = 0.2
        GROQ_MAX_TOKENS: int = 1024
        GROQ_TIMEOUT_SECONDS: float = 30.0
        MAX_CONTEXT_CHUNKS: int = 5
        MAX_CONTEXT_CHARS: int = 4000
        MAX_HISTORY_TURNS: int = 5
        UPLOAD_DIR: str = "uploads"
        MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
        TESSERACT_CMD: Optional[str] = None
        CHUNK_SIZE: int = 500
        CHUNK_OVERLAP: int = 50
        HYBRID_BM25_WEIGHT: float = 0.5
        HYBRID_VECTOR_WEIGHT: float = 0.5
        HOST: str = "0.0.0.0"
        PORT: int = 5000
        DEBUG: bool = False

        @property
        def is_production(self) -> bool:
            return self.ENVIRONMENT.strip().lower() in ("production", "prod")

        @property
        def is_development(self) -> bool:
            return self.ENVIRONMENT.strip().lower() in ("development", "dev", "local")

        @property
        def is_testing(self) -> bool:
            return self.ENVIRONMENT.strip().lower() in ("testing", "test")

        @model_validator(mode="after")
        def validate_production(self):
            _validate_production_settings(
                environment=self.ENVIRONMENT,
                database_url=self.DATABASE_URL,
                groq_api_key=self.GROQ_API_KEY,
                secret_key=self.SECRET_KEY,
            )
            return self

except ImportError:
    # Fallback when pydantic-settings is not yet installed in the current environment
    class Settings:  # type: ignore
        def __init__(self):
            self.ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development")
            self.DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./dev.db")
            self.GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
            self.SECRET_KEY: str = os.environ.get(
                "SECRET_KEY", "dev-secret-key-do-not-use-in-production"
            )
            self.GROQ_MODEL: str = os.environ.get(
                "GROQ_MODEL", "openai/gpt-oss-20b"
            )
            self.GROQ_API_URL: str = os.environ.get(
                "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
            )
            self.GROQ_TEMPERATURE: float = float(os.environ.get("GROQ_TEMPERATURE", "0.2"))
            self.GROQ_MAX_TOKENS: int = int(os.environ.get("GROQ_MAX_TOKENS", "1024"))
            self.GROQ_TIMEOUT_SECONDS: float = float(
                os.environ.get("GROQ_TIMEOUT_SECONDS", "30.0")
            )
            self.MAX_CONTEXT_CHUNKS: int = int(os.environ.get("MAX_CONTEXT_CHUNKS", "5"))
            self.MAX_CONTEXT_CHARS: int = int(os.environ.get("MAX_CONTEXT_CHARS", "4000"))
            self.MAX_HISTORY_TURNS: int = int(os.environ.get("MAX_HISTORY_TURNS", "5"))
            self.UPLOAD_DIR: str = os.environ.get("UPLOAD_DIR", "uploads")
            self.MAX_UPLOAD_SIZE_BYTES: int = int(
                os.environ.get("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024))
            )
            self.TESSERACT_CMD: Optional[str] = os.environ.get("TESSERACT_CMD")
            self.CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "500"))
            self.CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "50"))
            self.HYBRID_BM25_WEIGHT: float = float(os.environ.get("HYBRID_BM25_WEIGHT", "0.5"))
            self.HYBRID_VECTOR_WEIGHT: float = float(os.environ.get("HYBRID_VECTOR_WEIGHT", "0.5"))
            self.HOST: str = os.environ.get("HOST", "0.0.0.0")
            self.PORT: int = int(os.environ.get("PORT", "5000"))
            self.DEBUG: bool = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")

            _validate_production_settings(
                environment=self.ENVIRONMENT,
                database_url=self.DATABASE_URL,
                groq_api_key=self.GROQ_API_KEY,
                secret_key=self.SECRET_KEY,
            )

        @property
        def is_production(self) -> bool:
            return self.ENVIRONMENT.strip().lower() in ("production", "prod")

        @property
        def is_development(self) -> bool:
            return self.ENVIRONMENT.strip().lower() in ("development", "dev", "local")

        @property
        def is_testing(self) -> bool:
            return self.ENVIRONMENT.strip().lower() in ("testing", "test")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()


settings = get_settings()
