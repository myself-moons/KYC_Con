"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Validated configuration for local development and Firebase deployments."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    firebase_project_id: str = "demo-kyc"
    use_emulator: bool = True
    firestore_emulator_host: str | None = "127.0.0.1:8080"
    google_application_credentials: str | None = None
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = Field(default=20, gt=0)
    allowed_extensions: Annotated[tuple[str, ...], NoDecode] = (
        ".pdf",
        ".docx",
        ".jpg",
        ".jpeg",
        ".png",
    )
    max_files_per_case: int = Field(default=15, gt=0)
    document_retention_minutes: int = Field(default=0, ge=0)
    pdf_min_text_chars_per_page: int = Field(default=50, ge=0)
    pdf_min_alnum_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    ocr_engine: Literal["paddleocr", "tesseract"] = "paddleocr"

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, value: object) -> tuple[str, ...]:
        """Accept a comma-separated environment value or a sequence."""
        if isinstance(value, str):
            extensions = value.split(",")
        elif isinstance(value, (list, tuple)):
            extensions = value
        else:
            raise ValueError("allowed_extensions must be a comma-separated string")

        normalized = tuple(str(extension).strip().lower() for extension in extensions)
        if not normalized or any(
            not item.startswith(".") or item == "." for item in normalized
        ):
            raise ValueError("each allowed extension must start with a dot")
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_extensions must not contain duplicates")
        return normalized


@lru_cache
def get_settings() -> Settings:
    """Return the cached process settings."""
    return Settings()
