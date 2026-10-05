"""Document metadata and processing status models."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.extraction import ExtractionError


class DocumentStatus(StrEnum):
    """Lifecycle states for an uploaded document."""

    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"


class Document(BaseModel):
    """Document metadata stored beneath its parent Firestore case."""

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    case_id: str
    filename: str
    mime_type: str
    size: int = Field(ge=0)
    storage_path: str
    status: DocumentStatus = DocumentStatus.PROCESSING
    page_count: int = Field(default=0, ge=0)
    extraction_method: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    error: ExtractionError | None = None
