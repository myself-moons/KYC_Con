"""Case metadata stored in the Firestore cases collection."""

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class Case(BaseModel):
    """A KYC processing case and its Firestore metadata."""

    case_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = "processing"
    document_count: int = Field(default=0, ge=0)
    expires_at: datetime | None = None
