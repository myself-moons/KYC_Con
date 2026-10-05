"""Document extractor interface and result contract."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.models.document import Document, DocumentStatus
from app.models.extraction import ExtractionError, FieldExtractionResult, PageResult


class ExtractionResult(BaseModel):
    """Results from a single document extraction attempt."""

    document_id: str
    status: DocumentStatus
    pages: list[PageResult] = Field(default_factory=list)
    field_outcomes: list[FieldExtractionResult] = Field(default_factory=list)
    error: ExtractionError | None = None
    extraction_method: str


class DocumentExtractor(ABC):
    """Interface implemented by document-specific extraction engines."""

    @abstractmethod
    def extract(self, document: Document, content: bytes) -> ExtractionResult:
        """Extract page results from document bytes without persisting them."""
