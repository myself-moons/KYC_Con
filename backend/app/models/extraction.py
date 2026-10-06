"""Page and field extraction models with source provenance."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from app.core.exceptions import ExtractionErrorCode


class BoundingBox(BaseModel):
    """A source rectangle; coordinates are nullable at the field level."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class WordBox(BaseModel):
    """A word and its source bounding box on a page."""

    text: str
    bbox: BoundingBox
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    low_confidence: bool = False


class OCRLine(BaseModel):
    """A recognized line with normalized coordinates and word confidences."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    words: list[WordBox] = Field(default_factory=list)
    low_confidence: bool = False


class PageResult(BaseModel):
    """Page-level extracted text stored beneath a Firestore document."""

    document_id: str
    page_number: int = Field(ge=1)
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    needs_ocr: bool = False
    word_boxes: list[WordBox] = Field(default_factory=list)
    ocr_lines: list[OCRLine] = Field(default_factory=list)
    extraction_method: str | None = None
    ocr_engine_used: str | None = None
    ocr_primary_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ocr_fallback_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    low_confidence: bool = False


class ExtractionError(BaseModel):
    """A structured, user-readable processing error."""

    code: ExtractionErrorCode
    message: str
    document_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    retryable: bool = False


class ExtractedField(BaseModel):
    """An extracted value with complete document and page provenance."""

    field_id: str = Field(default_factory=lambda: str(uuid4()))
    field: str
    value: str
    case_id: str
    document_id: str
    filename: str
    page: int = Field(ge=1)
    source_text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_bbox: BoundingBox | None = None
    extraction_method: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FieldExtractionStatus(StrEnum):
    """Outcome for a field lookup, distinguishing absence from extraction failure."""

    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    COULD_NOT_EXTRACT = "COULD_NOT_EXTRACT"


class FieldExtractionResult(BaseModel):
    """The result of attempting to locate one field in source material."""

    field: str
    status: FieldExtractionStatus
    extracted_field: ExtractedField | None = None
    error: ExtractionError | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "FieldExtractionResult":
        """Require payloads that preserve the distinction between outcomes."""
        if self.status is FieldExtractionStatus.FOUND:
            if self.extracted_field is None or self.error is not None:
                raise ValueError("FOUND requires a field and cannot include an error")
        elif self.extracted_field is not None:
            raise ValueError("non-FOUND outcomes cannot include an extracted field")
        elif (
            self.status is FieldExtractionStatus.COULD_NOT_EXTRACT
            and self.error is None
        ):
            raise ValueError("COULD_NOT_EXTRACT requires a structured error")
        elif self.status is FieldExtractionStatus.NOT_FOUND and self.error is not None:
            raise ValueError("NOT_FOUND cannot include an extraction error")
        return self
