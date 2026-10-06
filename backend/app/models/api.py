"""Typed API response models for case and document endpoints."""

from datetime import datetime

from pydantic import BaseModel

from app.models.document import Document, DocumentStatus
from app.models.extraction import ExtractedField, ExtractionError, PageResult


class ApiErrorResponse(BaseModel):
    """Consistent error response payload returned by API routes."""

    code: str
    message: str
    document_id: str | None = None
    page: int | None = None
    retryable: bool = False


class CaseCreatedResponse(BaseModel):
    """Response containing the newly created case identifier."""

    case_id: str


class UploadedDocumentResponse(BaseModel):
    """Validation and initial processing status for one uploaded file."""

    document_id: str
    filename: str
    status: DocumentStatus
    error: ExtractionError | None = None


class UploadDocumentsResponse(BaseModel):
    """One independent validation result for every uploaded file."""

    case_id: str
    documents: list[UploadedDocumentResponse]


class CaseStatusResponse(BaseModel):
    """Document processing counts for one case."""

    case_id: str
    total_documents: int
    processing: int
    successful: int
    partial: int
    failed: int
    unsupported: int


class DocumentResponse(BaseModel):
    """Public document metadata without its private storage path."""

    document_id: str
    case_id: str
    filename: str
    mime_type: str
    size: int
    status: DocumentStatus
    page_count: int
    extraction_method: str | None = None
    created_at: datetime
    expires_at: datetime | None = None
    error: ExtractionError | None = None
    file_url: str | None = None

    @classmethod
    def from_document(cls, document: Document) -> "DocumentResponse":
        """Build the public response while withholding internal storage keys."""
        return cls(
            document_id=document.document_id,
            case_id=document.case_id,
            filename=document.filename,
            mime_type=document.mime_type,
            size=document.size,
            status=document.status,
            page_count=document.page_count,
            extraction_method=document.extraction_method,
            created_at=document.created_at,
            expires_at=document.expires_at,
            error=document.error,
            file_url=(
                f"/api/v1/cases/{document.case_id}/documents/"
                f"{document.document_id}/file"
                if document.storage_path
                else None
            ),
        )


class DocumentExtractionsResponse(BaseModel):
    """All extracted fields recorded for a case."""

    case_id: str
    fields: list[ExtractedField]


class PageTextResponse(PageResult):
    """Page response using the same schema as the stored page result."""


class RetryResponse(BaseModel):
    """Acknowledgement and current status after a retry is scheduled."""

    document_id: str
    status: DocumentStatus
