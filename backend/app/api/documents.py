"""Private document file route; upload and extraction APIs remain deferred."""

from typing import Annotated
from urllib.parse import quote

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    get_document_service,
    get_repository,
    get_storage_backend,
)
from app.core.exceptions import ExtractionErrorCode
from app.models.api import (
    ApiErrorResponse,
    DocumentResponse,
    PageTextResponse,
    RetryResponse,
    UploadDocumentsResponse,
    UploadedDocumentResponse,
)
from app.models.document import Document
from app.models.extraction import ExtractionError, PageResult
from app.repositories.base import Repository, StorageBackend
from app.services.document_service import DocumentService

router = APIRouter()


def require_authenticated_user(request: Request) -> object:
    """Require authentication middleware to set a principal on the request."""
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail=ApiErrorResponse(
                code="AUTHENTICATION_REQUIRED",
                message="Authentication required.",
            ).model_dump(mode="json"),
        )
    return principal


@router.get(
    "/api/v1/cases/{case_id}/documents/{document_id}/file",
    response_class=StreamingResponse,
)
def get_document_file(
    case_id: str,
    document_id: str,
    _principal: Annotated[object, Depends(require_authenticated_user)],
    repository: Annotated[Repository, Depends(get_repository)],
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
) -> StreamingResponse:
    """Stream a private document after its case ownership is verified."""
    document: Document | None = repository.get_document(case_id, document_id)
    if document is None:
        _raise_api_error(
            404,
            ExtractionErrorCode.DOCUMENT_NOT_FOUND,
            "Document not found.",
            document_id,
        )
    try:
        if not storage.exists(document.storage_path):
            _raise_api_error(
                404,
                ExtractionErrorCode.FILE_UNAVAILABLE,
                "Document file not found.",
                document_id,
            )
        content = storage.stream(document.storage_path)
    except (OSError, ValueError):
        _raise_api_error(
            404,
            ExtractionErrorCode.FILE_UNAVAILABLE,
            "Document file not found.",
            document_id,
        )

    filename = quote(document.filename, safe="")
    return StreamingResponse(
        content,
        media_type=document.mime_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/api/v1/cases/{case_id}/documents",
    response_model=UploadDocumentsResponse,
)
async def upload_documents(
    case_id: str,
    background_tasks: BackgroundTasks,
    service: Annotated[DocumentService, Depends(get_document_service)],
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> UploadDocumentsResponse:
    """Validate every uploaded file independently and schedule valid documents."""
    upload_files = files or []
    service.validate_upload_request(case_id, len(upload_files))
    max_bytes = service.max_upload_bytes
    outcomes: list[UploadedDocumentResponse] = []
    for upload in upload_files:
        try:
            content = await upload.read(max_bytes + 1)
            document = service.accept_upload(
                case_id,
                upload.filename or "",
                upload.content_type,
                content,
            )
        except Exception:
            document = service.record_unexpected_upload_failure(
                case_id,
                upload.filename or "",
                upload.content_type,
            )
        finally:
            await upload.close()
        outcomes.append(
            UploadedDocumentResponse(
                document_id=document.document_id,
                filename=document.filename,
                status=document.status,
                error=document.error,
            )
        )
        if document.status.value == "PROCESSING":
            background_tasks.add_task(
                service.process_document, case_id, document.document_id
            )
    return UploadDocumentsResponse(case_id=case_id, documents=outcomes)


@router.get(
    "/api/v1/cases/{case_id}/documents/{document_id}",
    response_model=DocumentResponse,
)
def get_document(
    case_id: str,
    document_id: str,
    repository: Annotated[Repository, Depends(get_repository)],
) -> DocumentResponse:
    """Return public metadata for one document without its storage key."""
    document = repository.get_document(case_id, document_id)
    if document is None:
        _raise_api_error(
            404,
            ExtractionErrorCode.DOCUMENT_NOT_FOUND,
            "Document not found.",
            document_id,
        )
    return DocumentResponse.from_document(document)


@router.get(
    "/api/v1/cases/{case_id}/documents/{document_id}/pages/{page_number}",
    response_model=PageTextResponse,
)
def get_page_text(
    case_id: str,
    document_id: str,
    page_number: int,
    repository: Annotated[Repository, Depends(get_repository)],
) -> PageResult:
    """Return stored text and source word boxes for one page."""
    document = repository.get_document(case_id, document_id)
    if document is None:
        _raise_api_error(
            404,
            ExtractionErrorCode.DOCUMENT_NOT_FOUND,
            "Document not found.",
            document_id,
        )
    page = repository.get_page(case_id, document_id, page_number)
    if page is None:
        _raise_api_error(
            404,
            ExtractionErrorCode.NO_TEXT_DETECTED,
            "Page extraction result not found.",
            document_id,
            page_number,
        )
    return page


@router.post(
    "/api/v1/cases/{case_id}/documents/{document_id}/retry",
    response_model=RetryResponse,
    status_code=202,
)
def retry_document(
    case_id: str,
    document_id: str,
    background_tasks: BackgroundTasks,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> RetryResponse:
    """Clear prior results and schedule a retry for one failed document."""
    document = service.retry_document(case_id, document_id)
    background_tasks.add_task(service.process_document, case_id, document_id)
    return RetryResponse(document_id=document_id, status=document.status)


def _raise_api_error(
    status_code: int,
    code: ExtractionErrorCode,
    message: str,
    document_id: str | None = None,
    page: int | None = None,
) -> None:
    error = ExtractionError(
        code=code,
        message=message,
        document_id=document_id,
        page=page,
        retryable=False,
    )
    raise HTTPException(status_code=status_code, detail=error.model_dump(mode="json"))
