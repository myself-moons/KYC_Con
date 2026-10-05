"""Private document file route; upload and extraction APIs remain deferred."""

from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.models.document import Document
from app.repositories.base import Repository, StorageBackend
from app.repositories.firestore import FirestoreRepository
from app.repositories.storage import LocalStorageBackend

router = APIRouter()


def require_authenticated_user(request: Request) -> object:
    """Require authentication middleware to set a principal on the request."""
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return principal


def get_repository(request: Request) -> Repository:
    """Resolve a repository from app state or lazily create Firestore access."""
    repository = getattr(request.app.state, "repository", None)
    if repository is None:
        repository = FirestoreRepository()
        request.app.state.repository = repository
    return repository


def get_storage_backend(request: Request) -> StorageBackend:
    """Resolve local private storage from app state or create its default."""
    backend = getattr(request.app.state, "storage_backend", None)
    if backend is None:
        backend = LocalStorageBackend()
        request.app.state.storage_backend = backend
    return backend


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
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        if not storage.exists(document.storage_path):
            raise HTTPException(status_code=404, detail="Document file not found")
        content = storage.stream(document.storage_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Document file not found") from exc

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
