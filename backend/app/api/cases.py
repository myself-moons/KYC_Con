"""Case creation and processing status routes."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_document_service, get_repository
from app.core.exceptions import ExtractionErrorCode
from app.models.api import CaseCreatedResponse, CaseStatusResponse
from app.models.document import DocumentStatus
from app.models.extraction import ExtractionError
from app.repositories.base import Repository
from app.services.document_service import DocumentService, DocumentServiceError

router = APIRouter()


@router.post("/api/v1/cases", response_model=CaseCreatedResponse, status_code=201)
def create_case(
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> CaseCreatedResponse:
    """Create an empty KYC document case."""
    case = service.create_case()
    return CaseCreatedResponse(case_id=case.case_id)


@router.get("/api/v1/cases/{case_id}/status", response_model=CaseStatusResponse)
def get_case_status(
    case_id: str,
    repository: Annotated[Repository, Depends(get_repository)],
) -> CaseStatusResponse:
    """Return counts for each document status in the case."""
    if repository.get_case(case_id) is None:
        raise DocumentServiceError(
            404,
            _error(ExtractionErrorCode.CASE_NOT_FOUND, "Case not found."),
        )
    documents = repository.list_documents(case_id)
    counts = {status: 0 for status in DocumentStatus}
    for document in documents:
        counts[document.status] += 1
    return CaseStatusResponse(
        case_id=case_id,
        total_documents=len(documents),
        processing=counts[DocumentStatus.PROCESSING],
        successful=counts[DocumentStatus.SUCCESS],
        partial=counts[DocumentStatus.PARTIAL],
        failed=counts[DocumentStatus.FAILED],
        unsupported=counts[DocumentStatus.UNSUPPORTED],
    )


def _error(code: ExtractionErrorCode, message: str):
    return ExtractionError(code=code, message=message)
