"""Case extraction result routes."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_repository
from app.core.exceptions import ExtractionErrorCode
from app.models.api import DocumentExtractionsResponse
from app.models.extraction import ExtractionError
from app.repositories.base import Repository
from app.services.document_service import DocumentServiceError

router = APIRouter()


@router.get(
    "/api/v1/cases/{case_id}/extractions",
    response_model=DocumentExtractionsResponse,
)
def get_extractions(
    case_id: str,
    repository: Annotated[Repository, Depends(get_repository)],
) -> DocumentExtractionsResponse:
    """Return all provenance-bearing field results for a case."""
    if repository.get_case(case_id) is None:
        raise DocumentServiceError(
            404,
            ExtractionError(
                code=ExtractionErrorCode.CASE_NOT_FOUND,
                message="Case not found.",
            ),
        )
    return DocumentExtractionsResponse(
        case_id=case_id,
        fields=repository.list_fields(case_id),
    )
