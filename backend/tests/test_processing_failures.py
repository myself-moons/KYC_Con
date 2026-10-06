"""Timeout and unexpected processing failures remain document-local."""

import time
from uuid import uuid4

from app.core.config import Settings
from app.core.exceptions import ExtractionErrorCode
from app.extractors.base import DocumentExtractor, ExtractionResult
from app.models.case import Case
from app.models.document import Document, DocumentStatus
from app.models.extraction import PageResult
from app.services.document_service import DocumentService
from tests.fakes import InMemoryRepository, InMemoryStorage


class FailingExtractor(DocumentExtractor):
    """Synthetic extractor that fails only a named test document."""

    def extract(self, document: Document, content: bytes) -> ExtractionResult:
        if document.filename == "broken.pdf":
            raise RuntimeError("Synthetic PAN-like text must not leak into logs")
        return ExtractionResult(
            document_id=document.document_id,
            status=DocumentStatus.SUCCESS,
            pages=[
                PageResult(
                    document_id=document.document_id,
                    page_number=1,
                    text="Synthetic successful test page.",
                )
            ],
            extraction_method="test",
        )


class SlowExtractor(DocumentExtractor):
    """Synthetic extractor used to trigger the configured timeout."""

    def extract(self, document: Document, content: bytes) -> ExtractionResult:
        time.sleep(0.1)
        return ExtractionResult(
            document_id=document.document_id,
            status=DocumentStatus.SUCCESS,
            extraction_method="test",
        )


class SelectivelyUnavailableRepository(InMemoryRepository):
    """Test repository that fails reads for one document identifier."""

    def __init__(self, failing_document_id: str) -> None:
        super().__init__()
        self._failing_document_id = failing_document_id

    def get_document(self, case_id: str, document_id: str) -> Document | None:
        if document_id == self._failing_document_id:
            raise RuntimeError("Synthetic repository failure")
        return super().get_document(case_id, document_id)


def _pending_document(
    repository: InMemoryRepository,
    storage: InMemoryStorage,
    filename: str = "synthetic.pdf",
) -> tuple[str, str]:
    case_id = str(uuid4())
    document_id = str(uuid4())
    repository.save_case(Case(case_id=case_id))
    path = storage.upload(
        case_id, document_id, filename, b"synthetic test bytes", "application/pdf"
    )
    repository.save_document(
        Document(
            document_id=document_id,
            case_id=case_id,
            filename=filename,
            mime_type="application/pdf",
            size=20,
            storage_path=path,
            status=DocumentStatus.PROCESSING,
        )
    )
    return case_id, document_id


def test_unexpected_exception_is_retryable_and_document_local(
    caplog,
) -> None:
    repository = InMemoryRepository()
    storage = InMemoryStorage()
    broken_case, broken_id = _pending_document(repository, storage, "broken.pdf")
    good_case, good_id = _pending_document(repository, storage, "good.pdf")
    service = DocumentService(
        repository,
        storage,
        Settings(_env_file=None),
        pdf_extractor=FailingExtractor(),
    )

    with caplog.at_level("INFO"):
        failed = service.process_document(broken_case, broken_id)
        succeeded = service.process_document(good_case, good_id)

    assert failed is not None
    assert failed.status is DocumentStatus.FAILED
    assert failed.error is not None
    assert failed.error.code is ExtractionErrorCode.UNKNOWN_EXTRACTION_ERROR
    assert failed.error.retryable is True
    assert succeeded is not None
    assert succeeded.status is DocumentStatus.SUCCESS
    assert "Synthetic PAN-like text" not in caplog.text


def test_processing_timeout_becomes_retryable_failure() -> None:
    repository = InMemoryRepository()
    storage = InMemoryStorage()
    case_id, document_id = _pending_document(repository, storage)
    service = DocumentService(
        repository,
        storage,
        Settings(_env_file=None, processing_timeout_seconds=0.005),
        pdf_extractor=SlowExtractor(),
    )

    document = service.process_document(case_id, document_id)

    assert document is not None
    assert document.status is DocumentStatus.FAILED
    assert document.error is not None
    assert document.error.code is ExtractionErrorCode.PROCESSING_TIMEOUT
    assert document.error.retryable is True


def test_repository_failure_does_not_prevent_processing_next_document() -> None:
    storage = InMemoryStorage()
    repository = InMemoryRepository()
    broken_case, broken_id = _pending_document(repository, storage, "broken.pdf")
    good_case, good_id = _pending_document(repository, storage, "good.pdf")
    failing_repository = SelectivelyUnavailableRepository(broken_id)
    failing_repository.cases = repository.cases
    failing_repository.documents = repository.documents
    service = DocumentService(
        failing_repository,
        storage,
        Settings(_env_file=None),
        pdf_extractor=FailingExtractor(),
    )

    assert service.process_document(broken_case, broken_id) is None
    succeeded = service.process_document(good_case, good_id)

    assert succeeded is not None
    assert succeeded.status is DocumentStatus.SUCCESS
