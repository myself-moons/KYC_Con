"""Firestore-only emulator integration coverage."""

import socket
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from app.core.config import Settings
from app.models.case import Case
from app.models.document import Document, DocumentStatus
from app.models.extraction import BoundingBox, ExtractedField, PageResult, WordBox
from app.repositories.firestore import FirestoreRepository

pytestmark = pytest.mark.emulator


def _host_is_reachable(host: str | None) -> bool:
    if not host:
        return False
    parsed = urlsplit(host if "://" in host else f"//{host}")
    if parsed.hostname is None or parsed.port is None:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=0.25):
            return True
    except OSError:
        return False


def test_firestore_repository_round_trip() -> None:
    settings = Settings()
    if not settings.use_emulator:
        pytest.skip("Firestore emulator tests require USE_EMULATOR=true")
    if not _host_is_reachable(settings.firestore_emulator_host):
        pytest.skip("Firestore emulator is not reachable")

    repository = FirestoreRepository(settings)
    case = Case(case_id=str(uuid4()))
    document = Document(
        document_id=str(uuid4()),
        case_id=case.case_id,
        filename="synthetic.pdf",
        mime_type="application/pdf",
        size=100,
        storage_path="synthetic/object-key",
        status=DocumentStatus.SUCCESS,
        page_count=1,
    )
    page = PageResult(
        document_id=document.document_id,
        page_number=1,
        text="Synthetic emulator fixture text.",
        word_boxes=[
            WordBox(
                text="Synthetic",
                bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.03),
            )
        ],
    )
    field = ExtractedField(
        field="SampleField",
        value="SyntheticValue",
        case_id=case.case_id,
        document_id=document.document_id,
        filename=document.filename,
        page=1,
        source_text="SampleField: SyntheticValue",
        extraction_method="synthetic_test",
    )
    try:
        repository.save_case(case)
        assert repository.get_case(case.case_id) == case
        repository.save_document(document)
        assert repository.get_document(case.case_id, document.document_id) == document
        repository.save_page(case.case_id, page)
        assert repository.get_page(case.case_id, document.document_id, 1) == page
        repository.save_field(field)
        assert repository.list_fields(case.case_id) == [field]
        repository.delete_pages(case.case_id, document.document_id)
        repository.delete_fields(case.case_id, document.document_id)
        assert repository.list_pages(case.case_id, document.document_id) == []
        assert repository.list_fields(case.case_id) == []
    finally:
        repository._client.collection("cases").document(case.case_id).collection(
            "documents"
        ).document(document.document_id).delete()
        repository._client.collection("cases").document(case.case_id).delete()
