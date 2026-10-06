"""End-to-end document status, page access, and retry behavior."""

from app.models.extraction import ExtractedField, PageResult
from fastapi.testclient import TestClient
from tests.fakes import InMemoryRepository, InMemoryStorage


def _new_case(client: TestClient) -> str:
    response = client.post("/api/v1/cases")
    assert response.status_code == 201
    return response.json()["case_id"]


def _upload(client: TestClient, case_id: str, filename: str, data: bytes) -> dict:
    response = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[("files", (filename, data, "application/pdf"))],
    )
    assert response.status_code == 200
    return response.json()["documents"][0]


def test_status_document_page_and_extractions_endpoints(
    client: TestClient, digital_pdf_bytes: bytes
) -> None:
    case_id = _new_case(client)
    upload = _upload(client, case_id, "synthetic.pdf", digital_pdf_bytes)
    document_id = upload["document_id"]

    status = client.get(f"/api/v1/cases/{case_id}/status")
    document = client.get(f"/api/v1/cases/{case_id}/documents/{document_id}")
    page = client.get(f"/api/v1/cases/{case_id}/documents/{document_id}/pages/1")
    extractions = client.get(f"/api/v1/cases/{case_id}/extractions")

    assert status.status_code == 200
    assert status.json()["total_documents"] == 1
    assert status.json()["successful"] == 1
    assert document.status_code == 200
    assert document.json()["status"] == "SUCCESS"
    assert "storage_path" not in document.json()
    assert document.json()["file_url"].endswith(f"/{document_id}/file")
    assert page.status_code == 200
    assert page.json()["page_number"] == 1
    assert page.json()["text"]
    assert page.json()["word_boxes"]
    assert extractions.status_code == 200
    assert extractions.json() == {"case_id": case_id, "fields": []}


def test_mixed_processing_outcomes_are_counted_independently(
    client: TestClient, digital_pdf_bytes: bytes, scanned_pdf_bytes: bytes
) -> None:
    case_id = _new_case(client)
    response = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[
            ("files", ("digital.pdf", digital_pdf_bytes, "application/pdf")),
            ("files", ("scanned.pdf", scanned_pdf_bytes, "application/pdf")),
            ("files", ("unsupported.xyz", b"synthetic", "text/plain")),
        ],
    )

    assert response.status_code == 200
    status = client.get(f"/api/v1/cases/{case_id}/status").json()
    assert status == {
        "case_id": case_id,
        "total_documents": 3,
        "processing": 0,
        "successful": 1,
        "partial": 0,
        "failed": 1,
        "unsupported": 1,
    }


def test_retry_clears_previous_pages_and_fields(
    client: TestClient,
    digital_pdf_bytes: bytes,
    memory_repository: InMemoryRepository,
    memory_storage: InMemoryStorage,
) -> None:
    case_id = _new_case(client)
    upload = _upload(client, case_id, "retry.pdf", digital_pdf_bytes)
    document_id = upload["document_id"]
    document = memory_repository.get_document(case_id, document_id)
    assert document is not None

    memory_repository.save_page(
        case_id,
        PageResult(document_id=document_id, page_number=99, text="stale result"),
    )
    memory_repository.save_field(
        ExtractedField(
            field="SyntheticField",
            value="stale value",
            case_id=case_id,
            document_id=document_id,
            filename=document.filename,
            page=99,
            source_text="stale source",
            extraction_method="test",
        )
    )
    document.status = document.status.FAILED
    memory_repository.save_document(document)

    response = client.post(f"/api/v1/cases/{case_id}/documents/{document_id}/retry")

    assert response.status_code == 202
    assert response.json()["status"] == "PROCESSING"
    assert memory_repository.get_page(case_id, document_id, 99) is None
    assert memory_repository.list_fields(case_id) == []
    assert memory_storage.exists(document.storage_path)


def test_retry_returns_expired_message_if_file_is_missing(
    client: TestClient,
    scanned_pdf_bytes: bytes,
    memory_repository: InMemoryRepository,
    memory_storage: InMemoryStorage,
) -> None:
    case_id = _new_case(client)
    upload = _upload(client, case_id, "expired.pdf", scanned_pdf_bytes)
    document_id = upload["document_id"]
    document = memory_repository.get_document(case_id, document_id)
    assert document is not None
    memory_storage.delete(document.storage_path)

    response = client.post(f"/api/v1/cases/{case_id}/documents/{document_id}/retry")

    assert response.status_code == 410
    error_message = response.json()["detail"]["message"]
    assert "temporary retention period has expired" in error_message


def test_retry_only_accepts_failed_documents(
    client: TestClient, digital_pdf_bytes: bytes
) -> None:
    case_id = _new_case(client)
    upload = _upload(client, case_id, "successful.pdf", digital_pdf_bytes)

    response = client.post(
        f"/api/v1/cases/{case_id}/documents/{upload['document_id']}/retry"
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "DOCUMENT_NOT_RETRYABLE"


def test_failed_document_does_not_block_valid_document(
    client: TestClient,
    digital_pdf_bytes: bytes,
    memory_repository: InMemoryRepository,
) -> None:
    case_id = _new_case(client)
    response = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[
            ("files", ("good.pdf", digital_pdf_bytes, "application/pdf")),
            ("files", ("broken.pdf", b"%PDF-1.7 broken", "application/pdf")),
        ],
    )

    assert response.status_code == 200
    outcomes = response.json()["documents"]
    assert outcomes[0]["status"] == "PROCESSING"
    assert outcomes[1]["status"] == "FAILED"
    status = client.get(f"/api/v1/cases/{case_id}/status").json()
    assert status["successful"] == 1
    assert status["failed"] == 1
    assert len(memory_repository.list_documents(case_id)) == 2
