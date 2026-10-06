"""Per-file upload validation through the public API."""

import pytest
from fastapi.testclient import TestClient


def _create_case(client: TestClient) -> str:
    response = client.post("/api/v1/cases")
    assert response.status_code == 201
    return response.json()["case_id"]


def test_mixed_good_and_bad_files_are_reported_independently(
    client: TestClient, digital_pdf_bytes: bytes
) -> None:
    case_id = _create_case(client)
    response = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[
            ("files", ("synthetic.pdf", digital_pdf_bytes, "application/pdf")),
            ("files", ("unsupported.xyz", b"synthetic bytes", "text/plain")),
        ],
    )

    assert response.status_code == 200
    outcomes = {item["filename"]: item for item in response.json()["documents"]}
    assert outcomes["synthetic.pdf"]["status"] == "PROCESSING"
    assert outcomes["unsupported.xyz"]["status"] == "UNSUPPORTED"
    assert outcomes["unsupported.xyz"]["error"]["code"] == "UNSUPPORTED_FILE_TYPE"

    status = client.get(f"/api/v1/cases/{case_id}/status").json()
    assert status["total_documents"] == 2
    assert status["successful"] == 1
    assert status["unsupported"] == 1


def test_oversized_file_is_reported_per_file(client: TestClient) -> None:
    case_id = _create_case(client)
    response = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[("files", ("large.pdf", b"x" * (1024 * 1024 + 1), "application/pdf"))],
    )

    assert response.status_code == 200
    uploaded = response.json()["documents"][0]
    assert uploaded["status"] == "FAILED"
    assert uploaded["error"]["code"] == "FILE_TOO_LARGE"


@pytest.mark.parametrize(
    ("filename", "content", "content_type", "expected_code"),
    [
        (
            "corrupted.pdf",
            b"%PDF-1.7\nnot a valid pdf structure",
            "application/pdf",
            "FILE_CORRUPTED",
        ),
        ("empty.pdf", b"", "application/pdf", "NO_TEXT_DETECTED"),
        (
            "fake.pdf",
            b"this is actually plain text",
            "application/pdf",
            "FILE_CORRUPTED",
        ),
    ],
)
def test_unreadable_empty_and_signature_mismatch_are_per_file_failures(
    client: TestClient,
    filename: str,
    content: bytes,
    content_type: str,
    expected_code: str,
) -> None:
    case_id = _create_case(client)
    response = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[("files", (filename, content, content_type))],
    )

    assert response.status_code == 200
    outcome = response.json()["documents"][0]
    assert outcome["status"] == "FAILED"
    assert outcome["error"]["code"] == expected_code


def test_password_protected_pdf_is_unreadable(
    client: TestClient, encrypted_pdf_bytes: bytes
) -> None:
    case_id = _create_case(client)
    response = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[
            (
                "files",
                ("protected.pdf", encrypted_pdf_bytes, "application/pdf"),
            )
        ],
    )

    assert response.status_code == 200
    outcome = response.json()["documents"][0]
    assert outcome["status"] == "FAILED"
    assert outcome["error"]["code"] == "PDF_PARSE_FAILED"
    assert "password-protected" in outcome["error"]["message"]


def test_no_files_too_many_files_and_missing_case_are_request_errors(
    client: TestClient,
) -> None:
    case_id = _create_case(client)

    no_files = client.post(f"/api/v1/cases/{case_id}/documents")
    too_many = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[
            ("files", (f"{index}.txt", b"data", "text/plain")) for index in range(4)
        ],
    )
    missing_case = client.post(
        "/api/v1/cases/missing/documents",
        files=[("files", ("file.pdf", b"", "application/pdf"))],
    )

    assert no_files.status_code == 400
    assert no_files.json()["detail"]["code"] == "NO_FILES"
    assert too_many.status_code == 413
    assert too_many.json()["detail"]["code"] == "TOO_MANY_FILES"
    assert missing_case.status_code == 404
    assert missing_case.json()["detail"]["code"] == "CASE_NOT_FOUND"


def test_valid_docx_requires_zip_magic_and_document_xml(client: TestClient) -> None:
    from io import BytesIO
    from zipfile import ZIP_DEFLATED, ZipFile

    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "<w:document/>")
    case_id = _create_case(client)

    response = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[
            (
                "files",
                (
                    "synthetic.docx",
                    archive_bytes.getvalue(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            )
        ],
    )

    assert response.status_code == 200
    outcome = response.json()["documents"][0]
    assert outcome["status"] == "PROCESSING"


def test_pdf_content_type_must_match_extension(client: TestClient) -> None:
    case_id = _create_case(client)
    response = client.post(
        f"/api/v1/cases/{case_id}/documents",
        files=[("files", ("mismatch.pdf", b"%PDF-1.7", "image/png"))],
    )

    assert response.status_code == 200
    outcome = response.json()["documents"][0]
    assert outcome["status"] == "UNSUPPORTED"
    assert outcome["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
