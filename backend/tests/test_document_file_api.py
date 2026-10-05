"""Tests for private, authenticated-ready document file streaming."""

import pytest
from app.api.documents import (
    get_repository,
    get_storage_backend,
    require_authenticated_user,
)
from app.main import app
from app.models.document import Document
from fastapi.testclient import TestClient
from tests.fakes import InMemoryRepository, InMemoryStorage


def test_file_endpoint_requires_principal(client: TestClient) -> None:
    response = client.get("/api/v1/cases/case-1/documents/document-1/file")

    assert response.status_code == 401


def test_file_endpoint_streams_private_file(
    client: TestClient,
    memory_repository: InMemoryRepository,
    memory_storage: InMemoryStorage,
) -> None:
    case_id = "case-1"
    document_id = "document-1"
    content = b"synthetic private document bytes"
    storage_path = memory_storage.upload(
        case_id, document_id, "source.pdf", content, "application/pdf"
    )
    memory_repository.save_document(
        Document(
            document_id=document_id,
            case_id=case_id,
            filename="source.pdf",
            mime_type="application/pdf",
            size=len(content),
            storage_path=storage_path,
        )
    )
    app.dependency_overrides[require_authenticated_user] = lambda: object()
    app.dependency_overrides[get_repository] = lambda: memory_repository
    app.dependency_overrides[get_storage_backend] = lambda: memory_storage
    try:
        response = client.get(f"/api/v1/cases/{case_id}/documents/{document_id}/file")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.content == content
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_file_endpoint_does_not_serve_unknown_document(
    client: TestClient,
    memory_repository: InMemoryRepository,
    memory_storage: InMemoryStorage,
) -> None:
    app.dependency_overrides[require_authenticated_user] = lambda: object()
    app.dependency_overrides[get_repository] = lambda: memory_repository
    app.dependency_overrides[get_storage_backend] = lambda: memory_storage
    try:
        response = client.get("/api/v1/cases/case-1/documents/missing/file")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
