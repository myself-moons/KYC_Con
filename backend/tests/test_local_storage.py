"""Security and behavior tests for UUID-based local file storage."""

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from app.repositories.storage import LocalStorageBackend


def test_local_storage_saves_reads_streams_and_deletes(tmp_path: Path) -> None:
    storage = LocalStorageBackend(upload_dir=tmp_path)
    case_id = str(uuid4())
    document_id = str(uuid4())
    content = b"synthetic sample document"

    storage_path = storage.upload(
        case_id, document_id, "sample.pdf", content, "application/pdf"
    )

    assert storage.exists(storage_path)
    assert storage.download(storage_path) == content
    assert b"".join(storage.stream(storage_path)) == content
    assert Path(storage_path).parts[:2] == (case_id, document_id)
    assert UUID(Path(storage_path).name).hex == Path(storage_path).name
    assert "sample.pdf" not in storage_path
    storage.delete(storage_path)
    assert not storage.exists(storage_path)


@pytest.mark.parametrize(
    "filename", ["../outside.pdf", "/tmp/outside.pdf", "..\\outside.pdf"]
)
def test_upload_rejects_path_traversal_filename(tmp_path: Path, filename: str) -> None:
    storage = LocalStorageBackend(upload_dir=tmp_path)

    with pytest.raises(ValueError, match="safe base name"):
        storage.upload(str(uuid4()), str(uuid4()), filename, b"data", "application/pdf")


@pytest.mark.parametrize(
    "storage_path",
    ["../outside/file", "/tmp/outside/file", "not-a-uuid/doc-id/file"],
)
def test_download_rejects_unsafe_storage_path(
    tmp_path: Path, storage_path: str
) -> None:
    storage = LocalStorageBackend(upload_dir=tmp_path)

    with pytest.raises(ValueError):
        storage.download(storage_path)


def test_missing_file_is_not_found_and_delete_is_idempotent(tmp_path: Path) -> None:
    storage = LocalStorageBackend(upload_dir=tmp_path)
    missing_path = f"{uuid4()}/{uuid4()}/{uuid4().hex}"

    assert not storage.exists(missing_path)
    with pytest.raises(FileNotFoundError):
        storage.download(missing_path)
    storage.delete(missing_path)
