"""Tests for explicit, local-file-only cleanup behavior."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from app.core.config import Settings
from app.repositories.storage import LocalStorageBackend
from app.services.cleanup_service import CleanupService


def test_retention_zero_disables_cleanup(tmp_path: Path) -> None:
    storage = LocalStorageBackend(upload_dir=tmp_path)
    path = storage.upload(
        str(uuid4()), str(uuid4()), "synthetic.pdf", b"data", "application/pdf"
    )
    service = CleanupService(storage, Settings(_env_file=None, upload_dir=tmp_path))

    report = service.run(dry_run=False)

    assert report.disabled
    assert storage.exists(path)


def test_manual_cleanup_dry_run_keeps_file(tmp_path: Path) -> None:
    storage = LocalStorageBackend(upload_dir=tmp_path)
    path = storage.upload(
        str(uuid4()), str(uuid4()), "synthetic.pdf", b"data", "application/pdf"
    )
    full_path = tmp_path / path
    old_timestamp = (datetime.now(UTC) - timedelta(minutes=20)).timestamp()
    os.utime(full_path, (old_timestamp, old_timestamp))
    service = CleanupService(storage, Settings(_env_file=None, upload_dir=tmp_path))

    report = service.run(older_than_minutes=10, dry_run=True)

    assert report.candidates == (path,)
    assert report.deleted == ()
    assert storage.exists(path)


def test_manual_cleanup_deletes_only_after_explicit_non_dry_run(
    tmp_path: Path,
) -> None:
    storage = LocalStorageBackend(upload_dir=tmp_path)
    path = storage.upload(
        str(uuid4()), str(uuid4()), "synthetic.pdf", b"data", "application/pdf"
    )
    full_path = tmp_path / path
    old_timestamp = (datetime.now(UTC) - timedelta(minutes=20)).timestamp()
    os.utime(full_path, (old_timestamp, old_timestamp))
    service = CleanupService(storage, Settings(_env_file=None, upload_dir=tmp_path))

    report = service.run(older_than_minutes=10, dry_run=False)

    assert report.deleted == (path,)
    assert not storage.exists(path)
