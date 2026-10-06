"""Tests for explicit, local-file-only cleanup behavior."""

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from app.core.config import Settings, get_settings
from app.repositories.storage import LocalStorageBackend
from app.scripts.cleanup import main as cleanup_main
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


def test_cleanup_cli_reports_retention_zero_without_deleting(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    upload_dir = tmp_path / "uploads"
    storage = LocalStorageBackend(upload_dir=upload_dir)
    path = storage.upload(
        str(uuid4()), str(uuid4()), "synthetic.pdf", b"synthetic", "application/pdf"
    )
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("DOCUMENT_RETENTION_MINUTES", "0")
    monkeypatch.setattr(sys, "argv", ["cleanup"])
    get_settings.cache_clear()
    try:
        cleanup_main()
    finally:
        get_settings.cache_clear()

    assert "Cleanup disabled" in capsys.readouterr().out
    assert storage.exists(path)


@pytest.mark.parametrize("dry_run", [True, False])
def test_cleanup_cli_only_deletes_without_dry_run(
    tmp_path: Path,
    monkeypatch,
    capsys,
    dry_run: bool,
) -> None:
    upload_dir = tmp_path / "uploads"
    storage = LocalStorageBackend(upload_dir=upload_dir)
    path = storage.upload(
        str(uuid4()), str(uuid4()), "synthetic.pdf", b"synthetic", "application/pdf"
    )
    file_path = upload_dir / path
    old_timestamp = (datetime.now(UTC) - timedelta(minutes=20)).timestamp()
    os.utime(file_path, (old_timestamp, old_timestamp))
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("DOCUMENT_RETENTION_MINUTES", "0")
    arguments = ["cleanup", "--older-than-minutes", "10"]
    if dry_run:
        arguments.append("--dry-run")
    monkeypatch.setattr(sys, "argv", arguments)
    get_settings.cache_clear()
    try:
        cleanup_main()
    finally:
        get_settings.cache_clear()

    output = capsys.readouterr().out
    assert path in output
    assert storage.exists(path) is dry_run
