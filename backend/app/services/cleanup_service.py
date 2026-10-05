"""Explicit local-file cleanup service with no automatic execution path."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.core.config import Settings, get_settings
from app.repositories.storage import LocalStorageBackend


@dataclass(frozen=True)
class CleanupReport:
    """Outcome of one manually requested cleanup run."""

    disabled: bool
    dry_run: bool
    candidates: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()


class CleanupService:
    """Delete expired local uploads only when an operator calls this service."""

    def __init__(
        self,
        storage: LocalStorageBackend,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._settings = settings or get_settings()

    def run(
        self,
        *,
        older_than_minutes: int | None = None,
        dry_run: bool,
    ) -> CleanupReport:
        """Preview or delete files older than an explicit/configured threshold."""
        retention = (
            self._settings.document_retention_minutes
            if older_than_minutes is None
            else older_than_minutes
        )
        if retention < 0:
            raise ValueError("older_than_minutes cannot be negative")
        if retention == 0:
            return CleanupReport(disabled=True, dry_run=dry_run)

        cutoff = datetime.now(UTC) - timedelta(minutes=retention)
        candidates = tuple(self._storage.list_files_older_than(cutoff))
        deleted: tuple[str, ...] = ()
        if not dry_run:
            for storage_path in candidates:
                self._storage.delete(storage_path)
            deleted = candidates
        return CleanupReport(
            disabled=False,
            dry_run=dry_run,
            candidates=candidates,
            deleted=deleted,
        )
