"""Private local file storage implementation."""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from app.core.config import Settings, get_settings
from app.repositories.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Store uploads outside served paths using UUID-only file names."""

    def __init__(
        self, settings: Settings | None = None, upload_dir: Path | str | None = None
    ) -> None:
        self._settings = settings or get_settings()
        configured_dir = (
            upload_dir if upload_dir is not None else self._settings.upload_dir
        )
        self._root = Path(configured_dir).expanduser().resolve()

    def upload(
        self,
        case_id: str,
        document_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        self._validate_original_filename(filename)
        case_segment = self._uuid_segment(case_id, "case_id")
        document_segment = self._uuid_segment(document_id, "document_id")
        storage_path = PurePosixPath(
            case_segment, document_segment, uuid4().hex
        ).as_posix()
        destination = self._resolve(storage_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as upload_file:
            upload_file.write(content)
        return storage_path

    def download(self, storage_path: str) -> bytes:
        return self._resolve(storage_path).read_bytes()

    def stream(self, storage_path: str) -> Iterator[bytes]:
        with self._resolve(storage_path).open("rb") as stored_file:
            while chunk := stored_file.read(64 * 1024):
                yield chunk

    def delete(self, storage_path: str) -> None:
        path = self._resolve(storage_path)
        path.unlink(missing_ok=True)
        for parent in (path.parent, path.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                break

    def exists(self, storage_path: str) -> bool:
        return self._resolve(storage_path).is_file()

    def list_files_older_than(self, cutoff: datetime) -> list[str]:
        """List safe object keys whose modification time is at or before cutoff."""
        if not self._root.exists():
            return []
        candidates: list[str] = []
        for path in self._root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative_path = path.relative_to(self._root).as_posix()
            try:
                resolved = self._resolve(relative_path)
            except ValueError:
                continue
            if resolved.stat().st_mtime <= cutoff.timestamp():
                candidates.append(relative_path)
        return candidates

    def _resolve(self, storage_path: str) -> Path:
        """Resolve an object key and reject traversal or symlink escapes."""
        if "\\" in storage_path:
            raise ValueError("storage path must use POSIX separators")
        relative = PurePosixPath(storage_path)
        if relative.is_absolute() or len(relative.parts) != 3:
            raise ValueError(
                "storage path must contain exactly three relative segments"
            )
        case_segment = self._uuid_segment(relative.parts[0], "case_id")
        document_segment = self._uuid_segment(relative.parts[1], "document_id")
        file_segment = self._uuid_segment(relative.parts[2], "stored filename")
        resolved = (
            self._root / case_segment / document_segment / file_segment
        ).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError("storage path escapes the upload directory")
        return resolved

    @staticmethod
    def _validate_original_filename(filename: str) -> None:
        if (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
        ):
            raise ValueError("filename must be a safe base name")

    @staticmethod
    def _uuid_segment(value: str, name: str) -> str:
        try:
            parsed = UUID(value)
            return parsed.hex if name == "stored filename" else str(parsed)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"{name} must be a UUID") from exc
