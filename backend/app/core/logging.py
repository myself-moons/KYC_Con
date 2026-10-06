"""Logging helpers that accept only non-sensitive processing metadata."""

import logging
import re
from enum import StrEnum
from typing import Any

_SAFE_METADATA_FIELDS = frozenset(
    {
        "case_id",
        "document_id",
        "filename",
        "status",
        "error_code",
        "extractor",
        "duration",
    }
)
_PAN_PATTERN = re.compile(r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])", re.I)
_AADHAAR_PATTERN = re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])")


class LogEvent(StrEnum):
    """Approved fixed event names for document processing logs."""

    UPLOAD_VALIDATED = "upload_validated"
    DOCUMENT_PROCESSING_STARTED = "document_processing_started"
    DOCUMENT_PROCESSING_FINISHED = "document_processing_finished"
    DOCUMENT_PROCESSING_FAILED = "document_processing_failed"
    STORAGE_OPERATION = "storage_operation"


def log_document_event(
    logger: logging.Logger, event: LogEvent, **metadata: Any
) -> None:
    """Log allowlisted metadata, dropping document contents and unknown fields."""
    safe_metadata = {}
    for key, value in metadata.items():
        if key not in _SAFE_METADATA_FIELDS:
            continue
        if key == "filename" and isinstance(value, str):
            value = _PAN_PATTERN.sub("[REDACTED]", value)
            value = _AADHAAR_PATTERN.sub("[REDACTED]", value)
        safe_metadata[key] = value
    logger.info("%s %s", event.value, safe_metadata)
