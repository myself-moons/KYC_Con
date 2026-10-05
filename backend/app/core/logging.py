"""Logging helpers that accept only non-sensitive processing metadata."""

import logging
from enum import StrEnum
from typing import Any

_SAFE_METADATA_FIELDS = frozenset(
    {
        "case_id",
        "document_id",
        "status",
        "error_code",
        "extractor_used",
        "page",
        "processing_duration",
    }
)


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
    safe_metadata = {
        key: value for key, value in metadata.items() if key in _SAFE_METADATA_FIELDS
    }
    logger.info("%s %s", event.value, safe_metadata)
