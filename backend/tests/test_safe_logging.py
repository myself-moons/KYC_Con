"""Ensure sensitive document content is rejected by the logging helper."""

import logging

from app.core.logging import LogEvent, log_document_event


def test_logging_helper_keeps_only_safe_metadata(caplog) -> None:
    logger = logging.getLogger("test.document_metadata")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_document_event(
            logger,
            LogEvent.DOCUMENT_PROCESSING_FINISHED,
            case_id="case-1",
            document_id="doc-1",
            status="SUCCESS",
            filename="PAN-SYNTHETIC-PAN-SECRET.pdf",
            pan="SYNTHETIC-PAN-SECRET",
            aadhaar="SYNTHETIC-AADHAAR-SECRET",
            document_text="SYNTHETIC-DOCUMENT-SECRET",
        )

    assert "case-1" in caplog.text
    assert "doc-1" in caplog.text
    assert "SUCCESS" in caplog.text
    assert "SYNTHETIC" not in caplog.text
