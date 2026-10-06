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
            filename="source_ABCDE1234F_123456789012.pdf",
            status="SUCCESS",
            extractor="pdf_text",
            duration=0.1,
            pan="ABCDE1234F",
            aadhaar="000000000000",
            extracted_text="Permanent Account Number ABCDE1234F",
        )

    assert "case-1" in caplog.text
    assert "doc-1" in caplog.text
    assert "source_[REDACTED]_[REDACTED].pdf" in caplog.text
    assert "pdf_text" in caplog.text
    assert "SUCCESS" in caplog.text
    assert "ABCDE1234F" not in caplog.text
    assert "Permanent Account Number" not in caplog.text
