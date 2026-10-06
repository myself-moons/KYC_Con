"""B1 settings, status/error, provenance, and in-memory contract tests."""

from datetime import UTC, datetime

import pytest
from app.core.config import Settings
from app.core.exceptions import ExtractionErrorCode
from app.models.case import Case
from app.models.document import Document, DocumentStatus
from app.models.extraction import (
    BoundingBox,
    ExtractedField,
    ExtractionError,
    FieldExtractionResult,
    FieldExtractionStatus,
    PageResult,
)
from pydantic import ValidationError
from tests.fakes import InMemoryRepository, InMemoryStorage


def test_settings_defaults_and_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EXTENSIONS", ".PDF,.png,.jpg")
    settings = Settings(_env_file=None)

    assert settings.document_retention_minutes == 0
    assert settings.upload_dir.as_posix() == "data/uploads"
    assert settings.max_upload_mb == 20
    assert settings.max_files_per_case == 15
    assert settings.pdf_min_text_chars_per_page == 50
    assert settings.pdf_min_alnum_ratio == 0.5
    assert settings.ocr_engine == "paddleocr"
    assert settings.ocr_dpi == 300
    assert settings.ocr_min_confidence == 0.5
    assert settings.ocr_line_min_confidence == 0.5
    assert settings.ocr_preprocess_grayscale is True
    assert settings.ocr_preprocess_denoise is True
    assert settings.ocr_preprocess_deskew is True
    assert settings.ocr_preprocess_adaptive_threshold is False
    assert settings.processing_timeout_seconds == 30.0
    assert settings.use_emulator is True
    assert settings.allowed_extensions == (".pdf", ".png", ".jpg")


def test_error_enum_includes_srs_and_b2_codes() -> None:
    srs_codes = {
        "UNSUPPORTED_FILE_TYPE",
        "FILE_TOO_LARGE",
        "FILE_CORRUPTED",
        "PDF_PARSE_FAILED",
        "DOCX_PARSE_FAILED",
        "OCR_FAILED",
        "OCR_LOW_CONFIDENCE",
        "NO_TEXT_DETECTED",
        "PARTIAL_EXTRACTION",
        "PROCESSING_TIMEOUT",
        "STORAGE_ERROR",
        "UNKNOWN_EXTRACTION_ERROR",
    }
    assert srs_codes <= {code.value for code in ExtractionErrorCode}
    assert "OCR_NOT_AVAILABLE" not in {code.value for code in ExtractionErrorCode}
    assert "FILE_UNAVAILABLE" in {code.value for code in ExtractionErrorCode}
    error = ExtractionError(
        code=ExtractionErrorCode.FILE_CORRUPTED,
        message="Corrupted synthetic fixture",
    )
    assert error.model_dump(mode="json")["code"] == "FILE_CORRUPTED"


def test_document_status_values_and_provenance() -> None:
    assert {status.value for status in DocumentStatus} == {
        "PROCESSING",
        "SUCCESS",
        "PARTIAL",
        "FAILED",
        "UNSUPPORTED",
    }
    field = ExtractedField(
        field="CompanyName",
        value="Example Co",
        case_id="case-1",
        document_id="document-1",
        filename="synthetic.pdf",
        page=1,
        source_text="Company: Example Co",
        confidence=0.9,
        source_bbox=None,
        extraction_method="pdf_text",
        timestamp=datetime.now(UTC),
    )

    assert field.source_bbox is None
    assert field.timestamp.tzinfo == UTC
    with pytest.raises(ValidationError):
        BoundingBox(x=0, y=0, width=0, height=1)


def test_not_found_and_could_not_extract_are_distinct() -> None:
    not_found = FieldExtractionResult(
        field="PAN", status=FieldExtractionStatus.NOT_FOUND
    )
    failure = FieldExtractionResult(
        field="PAN",
        status=FieldExtractionStatus.COULD_NOT_EXTRACT,
        error=ExtractionError(
            code=ExtractionErrorCode.NO_TEXT_DETECTED,
            message="No text in synthetic fixture",
        ),
    )

    assert not_found.error is None
    assert failure.error is not None
    with pytest.raises(ValidationError):
        FieldExtractionResult(
            field="PAN", status=FieldExtractionStatus.COULD_NOT_EXTRACT
        )


def test_in_memory_repository_round_trips_records() -> None:
    repository = InMemoryRepository()
    case = Case(case_id="case-test")
    document = Document(
        document_id="doc-test",
        case_id=case.case_id,
        filename="fixture.pdf",
        mime_type="application/pdf",
        size=4,
        storage_path="fixture-object",
    )
    page = PageResult(document_id=document.document_id, page_number=1, text="text")

    repository.save_case(case)
    repository.save_document(document)
    repository.save_page(case.case_id, page)

    assert repository.get_case(case.case_id) == case
    assert repository.get_document(case.case_id, document.document_id) == document
    assert repository.get_page(case.case_id, document.document_id, 1) == page


def test_in_memory_storage_round_trips_and_streams() -> None:
    storage = InMemoryStorage()
    path = storage.upload(
        "case", "document", "fixture.pdf", b"fake bytes", "application/pdf"
    )

    assert storage.download(path) == b"fake bytes"
    assert b"".join(storage.stream(path)) == b"fake bytes"
    assert storage.exists(path)
    storage.delete(path)
    assert not storage.exists(path)
