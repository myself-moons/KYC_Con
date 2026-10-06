"""OCR page rendering and original-image preservation tests."""

from io import BytesIO

import pymupdf
import pytest
from app.core.config import Settings
from app.services.ocr_service import OCRPreprocessingError, OCRService
from PIL import Image, ImageChops


def _settings_without_preprocessing() -> Settings:
    return Settings(
        _env_file=None,
        ocr_preprocess_grayscale=False,
        ocr_preprocess_denoise=False,
        ocr_preprocess_deskew=False,
        ocr_preprocess_adaptive_threshold=False,
    )


def test_image_preprocessing_keeps_an_unmodified_original() -> None:
    source = Image.new("RGB", (90, 60), (210, 185, 140))
    stream = BytesIO()
    source.save(stream, format="PNG")
    prepared = OCRService(_settings_without_preprocessing()).load_image(
        stream.getvalue()
    )

    assert prepared.original_image is not prepared.processed_image
    assert (
        ImageChops.difference(
            prepared.original_image, prepared.processed_image
        ).getbbox()
        is None
    )
    assert prepared.dpi is None


def test_pdf_page_is_rendered_at_configured_ocr_dpi() -> None:
    pdf = pymupdf.open()
    page = pdf.new_page(width=72, height=72)
    page.insert_text((10, 30), "Synthetic OCR render test")
    content = pdf.tobytes()
    pdf.close()
    prepared = OCRService(_settings_without_preprocessing()).render_pdf_page(
        content, page_index=0
    )

    assert prepared.dpi == 300
    assert prepared.original_image.size == (300, 300)
    assert prepared.processed_image.size == prepared.original_image.size


def test_invalid_image_returns_a_processing_error() -> None:
    with pytest.raises(OCRPreprocessingError, match="could not be decoded"):
        OCRService(_settings_without_preprocessing()).load_image(b"not an image")


def test_enabled_preprocessing_is_applied_or_reports_runtime_dependency() -> None:
    source = Image.new("RGB", (120, 80), (210, 185, 140))
    service = OCRService(Settings(_env_file=None))
    try:
        import cv2  # noqa: F401
    except Exception:
        with pytest.raises(OCRPreprocessingError, match="OpenCV could not be imported"):
            service.prepare_image(source, dpi=None)
    else:
        prepared = service.prepare_image(source, dpi=None)
        assert prepared.original_image is not prepared.processed_image
        assert prepared.original_image.size == prepared.processed_image.size
