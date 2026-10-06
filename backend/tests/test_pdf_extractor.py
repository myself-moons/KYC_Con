"""Page-level PyMuPDF extraction behavior using synthetic PDFs only."""

from collections.abc import Callable

from app.core.config import Settings
from app.core.exceptions import ExtractionErrorCode
from app.extractors.pdf import PDFTextExtractor
from app.models.document import Document, DocumentStatus


def _document() -> Document:
    return Document(
        document_id="pdf-test-document",
        case_id="pdf-test-case",
        filename="synthetic.pdf",
        mime_type="application/pdf",
        size=1,
        storage_path="synthetic/object",
    )


def test_digital_pdf_extracts_pages_and_normalized_word_boxes(
    pdf_factory: Callable[..., bytes], settings: Settings
) -> None:
    useful_text = (
        "Synthetic digital page with enough ordinary alphanumeric words for "
        "the minimum embedded text threshold."
    )
    second_text = "Another generated page contains sufficiently long readable text."
    extractor = PDFTextExtractor(settings)

    result = extractor.extract(_document(), pdf_factory([useful_text, second_text]))

    assert result.status is DocumentStatus.SUCCESS
    assert len(result.pages) == 2
    assert result.pages[0].text
    assert result.pages[0].needs_ocr is False
    assert result.pages[0].word_boxes
    for word_box in result.pages[0].word_boxes:
        assert 0 <= word_box.bbox.x <= 1
        assert 0 <= word_box.bbox.y <= 1
        assert 0 < word_box.bbox.width <= 1
        assert 0 < word_box.bbox.height <= 1


def test_image_only_pdf_is_failed_and_needs_ocr(
    pdf_factory: Callable[..., bytes], settings: Settings
) -> None:
    result = PDFTextExtractor(settings).extract(
        _document(), pdf_factory([None], image_only_pages={0})
    )

    assert result.status is DocumentStatus.FAILED
    assert result.pages[0].needs_ocr is True
    assert result.error is not None
    assert result.error.code is ExtractionErrorCode.OCR_NOT_AVAILABLE


def test_mixed_digital_and_image_pages_are_partial(
    pdf_factory: Callable[..., bytes], settings: Settings
) -> None:
    readable_text = (
        "Synthetic readable page contains enough alphanumeric text to pass checks."
    )
    result = PDFTextExtractor(settings).extract(
        _document(), pdf_factory([readable_text, None], image_only_pages={1})
    )

    assert result.status is DocumentStatus.PARTIAL
    assert [page.needs_ocr for page in result.pages] == [False, True]
    assert result.error is not None
    assert result.error.code is ExtractionErrorCode.OCR_NOT_AVAILABLE


def test_junk_text_layer_fails_alphanumeric_ratio(
    pdf_factory: Callable[..., bytes], settings: Settings
) -> None:
    junk_text = "-_=+!? " * 30 + "Synthetic"
    result = PDFTextExtractor(settings).extract(_document(), pdf_factory([junk_text]))

    assert result.status is DocumentStatus.FAILED
    assert result.pages[0].needs_ocr is True
    assert result.error is not None
    assert result.error.code is ExtractionErrorCode.OCR_NOT_AVAILABLE
