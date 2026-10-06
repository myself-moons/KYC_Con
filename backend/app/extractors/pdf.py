"""Page-level embedded text extraction with normalized word coordinates."""

from collections.abc import Iterable

import pymupdf

from app.core.config import Settings, get_settings
from app.core.exceptions import ExtractionErrorCode
from app.extractors.base import DocumentExtractor, ExtractionResult
from app.models.document import Document, DocumentStatus
from app.models.extraction import BoundingBox, ExtractionError, PageResult, WordBox


class PDFTextExtractor(DocumentExtractor):
    """Extract embedded PDF text and mark pages that need OCR."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def extract(self, document: Document, content: bytes) -> ExtractionResult:
        """Extract text and word boxes for each page of a digital PDF."""
        try:
            pdf = pymupdf.open(stream=content, filetype="pdf")
        except Exception:
            return self._failed(
                document.document_id,
                ExtractionErrorCode.PDF_PARSE_FAILED,
                "The PDF could not be opened or parsed.",
            )

        with pdf:
            if pdf.needs_pass:
                return self._failed(
                    document.document_id,
                    ExtractionErrorCode.PDF_PARSE_FAILED,
                    "The PDF is password-protected and cannot be read.",
                )
            if not pdf.page_count:
                return self._failed(
                    document.document_id,
                    ExtractionErrorCode.NO_TEXT_DETECTED,
                    "The PDF contains no pages or readable text.",
                )

            pages: list[PageResult] = []
            for page_index in range(pdf.page_count):
                page = pdf.load_page(page_index)
                text = page.get_text("text", sort=True).strip()
                needs_ocr = not self._is_usable(text)
                pages.append(
                    PageResult(
                        document_id=document.document_id,
                        page_number=page_index + 1,
                        text=text,
                        needs_ocr=needs_ocr,
                        extraction_method="pdf_text",
                        word_boxes=self._word_boxes(page.get_text("words"), page.rect),
                    )
                )

        needs_ocr_count = sum(page.needs_ocr for page in pages)
        if needs_ocr_count:
            usable_page_count = len(pages) - needs_ocr_count
            status = (
                DocumentStatus.PARTIAL if usable_page_count else DocumentStatus.FAILED
            )
            error = None
        else:
            status = DocumentStatus.SUCCESS
            error = None

        return ExtractionResult(
            document_id=document.document_id,
            status=status,
            pages=pages,
            error=error,
            extraction_method="pdf_text",
        )

    def _is_usable(self, text: str) -> bool:
        if len(text) < self._settings.pdf_min_text_chars_per_page:
            return False
        non_whitespace = [character for character in text if not character.isspace()]
        if not non_whitespace:
            return False
        alphanumeric_count = sum(character.isalnum() for character in non_whitespace)
        alphanumeric_ratio = alphanumeric_count / len(non_whitespace)
        return alphanumeric_ratio >= self._settings.pdf_min_alnum_ratio

    @staticmethod
    def _word_boxes(words: Iterable[tuple], page_rect: pymupdf.Rect) -> list[WordBox]:
        normalized: list[WordBox] = []
        for word in words:
            x0, y0, x1, y1, text = word[:5]
            width = float(page_rect.width)
            height = float(page_rect.height)
            if width <= 0 or height <= 0 or x1 <= x0 or y1 <= y0:
                continue
            normalized.append(
                WordBox(
                    text=str(text),
                    bbox=BoundingBox(
                        x=max(0.0, (x0 - page_rect.x0) / width),
                        y=max(0.0, (y0 - page_rect.y0) / height),
                        width=(x1 - x0) / width,
                        height=(y1 - y0) / height,
                    ),
                )
            )
        return normalized

    @staticmethod
    def _failed(
        document_id: str, code: ExtractionErrorCode, message: str
    ) -> ExtractionResult:
        return ExtractionResult(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            error=ExtractionError(
                code=code,
                message=message,
                document_id=document_id,
                retryable=False,
            ),
            extraction_method="pdf_text",
        )
