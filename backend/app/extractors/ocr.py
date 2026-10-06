"""Lazy PaddleOCR/Tesseract engines and OCR extraction orchestration."""

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache, partial
from pathlib import PurePath
from threading import Lock
from time import monotonic
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.exceptions import ExtractionErrorCode
from app.extractors.base import DocumentExtractor, ExtractionResult
from app.extractors.pdf import PDFTextExtractor
from app.models.document import Document, DocumentStatus
from app.models.extraction import (
    BoundingBox,
    ExtractionError,
    OCRLine,
    PageResult,
    WordBox,
)
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)
_PADDLE_MODELS: dict[str, Any] = {}
_PADDLE_MODEL_LOCK = Lock()


class OCREngineResult(BaseModel):
    """Recognized lines/words, confidence, and elapsed time from one engine."""

    engine: str
    lines: list[OCRLine] = Field(default_factory=list)
    words: list[WordBox] = Field(default_factory=list)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    elapsed_seconds: float = Field(ge=0.0)

    @property
    def text(self) -> str:
        """Return recognized text in reading-line order."""
        return "\n".join(line.text for line in self.lines if line.text.strip())


class OCREngine(ABC):
    """Interface for OCR providers returning normalized page coordinates."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable short identifier for result metadata."""

    @abstractmethod
    def recognize(self, image: Image.Image) -> OCREngineResult:
        """Recognize lines and words on an image using normalized boxes."""


class PaddleOCREngine(OCREngine):
    """CPU PaddleOCR engine; the heavy model is initialized once on first use."""

    def __init__(self, language: str = "en") -> None:
        self._language = language

    @property
    def name(self) -> str:
        return "paddleocr"

    def recognize(self, image: Image.Image) -> OCREngineResult:
        """Run PaddleOCR and adapt current or legacy result objects."""
        started = monotonic()
        model = _load_paddle_model(self._language)
        image_array = np.asarray(image.convert("RGB"))
        if hasattr(model, "predict"):
            raw_results = model.predict(image_array)
            lines = self._parse_predict_results(raw_results, image.size)
        else:
            raw_results = model.ocr(image_array, cls=True)
            lines = self._parse_legacy_results(raw_results, image.size)
        mean = sum(line.confidence for line in lines) / len(lines) if lines else 0.0
        return OCREngineResult(
            engine=self.name,
            lines=lines,
            words=[word for line in lines for word in line.words],
            mean_confidence=mean,
            elapsed_seconds=max(0.0, monotonic() - started),
        )

    def _parse_predict_results(
        self, raw_results: Any, image_size: tuple[int, int]
    ) -> list[OCRLine]:
        lines: list[OCRLine] = []
        for raw_result in raw_results or []:
            data = _result_dict(raw_result)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_polys")
            if boxes is None:
                boxes = data.get("dt_polys")
            if boxes is None:
                boxes = data.get("rec_boxes", [])
            for index, text in enumerate(texts):
                if index >= len(scores) or index >= len(boxes) or not str(text).strip():
                    continue
                confidence = _confidence(scores[index])
                bbox = _normalized_bbox(boxes[index], image_size)
                if bbox is not None:
                    lines.append(
                        OCRLine(
                            text=str(text).strip(),
                            confidence=confidence,
                            bbox=bbox,
                        )
                    )
        return lines

    @staticmethod
    def _parse_legacy_results(
        raw_results: Any, image_size: tuple[int, int]
    ) -> list[OCRLine]:
        lines: list[OCRLine] = []
        for page_result in raw_results or []:
            for item in page_result or []:
                if len(item) < 2:
                    continue
                polygon, recognition = item
                text, score = recognition
                bbox = _normalized_bbox(polygon, image_size)
                if bbox is not None and str(text).strip():
                    lines.append(
                        OCRLine(
                            text=str(text).strip(),
                            confidence=_confidence(score),
                            bbox=bbox,
                        )
                    )
        return lines


class TesseractEngine(OCREngine):
    """Tesseract adapter using word-level boxes and configured language data."""

    def __init__(self, languages: str = "eng+hin") -> None:
        self._languages = languages

    @property
    def name(self) -> str:
        return "tesseract"

    def recognize(self, image: Image.Image) -> OCREngineResult:
        """Run the external Tesseract binary via pytesseract and group lines."""
        import pytesseract
        from pytesseract import Output

        started = monotonic()
        data = pytesseract.image_to_data(
            image.convert("RGB"),
            lang=self._languages,
            config="--oem 1 --psm 11",
            output_type=Output.DICT,
        )
        width, height = image.size
        grouped: dict[tuple[int, int, int], list[WordBox]] = defaultdict(list)
        for index, text in enumerate(data.get("text", [])):
            normalized_text = str(text).strip()
            if not normalized_text:
                continue
            try:
                raw_confidence = float(data["conf"][index])
            except (KeyError, TypeError, ValueError):
                continue
            if raw_confidence < 0:
                continue
            bbox = _pixel_bbox(
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
                (width, height),
            )
            if bbox is None:
                continue
            word = WordBox(
                text=normalized_text,
                bbox=bbox,
                confidence=_confidence(raw_confidence / 100.0),
            )
            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            grouped[key].append(word)

        lines: list[OCRLine] = []
        words: list[WordBox] = []
        for line_words in grouped.values():
            line_bbox = _union_boxes([word.bbox for word in line_words])
            if line_bbox is None:
                continue
            words.extend(line_words)
            lines.append(
                OCRLine(
                    text=" ".join(word.text for word in line_words),
                    confidence=sum(word.confidence or 0.0 for word in line_words)
                    / len(line_words),
                    bbox=line_bbox,
                    words=line_words,
                )
            )
        mean = sum(line.confidence for line in lines) / len(lines) if lines else 0.0
        return OCREngineResult(
            engine=self.name,
            lines=lines,
            words=words,
            mean_confidence=mean,
            elapsed_seconds=max(0.0, monotonic() - started),
        )


@lru_cache(maxsize=4)
def get_paddle_engine(language: str = "en") -> PaddleOCREngine:
    """Return one reusable Paddle engine per language; model loading stays lazy."""
    return PaddleOCREngine(language)


@lru_cache(maxsize=4)
def get_tesseract_engine(languages: str = "eng+hin") -> TesseractEngine:
    """Return one reusable Tesseract adapter per language configuration."""
    return TesseractEngine(languages)


class OCRExtractor(DocumentExtractor):
    """OCR scanned PDF pages and image documents with confidence-based fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        primary_engine: OCREngine | None = None,
        fallback_engine: OCREngine | None = None,
        pdf_extractor: DocumentExtractor | None = None,
        ocr_service: OCRService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if primary_engine is not None and fallback_engine is not None:
            self._primary_engine = primary_engine
            self._fallback_engine = fallback_engine
        elif self._settings.ocr_engine == "paddleocr":
            self._primary_engine = primary_engine or get_paddle_engine(
                self._settings.ocr_paddle_language
            )
            self._fallback_engine = fallback_engine or get_tesseract_engine(
                self._settings.ocr_languages
            )
        else:
            self._primary_engine = primary_engine or get_tesseract_engine(
                self._settings.ocr_languages
            )
            self._fallback_engine = fallback_engine or get_paddle_engine(
                self._settings.ocr_paddle_language
            )
        self._pdf_extractor = pdf_extractor or PDFTextExtractor(self._settings)
        self._ocr_service = ocr_service or OCRService(self._settings)

    def extract(self, document: Document, content: bytes) -> ExtractionResult:
        """Use embedded text where usable, OCR only the pages that need it."""
        extension = PurePath(document.filename).suffix.lower()
        if extension == ".pdf":
            base_result = self._pdf_extractor.extract(document, content)
            if base_result.error is not None or not base_result.pages:
                return base_result
            pages = base_result.pages
            page_inputs = [
                (
                    page,
                    partial(
                        self._ocr_service.render_pdf_page,
                        content,
                        page.page_number - 1,
                    ),
                )
                for page in pages
                if page.needs_ocr
            ]
        elif extension in {".jpg", ".jpeg", ".png"}:
            page = PageResult(
                document_id=document.document_id,
                page_number=1,
                text="",
                needs_ocr=True,
            )
            pages = [page]
            page_inputs = [(page, lambda: self._ocr_service.load_image(content))]
        else:
            return self._failure(
                document.document_id,
                ExtractionErrorCode.UNSUPPORTED_FILE_TYPE,
                "OCR supports PDF pages and JPG/PNG images.",
                DocumentStatus.UNSUPPORTED,
            )

        failed_pages: list[int] = []
        low_confidence_pages: list[int] = []
        engine_errors: list[int] = []
        for page, prepare in page_inputs:
            try:
                prepared = prepare()
                selected, primary_score, fallback_score, engines_failed = (
                    self._recognize_with_fallback(prepared.processed_image)
                )
            except Exception:
                failed_pages.append(page.page_number)
                engine_errors.append(page.page_number)
                continue
            if selected is None or not selected.text.strip():
                failed_pages.append(page.page_number)
                if engines_failed:
                    engine_errors.append(page.page_number)
                page.extraction_method = None
                page.ocr_primary_confidence = primary_score
                page.ocr_fallback_confidence = fallback_score
                continue

            lines = [
                self._apply_line_threshold(line)
                for line in selected.lines
                if line.text.strip()
            ]
            page.text = "\n".join(line.text for line in lines)
            page.ocr_lines = lines
            page.word_boxes = [word for line in lines for word in line.words]
            page.confidence = selected.mean_confidence
            page.extraction_method = selected.engine
            page.ocr_engine_used = selected.engine
            page.ocr_primary_confidence = primary_score
            page.ocr_fallback_confidence = fallback_score
            page.low_confidence = (
                selected.mean_confidence < self._settings.ocr_min_confidence
                or any(line.low_confidence for line in lines)
            )
            if page.low_confidence:
                low_confidence_pages.append(page.page_number)

        result_error: ExtractionError | None = None
        text_pages = [page for page in pages if page.text.strip()]
        if failed_pages and not text_pages:
            if low_confidence_pages:
                code = ExtractionErrorCode.OCR_LOW_CONFIDENCE
                message = self._page_message(
                    "OCR confidence was too low", low_confidence_pages
                )
            elif engine_errors:
                code = ExtractionErrorCode.OCR_FAILED
                message = self._page_message("OCR failed", engine_errors)
            else:
                code = ExtractionErrorCode.NO_TEXT_DETECTED
                message = self._page_message("No text was detected", failed_pages)
            status = DocumentStatus.FAILED
            result_error = ExtractionError(
                code=code,
                message=message,
                document_id=document.document_id,
                retryable=code in {ExtractionErrorCode.OCR_FAILED},
            )
        elif failed_pages or low_confidence_pages:
            status = DocumentStatus.PARTIAL
            affected_pages = sorted(set(failed_pages + low_confidence_pages))
            code = (
                ExtractionErrorCode.OCR_LOW_CONFIDENCE
                if low_confidence_pages
                else (
                    ExtractionErrorCode.OCR_FAILED
                    if engine_errors
                    else ExtractionErrorCode.NO_TEXT_DETECTED
                )
            )
            reason = (
                "OCR confidence was too low"
                if low_confidence_pages
                else "OCR failed" if engine_errors else "No text was detected"
            )
            result_error = ExtractionError(
                code=code,
                message=self._page_message(reason, affected_pages),
                document_id=document.document_id,
                page=affected_pages[0] if len(affected_pages) == 1 else None,
                retryable=code is ExtractionErrorCode.OCR_FAILED,
            )
        else:
            status = DocumentStatus.SUCCESS

        methods = {page.extraction_method for page in pages if page.extraction_method}
        method = methods.pop() if len(methods) == 1 else "mixed"
        return ExtractionResult(
            document_id=document.document_id,
            status=status,
            pages=pages,
            error=result_error,
            extraction_method=method,
        )

    def _recognize_with_fallback(
        self, image: Image.Image
    ) -> tuple[OCREngineResult | None, float | None, float | None, bool]:
        primary_result: OCREngineResult | None = None
        fallback_result: OCREngineResult | None = None
        try:
            primary_result = self._primary_engine.recognize(image)
        except Exception:
            logger.warning("Primary OCR engine %s failed", self._primary_engine.name)

        primary_score = (
            primary_result.mean_confidence if primary_result is not None else None
        )
        should_fallback = (
            primary_result is None
            or primary_result.mean_confidence < self._settings.ocr_min_confidence
        )
        if should_fallback:
            try:
                fallback_result = self._fallback_engine.recognize(image)
            except Exception:
                logger.warning(
                    "Fallback OCR engine %s failed", self._fallback_engine.name
                )

        fallback_score = (
            fallback_result.mean_confidence if fallback_result is not None else None
        )
        candidates = [
            result for result in (primary_result, fallback_result) if result is not None
        ]
        selected = max(
            candidates,
            key=lambda result: (bool(result.text.strip()), result.mean_confidence),
            default=None,
        )
        failed = primary_result is None and fallback_result is None
        return selected, primary_score, fallback_score, failed

    def _apply_line_threshold(self, line: OCRLine) -> OCRLine:
        low_line = line.confidence < self._settings.ocr_line_min_confidence
        words = [
            word.model_copy(
                update={
                    "low_confidence": word.confidence is not None
                    and word.confidence < self._settings.ocr_line_min_confidence
                }
            )
            for word in line.words
        ]
        return line.model_copy(update={"low_confidence": low_line, "words": words})

    @staticmethod
    def _page_message(reason: str, pages: list[int]) -> str:
        return f"{reason} on page(s): {', '.join(map(str, sorted(set(pages))))}."

    @staticmethod
    def _failure(
        document_id: str,
        code: ExtractionErrorCode,
        message: str,
        status: DocumentStatus = DocumentStatus.FAILED,
    ) -> ExtractionResult:
        return ExtractionResult(
            document_id=document_id,
            status=status,
            error=ExtractionError(
                code=code,
                message=message,
                document_id=document_id,
                retryable=code is ExtractionErrorCode.OCR_FAILED,
            ),
            extraction_method="ocr",
        )


def _load_paddle_model(language: str) -> Any:
    """Initialize one CPU PaddleOCR model lazily and reuse it across requests."""
    with _PADDLE_MODEL_LOCK:
        if language not in _PADDLE_MODELS:
            os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
            from paddleocr import PaddleOCR

            try:
                model = PaddleOCR(
                    lang=language,
                    device="cpu",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except TypeError:
                model = PaddleOCR(lang=language, use_angle_cls=False)
            _PADDLE_MODELS[language] = model
        return _PADDLE_MODELS[language]


def _result_dict(value: Any) -> dict[str, Any]:
    """Adapt PaddleX result wrappers and legacy dicts into a plain dictionary."""
    result = value
    json_value = getattr(result, "json", None)
    if json_value is not None:
        result = json_value() if callable(json_value) else json_value
        if isinstance(result, str):
            result = json.loads(result)
    if not isinstance(result, dict):
        result = dict(result)
    if isinstance(result.get("res"), dict):
        result = result["res"]
    return result


def _confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _normalized_bbox(points: Any, image_size: tuple[int, int]) -> BoundingBox | None:
    array = np.asarray(points, dtype=float)
    width, height = image_size
    if width <= 0 or height <= 0 or array.size < 4:
        return None
    if array.ndim == 1 and array.size == 4:
        x0, y0, x1, y1 = array.tolist()
        coordinates = np.array([[x0, y0], [x1, y1]], dtype=float)
    else:
        coordinates = array.reshape(-1, 2)
    x0 = max(0.0, min(float(width), float(coordinates[:, 0].min())))
    y0 = max(0.0, min(float(height), float(coordinates[:, 1].min())))
    x1 = max(x0 + 1e-3, min(float(width), float(coordinates[:, 0].max())))
    y1 = max(y0 + 1e-3, min(float(height), float(coordinates[:, 1].max())))
    return BoundingBox(
        x=x0 / width,
        y=y0 / height,
        width=min(1.0 - x0 / width, (x1 - x0) / width),
        height=min(1.0 - y0 / height, (y1 - y0) / height),
    )


def _pixel_bbox(
    left: int,
    top: int,
    box_width: int,
    box_height: int,
    image_size: tuple[int, int],
) -> BoundingBox | None:
    if box_width <= 0 or box_height <= 0:
        return None
    return _normalized_bbox([left, top, left + box_width, top + box_height], image_size)


def _union_boxes(boxes: list[BoundingBox]) -> BoundingBox | None:
    if not boxes:
        return None
    x0 = min(box.x for box in boxes)
    y0 = min(box.y for box in boxes)
    x1 = max(box.x + box.width for box in boxes)
    y1 = max(box.y + box.height for box in boxes)
    return BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
