"""Fast OCRExtractor tests using fake engines and a fake image preprocessor."""

import sys
from types import SimpleNamespace

import pytest
from app.core.config import Settings
from app.core.exceptions import ExtractionErrorCode
from app.extractors.ocr import OCREngine, OCREngineResult, OCRExtractor
from app.models.document import Document, DocumentStatus
from app.models.extraction import BoundingBox, OCRLine, WordBox
from app.services.ocr_service import PreparedOCRPage
from PIL import Image


class FakeEngine(OCREngine):
    """Return a controlled result or raise, while recording invocation count."""

    def __init__(self, engine_name: str, result: OCREngineResult | Exception) -> None:
        self._engine_name = engine_name
        self._result = result
        self.calls = 0

    @property
    def name(self) -> str:
        return self._engine_name

    def recognize(self, image: Image.Image) -> OCREngineResult:
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeOCRService:
    """Return a synthetic image pair without importing OpenCV."""

    def __init__(self) -> None:
        self.image = Image.new("RGB", (100, 80), "white")

    def render_pdf_page(self, pdf_bytes: bytes, page_index: int) -> PreparedOCRPage:
        return PreparedOCRPage(self.image.copy(), self.image.copy(), 300)

    def load_image(self, image_bytes: bytes) -> PreparedOCRPage:
        return PreparedOCRPage(self.image.copy(), self.image.copy(), None)


def _document(filename: str = "scan.png") -> Document:
    return Document(
        document_id="ocr-test-document",
        case_id="ocr-test-case",
        filename=filename,
        mime_type="image/png" if filename.endswith(".png") else "application/pdf",
        size=10,
        storage_path="synthetic/ocr-test",
    )


def _recognition(engine: str, text: str, confidence: float) -> OCREngineResult:
    line = OCRLine(
        text=text,
        confidence=confidence,
        bbox=BoundingBox(x=0.1, y=0.2, width=0.5, height=0.1),
        words=[
            WordBox(
                text=text,
                confidence=confidence,
                bbox=BoundingBox(x=0.1, y=0.2, width=0.5, height=0.1),
            )
        ],
    )
    return OCREngineResult(
        engine=engine,
        lines=[line],
        words=line.words,
        mean_confidence=confidence,
        elapsed_seconds=0.01,
    )


def test_primary_low_confidence_uses_better_fallback_and_records_both_scores() -> None:
    primary = FakeEngine("paddleocr", _recognition("paddleocr", "AAAAA0000A", 0.31))
    fallback = FakeEngine("tesseract", _recognition("tesseract", "AAAAA0000A", 0.91))
    extractor = OCRExtractor(
        settings=Settings(_env_file=None, ocr_min_confidence=0.6),
        primary_engine=primary,
        fallback_engine=fallback,
        ocr_service=FakeOCRService(),
    )

    result = extractor.extract(_document(), b"fake png bytes")

    assert result.status is DocumentStatus.SUCCESS
    assert result.pages[0].text == "AAAAA0000A"
    assert result.pages[0].extraction_method == "tesseract"
    assert result.pages[0].ocr_engine_used == "tesseract"
    assert result.pages[0].ocr_primary_confidence == 0.31
    assert result.pages[0].ocr_fallback_confidence == 0.91
    assert result.pages[0].word_boxes[0].bbox.x == 0.1
    assert primary.calls == 1
    assert fallback.calls == 1


def test_primary_failure_runs_fallback_and_does_not_fail_document() -> None:
    primary = FakeEngine("paddleocr", RuntimeError("Synthetic model failure"))
    fallback = FakeEngine(
        "tesseract", _recognition("tesseract", "27AAAAA0000A1Z5", 0.8)
    )
    extractor = OCRExtractor(
        Settings(_env_file=None),
        primary_engine=primary,
        fallback_engine=fallback,
        ocr_service=FakeOCRService(),
    )

    result = extractor.extract(_document(), b"fake png bytes")

    assert result.status is DocumentStatus.SUCCESS
    assert result.pages[0].ocr_primary_confidence is None
    assert result.pages[0].ocr_fallback_confidence == 0.8
    assert result.pages[0].ocr_engine_used == "tesseract"


def test_high_confidence_primary_skips_fallback() -> None:
    primary = FakeEngine("paddleocr", _recognition("paddleocr", "AAAAA0000A", 0.93))
    fallback = FakeEngine("tesseract", RuntimeError("Must not run"))
    extractor = OCRExtractor(
        Settings(_env_file=None),
        primary_engine=primary,
        fallback_engine=fallback,
        ocr_service=FakeOCRService(),
    )

    result = extractor.extract(_document(), b"fake png bytes")

    assert result.status is DocumentStatus.SUCCESS
    assert result.pages[0].extraction_method == "paddleocr"
    assert result.pages[0].ocr_fallback_confidence is None
    assert fallback.calls == 0


def test_low_confidence_page_is_partial_with_explicit_error() -> None:
    low_primary = FakeEngine("paddleocr", _recognition("paddleocr", "AAAAA0000A", 0.22))
    low_fallback = FakeEngine("tesseract", _recognition("tesseract", "AAAAA0000A", 0.3))
    extractor = OCRExtractor(
        Settings(_env_file=None, ocr_min_confidence=0.6),
        primary_engine=low_primary,
        fallback_engine=low_fallback,
        ocr_service=FakeOCRService(),
    )

    result = extractor.extract(_document(), b"fake png bytes")

    assert result.status is DocumentStatus.PARTIAL
    assert result.error is not None
    assert result.error.code is ExtractionErrorCode.OCR_LOW_CONFIDENCE
    assert "page(s): 1" in result.error.message
    assert result.pages[0].ocr_lines[0].low_confidence is True


def test_no_text_returns_no_text_detected() -> None:
    no_text = OCREngineResult(
        engine="paddleocr",
        lines=[],
        words=[],
        mean_confidence=0.0,
        elapsed_seconds=0.01,
    )
    primary = FakeEngine("paddleocr", no_text)
    fallback = FakeEngine(
        "tesseract", no_text.model_copy(update={"engine": "tesseract"})
    )
    extractor = OCRExtractor(
        Settings(_env_file=None),
        primary_engine=primary,
        fallback_engine=fallback,
        ocr_service=FakeOCRService(),
    )

    result = extractor.extract(_document(), b"fake png bytes")

    assert result.status is DocumentStatus.FAILED
    assert result.error is not None
    assert result.error.code is ExtractionErrorCode.NO_TEXT_DETECTED


def test_mixed_pdf_keeps_digital_page_and_ocr_method_per_page(
    pdf_factory,
) -> None:
    direct_text = (
        "Synthetic digital text page passes configured threshold and remains "
        "searchable."
    )
    pdf_bytes = pdf_factory([direct_text, None], image_only_pages={1})
    document = _document("mixed.pdf")
    primary = FakeEngine("paddleocr", _recognition("paddleocr", "AAAAA0000A", 0.9))
    fallback = FakeEngine("tesseract", RuntimeError("Should not run"))
    extractor = OCRExtractor(
        Settings(_env_file=None),
        primary_engine=primary,
        fallback_engine=fallback,
        ocr_service=FakeOCRService(),
    )

    result = extractor.extract(document, pdf_bytes)

    assert result.status is DocumentStatus.SUCCESS
    assert [page.extraction_method for page in result.pages] == [
        "pdf_text",
        "paddleocr",
    ]
    assert [page.needs_ocr for page in result.pages] == [False, True]
    assert result.extraction_method == "mixed"
    assert fallback.calls == 0


def test_mixed_pdf_engine_failure_is_partial_and_retryable(
    pdf_factory,
) -> None:
    digital_text = (
        "Synthetic digital page contains enough text to pass configured checks."
    )
    primary = FakeEngine("paddleocr", RuntimeError("synthetic engine failure"))
    fallback = FakeEngine("tesseract", RuntimeError("synthetic fallback failure"))
    extractor = OCRExtractor(
        Settings(_env_file=None),
        primary_engine=primary,
        fallback_engine=fallback,
        ocr_service=FakeOCRService(),
    )

    result = extractor.extract(
        _document("mixed.pdf"), pdf_factory([digital_text, None], image_only_pages={1})
    )

    assert result.status is DocumentStatus.PARTIAL
    assert result.pages[0].extraction_method == "pdf_text"
    assert result.error is not None
    assert result.error.code is ExtractionErrorCode.OCR_FAILED
    assert result.error.retryable is True


def test_paddle_adapter_normalizes_v3_polygon_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.extractors import ocr as ocr_module

    class PredictResult:
        json = {
            "res": {
                "rec_texts": ["AAAAA0000A"],
                "rec_scores": [0.88],
                "rec_polys": [[[10, 20], [50, 20], [50, 40], [10, 40]]],
            }
        }

    class PredictModel:
        def predict(self, image: Image.Image) -> list[PredictResult]:
            return [PredictResult()]

    monkeypatch.setattr(
        ocr_module, "_load_paddle_model", lambda language: PredictModel()
    )
    result = ocr_module.PaddleOCREngine().recognize(Image.new("RGB", (100, 100)))

    assert result.text == "AAAAA0000A"
    assert result.mean_confidence == 0.88
    assert result.lines[0].bbox.x == 0.1
    assert result.lines[0].bbox.width == 0.4


def test_tesseract_adapter_groups_words_and_normalizes_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pytesseract
    from app.extractors.ocr import TesseractEngine

    captured: dict[str, object] = {}

    def fake_image_to_data(*args, **kwargs):
        captured.update(kwargs)
        return {
            "text": ["PAN:", "AAAAA0000A"],
            "conf": ["90", "80"],
            "left": [10, 50],
            "top": [20, 20],
            "width": [30, 40],
            "height": [10, 10],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
        }

    monkeypatch.setattr(
        pytesseract,
        "image_to_data",
        fake_image_to_data,
    )

    result = TesseractEngine().recognize(Image.new("RGB", (100, 100)))

    assert result.text == "PAN: AAAAA0000A"
    assert len(result.words) == 2
    assert result.words[0].bbox.x == 0.1
    assert result.words[0].confidence == 0.9
    assert captured["config"] == "--oem 1 --psm 11"


def test_paddle_model_is_loaded_once_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.extractors import ocr as ocr_module

    constructors: list[str] = []
    monkeypatch.delenv("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", raising=False)

    class FakePaddleModel:
        def __init__(self, **options) -> None:
            constructors.append(options["lang"])

    monkeypatch.setattr(ocr_module, "_PADDLE_MODELS", {})
    monkeypatch.setitem(
        sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleModel)
    )

    first = ocr_module._load_paddle_model("en")
    second = ocr_module._load_paddle_model("en")

    assert first is second
    assert constructors == ["en"]
    assert __import__("os").environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] == "False"


def test_ocr_engine_setting_controls_primary_and_fallback_order() -> None:
    extractor = OCRExtractor(Settings(_env_file=None, ocr_engine="tesseract"))

    assert extractor._primary_engine.name == "tesseract"
    assert extractor._fallback_engine.name == "paddleocr"
