"""Real OCR recovery checks against generated fake documents."""

import re
import shutil

import pytest
from app.core.config import Settings
from app.extractors.ocr import PaddleOCREngine, TesseractEngine
from app.services.ocr_service import OCRService

pytestmark = pytest.mark.slow
RECOVERY_PRESETS = ("clean", "light_scan", "xerox_medium")


def _normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _require_engine(engine_name: str) -> None:
    if engine_name == "paddleocr":
        try:
            import cv2  # noqa: F401
            import paddle  # noqa: F401
        except Exception as exc:
            pytest.skip(f"PaddleOCR runtime unavailable: {type(exc).__name__}: {exc}")
        return

    if shutil.which("tesseract") is None:
        pytest.skip("Tesseract system binary is not installed")
    import pytesseract

    try:
        languages = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        pytest.skip(f"Tesseract language data unavailable: {exc}")
    if not {"eng", "hin"} <= languages:
        pytest.skip("Tesseract requires both eng and hin language data for this test")


def _run_recovery_checks(
    engine, synthetic_kyc_samples_factory, record_property
) -> None:
    settings = Settings(_env_file=None)
    preprocessing = OCRService(settings)
    for preset in RECOVERY_PRESETS:
        outputs = synthetic_kyc_samples_factory(preset=preset, seed=81)
        for stem, field in (
            ("pan_card", "PAN"),
            ("gst_certificate", "GSTIN"),
        ):
            truth = __import__("json").loads(
                outputs[f"{stem}.json"].read_text(encoding="utf-8")
            )
            expected = truth["ground_truth"][field]
            with outputs[f"{stem}_page_1.png"].open("rb") as source_image_file:
                prepared = preprocessing.load_image(source_image_file.read())
            result = engine.recognize(prepared.processed_image)
            assert _normalize(expected) in _normalize(
                result.text
            ), f"{engine.name} failed to recover {field} on {preset}"

    heavy_outputs = synthetic_kyc_samples_factory(preset="xerox_heavy", seed=81)
    heavy_recovered: dict[str, bool] = {}
    for stem, field in (("pan_card", "PAN"), ("gst_certificate", "GSTIN")):
        truth = __import__("json").loads(
            heavy_outputs[f"{stem}.json"].read_text(encoding="utf-8")
        )
        with heavy_outputs[f"{stem}_page_1.png"].open("rb") as source_image_file:
            prepared = preprocessing.load_image(source_image_file.read())
        result = engine.recognize(prepared.processed_image)
        heavy_recovered[field] = _normalize(truth["ground_truth"][field]) in _normalize(
            result.text
        )
    record_property(f"{engine.name}_xerox_heavy_recovery", heavy_recovered)
    print(f"{engine.name} xerox_heavy recovery (informational): {heavy_recovered}")


def test_paddleocr_recovers_pan_and_gstin(
    synthetic_kyc_samples_factory, record_property
) -> None:
    _require_engine("paddleocr")
    _run_recovery_checks(
        PaddleOCREngine(Settings(_env_file=None).ocr_paddle_language),
        synthetic_kyc_samples_factory,
        record_property,
    )


def test_tesseract_recovers_pan_and_gstin(
    synthetic_kyc_samples_factory, record_property
) -> None:
    _require_engine("tesseract")
    _run_recovery_checks(
        TesseractEngine(Settings(_env_file=None).ocr_languages),
        synthetic_kyc_samples_factory,
        record_property,
    )
