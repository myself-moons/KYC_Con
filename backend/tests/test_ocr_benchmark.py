"""Fast benchmark math/output tests using a deterministic fake OCR engine."""

import json
from io import StringIO
from pathlib import Path

from app.extractors.ocr import OCREngine, OCREngineResult, OCRLine
from app.models.extraction import BoundingBox
from app.scripts.benchmark_ocr import benchmark_samples, render_table, write_csv
from app.services.ocr_service import OCRPreprocessingError, PreparedOCRPage
from PIL import Image


class ExactFakeEngine(OCREngine):
    """Return known text for deterministic benchmark arithmetic."""

    @property
    def name(self) -> str:
        return "fake"

    def recognize(self, image: Image.Image) -> OCREngineResult:
        text = "PAN: AAAAA0000A\nName: Aarav Sample"
        return OCREngineResult(
            engine=self.name,
            lines=[
                OCRLine(
                    text=text,
                    confidence=0.9,
                    bbox=BoundingBox(x=0.1, y=0.1, width=0.8, height=0.3),
                )
            ],
            mean_confidence=0.9,
            elapsed_seconds=0.02,
        )


class PassthroughOCRService:
    """Benchmark preprocessor fake that requires no OpenCV runtime."""

    def prepare_image(self, image: Image.Image, dpi: int | None) -> PreparedOCRPage:
        copy = image.copy()
        return PreparedOCRPage(copy, copy, dpi)


class BrokenOCRService:
    """Simulate an OpenCV import error before an OCR engine is invoked."""

    def prepare_image(self, image: Image.Image, dpi: int | None) -> PreparedOCRPage:
        raise OCRPreprocessingError("OpenCV runtime unavailable")


def test_benchmark_metrics_table_and_csv(tmp_path: Path) -> None:
    sample_dir = tmp_path / "xerox_medium"
    sample_dir.mkdir()
    Image.new("RGB", (20, 20), "white").save(sample_dir / "fixture_page_1.png")
    sidecar = {
        "synthetic_only": True,
        "preset": "xerox_medium",
        "pages": [
            {
                "text": "PAN: AAAAA0000A\nName: Aarav Sample",
                "fields": {"PAN": "AAAAA0000A", "Name": "Aarav Sample"},
            }
        ],
        "files": {"images": ["fixture_page_1.png"]},
    }
    (sample_dir / "synthetic.json").write_text(json.dumps(sidecar), encoding="utf-8")

    result = benchmark_samples(
        [sample_dir],
        {"fake": ExactFakeEngine()},
        ocr_service=PassthroughOCRService(),
    )[0]

    assert result.character_accuracy == 1.0
    assert result.mean_confidence == 0.9
    assert result.field_recovery_rate == 1.0
    assert result.pages == 1
    assert result.seconds_per_page >= 0.02
    table = render_table([result])
    assert "xerox_medium" in table
    assert "100.0%" in table
    csv_stream = StringIO()
    write_csv([result], csv_stream)
    assert "character_accuracy" in csv_stream.getvalue()
    assert "fake" in csv_stream.getvalue()


def test_preprocessing_failure_does_not_report_engine_time(tmp_path: Path) -> None:
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    Image.new("RGB", (20, 20), "white").save(sample_dir / "fixture_page_1.png")
    sidecar = {
        "synthetic_only": True,
        "preset": "xerox_heavy",
        "pages": [{"text": "synthetic", "fields": {"PAN": "AAAAA0000A"}}],
        "files": {"images": ["fixture_page_1.png"]},
    }
    (sample_dir / "synthetic.json").write_text(json.dumps(sidecar), encoding="utf-8")

    result = benchmark_samples(
        [sample_dir],
        {"fake": ExactFakeEngine()},
        ocr_service=BrokenOCRService(),
    )[0]

    assert result.character_accuracy is None
    assert result.seconds_per_page is None
    assert "OpenCV runtime unavailable" in result.error
