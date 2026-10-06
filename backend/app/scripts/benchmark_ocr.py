"""Benchmark OCR engines against synthetic image samples and sidecar truth."""

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import TextIO

from PIL import Image

from app.core.config import Settings, get_settings
from app.extractors.ocr import (
    OCREngine,
    PaddleOCREngine,
    TesseractEngine,
)
from app.services.ocr_service import OCRService

_CSV_FIELDS = (
    "preset",
    "engine",
    "character_accuracy",
    "mean_confidence",
    "field_recovery_rate",
    "seconds_per_page",
    "pages",
    "error",
)


@dataclass
class BenchmarkMetrics:
    """Aggregate one engine over a preset's generated pages."""

    preset: str
    engine: str
    character_accuracy: float | None = None
    mean_confidence: float | None = None
    field_recovery_rate: float | None = None
    seconds_per_page: float | None = None
    pages: int = 0
    error: str = ""


def benchmark_samples(
    sample_dirs: list[Path],
    engines: dict[str, OCREngine],
    ocr_service: OCRService | None = None,
) -> list[BenchmarkMetrics]:
    """Run each supplied engine over PNG/JPG pages with matching JSON truth."""
    service = ocr_service or OCRService()
    sidecars = sorted(
        sidecar
        for sample_dir in sample_dirs
        for sidecar in sample_dir.rglob("*.json")
        if sidecar.name != "ocr_benchmark.csv"
    )
    grouped_sidecars: dict[str, list[Path]] = {}
    for sidecar in sidecars:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if data.get("synthetic_only") is True:
            grouped_sidecars.setdefault(data.get("preset", "unknown"), []).append(
                sidecar
            )

    results: list[BenchmarkMetrics] = []
    for preset, preset_sidecars in sorted(grouped_sidecars.items()):
        for engine_name, engine in engines.items():
            metrics = BenchmarkMetrics(preset=preset, engine=engine_name)
            total_characters = 0
            total_distance = 0
            recovered_fields = 0
            total_fields = 0
            confidence_sum = 0.0
            confidence_count = 0
            elapsed_sum = 0.0
            engine_page_count = 0
            page_count = 0
            error_messages: list[str] = []
            for sidecar_path in preset_sidecars:
                sample = json.loads(sidecar_path.read_text(encoding="utf-8"))
                for page_index, page in enumerate(sample.get("pages", [])):
                    image_path = _page_image_path(
                        sidecar_path.parent,
                        sample.get("files", {}).get("images", []),
                        page_index,
                    )
                    if image_path is None:
                        error_messages.append(f"{sidecar_path.name}: no page image")
                        continue
                    page_count += 1
                    try:
                        with Image.open(image_path) as image:
                            prepared = service.prepare_image(
                                image.convert("RGB"), dpi=None
                            )
                    except Exception as exc:
                        error_messages.append(f"{type(exc).__name__}: {exc}")
                        continue
                    engine_started = monotonic()
                    try:
                        recognition = engine.recognize(prepared.processed_image)
                    except Exception as exc:
                        elapsed_sum += monotonic() - engine_started
                        engine_page_count += 1
                        error_messages.append(f"{type(exc).__name__}: {exc}")
                        continue
                    elapsed_sum += max(
                        0.0,
                        monotonic() - engine_started,
                        recognition.elapsed_seconds,
                    )
                    engine_page_count += 1
                    predicted = recognition.text
                    expected = str(page.get("text", ""))
                    expected_normalized = _normalize_chars(expected)
                    predicted_normalized = _normalize_chars(predicted)
                    total_characters += len(expected_normalized)
                    total_distance += _levenshtein(
                        expected_normalized, predicted_normalized
                    )
                    recognized_fields = _normalize_field_text(predicted)
                    for value in page.get("fields", {}).values():
                        total_fields += 1
                        if _normalize_field_text(str(value)) in recognized_fields:
                            recovered_fields += 1
                    confidence_sum += recognition.mean_confidence
                    confidence_count += 1

            metrics.pages = page_count
            metrics.character_accuracy = (
                max(0.0, 1.0 - total_distance / total_characters)
                if total_characters
                else None
            )
            metrics.mean_confidence = (
                confidence_sum / confidence_count if confidence_count else None
            )
            metrics.field_recovery_rate = (
                recovered_fields / total_fields if total_fields else None
            )
            metrics.seconds_per_page = (
                elapsed_sum / engine_page_count if engine_page_count else None
            )
            metrics.error = "; ".join(dict.fromkeys(error_messages))
            results.append(metrics)
    return results


def write_csv(results: list[BenchmarkMetrics], stream: TextIO) -> None:
    """Write benchmark records using stable, machine-readable columns."""
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                **asdict(result),
                "character_accuracy": _metric(result.character_accuracy),
                "mean_confidence": _metric(result.mean_confidence),
                "field_recovery_rate": _metric(result.field_recovery_rate),
                "seconds_per_page": _metric(result.seconds_per_page),
            }
        )


def render_table(results: list[BenchmarkMetrics]) -> str:
    """Render results in a compact terminal table."""
    headers = (
        "Preset",
        "Engine",
        "Char accuracy",
        "Mean confidence",
        "Field recovery",
        "Seconds/page",
        "Pages",
        "Error",
    )
    rows = [
        [
            item.preset,
            item.engine,
            _percent(item.character_accuracy),
            _percent(item.mean_confidence),
            _percent(item.field_recovery_rate),
            _metric(item.seconds_per_page),
            str(item.pages),
            item.error[:42],
        ]
        for item in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(values: tuple[str, ...] | list[str]) -> str:
        return (
            "| "
            + " | ".join(
                values[index].ljust(widths[index]) for index in range(len(widths))
            )
            + " |"
        )

    lines = [border, format_row(headers), border]
    lines.extend(format_row(row) for row in rows)
    lines.append(border)
    return "\n".join(lines)


def build_engines(settings: Settings | None = None) -> dict[str, OCREngine]:
    """Create lazy engine adapters without loading models or binaries."""
    selected = settings or get_settings()
    return {
        "paddleocr": PaddleOCREngine(selected.ocr_paddle_language),
        "tesseract": TesseractEngine(selected.ocr_languages),
    }


def main(argv: list[str] | None = None) -> None:
    """Benchmark both engines and write terminal and CSV reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        nargs="+",
        type=Path,
        default=[Path("data/samples")],
        help="sample directories containing sidecar JSON and page images",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/samples/ocr_benchmark.csv"),
        help="CSV destination (default: data/samples/ocr_benchmark.csv)",
    )
    args = parser.parse_args(argv)
    missing_dirs = [path for path in args.samples if not path.is_dir()]
    if missing_dirs:
        parser.error(f"sample directory does not exist: {missing_dirs[0]}")
    results = benchmark_samples(args.samples, build_engines())
    if not results:
        parser.error("no synthetic sidecar JSON files found in the sample directories")
    print(render_table(results))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as output:
        write_csv(results, output)
    print(f"CSV written to {args.csv}")


def _page_image_path(
    directory: Path, image_names: list[str], page_index: int
) -> Path | None:
    candidates = [
        directory / name
        for name in image_names
        if name.endswith(".png") and f"page_{page_index + 1}." in name
    ]
    if not candidates:
        candidates = [
            directory / name
            for name in image_names
            if f"page_{page_index + 1}." in name
        ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _normalize_chars(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _normalize_field_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
