"""Check OCR runtime prerequisites without loading models or downloading weights."""

from dataclasses import dataclass
import importlib
from importlib import metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class EnvironmentCheck:
    """One environment requirement and its suggested remediation."""

    name: str
    status: str
    detail: str
    required: bool
    fix: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "OK" or not self.required


@dataclass(frozen=True)
class OCRCheckReport:
    """All checks and model-cache paths printed by the environment command."""

    python_version: str
    checks: tuple[EnvironmentCheck, ...]
    model_cache_locations: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 0 if all(check.passed for check in self.checks) else 1

    def render(self) -> str:
        """Format the environment report with exact remediation hints."""
        lines = [f"Python: {self.python_version}"]
        for check in self.checks:
            marker = check.status
            lines.append(f"{marker}: {check.name}: {check.detail}")
            if check.fix:
                lines.append(f"  Fix: {check.fix}")
        lines.append("PaddleOCR model cache locations:")
        lines.extend(f"  {location}" for location in self.model_cache_locations)
        lines.append(
            "Overall: " + ("READY" if self.exit_code == 0 else "NOT READY")
        )
        return "\n".join(lines)


def inspect_environment(
    *,
    python_version: tuple[int, int, int] | None = None,
    distributions: Callable[[], Iterable[Any]] | None = None,
    module_importer: Callable[[str], Any] | None = None,
    which: Callable[[str], str | None] | None = None,
    run: Callable[..., Any] | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> OCRCheckReport:
    """Inspect runtime prerequisites through injectable, model-free probes."""
    version = python_version or sys.version_info[:3]
    distribution_reader = distributions or metadata.distributions
    importer = module_importer or importlib.import_module
    executable_finder = which or shutil.which
    process_runner = run or subprocess.run
    environment = environ if environ is not None else os.environ
    home_path = home or Path.home()
    checks: list[EnvironmentCheck] = []

    version_text = ".".join(map(str, version))
    version_ok = (3, 11) <= version[:2] < (3, 13)
    checks.append(
        EnvironmentCheck(
            "Python version",
            "OK" if version_ok else "MISSING",
            version_text,
            required=True,
            fix=None
            if version_ok
            else "Create and use a Python 3.12 venv: python3.12 -m venv .venv",
        )
    )

    opencv_distributions = sorted(
        (
            str(distribution.metadata.get("Name", "")),
            str(distribution.version),
        )
        for distribution in distribution_reader()
        if "opencv" in str(distribution.metadata.get("Name", "")).lower()
    )
    variants = [f"{name}=={version_value}" for name, version_value in opencv_distributions]
    if len(variants) > 1:
        checks.append(
            EnvironmentCheck(
                "OpenCV distributions",
                "WARN",
                ", ".join(variants),
                required=False,
                fix=(
                    "After confirming PaddleOCR works, keep opencv-python-headless "
                    "and remove the duplicate contrib/python OpenCV distributions "
                    "inside this venv."
                ),
            )
        )
    elif variants:
        checks.append(
            EnvironmentCheck(
                "OpenCV distributions", "OK", ", ".join(variants), required=True
            )
        )
    else:
        checks.append(
            EnvironmentCheck(
                "OpenCV distributions",
                "MISSING",
                "no OpenCV distribution installed",
                required=True,
                fix="Run .venv/bin/python -m pip install -e '.[ocr]'",
            )
        )

    cv2_version: str | None = None
    try:
        cv2_module = importer("cv2")
        cv2_version = str(getattr(cv2_module, "__version__", "unknown"))
        checks.append(
            EnvironmentCheck("cv2 import", "OK", cv2_version, required=True)
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if "libGL.so.1" in str(exc):
            fix = "Install the system package libgl1 (and libglib2.0-0 on Debian/Ubuntu)."
        else:
            fix = "Run .venv/bin/python -m pip install -e '.[ocr]' and resolve the reported native-library error."
        checks.append(
            EnvironmentCheck("cv2 import", "MISSING", detail, required=True, fix=fix)
        )

    tesseract_path = executable_finder("tesseract")
    if tesseract_path is None:
        checks.append(
            EnvironmentCheck(
                "Tesseract binary",
                "MISSING",
                "tesseract executable not found on PATH",
                required=True,
                fix=(
                    "Install the system Tesseract OCR binary and reopen the shell; "
                    "pytesseract does not install it."
                ),
            )
        )
        for language in ("eng", "hin", "mar"):
            required = language in {"eng", "hin"}
            checks.append(
                EnvironmentCheck(
                    f"Tesseract language {language}",
                    "MISSING",
                    "cannot query language data without the Tesseract binary",
                    required=required,
                    fix=f"Install the Tesseract {language} language data package.",
                )
            )
    else:
        checks.append(
            EnvironmentCheck(
                "Tesseract binary", "OK", tesseract_path, required=True
            )
        )
        try:
            completed = process_runner(
                [tesseract_path, "--list-langs"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            language_output = f"{completed.stdout}\n{completed.stderr}"
            if completed.returncode != 0:
                raise RuntimeError(language_output.strip() or "--list-langs failed")
            available_languages = _parse_languages(language_output)
            for language in ("eng", "hin", "mar"):
                found = language in available_languages
                required = language in {"eng", "hin"}
                checks.append(
                    EnvironmentCheck(
                        f"Tesseract language {language}",
                        "OK" if found else "MISSING",
                        "language data found" if found else "language data not found",
                        required=required,
                        fix=None
                        if found
                        else f"Install Tesseract language data for {language}.",
                    )
                )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            checks.append(
                EnvironmentCheck(
                    "Tesseract language query",
                    "MISSING",
                    detail,
                    required=True,
                    fix="Run `tesseract --list-langs` and install eng and hin language data.",
                )
            )

    _check_import("paddle", importer, checks)
    _check_import("paddleocr", importer, checks)

    paddle_home = Path(environment.get("PADDLE_HOME", home_path / ".paddle"))
    paddlex_home = Path(environment.get("PADDLEX_HOME", home_path / ".paddlex"))
    model_cache_locations = (
        str(paddlex_home / "official_models"),
        str(paddle_home / "models"),
        str(home_path / ".paddleocr"),
    )
    return OCRCheckReport(
        python_version=version_text,
        checks=tuple(checks),
        model_cache_locations=model_cache_locations,
    )


def _check_import(
    module_name: str,
    importer: Callable[[str], Any],
    checks: list[EnvironmentCheck],
) -> None:
    try:
        module = importer(module_name)
        version = str(getattr(module, "__version__", "imported"))
        checks.append(
            EnvironmentCheck(module_name, "OK", version, required=True)
        )
    except Exception as exc:
        checks.append(
            EnvironmentCheck(
                module_name,
                "MISSING",
                f"{type(exc).__name__}: {exc}",
                required=True,
                fix="Run .venv/bin/python -m pip install -e '.[ocr]' and address the import error above.",
            )
        )


def _parse_languages(output: str) -> set[str]:
    """Extract language codes from Tesseract's `--list-langs` output."""
    languages: set[str] = set()
    for line in output.splitlines():
        token = line.strip()
        if token and token.isalpha() and len(token) in {3, 4}:
            languages.add(token)
    return languages


def main() -> int:
    """Print readiness and remediation; never instantiate OCR models."""
    report = inspect_environment()
    print(report.render())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())