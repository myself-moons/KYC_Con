"""Mocked environment checks for the OCR readiness command."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.scripts.check_ocr_env import inspect_environment


class FakeDistribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.version = version


def _distributions(*items: tuple[str, str]):
    return lambda: [FakeDistribution(name, version) for name, version in items]


def _imports(modules: dict[str, object], fail: dict[str, Exception] | None = None):
    failures = fail or {}

    def importer(name: str) -> object:
        if name in failures:
            raise failures[name]
        if name not in modules:
            raise ModuleNotFoundError(name)
        return modules[name]

    return importer


def _successful_run(stdout: str = "List of available languages (3):\neng\nhin\nmar\n"):
    return lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=stdout, stderr=""
    )


def _check(report, name: str):
    return next(check for check in report.checks if check.name == name)


def test_ready_environment_and_model_cache_do_not_load_models() -> None:
    report = inspect_environment(
        python_version=(3, 12, 3),
        distributions=_distributions(("opencv-python-headless", "5.0.0")),
        module_importer=_imports(
            {
                "cv2": SimpleNamespace(__version__="5.0.0"),
                "paddle": SimpleNamespace(__version__="3.3.1"),
                "paddleocr": SimpleNamespace(__version__="3.7.0"),
            }
        ),
        which=lambda _: "/usr/bin/tesseract",
        run=_successful_run(),
        environ={"PADDLE_HOME": "/tmp/paddle-cache", "PADDLEX_HOME": "/tmp/paddlex"},
        home=Path("/tmp/home"),
    )

    assert report.exit_code == 0
    assert _check(report, "Tesseract language mar").status == "OK"
    assert report.model_cache_locations[0] == "/tmp/paddlex/official_models"


def test_python_version_failure_has_exact_fix() -> None:
    report = inspect_environment(
        python_version=(3, 14, 0),
        distributions=_distributions(("opencv-python-headless", "5")),
        module_importer=_imports({"cv2": object(), "paddle": object(), "paddleocr": object()}),
        which=lambda _: "/usr/bin/tesseract",
        run=_successful_run(),
        environ={},
        home=Path("/tmp/home"),
    )

    check = _check(report, "Python version")
    assert check.status == "MISSING"
    assert "python3.12 -m venv .venv" in check.fix
    assert report.exit_code == 1


def test_multiple_opencv_variants_warn_without_hiding_cv2_status() -> None:
    report = inspect_environment(
        python_version=(3, 12, 0),
        distributions=_distributions(
            ("opencv-python-headless", "5"), ("opencv-contrib-python", "4")
        ),
        module_importer=_imports({"cv2": SimpleNamespace(__version__="4"), "paddle": object(), "paddleocr": object()}),
        which=lambda _: "/usr/bin/tesseract",
        run=_successful_run(),
        environ={},
        home=Path("/tmp/home"),
    )

    assert _check(report, "OpenCV distributions").status == "WARN"
    assert _check(report, "cv2 import").status == "OK"
    assert report.exit_code == 0


def test_cv2_failure_reports_native_library_fix() -> None:
    report = inspect_environment(
        python_version=(3, 12, 0),
        distributions=_distributions(("opencv-python-headless", "5")),
        module_importer=_imports(
            {"paddle": object(), "paddleocr": object()},
            {"cv2": ImportError("libGL.so.1 missing")},
        ),
        which=lambda _: "/usr/bin/tesseract",
        run=_successful_run(),
        environ={},
        home=Path("/tmp/home"),
    )

    check = _check(report, "cv2 import")
    assert check.status == "MISSING"
    assert "libgl1" in check.fix
    assert report.exit_code == 1


@pytest.mark.parametrize("module_name", ["paddle", "paddleocr"])
def test_paddle_import_failure_has_optional_extra_fix(module_name: str) -> None:
    report = inspect_environment(
        python_version=(3, 12, 0),
        distributions=_distributions(("opencv-python-headless", "5")),
        module_importer=_imports(
            {"cv2": object(), "paddle": object(), "paddleocr": object()},
            {module_name: ImportError("synthetic import failure")},
        ),
        which=lambda _: "/usr/bin/tesseract",
        run=_successful_run(),
        environ={},
        home=Path("/tmp/home"),
    )

    check = _check(report, module_name)
    assert check.status == "MISSING"
    assert ".[ocr]" in check.fix
    assert report.exit_code == 1


def test_tesseract_binary_failure_reports_system_fix() -> None:
    report = inspect_environment(
        python_version=(3, 12, 0),
        distributions=_distributions(("opencv-python-headless", "5")),
        module_importer=_imports({"cv2": object(), "paddle": object(), "paddleocr": object()}),
        which=lambda _: None,
        run=_successful_run(),
        environ={},
        home=Path("/tmp/home"),
    )

    assert _check(report, "Tesseract binary").status == "MISSING"
    assert _check(report, "Tesseract language eng").required is True
    assert _check(report, "Tesseract language mar").required is False
    assert report.exit_code == 1


@pytest.mark.parametrize("missing_language", ["eng", "hin", "mar"])
def test_tesseract_language_branches(missing_language: str) -> None:
    languages = {"eng", "hin", "mar"} - {missing_language}
    report = inspect_environment(
        python_version=(3, 12, 0),
        distributions=_distributions(("opencv-python-headless", "5")),
        module_importer=_imports({"cv2": object(), "paddle": object(), "paddleocr": object()}),
        which=lambda _: "/usr/bin/tesseract",
        run=_successful_run(
            "List of available languages (2):\n" + "\n".join(sorted(languages))
        ),
        environ={},
        home=Path("/tmp/home"),
    )

    check = _check(report, f"Tesseract language {missing_language}")
    assert check.status == "MISSING"
    assert bool(report.exit_code) is (missing_language != "mar")