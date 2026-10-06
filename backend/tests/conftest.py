"""Shared API fixtures and synthetic document factories."""

from collections.abc import Callable, Iterator
from io import BytesIO
from pathlib import Path

import pymupdf
import pytest
from app.core.config import Settings
from app.main import app
from fastapi.testclient import TestClient
from tests.fakes import InMemoryRepository, InMemoryStorage
from tests.fixtures.make_samples import generate_samples


@pytest.fixture
def settings() -> Settings:
    """Provide fast, isolated test settings with a one-megabyte upload cap."""
    return Settings(
        _env_file=None,
        max_upload_mb=1,
        max_files_per_case=3,
        processing_timeout_seconds=1.0,
    )


@pytest.fixture
def memory_repository() -> InMemoryRepository:
    """Provide an in-memory repository fake for unit tests."""
    return InMemoryRepository()


@pytest.fixture
def memory_storage() -> InMemoryStorage:
    """Provide an in-memory storage fake for unit tests."""
    return InMemoryStorage()


@pytest.fixture
def client(
    memory_repository: InMemoryRepository,
    memory_storage: InMemoryStorage,
    settings: Settings,
) -> Iterator[TestClient]:
    """Provide an API client configured with isolated in-memory adapters."""
    app.state.repository = memory_repository
    app.state.storage_backend = memory_storage
    app.state.settings = settings
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        for state_name in ("repository", "storage_backend", "settings"):
            if hasattr(app.state, state_name):
                delattr(app.state, state_name)


@pytest.fixture
def pdf_factory() -> Callable[..., bytes]:
    """Create deterministic synthetic PDFs without real KYC content."""

    def create_pdf(
        pages: list[str | None],
        image_only_pages: set[int] | None = None,
        user_password: str | None = None,
    ) -> bytes:
        image_only = image_only_pages or set()
        pdf = pymupdf.open()
        for page_index, text in enumerate(pages):
            page = pdf.new_page(width=612, height=792)
            if text:
                page.insert_textbox(pymupdf.Rect(40, 40, 570, 740), text, fontsize=12)
            if page_index in image_only:
                pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), 0)
                pixmap.clear_with(245)
                page.insert_image(page.rect, pixmap=pixmap)
        stream = BytesIO()
        save_options: dict[str, object] = {}
        if user_password:
            save_options = {
                "encryption": pymupdf.PDF_ENCRYPT_AES_256,
                "owner_pw": "synthetic-owner-password",
                "user_pw": user_password,
            }
        pdf.save(stream, **save_options)
        pdf.close()
        return stream.getvalue()

    return create_pdf


@pytest.fixture
def digital_pdf_bytes(pdf_factory: Callable[..., bytes]) -> bytes:
    """Return a single-page synthetic PDF with usable embedded text."""
    text = (
        "Synthetic KYC Review Sample. This generated document contains enough "
        "ordinary alphanumeric text to pass the configured embedded-text check."
    )
    return pdf_factory([text])


@pytest.fixture
def scanned_pdf_bytes(pdf_factory: Callable[..., bytes]) -> bytes:
    """Return an image-only PDF page that requires OCR."""
    return pdf_factory([None], image_only_pages={0})


@pytest.fixture
def encrypted_pdf_bytes(pdf_factory: Callable[..., bytes]) -> bytes:
    """Return a synthetic password-protected PDF."""
    return pdf_factory(["Synthetic protected page content."], user_password="test-pass")


@pytest.fixture
def synthetic_kyc_samples_factory(
    tmp_path: Path,
) -> Callable[..., dict[str, Path]]:
    """Generate fake OCR documents inside pytest's temporary directory."""

    def factory(
        preset: str = "xerox_medium",
        seed: int = 17,
        folder: str = "samples",
    ) -> dict[str, Path]:
        return generate_samples(tmp_path / folder, preset=preset, seed=seed)

    return factory
