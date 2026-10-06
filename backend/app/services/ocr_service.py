"""Render source pages and preprocess OCR copies while retaining originals."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pymupdf
from PIL import Image

from app.core.config import Settings, get_settings


class OCRPreprocessingError(RuntimeError):
    """Raised when the configured preprocessing runtime is unavailable."""


@dataclass(frozen=True)
class PreparedOCRPage:
    """Original page pixels for viewing and a separate OCR-ready image."""

    original_image: Image.Image
    processed_image: Image.Image
    dpi: int | None


class OCRService:
    """Prepare OCR inputs without mutating or replacing the original image."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def render_pdf_page(self, pdf_bytes: bytes, page_index: int) -> PreparedOCRPage:
        """Render one zero-based PDF page at configured OCR resolution."""
        try:
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
                page = pdf.load_page(page_index)
                pixmap = page.get_pixmap(dpi=self._settings.ocr_dpi, alpha=False)
                image = Image.frombytes(
                    "RGB", (pixmap.width, pixmap.height), pixmap.samples
                )
        except Exception as exc:
            raise OCRPreprocessingError(
                "PDF page could not be rendered for OCR"
            ) from exc
        return self.prepare_image(image, dpi=self._settings.ocr_dpi)

    def load_image(self, image_bytes: bytes) -> PreparedOCRPage:
        """Load an uploaded image and preprocess only a working copy."""
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                source_image = image.convert("RGB").copy()
        except Exception as exc:
            raise OCRPreprocessingError("Image could not be decoded for OCR") from exc
        return self.prepare_image(source_image, dpi=None)

    def prepare_image(
        self, source_image: Image.Image, dpi: int | None
    ) -> PreparedOCRPage:
        """Return original pixels and a distinct optionally preprocessed copy."""
        original = source_image.convert("RGB").copy()
        steps_enabled = any(
            (
                self._settings.ocr_preprocess_grayscale,
                self._settings.ocr_preprocess_denoise,
                self._settings.ocr_preprocess_deskew,
                self._settings.ocr_preprocess_adaptive_threshold,
            )
        )
        if not steps_enabled:
            return PreparedOCRPage(original, original.copy(), dpi)

        try:
            import cv2
        except ImportError as exc:
            raise OCRPreprocessingError(f"OpenCV could not be imported: {exc}") from exc

        source_array = np.asarray(original)
        grayscale = self._settings.ocr_preprocess_grayscale
        if grayscale:
            working = cv2.cvtColor(source_array, cv2.COLOR_RGB2GRAY)
        else:
            working = cv2.cvtColor(source_array, cv2.COLOR_RGB2BGR)

        if self._settings.ocr_preprocess_denoise:
            if grayscale:
                working = cv2.fastNlMeansDenoising(
                    working, None, h=10, templateWindowSize=7, searchWindowSize=21
                )
            else:
                working = cv2.fastNlMeansDenoisingColored(
                    working,
                    None,
                    h=10,
                    hColor=10,
                    templateWindowSize=7,
                    searchWindowSize=21,
                )
        if self._settings.ocr_preprocess_deskew:
            working = self._deskew(working, cv2)
        if self._settings.ocr_preprocess_adaptive_threshold:
            if working.ndim == 3:
                working = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
            block_size = min(35, min(working.shape[:2]) // 2 * 2 - 1)
            block_size = max(3, block_size if block_size % 2 else block_size - 1)
            working = cv2.adaptiveThreshold(
                working,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                11,
            )

        if working.ndim == 2:
            processed = Image.fromarray(working, mode="L")
        else:
            processed = Image.fromarray(cv2.cvtColor(working, cv2.COLOR_BGR2RGB))
        return PreparedOCRPage(original, processed, dpi)

    @staticmethod
    def _deskew(image: np.ndarray, cv2: object) -> np.ndarray:
        """Estimate small scan rotation from foreground pixels and correct it."""
        grayscale = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        )
        inverted = cv2.bitwise_not(
            cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        )
        coordinates = cv2.findNonZero(inverted)
        if coordinates is None or len(coordinates) < 10:
            return image
        angle = cv2.minAreaRect(coordinates)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if not 0.15 <= abs(angle) <= 8.0:
            return image
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        border_value: int | tuple[int, int, int] = (
            255 if image.ndim == 2 else (255, 255, 255)
        )
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
