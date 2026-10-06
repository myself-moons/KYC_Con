"""Generate synthetic KYC-style scan fixtures and sidecar ground truth."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

PRESETS = ("clean", "light_scan", "xerox_medium", "xerox_heavy")
EFFECT_NAMES = (
    "grayscale",
    "low_contrast",
    "gaussian_noise",
    "salt_pepper",
    "blur",
    "rotation",
    "perspective",
    "edge_borders",
    "shadow",
    "jpeg_compression",
    "low_dpi",
)


@dataclass(frozen=True)
class DegradationProfile:
    """Normalized 0..1 severities for each simulated scan artifact."""

    grayscale: float = 0.0
    low_contrast: float = 0.0
    gaussian_noise: float = 0.0
    salt_pepper: float = 0.0
    blur: float = 0.0
    rotation: float = 0.0
    perspective: float = 0.0
    edge_borders: float = 0.0
    shadow: float = 0.0
    jpeg_compression: float = 0.0
    low_dpi: float = 0.0

    def __post_init__(self) -> None:
        for effect, severity in asdict(self).items():
            if not 0.0 <= severity <= 1.0:
                raise ValueError(f"{effect} severity must be between 0 and 1")


@dataclass(frozen=True)
class SyntheticPage:
    """Printed page content and its machine-readable ground truth."""

    title: str
    fields: dict[str, str]
    note: str = ""


@dataclass(frozen=True)
class SyntheticDocument:
    """A synthetic document composed of one or more pages."""

    stem: str
    document_type: str
    width: int
    height: int
    pages: tuple[SyntheticPage, ...]


_PRESET_PROFILES: dict[str, DegradationProfile] = {
    "clean": DegradationProfile(),
    "light_scan": DegradationProfile(
        grayscale=0.55,
        low_contrast=0.15,
        gaussian_noise=0.12,
        salt_pepper=0.08,
        blur=0.12,
        rotation=0.2,
        perspective=0.12,
        edge_borders=0.12,
        shadow=0.12,
        jpeg_compression=0.1,
        low_dpi=0.1,
    ),
    "xerox_medium": DegradationProfile(
        grayscale=0.9,
        low_contrast=0.45,
        gaussian_noise=0.35,
        salt_pepper=0.32,
        blur=0.35,
        rotation=0.62,
        perspective=0.42,
        edge_borders=0.55,
        shadow=0.5,
        jpeg_compression=0.55,
        low_dpi=0.45,
    ),
    "xerox_heavy": DegradationProfile(
        grayscale=1.0,
        low_contrast=0.72,
        gaussian_noise=0.62,
        salt_pepper=0.58,
        blur=0.62,
        rotation=0.95,
        perspective=0.78,
        edge_borders=0.9,
        shadow=0.82,
        jpeg_compression=0.9,
        low_dpi=0.8,
    ),
}

_FONT_PATHS = {
    ("sans", False): "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ("sans", True): "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ("serif", False): "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ("serif", True): "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ("mono", False): "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ("mono", True): "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
}


def synthetic_documents() -> tuple[SyntheticDocument, ...]:
    """Return fake fixed-format documents containing no personal data."""
    pan = "AAAAA0000A"
    gstin = "27AAAAA0000A1Z5"
    return (
        SyntheticDocument(
            "pan_card",
            "synthetic_pan_card",
            1024,
            640,
            (
                SyntheticPage(
                    "SYNTHETIC TAX IDENTITY CARD",
                    {
                        "PAN": pan,
                        "Name": "Aarav Sample",
                        "Parent name": "Kiran Example",
                        "Date of birth": "01/01/1990",
                        "Card number": "0000 0000 0000",
                    },
                    "SPECIMEN - SYNTHETIC DATA - NOT AN OFFICIAL DOCUMENT",
                ),
            ),
        ),
        SyntheticDocument(
            "gst_certificate",
            "synthetic_gst_registration_certificate",
            1050,
            1360,
            (
                SyntheticPage(
                    "GOODS AND SERVICES TAX REGISTRATION",
                    {
                        "GSTIN": gstin,
                        "Legal name": "Example Sample Trading Private Limited",
                        "Trade name": "Sample Supply House",
                        "Constitution": "Private Limited Company",
                        "Principal place": "123 Example Road, Sample City, ZZ 000001",
                        "Registration date": "01/01/2020",
                    },
                    "SYNTHETIC CERTIFICATE - FOR OCR TESTING ONLY",
                ),
            ),
        ),
        SyntheticDocument(
            "certificate_of_incorporation",
            "synthetic_certificate_of_incorporation",
            1050,
            1360,
            (
                SyntheticPage(
                    "CERTIFICATE OF INCORPORATION",
                    {
                        "Company name": "Example Orbit Systems Private Limited",
                        "Corporate ID": "U00000ZZ2020PTC000001",
                        "Incorporation date": "01/01/2020",
                        "Registered office": (
                            "45 Placeholder Avenue, Sample City, ZZ 000002"
                        ),
                        "Registrar reference": "SYN-REG-000001",
                    },
                    "FICTIONAL COMPANY - SYNTHETIC SAMPLE",
                ),
            ),
        ),
        SyntheticDocument(
            "cancelled_cheque",
            "synthetic_cancelled_cheque",
            1200,
            560,
            (
                SyntheticPage(
                    "EXAMPLE COMMUNITY BANK",
                    {
                        "Pay to": "AARAV SAMPLE",
                        "Account number": "000000123456",
                        "IFSC": "EXMP0000123",
                        "Branch": "SAMPLE CITY - TEST BRANCH",
                        "MICR": "000000000",
                    },
                    "CANCELLED - VOID - SYNTHETIC SPECIMEN",
                ),
            ),
        ),
        SyntheticDocument(
            "multi_page_profile",
            "synthetic_multi_page_company_profile",
            1050,
            1360,
            (
                SyntheticPage(
                    "SYNTHETIC COMPANY PROFILE - PAGE 1",
                    {
                        "Company": "Example Sample Trading Private Limited",
                        "PAN": pan,
                        "GSTIN": gstin,
                        "Established": "01/01/2020",
                    },
                    "Generated sample page one of three.",
                ),
                SyntheticPage(
                    "REGISTERED OFFICE - PAGE 2",
                    {
                        "Address": "123 Example Road, Sample City, ZZ 000001",
                        "Mailing address": "PO Box 42, Sample City, ZZ 000001",
                        "Contact email": "not-a-real-address@example.invalid",
                        "Telephone": "+00 000 000 0000",
                    },
                    "All contact details are reserved or fictional values.",
                ),
                SyntheticPage(
                    "AUTHORIZED SIGNATORY - PAGE 3",
                    {
                        "Name": "Aarav Sample",
                        "Role": "Synthetic Test Signatory",
                        "Reference": "SAMPLE-REF-0003",
                        "Signature": "[printed placeholder]",
                    },
                    "No real signature or identity information is present.",
                ),
            ),
        ),
    )


def profile_for(
    preset: str, overrides: dict[str, float] | None = None
) -> DegradationProfile:
    """Return a named preset with optional individual effect overrides."""
    if preset not in _PRESET_PROFILES:
        raise ValueError(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    profile = _PRESET_PROFILES[preset]
    if not overrides:
        return profile
    invalid_names = set(overrides) - set(EFFECT_NAMES)
    if invalid_names:
        raise ValueError(
            f"unknown degradation effect(s): {', '.join(sorted(invalid_names))}"
        )
    return replace(profile, **overrides)


def degrade_image(
    image: Image.Image,
    preset: str = "xerox_medium",
    seed: int = 17,
    severity_overrides: dict[str, float] | None = None,
) -> Image.Image:
    """Apply deterministic xerox/scan artifacts with adjustable severities."""
    profile = profile_for(preset, severity_overrides)
    rng = random.Random(seed)
    result = image.convert("RGB")
    if profile.grayscale:
        gray = result.convert("L").convert("RGB")
        result = Image.blend(result, gray, profile.grayscale)
    if profile.low_contrast:
        result = ImageEnhance.Contrast(result).enhance(
            1.0 - 0.72 * profile.low_contrast
        )
    if profile.gaussian_noise:
        result = _add_gaussian_noise(result, 4 + 25 * profile.gaussian_noise, rng)
    if profile.salt_pepper:
        result = _add_speckle(result, profile.salt_pepper, rng)
    if profile.blur:
        result = result.filter(ImageFilter.GaussianBlur(radius=1.5 * profile.blur))
    if profile.rotation:
        max_angle = 1.0 + 3.0 * profile.rotation
        angle = rng.choice((-1.0, 1.0)) * rng.uniform(1.0, max_angle)
        result = result.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(242, 240, 233),
        )
    if profile.perspective:
        result = _apply_skew(result, profile.perspective, rng)
    if profile.edge_borders:
        result = _add_edge_borders(result, profile.edge_borders, rng)
    if profile.shadow:
        result = _add_uneven_lighting(result, profile.shadow, rng)
    if profile.low_dpi:
        result = _simulate_low_dpi(result, profile.low_dpi)
    if profile.jpeg_compression:
        output = BytesIO()
        quality = int(95 - 68 * profile.jpeg_compression)
        result.save(output, format="JPEG", quality=quality, optimize=False)
        output.seek(0)
        with Image.open(output) as compressed:
            result = compressed.convert("RGB")
    return result


def generate_samples(
    out_dir: str | Path,
    preset: str = "xerox_medium",
    seed: int = 17,
    severity_overrides: dict[str, float] | None = None,
) -> dict[str, Path]:
    """Write PNG/JPG pages, image-only/digital PDFs, and sidecar truth JSON."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_for(preset, severity_overrides)
    generated: dict[str, Path] = {}
    for document_index, document in enumerate(synthetic_documents()):
        clean_pages = [
            _render_page(document, page, page_index)
            for page_index, page in enumerate(document.pages)
        ]
        scan_pages = [
            degrade_image(
                page,
                preset,
                seed + document_index * 1009 + page_index * 9176,
                severity_overrides,
            )
            for page_index, page in enumerate(clean_pages)
        ]
        for page_index, (clean_page, scan_page) in enumerate(
            zip(clean_pages, scan_pages, strict=True), start=1
        ):
            image_stem = f"{document.stem}_page_{page_index}"
            png_path = output_dir / f"{image_stem}.png"
            jpg_path = output_dir / f"{image_stem}.jpg"
            scan_page.save(png_path, format="PNG", dpi=_output_dpi(profile))
            scan_page.save(
                jpg_path,
                format="JPEG",
                quality=_jpeg_quality(profile),
                dpi=_output_dpi(profile),
            )
            generated[png_path.name] = png_path
            generated[jpg_path.name] = jpg_path
            if document.stem == "pan_card" and page_index == 1:
                before_path = output_dir / "pan_card_before.png"
                after_path = output_dir / "pan_card_after.png"
                clean_page.save(before_path, format="PNG")
                scan_page.save(after_path, format="PNG", dpi=_output_dpi(profile))
                generated[before_path.name] = before_path
                generated[after_path.name] = after_path

        image_pdf_path = output_dir / f"{document.stem}_image_only.pdf"
        digital_pdf_path = output_dir / f"{document.stem}_digital.pdf"
        _write_image_pdf(image_pdf_path, scan_pages)
        _write_digital_pdf(digital_pdf_path, document)
        generated[image_pdf_path.name] = image_pdf_path
        generated[digital_pdf_path.name] = digital_pdf_path

        sidecar = {
            "synthetic_only": True,
            "document_type": document.document_type,
            "stem": document.stem,
            "preset": preset,
            "seed": seed,
            "page_count": len(document.pages),
            "degradation_severity": asdict(profile),
            "ground_truth": {
                key: value
                for page in document.pages
                for key, value in page.fields.items()
            },
            "pages": [
                {
                    "page_number": page_index,
                    "title": page.title,
                    "note": page.note,
                    "fields": page.fields,
                    "text": _page_text(page),
                }
                for page_index, page in enumerate(document.pages, start=1)
            ],
            "files": {
                "image_only_pdf": image_pdf_path.name,
                "digital_pdf": digital_pdf_path.name,
                "images": [
                    f"{document.stem}_page_{page_index}.{extension}"
                    for page_index in range(1, len(document.pages) + 1)
                    for extension in ("png", "jpg")
                ],
            },
        }
        sidecar_path = output_dir / f"{document.stem}.json"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        generated[sidecar_path.name] = sidecar_path
    return generated


def _render_page(
    document: SyntheticDocument, page: SyntheticPage, page_index: int
) -> Image.Image:
    image = Image.new("RGB", (document.width, document.height), (250, 249, 245))
    draw = ImageDraw.Draw(image)
    margin = max(36, document.width // 24)
    title_font = _font(
        "serif" if page_index % 2 else "sans", True, max(25, document.width // 34)
    )
    label_font = _font(
        "mono" if page_index % 2 else "sans", True, max(15, document.width // 80)
    )
    value_font = _font(
        "sans" if page_index % 2 else "serif", False, max(19, document.width // 58)
    )
    note_font = _font("mono", True, max(13, document.width // 105))
    draw.rounded_rectangle(
        (12, 12, document.width - 13, document.height - 13),
        radius=12,
        outline=(55, 70, 82),
        width=max(2, document.width // 700),
    )
    draw.rectangle(
        (margin, margin, document.width - margin, margin + 13), fill=(39, 91, 112)
    )
    y = margin + 45
    draw.text((margin, y), page.title, font=title_font, fill=(29, 46, 57))
    y += int(title_font.size * 1.8)
    draw.line((margin, y, document.width - margin, y), fill=(126, 141, 145), width=2)
    y += 30
    if document.stem == "pan_card":
        draw.rounded_rectangle(
            (document.width - 180, margin + 38, document.width - margin, margin + 145),
            radius=5,
            outline=(120, 130, 130),
            width=2,
            fill=(231, 234, 225),
        )
        draw.text(
            (document.width - 163, margin + 70),
            "PHOTO\nSAMPLE",
            font=label_font,
            fill=(90, 100, 97),
            spacing=5,
        )
    if document.stem == "cancelled_cheque":
        draw.rectangle(
            (margin, margin, document.width - margin, document.height - margin),
            outline=(146, 158, 166),
            width=2,
        )
        y += 12

    value_x = margin + max(220, document.width // 4)
    available_width = document.width - value_x - margin
    for field_index, (label, value) in enumerate(page.fields.items()):
        rendered_value = {
            "PAN": "AAAAA 0000 A",
            "GSTIN": "27 AAAAA 0000 A1Z5",
        }.get(label, value)
        if document.stem == "pan_card" and label == "PAN":
            label_font_local = _font("mono", True, value_font.size)
            value_font_local = _font("mono", False, max(48, value_font.size * 2))
        elif label == "GSTIN":
            label_font_local = _font("mono", True, value_font.size)
            value_font_local = _font("mono", False, max(54, value_font.size * 3))
        else:
            label_font_local = label_font
            value_font_local = value_font
        row_gap = max(55, int(value_font_local.size * 1.55))
        draw.text(
            (margin + 12, y),
            label.upper() + ":",
            font=label_font_local,
            fill=(64, 72, 78),
        )
        draw.text(
            (value_x, y), rendered_value, font=value_font_local, fill=(21, 30, 38)
        )
        if draw.textlength(rendered_value, font=value_font_local) > available_width:
            words = rendered_value.split()
            split_at = max(1, len(words) // 2)
            draw.text(
                (value_x, y + value_font_local.size + 3),
                " ".join(words[split_at:]),
                font=value_font_local,
                fill=(21, 30, 38),
            )
            y += value_font_local.size + 6
        y += row_gap
        if y > document.height - margin - 90:
            y = margin + 75 + (field_index % 2) * row_gap
            value_x = document.width // 2
    if document.stem == "cancelled_cheque":
        baseline = document.height - margin - 70
        draw.text(
            (margin + 10, baseline),
            "000000000  000000123456  123456789",
            font=_font("mono", False, 27),
            fill=(28, 48, 62),
        )
        draw.text(
            (document.width // 2, margin + 200),
            "CANCELLED",
            font=_font("sans", True, 64),
            fill=(150, 40, 40),
            stroke_width=1,
        )
    else:
        footer_y = document.height - margin - 65
        draw.line(
            (margin, footer_y - 15, document.width - margin, footer_y - 15),
            fill=(170, 177, 178),
            width=1,
        )
        draw.text((margin + 5, footer_y), page.note, font=note_font, fill=(105, 60, 60))
    return image


def _font(
    family: str, bold: bool, size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path(_FONT_PATHS[(family, bold)])
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _page_text(page: SyntheticPage) -> str:
    values = [
        page.title,
        *(f"{label}: {value}" for label, value in page.fields.items()),
    ]
    return "\n".join([*values, page.note])


def _add_gaussian_noise(
    image: Image.Image, sigma: float, rng: random.Random
) -> Image.Image:
    random_state = np.random.default_rng(rng.getrandbits(64))
    pixels = np.asarray(image, dtype=np.int16)
    noise = random_state.normal(
        loc=0.0, scale=sigma, size=(image.height, image.width, 1)
    )
    noisy_pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_pixels, mode="RGB")


def _add_speckle(
    image: Image.Image, severity: float, rng: random.Random
) -> Image.Image:
    probability = 0.001 + 0.018 * severity
    random_state = np.random.default_rng(rng.getrandbits(64))
    choices = random_state.random((image.height, image.width))
    pixels = np.asarray(image).copy()
    pixels[choices < probability] = (255, 255, 255)
    pixels[choices > 1.0 - probability] = (12, 12, 12)
    return Image.fromarray(pixels, mode="RGB")


def _apply_skew(image: Image.Image, severity: float, rng: random.Random) -> Image.Image:
    horizontal = rng.uniform(-0.018, 0.018) * severity
    vertical = rng.uniform(-0.013, 0.013) * severity
    perspective_x = rng.uniform(-0.000018, 0.000018) * severity
    perspective_y = rng.uniform(-0.000018, 0.000018) * severity
    coefficients = (
        1,
        horizontal,
        2 * severity,
        vertical,
        1,
        2 * severity,
        perspective_x,
        perspective_y,
    )
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(242, 240, 233),
    )


def _add_edge_borders(
    image: Image.Image, severity: float, rng: random.Random
) -> Image.Image:
    width, height = image.size
    border = max(1, int(min(width, height) * 0.003 * severity))
    draw = ImageDraw.Draw(image)
    color = (18, 20, 21)
    if rng.random() < 0.85:
        draw.rectangle((0, 0, width - 1, border), fill=color)
    if rng.random() < 0.7:
        draw.rectangle((0, height - border, width - 1, height - 1), fill=color)
    if rng.random() < 0.45:
        draw.rectangle((0, 0, border, height - 1), fill=color)
    if rng.random() < 0.3:
        draw.rectangle((width - border, 0, width - 1, height - 1), fill=color)
    return image


def _add_uneven_lighting(
    image: Image.Image, severity: float, rng: random.Random
) -> Image.Image:
    direction = Image.linear_gradient("L").resize(image.size)
    if rng.random() < 0.5:
        direction = direction.transpose(Image.Transpose.ROTATE_90).resize(image.size)
    if rng.random() < 0.5:
        direction = ImageChops.invert(direction)
    blend_mask = ImageEnhance.Brightness(direction).enhance(0.35 + 0.65 * severity)
    darkened = ImageEnhance.Brightness(image).enhance(1.0 - 0.28 * severity)
    return Image.composite(darkened, image, blend_mask)


def _simulate_low_dpi(image: Image.Image, severity: float) -> Image.Image:
    factor = 1.0 - 0.5 * severity
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.resize(size, Image.Resampling.BILINEAR).resize(
        image.size, Image.Resampling.BICUBIC
    )


def _output_dpi(profile: DegradationProfile) -> tuple[int, int]:
    dpi = max(72, round(300 - 190 * profile.low_dpi))
    return (dpi, dpi)


def _jpeg_quality(profile: DegradationProfile) -> int:
    return max(28, round(95 - 68 * profile.jpeg_compression))


def _write_image_pdf(path: Path, pages: list[Image.Image]) -> None:
    pdf = pymupdf.open()
    for image in pages:
        page = pdf.new_page(width=image.width * 0.75, height=image.height * 0.75)
        stream = BytesIO()
        image.save(stream, format="PNG")
        page.insert_image(page.rect, stream=stream.getvalue())
    pdf.save(path, garbage=4, deflate=True)
    pdf.close()


def _write_digital_pdf(path: Path, document: SyntheticDocument) -> None:
    pdf = pymupdf.open()
    for page_data in document.pages:
        page = pdf.new_page(width=document.width * 0.75, height=document.height * 0.75)
        page.insert_textbox(
            pymupdf.Rect(32, 32, page.rect.width - 32, page.rect.height - 32),
            _page_text(page_data),
            fontname="helv",
            fontsize=10,
            lineheight=1.3,
        )
    pdf.save(path, garbage=4, deflate=True)
    pdf.close()


def _parse_severity(value: str) -> tuple[str, float]:
    try:
        name, raw_level = value.split("=", maxsplit=1)
        level = float(raw_level)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "severity must use EFFECT=0..1 format"
        ) from exc
    if name not in EFFECT_NAMES:
        raise argparse.ArgumentTypeError(f"unknown effect {name!r}")
    if not 0.0 <= level <= 1.0:
        raise argparse.ArgumentTypeError("severity must be between 0 and 1")
    return name, level


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for generating reviewable synthetic OCR samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/samples"))
    parser.add_argument("--preset", choices=PRESETS, default="xerox_medium")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--severity",
        action="append",
        type=_parse_severity,
        metavar="EFFECT=0..1",
        help="override an effect severity; repeat to set multiple effects",
    )
    args = parser.parse_args(argv)
    overrides = dict(args.severity or [])
    try:
        outputs = generate_samples(args.out, args.preset, args.seed, overrides)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {len(synthetic_documents())} synthetic documents in {args.out}.")
    print(
        f"Before/after example: {args.out / 'pan_card_before.png'} -> "
        f"{args.out / 'pan_card_after.png'}"
    )
    print(f"Generated {len(outputs)} files; all values are synthetic.")


if __name__ == "__main__":
    main()
