"""Tests for deterministic synthetic scan and ground-truth generation."""

import json

import pymupdf
from PIL import Image, ImageChops
from tests.fixtures.make_samples import PRESETS, degrade_image, profile_for


def test_generated_documents_include_ground_truth_and_pdf_variants(
    synthetic_kyc_samples_factory,
) -> None:
    outputs = synthetic_kyc_samples_factory(preset="light_scan", seed=29)
    output_dir = outputs["pan_card.json"].parent

    expected_stems = {
        "pan_card",
        "gst_certificate",
        "certificate_of_incorporation",
        "cancelled_cheque",
        "multi_page_profile",
    }
    for stem in expected_stems:
        sidecar_path = output_dir / f"{stem}.json"
        truth = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert truth["synthetic_only"] is True
        assert truth["ground_truth"]
        assert truth["page_count"] >= 1
        assert (output_dir / truth["files"]["image_only_pdf"]).exists()
        assert (output_dir / truth["files"]["digital_pdf"]).exists()
        for image_name in truth["files"]["images"]:
            assert (output_dir / image_name).exists()
        for image_name in truth["files"]["images"]:
            assert (output_dir / image_name).suffix in {".png", ".jpg"}

        with pymupdf.open(output_dir / truth["files"]["image_only_pdf"]) as pdf:
            assert all(not page.get_text("text").strip() for page in pdf)
        with pymupdf.open(output_dir / truth["files"]["digital_pdf"]) as pdf:
            assert any(page.get_text("text").strip() for page in pdf)

    pan = json.loads((output_dir / "pan_card.json").read_text(encoding="utf-8"))
    gst = json.loads((output_dir / "gst_certificate.json").read_text(encoding="utf-8"))
    assert pan["ground_truth"]["PAN"] == "AAAAA0000A"
    gstin = gst["ground_truth"]["GSTIN"]
    assert len(gstin) == 15
    assert gstin.startswith("27AAAAA0000A")
    with pymupdf.open(output_dir / "multi_page_profile_digital.pdf") as pdf:
        assert pdf.page_count == 3


def test_degradation_changes_pixels_and_presets_are_available(
    synthetic_kyc_samples_factory,
) -> None:
    outputs = synthetic_kyc_samples_factory(preset="xerox_medium", seed=13)

    with Image.open(outputs["pan_card_before.png"]) as before_image:
        before = before_image.convert("RGB")
    with Image.open(outputs["pan_card_after.png"]) as after_image:
        after = after_image.convert("RGB")
    assert ImageChops.difference(before, after).getbbox() is not None
    assert PRESETS == ("clean", "light_scan", "xerox_medium", "xerox_heavy")


def test_degradation_seed_and_severity_overrides_are_deterministic() -> None:
    source = Image.new("RGB", (160, 100), (205, 195, 175))
    first = degrade_image(source, preset="xerox_medium", seed=41)
    second = degrade_image(source, preset="xerox_medium", seed=41)
    softened = profile_for("clean", {"blur": 0.5})

    assert first.tobytes() == second.tobytes()
    assert softened.blur == 0.5
    assert profile_for("clean").blur == 0.0
