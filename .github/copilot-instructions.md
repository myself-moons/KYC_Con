# Project: KYC Document Intelligence Platform, Phase 1

The source specification is `SRS.docx` (Markdown text with a historical `.docx` extension). Read it before making changes and do not implement out-of-scope features. `docs/DECISIONS.md` records user decisions that override conflicting SRS requirements.

## Stack and environment

- Python 3.11 or 3.12 (`requires-python = ">=3.11,<3.13"`), FastAPI, Pydantic v2, pytest; frontend work is deferred.
- Use the repository-root `.venv` for all Python commands and tests.
- Dependencies belong in `pyproject.toml`. Do not add PaddleOCR or Tesseract packages until the OCR implementation phase.
- Use free/open-source tooling; no paid cloud APIs.
- Configure the application through `pydantic-settings` and environment variables. Never commit credentials, service-account files, or secrets.

## Architecture

- Layers: API -> services -> extractors / repositories. Keep business logic out of API routes.
- Extraction engines implement `DocumentExtractor` in `app/extractors/base.py`.
- Keep `StorageBackend` and `Repository` interfaces. Use local disk under the gitignored `UPLOAD_DIR` for private uploaded files and Firestore for JSON metadata/results. In-memory implementations are test-only.
- Firestore is the only Firebase product and must remain on the free Spark plan. Support only the Firestore emulator through environment configuration; do not use Firebase Cloud Storage, its emulator, Realtime Database, or SQLite.
- Planned OCR is pretrained PaddleOCR (PP-OCR) first, Tesseract fallback, behind `OCREngine`; do not implement OCR before its planned phase.
- `DOCUMENT_RETENTION_MINUTES=0` is the default and disables cleanup. Cleanup is manual only through the CLI; no scheduler or automatic trigger. Never delete files except on an explicit non-dry-run CLI invocation.

## Required behavior

- Every document must end in `SUCCESS`, `PARTIAL`, `FAILED`, or `UNSUPPORTED` (and may be `PROCESSING` while active), with structured errors using SRS section 31 codes where applicable.
- Distinguish “not found” from “could not extract”. Do not silently represent extraction failures as empty successful results.
- Process documents independently. A failing document must not fail its case or block other uploads.
- Report each file's validation outcome, including unsupported type, too large, corrupted, unreadable, and empty.
- Every extracted field carries case/document IDs, filename, page, value, raw source text, confidence, nullable bbox, extraction method, and timestamp. Never fabricate a bbox.
- PDF embedded text usability uses configurable `PDF_MIN_TEXT_CHARS_PER_PAGE=50` and `PDF_MIN_ALPHANUMERIC_RATIO=0.5`; pages failing either check need OCR.
- OCR confidence is not validity. Do not validate PAN/GST against external systems or make KYC decisions.
- Never log document contents, PAN/Aadhaar values, addresses, or other sensitive data. Log metadata only.

## Scope and quality

- Follow the user's current phase boundary; stop for review at each requested phase.
- Do not implement OCR, DOCX extraction, field extraction, or frontend unless explicitly requested. Local document-file storage is the approved production choice for now; metadata remains in Firestore.
- Use type hints, small functions, public-class docstrings, Ruff, and Black.
- Every implemented feature needs pytest coverage. Run tests in `.venv` and report actual output.
