# Project: KYC Document Intelligence Platform – Phase 1
Source of truth: docs/SRS.md. Read it before any task. Do not implement anything listed as out of scope.

## Stack
Python 3.11+, FastAPI, Pydantic v2, pytest. Frontend: React + TypeScript (Vite) added later.
Everything must use free/open-source tools only. No paid cloud APIs. OCR runs locally.

## Architecture rules
- Layers: api -> services -> extractors / repositories. No business logic in API routes.
- Extraction engines implement DocumentExtractor (extractors/base.py). Concrete: PDFTextExtractor (PyMuPDF), DOCXExtractor (python-docx), OCRExtractor (PaddleOCR primary, Tesseract fallback behind an OCREngine interface).
- Storage and database access go through interfaces (StorageBackend, Repository) with a local implementation first (filesystem + SQLite) and Firebase implementations later. Selected via env config.
- All config via pydantic-settings and env vars. No hardcoded secrets or paths.
- DOCUMENT_RETENTION_MINUTES: implement the cleanup service, but 0 means disabled and is the default.

## Non-negotiable behaviours
- Never silently return empty results. Every document ends in SUCCESS, PARTIAL, FAILED or UNSUPPORTED with a structured error (codes from SRS section 31).
- Distinguish "not found" from "could not extract".
- Every extracted field carries provenance: case_id, document_id, filename, page, value, raw source text, confidence, bbox (nullable), extraction method, timestamp. Never fabricate a bbox.
- OCR confidence is not validity. Do no PAN/GST validation against external systems.
- One document failing must not fail the case; process documents independently.
- Never log document contents or PAN/Aadhaar values. Log metadata only.

## Code quality
- Type hints everywhere, small functions, docstrings on public classes.
- Every feature ships with pytest tests. Run the tests and fix failures before finishing a task.
- Use ruff and black. Keep dependencies minimal and list them in pyproject.toml.
