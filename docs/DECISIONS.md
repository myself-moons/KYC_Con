# Implementation Decisions

These user-approved decisions override conflicting requirements in the SRS and
`.github/copilot-instructions.md`.

1. **Scope:** SRS Steps 4 and 5 (DOCX extraction and pretrained OCR) are now
   explicitly approved in addition to the earlier Steps 1–3 and Step 6
   status/failure system. Frontend and structured field extraction remain out
   of scope until separately approved.
2. **Persistence and cost:** Firestore is the only Firebase product and must
   remain on the free Spark plan. Firebase Cloud Storage is dropped because it
   requires the paid Blaze plan. Store all JSON/metadata (cases, documents,
   pages, and extracted fields) in Firestore following SRS section 23. Store
   uploaded files on local disk under `UPLOAD_DIR` (default `./data/uploads`),
   outside served/static paths. Keep the `StorageBackend` interface so another
   storage provider can be substituted later. Do not use Realtime Database or
   SQLite.
3. **Firestore emulator and credentials:** Support the Firestore emulator via
   `USE_EMULATOR` and `FIRESTORE_EMULATOR_HOST`; do not configure a Storage
   emulator. Switching between emulator and real Firestore must require only
   environment changes. Credentials come from environment/ADC or a
   service-account path; credential files must be ignored by Git and never
   committed.
4. **PDF text usability:** Configure `PDF_MIN_TEXT_CHARS_PER_PAGE=50` and
   `PDF_MIN_ALPHANUMERIC_RATIO=0.5` as conservative defaults. A page that fails
   either check is marked `needs_ocr` and routed to the approved OCR fallback.
5. **Retention:** `DOCUMENT_RETENTION_MINUTES=0` is the default and disables
   cleanup. Local files are deleted only by an explicitly run manual CLI, for
   example
   `python -m app.scripts.cleanup --older-than-minutes N --dry-run`. There is no
   scheduler, background job, or automatic trigger. Retain `expires_at` on
   documents. A dry run never deletes; deletion occurs only on an explicit
   non-dry-run invocation. Without an explicit age override, a zero retention
   setting makes the service report that cleanup is disabled.
6. **Upload validation:** Report each validation outcome per file, including
   unsupported type, excessive size, corruption, unreadability, and empty
   files. Only case-level problems (missing case, no files, or too many files)
   fail the complete request. One bad file never blocks other files.
7. **OCR decision:** Use pretrained PaddleOCR (PP-OCR) as primary and Tesseract
   as fallback behind an `OCREngine` interface. Do not train models. OCR
   packages are in the optional `ocr` extra. The pytesseract wrapper does not
   install the Tesseract system binary or language data; PaddleOCR downloads
   pretrained models on first use.
8. **Python environment and dependencies:** Use Python 3.12 in the
   repository-root `.venv` (project range `>=3.11,<3.13`) for all Python
   commands and tests to preserve compatibility with future OCR dependencies.
   Document venv creation/activation for Windows and macOS/Linux. Ignore `.venv`
   and credential files. Base dependencies include FastAPI, Uvicorn,
   pydantic-settings, python-multipart, PyMuPDF, python-docx, firebase-admin,
   pytest, pytest-asyncio, httpx, Pillow, NumPy, Ruff, and Black. OCR-only
   dependencies (CPU PaddlePaddle, PaddleOCR, pytesseract, and OpenCV contrib)
   are in the `ocr` extra. PaddleX's OCR-core check rejected the headless-only
   build, so the working environment keeps only the required contrib OpenCV
   distribution. Bound FastAPI, Starlette, and httpx to compatible versions to
   avoid TestClient deprecation warnings.
9. **Local file access:** Store files using server-generated UUID names in
   per-case directories. Reject unsafe paths, never expose the upload directory
   as static content, and serve documents through a private-ready API streaming
   endpoint. Tests use only synthetic content.
10. **Review gates:** Complete Phase A foundation and stop for review; after
   approval, complete Phase B core code and stop; after approval, complete
   Phase C tests. Use synthetic/fake documents only. Never commit real PAN or
   Aadhaar data.