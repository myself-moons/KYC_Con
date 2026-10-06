# KYC Document Intelligence

Phase 1 backend foundation for KYC document text extraction and source-linked
review. User decisions that override the SRS are recorded in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Requirements

- Python 3.11 or 3.12 (Python 3.12 is recommended for future OCR dependency compatibility)
- Java, Node.js, and Firebase CLI only when running the Firestore emulator

## Create the Python environment

From the repository root:

```sh
python3.12 -m venv .venv
```

On Windows, create the environment with `py -3.12 -m venv .venv`.

Activate it on macOS/Linux:

```sh
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt, use `\.venv\Scripts\activate.bat`.

Install the project and its dependencies:

```sh
python -m pip install --upgrade pip
python -m pip install -e .
```

Do not commit `.venv`, `.env`, or Firebase service-account credentials. Copy
`.env.example` to `.env` and set values for your environment. Uploaded files are
stored under `UPLOAD_DIR` (default `./data/uploads`), which is gitignored and is
not mounted as a static directory.

Install the OCR engines in the Python 3.12 environment with:

```sh
python -m pip install -e '.[ocr]'
```

This extra installs CPU PaddlePaddle, PaddleOCR, pytesseract, OpenCV contrib,
and NumPy. PaddleOCR's installed PaddleX OCR-core dependency requires the
contrib distribution; a headless-only OpenCV trial failed PaddleX's runtime
dependency check. This setup keeps exactly one OpenCV distribution installed.
PaddleOCR downloads its pretrained models on first use; no model is
trained by this project. `pytesseract` is only a Python wrapper: the Tesseract
system binary must be installed separately, along with language data for
`eng` and `hin` (optionally `mar`). For example, Debian/Ubuntu packages are
`tesseract-ocr`, `tesseract-ocr-eng`, `tesseract-ocr-hin`, and
`tesseract-ocr-mar`; macOS users can install `tesseract` and the desired
language data through Homebrew. Set `OCR_LANGUAGES=eng+hin+mar` when Marathi
data is installed.

## Run the backend

With the virtual environment active:

```sh
uvicorn app.main:app --app-dir backend --reload
```

The health endpoint is `http://127.0.0.1:8000/health`.
Documents are served through the API file endpoint, never directly from the
upload directory: `GET /api/v1/cases/{case_id}/documents/{document_id}/file`.
The endpoint requires authentication middleware to populate
`request.state.principal`; it returns `401` until that integration is configured.

## API Endpoints

- `POST /api/v1/cases` creates a case.
- `POST /api/v1/cases/{case_id}/documents` accepts multiple files and returns
	one validation result per file.
- `GET /api/v1/cases/{case_id}/status` returns counts by document status.
- `GET /api/v1/cases/{case_id}/extractions` returns stored provenance fields.
- `GET /api/v1/cases/{case_id}/documents/{document_id}` returns document
	metadata without the private storage path.
- `GET /api/v1/cases/{case_id}/documents/{document_id}/file` streams a private
	file after authentication middleware supplies a principal.
- `GET /api/v1/cases/{case_id}/documents/{document_id}/pages/{page_number}`
	returns page text and normalized word boxes.
- `POST /api/v1/cases/{case_id}/documents/{document_id}/retry` retries a failed
	document while its source file is available.

PDF embedded-text extraction, OCR fallback for scans/images, and page-1 DOCX
text extraction are supported. OCR preprocessing preserves the original upload
and uses a separate processed image at configured `OCR_DPI`.

## Run tests

```sh
pytest
```

Run the fast local/unit tests with `pytest -m 'not emulator and not slow'`.
Firestore emulator tests are marked `emulator` and can be run separately with
`pytest -m emulator`. Real OCR recovery tests are marked `slow` and run with
`pytest -m slow`; they require the OCR extra, Tesseract binary/language data,
OpenCV native libraries, and PaddleOCR model downloads.

## Firebase Emulator Suite

Firestore is the only Firebase product in use and is configured for the free
Spark plan. Install the Firebase CLI separately
(`npm install -g firebase-tools`), then from the repository root run:

```sh
firebase emulators:start --only firestore --project demo-kyc
```

The Firestore emulator listens on `127.0.0.1:8080`. Set
`USE_EMULATOR=true` and `FIRESTORE_EMULATOR_HOST=127.0.0.1:8080` for local
development. Firestore rules deny client access; the backend uses the Firebase
Admin SDK. For a real Firestore project on Spark, set the project and credential
environment variables instead. Keep service-account files outside version
control. Source files are stored locally under the private `UPLOAD_DIR`.

## Manual cleanup

No cleanup runs automatically. With `DOCUMENT_RETENTION_MINUTES=0`, cleanup is
disabled unless an age is explicitly supplied. Preview eligible local files
with:

```sh
python -m app.scripts.cleanup --older-than-minutes 10 --dry-run
```

Run the same command without `--dry-run` only when you intend to delete the
listed files.

## Synthetic OCR Samples

Generate fake document scans with sidecar ground truth using:

```sh
python -m tests.fixtures.make_samples --out data/samples --preset xerox_medium
```

The generated `data/samples/` output is gitignored. The tool also supports
`clean`, `light_scan`, and `xerox_heavy` presets, a fixed random seed, and
per-effect severity overrides such as `--severity blur=0.4`.

Generate the heavy preset separately and benchmark all presets found below
`data/samples/` with:

```sh
python -m tests.fixtures.make_samples --out data/samples/xerox_heavy --preset xerox_heavy
python -m app.scripts.benchmark_ocr --samples data/samples --csv data/samples/ocr_benchmark.csv
```

The benchmark prints a table and writes CSV metrics for character accuracy,
mean confidence, field recovery rate, and seconds per page for PaddleOCR and
Tesseract.
