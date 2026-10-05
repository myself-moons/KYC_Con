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

## Run tests

```sh
pytest
```

Run local/unit tests without Firebase with `pytest -m 'not emulator'`. Firestore
emulator tests are marked `emulator` and can be run separately with
`pytest -m emulator`.

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
