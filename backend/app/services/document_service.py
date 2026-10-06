"""Document validation, local storage, and independent extraction processing."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import PurePath
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import pymupdf
from fastapi import HTTPException

from app.core.config import Settings, get_settings
from app.core.exceptions import ExtractionErrorCode
from app.core.logging import LogEvent, log_document_event
from app.extractors.base import DocumentExtractor, ExtractionResult
from app.extractors.pdf import PDFTextExtractor
from app.models.case import Case
from app.models.document import Document, DocumentStatus
from app.models.extraction import ExtractionError
from app.repositories.base import Repository, StorageBackend

logger = logging.getLogger(__name__)
_DOCX_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/zip",
}
_MIME_BY_EXTENSION = {
    ".pdf": {"application/pdf"},
    ".docx": _DOCX_MIMES,
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".png": {"image/png"},
}


class DocumentServiceError(HTTPException):
    """A case/document-level API failure with a structured response payload."""

    def __init__(self, status_code: int, error: ExtractionError) -> None:
        super().__init__(
            status_code=status_code,
            detail=error.model_dump(mode="json"),
        )
        self.status_code = status_code
        self.error = error


class DocumentService:
    """Own per-file validation and processing, isolating every document failure."""

    def __init__(
        self,
        repository: Repository,
        storage: StorageBackend,
        settings: Settings | None = None,
        pdf_extractor: DocumentExtractor | None = None,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._settings = settings or get_settings()
        self._pdf_extractor = pdf_extractor or PDFTextExtractor(self._settings)

    @property
    def max_upload_bytes(self) -> int:
        """Return the per-file upload limit in bytes."""
        return self._settings.max_upload_mb * 1024 * 1024

    def create_case(self) -> Case:
        """Create and persist an empty case."""
        case = Case()
        self._repository.save_case(case)
        return case

    def accept_upload(
        self,
        case_id: str,
        filename: str,
        mime_type: str | None,
        content: bytes,
    ) -> Document:
        """Validate and persist one file, returning its independent outcome."""
        case = self._repository.get_case(case_id)
        if case is None:
            raise DocumentServiceError(
                404,
                ExtractionError(
                    code=ExtractionErrorCode.CASE_NOT_FOUND,
                    message="Case not found.",
                ),
            )

        safe_mime = (mime_type or "").split(";")[0].strip().lower()
        extension = PurePath(filename or "").suffix.lower()
        document_id = str(uuid4())
        validation_error = self._validate_file(filename, extension, safe_mime, content)
        storage_path = ""
        status = DocumentStatus.PROCESSING
        if validation_error is not None:
            status = (
                DocumentStatus.UNSUPPORTED
                if validation_error.code is ExtractionErrorCode.UNSUPPORTED_FILE_TYPE
                else DocumentStatus.FAILED
            )
            validation_error.document_id = document_id
        else:
            try:
                storage_path = self._storage.upload(
                    case_id,
                    document_id,
                    filename,
                    content,
                    safe_mime,
                )
            except Exception:
                status = DocumentStatus.FAILED
                validation_error = ExtractionError(
                    code=ExtractionErrorCode.STORAGE_ERROR,
                    message="The uploaded file could not be stored.",
                    document_id=document_id,
                    retryable=True,
                )

        document = Document(
            document_id=document_id,
            case_id=case_id,
            filename=filename or "",
            mime_type=safe_mime or "application/octet-stream",
            size=len(content),
            storage_path=storage_path,
            status=status,
            created_at=datetime.now(UTC),
            expires_at=self._expires_at(),
            error=validation_error,
        )
        self._repository.save_document(document)
        case.document_count += 1
        self._repository.save_case(case)
        log_document_event(
            logger,
            LogEvent.UPLOAD_VALIDATED,
            case_id=case_id,
            document_id=document_id,
            filename=document.filename,
            status=document.status.value,
            error_code=document.error.code.value if document.error else None,
        )
        return document

    def record_unexpected_upload_failure(
        self, case_id: str, filename: str, mime_type: str | None
    ) -> Document:
        """Persist a per-file failure if reading or validation unexpectedly fails."""
        case = self._repository.get_case(case_id)
        if case is None:
            raise DocumentServiceError(
                404,
                ExtractionError(
                    code=ExtractionErrorCode.CASE_NOT_FOUND,
                    message="Case not found.",
                ),
            )
        document_id = str(uuid4())
        document = Document(
            document_id=document_id,
            case_id=case_id,
            filename=filename,
            mime_type=mime_type or "application/octet-stream",
            size=0,
            storage_path="",
            status=DocumentStatus.FAILED,
            error=ExtractionError(
                code=ExtractionErrorCode.UNKNOWN_EXTRACTION_ERROR,
                message="The uploaded file could not be read.",
                document_id=document_id,
                retryable=True,
            ),
            expires_at=self._expires_at(),
        )
        self._repository.save_document(document)
        case.document_count += 1
        self._repository.save_case(case)
        return document

    def process_document(self, case_id: str, document_id: str) -> Document | None:
        """Extract a single document, persisting a terminal status on all exits."""
        try:
            document = self._repository.get_document(case_id, document_id)
        except Exception:
            log_document_event(
                logger,
                LogEvent.DOCUMENT_PROCESSING_FAILED,
                case_id=case_id,
                document_id=document_id,
                status=DocumentStatus.FAILED.value,
                error_code=ExtractionErrorCode.STORAGE_ERROR.value,
            )
            return None
        if document is None or document.status is not DocumentStatus.PROCESSING:
            return document

        started_at = datetime.now(UTC)
        log_document_event(
            logger,
            LogEvent.DOCUMENT_PROCESSING_STARTED,
            case_id=case_id,
            document_id=document_id,
            filename=document.filename,
        )
        try:
            if not document.storage_path or not self._storage.exists(
                document.storage_path
            ):
                raise DocumentServiceError(
                    410,
                    self._file_unavailable_error(document_id),
                )
            content = self._storage.download(document.storage_path)
            extension = PurePath(document.filename).suffix.lower()
            if extension != ".pdf":
                result = self._non_pdf_result(document, extension)
            else:
                result = self._extract_with_timeout(document, content)
            for page in result.pages:
                self._repository.save_page(case_id, page)
            document.status = result.status
            document.page_count = len(result.pages)
            document.extraction_method = result.extraction_method
            document.error = result.error
        except DocumentServiceError as exc:
            document.status = DocumentStatus.FAILED
            document.error = exc.error
        except TimeoutError:
            document.status = DocumentStatus.FAILED
            document.error = ExtractionError(
                code=ExtractionErrorCode.PROCESSING_TIMEOUT,
                message="Document processing exceeded the configured time limit.",
                document_id=document_id,
                retryable=True,
            )
        except Exception:
            document.status = DocumentStatus.FAILED
            document.error = ExtractionError(
                code=ExtractionErrorCode.UNKNOWN_EXTRACTION_ERROR,
                message="An unexpected error occurred while processing this document.",
                document_id=document_id,
                retryable=True,
            )

        duration = (datetime.now(UTC) - started_at).total_seconds()
        try:
            self._repository.save_document(document)
        except Exception:
            log_document_event(
                logger,
                LogEvent.DOCUMENT_PROCESSING_FAILED,
                case_id=case_id,
                document_id=document_id,
                filename=document.filename,
                status=DocumentStatus.FAILED.value,
                error_code=ExtractionErrorCode.STORAGE_ERROR.value,
                duration=duration,
            )
            return None
        log_document_event(
            logger,
            LogEvent.DOCUMENT_PROCESSING_FINISHED,
            case_id=case_id,
            document_id=document_id,
            filename=document.filename,
            status=document.status.value,
            error_code=document.error.code.value if document.error else None,
            extractor=document.extraction_method,
            duration=duration,
        )
        return document

    def retry_document(self, case_id: str, document_id: str) -> Document:
        """Clear old results and return a failed document to PROCESSING."""
        document = self._repository.get_document(case_id, document_id)
        if document is None:
            raise DocumentServiceError(
                404,
                ExtractionError(
                    code=ExtractionErrorCode.DOCUMENT_NOT_FOUND,
                    message="Document not found.",
                    document_id=document_id,
                ),
            )
        if document.status is not DocumentStatus.FAILED:
            raise DocumentServiceError(
                409,
                ExtractionError(
                    code=ExtractionErrorCode.DOCUMENT_NOT_RETRYABLE,
                    message="Only failed documents can be retried.",
                    document_id=document_id,
                    retryable=False,
                ),
            )
        if not document.storage_path or not self._storage.exists(document.storage_path):
            raise DocumentServiceError(410, self._file_unavailable_error(document_id))

        self._repository.delete_pages(case_id, document_id)
        self._repository.delete_fields(case_id, document_id)
        document.status = DocumentStatus.PROCESSING
        document.page_count = 0
        document.error = None
        self._repository.save_document(document)
        return document

    def validate_upload_request(self, case_id: str, file_count: int) -> None:
        """Reject whole-request conditions before processing any individual file."""
        if self._repository.get_case(case_id) is None:
            raise DocumentServiceError(
                404,
                ExtractionError(
                    code=ExtractionErrorCode.CASE_NOT_FOUND,
                    message="Case not found.",
                ),
            )
        if file_count == 0:
            raise DocumentServiceError(
                400,
                ExtractionError(
                    code=ExtractionErrorCode.NO_FILES,
                    message="At least one file is required.",
                ),
            )
        if file_count > self._settings.max_files_per_case:
            raise DocumentServiceError(
                413,
                ExtractionError(
                    code=ExtractionErrorCode.TOO_MANY_FILES,
                    message=(
                        "The request exceeds the maximum of "
                        f"{self._settings.max_files_per_case} files."
                    ),
                ),
            )

    def _validate_file(
        self, filename: str, extension: str, mime_type: str, content: bytes
    ) -> ExtractionError | None:
        unsafe_filename = (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
        )
        if unsafe_filename:
            return self._error(
                ExtractionErrorCode.UNSUPPORTED_FILE_TYPE,
                "The filename must be a plain basename.",
            )
        if extension not in self._settings.allowed_extensions:
            return self._error(
                ExtractionErrorCode.UNSUPPORTED_FILE_TYPE,
                "The file extension is not supported.",
            )
        allowed_mimes = _MIME_BY_EXTENSION.get(extension, set())
        if mime_type not in allowed_mimes:
            return self._error(
                ExtractionErrorCode.UNSUPPORTED_FILE_TYPE,
                "The declared MIME type does not match the file extension.",
            )
        if not content:
            return self._error(
                ExtractionErrorCode.NO_TEXT_DETECTED,
                "The uploaded file is empty.",
            )
        if len(content) > self._settings.max_upload_mb * 1024 * 1024:
            return self._error(
                ExtractionErrorCode.FILE_TOO_LARGE,
                f"The file exceeds the {self._settings.max_upload_mb} MB limit.",
            )
        if not self._has_matching_signature(extension, content):
            return self._error(
                ExtractionErrorCode.FILE_CORRUPTED,
                "The file content does not match its extension or is corrupted.",
            )
        if extension == ".pdf":
            return self._validate_pdf(content)
        return None

    def _validate_pdf(self, content: bytes) -> ExtractionError | None:
        try:
            with pymupdf.open(stream=content, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    return self._error(
                        ExtractionErrorCode.PDF_PARSE_FAILED,
                        "The PDF is password-protected and cannot be read.",
                    )
                for page_number in range(pdf.page_count):
                    pdf.load_page(page_number)
        except Exception:
            return self._error(
                ExtractionErrorCode.FILE_CORRUPTED,
                "The PDF is unreadable or corrupted.",
            )
        return None

    @staticmethod
    def _has_matching_signature(extension: str, content: bytes) -> bool:
        if extension == ".pdf":
            return b"%PDF-" in content[:1024]
        if extension == ".png":
            return content.startswith(b"\x89PNG\r\n\x1a\n")
        if extension in {".jpg", ".jpeg"}:
            return content.startswith(b"\xff\xd8\xff")
        if extension == ".docx":
            try:
                with ZipFile(BytesIO(content)) as archive:
                    return "word/document.xml" in archive.namelist()
            except (BadZipFile, OSError):
                return False
        return False

    def _extract_with_timeout(
        self, document: Document, content: bytes
    ) -> ExtractionResult:
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="document-extract"
        )
        future: Future[ExtractionResult] = executor.submit(
            self._pdf_extractor.extract, document, content
        )
        try:
            return future.result(timeout=self._settings.processing_timeout_seconds)
        except TimeoutError:
            future.cancel()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _non_pdf_result(document: Document, extension: str) -> ExtractionResult:
        if extension == ".docx":
            code = ExtractionErrorCode.DOCX_PARSE_FAILED
            message = "DOCX extraction is not available in this processing phase."
        else:
            code = ExtractionErrorCode.OCR_NOT_AVAILABLE
            message = "OCR is not available for image documents in this phase."
        return ExtractionResult(
            document_id=document.document_id,
            status=DocumentStatus.FAILED,
            error=ExtractionError(
                code=code,
                message=message,
                document_id=document.document_id,
                retryable=False,
            ),
            extraction_method="none",
        )

    def _expires_at(self) -> datetime | None:
        if not self._settings.document_retention_minutes:
            return None
        return datetime.now(UTC) + timedelta(
            minutes=self._settings.document_retention_minutes
        )

    @staticmethod
    def _error(code: ExtractionErrorCode, message: str) -> ExtractionError:
        return ExtractionError(code=code, message=message, retryable=False)

    @staticmethod
    def _file_unavailable_error(document_id: str) -> ExtractionError:
        return ExtractionError(
            code=ExtractionErrorCode.FILE_UNAVAILABLE,
            message=(
                "Original document is no longer available. The temporary retention "
                "period has expired. Please upload the document again if "
                "source-level review is required."
            ),
            document_id=document_id,
            retryable=False,
        )
