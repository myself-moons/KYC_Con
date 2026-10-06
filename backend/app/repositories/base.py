"""Abstract persistence interfaces for metadata and private file storage."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.models.case import Case
from app.models.document import Document
from app.models.extraction import ExtractedField, PageResult


class Repository(ABC):
    """Persistence contract for Firestore-compatible KYC metadata."""

    @abstractmethod
    def save_case(self, case: Case) -> None:
        """Create or replace case metadata."""

    @abstractmethod
    def get_case(self, case_id: str) -> Case | None:
        """Fetch case metadata by identifier."""

    @abstractmethod
    def save_document(self, document: Document) -> None:
        """Create or replace document metadata."""

    @abstractmethod
    def get_document(self, case_id: str, document_id: str) -> Document | None:
        """Fetch document metadata from its parent case."""

    @abstractmethod
    def list_documents(self, case_id: str) -> list[Document]:
        """Return the documents belonging to a case."""

    @abstractmethod
    def save_page(self, case_id: str, page: PageResult) -> None:
        """Create or replace a page-level extraction result."""

    @abstractmethod
    def get_page(
        self, case_id: str, document_id: str, page_number: int
    ) -> PageResult | None:
        """Fetch a page-level extraction result."""

    @abstractmethod
    def list_pages(self, case_id: str, document_id: str) -> list[PageResult]:
        """Return all pages for a document in page order."""

    @abstractmethod
    def delete_pages(self, case_id: str, document_id: str) -> None:
        """Delete all page results for a document."""

    @abstractmethod
    def save_field(self, field: ExtractedField) -> None:
        """Create or replace an extracted-field record."""

    @abstractmethod
    def list_fields(self, case_id: str) -> list[ExtractedField]:
        """Return all extracted fields in a case."""

    @abstractmethod
    def delete_fields(self, case_id: str, document_id: str) -> None:
        """Delete all extracted fields for one document."""


class StorageBackend(ABC):
    """Private uploaded-file storage contract."""

    @abstractmethod
    def upload(
        self,
        case_id: str,
        document_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """Store bytes privately and return their storage path."""

    @abstractmethod
    def download(self, storage_path: str) -> bytes:
        """Return the stored file bytes."""

    @abstractmethod
    def stream(self, storage_path: str) -> Iterator[bytes]:
        """Yield stored file content in chunks for API streaming responses."""

    @abstractmethod
    def delete(self, storage_path: str) -> None:
        """Delete a stored file."""

    @abstractmethod
    def exists(self, storage_path: str) -> bool:
        """Return whether a stored file exists."""
