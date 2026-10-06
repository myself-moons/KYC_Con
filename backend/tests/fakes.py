"""In-memory persistence fakes for isolated unit tests only."""

from collections.abc import Iterator

from app.models.case import Case
from app.models.document import Document
from app.models.extraction import ExtractedField, PageResult
from app.repositories.base import Repository, StorageBackend


class InMemoryRepository(Repository):
    """A test-only repository fake with the same contract as Firestore."""

    def __init__(self) -> None:
        self.cases: dict[str, Case] = {}
        self.documents: dict[tuple[str, str], Document] = {}
        self.pages: dict[tuple[str, str, int], PageResult] = {}
        self.fields: dict[tuple[str, str], ExtractedField] = {}

    def save_case(self, case: Case) -> None:
        self.cases[case.case_id] = case.model_copy(deep=True)

    def get_case(self, case_id: str) -> Case | None:
        case = self.cases.get(case_id)
        return case.model_copy(deep=True) if case else None

    def save_document(self, document: Document) -> None:
        key = (document.case_id, document.document_id)
        self.documents[key] = document.model_copy(deep=True)

    def get_document(self, case_id: str, document_id: str) -> Document | None:
        document = self.documents.get((case_id, document_id))
        return document.model_copy(deep=True) if document else None

    def list_documents(self, case_id: str) -> list[Document]:
        return [
            document.model_copy(deep=True)
            for (stored_case_id, _), document in self.documents.items()
            if stored_case_id == case_id
        ]

    def save_page(self, case_id: str, page: PageResult) -> None:
        key = (case_id, page.document_id, page.page_number)
        self.pages[key] = page.model_copy(deep=True)

    def get_page(
        self, case_id: str, document_id: str, page_number: int
    ) -> PageResult | None:
        page = self.pages.get((case_id, document_id, page_number))
        return page.model_copy(deep=True) if page else None

    def list_pages(self, case_id: str, document_id: str) -> list[PageResult]:
        pages = [
            page.model_copy(deep=True)
            for (stored_case_id, stored_document_id, _), page in self.pages.items()
            if stored_case_id == case_id and stored_document_id == document_id
        ]
        return sorted(pages, key=lambda page: page.page_number)

    def delete_pages(self, case_id: str, document_id: str) -> None:
        self.pages = {
            key: page
            for key, page in self.pages.items()
            if key[:2] != (case_id, document_id)
        }

    def save_field(self, field: ExtractedField) -> None:
        self.fields[(field.case_id, field.field_id)] = field.model_copy(deep=True)

    def list_fields(self, case_id: str) -> list[ExtractedField]:
        return [
            field.model_copy(deep=True)
            for (stored_case_id, _), field in self.fields.items()
            if stored_case_id == case_id
        ]

    def delete_fields(self, case_id: str, document_id: str) -> None:
        self.fields = {
            key: field
            for key, field in self.fields.items()
            if (field.case_id, field.document_id) != (case_id, document_id)
        }


class InMemoryStorage(StorageBackend):
    """A test-only byte store; never used as a production backend."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    def upload(
        self,
        case_id: str,
        document_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        storage_path = f"cases/{case_id}/documents/{document_id}/{filename}"
        self.objects[storage_path] = (bytes(content), content_type)
        return storage_path

    def download(self, storage_path: str) -> bytes:
        return self.objects[storage_path][0]

    def stream(self, storage_path: str) -> Iterator[bytes]:
        yield self.download(storage_path)

    def delete(self, storage_path: str) -> None:
        self.objects.pop(storage_path, None)

    def exists(self, storage_path: str) -> bool:
        return storage_path in self.objects
