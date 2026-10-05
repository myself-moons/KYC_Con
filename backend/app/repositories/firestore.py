"""Firestore implementation of the metadata repository interface."""

from firebase_admin import firestore
from google.cloud.firestore_v1 import Client as FirestoreClient

from app.core.config import Settings, get_settings
from app.core.firebase import get_firebase_app
from app.models.case import Case
from app.models.document import Document
from app.models.extraction import ExtractedField, PageResult
from app.repositories.base import Repository


class FirestoreRepository(Repository):
    """Persist cases, documents, pages, and fields in Firestore."""

    def __init__(
        self, settings: Settings | None = None, client: FirestoreClient | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or firestore.client(get_firebase_app(self._settings))

    def save_case(self, case: Case) -> None:
        self._client.collection("cases").document(case.case_id).set(
            case.model_dump(mode="json")
        )

    def get_case(self, case_id: str) -> Case | None:
        snapshot = self._client.collection("cases").document(case_id).get()
        data = snapshot.to_dict()
        return Case.model_validate(data) if data is not None else None

    def save_document(self, document: Document) -> None:
        self._case_ref(document.case_id).collection("documents").document(
            document.document_id
        ).set(document.model_dump(mode="json", exclude_none=True))

    def get_document(self, case_id: str, document_id: str) -> Document | None:
        snapshot = self._document_ref(case_id, document_id).get()
        data = snapshot.to_dict()
        return Document.model_validate(data) if data is not None else None

    def list_documents(self, case_id: str) -> list[Document]:
        snapshots = self._case_ref(case_id).collection("documents").stream()
        return [Document.model_validate(snapshot.to_dict()) for snapshot in snapshots]

    def save_page(self, case_id: str, page: PageResult) -> None:
        self._document_ref(case_id, page.document_id).collection("pages").document(
            str(page.page_number)
        ).set(page.model_dump(mode="json"))

    def get_page(
        self, case_id: str, document_id: str, page_number: int
    ) -> PageResult | None:
        snapshot = (
            self._document_ref(case_id, document_id)
            .collection("pages")
            .document(str(page_number))
            .get()
        )
        data = snapshot.to_dict()
        return PageResult.model_validate(data) if data is not None else None

    def list_pages(self, case_id: str, document_id: str) -> list[PageResult]:
        snapshots = (
            self._document_ref(case_id, document_id).collection("pages").stream()
        )
        pages = [
            PageResult.model_validate(snapshot.to_dict()) for snapshot in snapshots
        ]
        return sorted(pages, key=lambda page: page.page_number)

    def save_field(self, field: ExtractedField) -> None:
        self._case_ref(field.case_id).collection("extracted_fields").document(
            field.field_id
        ).set(field.model_dump(mode="json", exclude_none=True))

    def list_fields(self, case_id: str) -> list[ExtractedField]:
        snapshots = self._case_ref(case_id).collection("extracted_fields").stream()
        return [
            ExtractedField.model_validate(snapshot.to_dict()) for snapshot in snapshots
        ]

    def _case_ref(self, case_id: str):
        return self._client.collection("cases").document(case_id)

    def _document_ref(self, case_id: str, document_id: str):
        return self._case_ref(case_id).collection("documents").document(document_id)
