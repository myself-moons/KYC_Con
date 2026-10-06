"""Dependency providers for repositories, storage, and document workflows."""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.repositories.base import Repository, StorageBackend
from app.repositories.firestore import FirestoreRepository
from app.repositories.storage import LocalStorageBackend
from app.services.document_service import DocumentService


def get_repository(request: Request) -> Repository:
    """Resolve the app-scoped repository or lazily create Firestore access."""
    repository = getattr(request.app.state, "repository", None)
    if repository is None:
        repository = FirestoreRepository()
        request.app.state.repository = repository
    return repository


def get_storage_backend(request: Request) -> StorageBackend:
    """Resolve the app-scoped local storage backend."""
    backend = getattr(request.app.state, "storage_backend", None)
    if backend is None:
        backend = LocalStorageBackend()
        request.app.state.storage_backend = backend
    return backend


def get_app_settings(request: Request) -> Settings:
    """Use app-overridden settings in tests, otherwise environment settings."""
    return getattr(request.app.state, "settings", None) or get_settings()


def get_document_service(
    request: Request,
    repository: Annotated[Repository, Depends(get_repository)],
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
) -> DocumentService:
    """Build the document workflow around the resolved persistence adapters."""
    return DocumentService(
        repository=repository,
        storage=storage,
        settings=get_app_settings(request),
    )
