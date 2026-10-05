"""Shared test fixtures for API and service tests."""

import pytest
from app.main import app
from fastapi.testclient import TestClient
from tests.fakes import InMemoryRepository, InMemoryStorage


@pytest.fixture
def client() -> TestClient:
    """Provide a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def memory_repository() -> InMemoryRepository:
    """Provide an in-memory repository fake for unit tests."""
    return InMemoryRepository()


@pytest.fixture
def memory_storage() -> InMemoryStorage:
    """Provide an in-memory storage fake for unit tests."""
    return InMemoryStorage()
