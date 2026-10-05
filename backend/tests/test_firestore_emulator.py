"""Firestore-only emulator integration coverage."""

import socket
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from app.core.config import Settings
from app.models.case import Case
from app.repositories.firestore import FirestoreRepository

pytestmark = pytest.mark.emulator


def _host_is_reachable(host: str | None) -> bool:
    if not host:
        return False
    parsed = urlsplit(host if "://" in host else f"//{host}")
    if parsed.hostname is None or parsed.port is None:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=0.25):
            return True
    except OSError:
        return False


def test_firestore_repository_round_trip() -> None:
    settings = Settings()
    if not settings.use_emulator:
        pytest.skip("Firestore emulator tests require USE_EMULATOR=true")
    if not _host_is_reachable(settings.firestore_emulator_host):
        pytest.skip("Firestore emulator is not reachable")

    repository = FirestoreRepository(settings)
    case = Case(case_id=f"test-{uuid4()}")
    try:
        repository.save_case(case)
        assert repository.get_case(case.case_id) == case
    finally:
        repository._client.collection("cases").document(case.case_id).delete()
