"""Firebase Admin initialization shared by Firebase-backed adapters."""

import os
from threading import Lock

import firebase_admin
import google.auth.credentials
from firebase_admin import credentials

from app.core.config import Settings, get_settings

_APP_LOCK = Lock()


class _EmulatorCredential(credentials.Base):
    """Anonymous credential wrapper for local Firebase emulators."""

    def __init__(self, project_id: str) -> None:
        self._project_id = project_id

    def get_credential(self) -> google.auth.credentials.AnonymousCredentials:
        """Return credentials accepted by local emulators without secrets."""
        return google.auth.credentials.AnonymousCredentials()

    def get_project_id(self) -> str:
        """Return the configured emulator project identifier."""
        return self._project_id


def configure_emulator_environment(settings: Settings) -> None:
    """Translate application emulator settings to Google client environment."""
    if not settings.use_emulator:
        return
    if settings.firestore_emulator_host:
        os.environ["FIRESTORE_EMULATOR_HOST"] = settings.firestore_emulator_host


def get_firebase_app(settings: Settings | None = None) -> firebase_admin.App:
    """Return the default Firebase app configured for production or emulators."""
    selected = settings or get_settings()
    configure_emulator_environment(selected)
    with _APP_LOCK:
        try:
            return firebase_admin.get_app()
        except ValueError:
            options = {"projectId": selected.firebase_project_id}
            if selected.use_emulator:
                credential = _EmulatorCredential(selected.firebase_project_id)
            elif selected.google_application_credentials:
                credential = credentials.Certificate(
                    selected.google_application_credentials
                )
            else:
                credential = None
            return firebase_admin.initialize_app(credential, options=options)
