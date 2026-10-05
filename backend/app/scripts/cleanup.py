"""Manually preview or delete aged local upload files."""

import argparse

from app.repositories.storage import LocalStorageBackend
from app.services.cleanup_service import CleanupService


def main() -> None:
    """Run cleanup only when this module is explicitly invoked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--older-than-minutes",
        type=int,
        help="override DOCUMENT_RETENTION_MINUTES for this manual run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list eligible files without deleting them",
    )
    args = parser.parse_args()
    if args.older_than_minutes is not None and args.older_than_minutes <= 0:
        parser.error("--older-than-minutes must be a positive integer")

    report = CleanupService(LocalStorageBackend()).run(
        older_than_minutes=args.older_than_minutes,
        dry_run=args.dry_run,
    )
    if report.disabled:
        print("Cleanup disabled: DOCUMENT_RETENTION_MINUTES is 0.")
        return

    action = "Would delete" if report.dry_run else "Deleted"
    print(f"{action} {len(report.candidates)} file(s).")
    for storage_path in report.candidates:
        print(storage_path)


if __name__ == "__main__":
    main()
