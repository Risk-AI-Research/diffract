"""Offline upgrade for persistent Diffract stores with name-addressed values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from diffract.core.data.metadata.store_migration import (
    MIGRATION_CLI_ERRORS,
    migrate_store,
)


def main() -> int:
    """Run the store migration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "store", type=Path, help="Metadata index or its containing directory"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="INI or JSON with the physical storage configuration",
    )
    parser.add_argument("--dry-run", action="store_true")
    backups = parser.add_mutually_exclusive_group()
    backups.add_argument("--backup-dir", type=Path)
    backups.add_argument("--i-have-a-backup", action="store_true")
    parser.add_argument(
        "--keep", choices=("newest", "oldest", "abort"), default="abort"
    )
    parser.add_argument(
        "--rename-map", type=Path, help="JSON with models and parameters name maps"
    )
    parser.add_argument("--parse-step-suffix", action="store_true")
    args = parser.parse_args()
    try:
        renames = json.loads(args.rename_map.read_text()) if args.rename_map else None
        report = migrate_store(
            args.store,
            config=args.config,
            dry_run=args.dry_run,
            backup_dir=args.backup_dir,
            have_backup=args.i_have_a_backup,
            keep=args.keep,
            rename=renames,
            parse_step_suffix=args.parse_step_suffix,
        )
    except MIGRATION_CLI_ERRORS as exc:
        parser.exit(1, f"{exc}\n")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return int(report["status"] == "blocked")


if __name__ == "__main__":
    raise SystemExit(main())
