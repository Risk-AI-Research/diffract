"""Inspect or collect values in an existing Diffract store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from diffract.core.data.metadata.store_migration import MIGRATION_CLI_ERRORS
from diffract.core.storage.maintenance import check_store, collect_store


def main() -> int:
    """Run a maintenance command and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fsck = commands.add_parser("fsck", help="Check catalog and value integrity")
    gc = commands.add_parser("gc", help="Collect tombstoned and orphaned values")
    for command in (fsck, gc):
        command.add_argument(
            "store", type=Path, help="Metadata index or its containing directory"
        )
        command.add_argument(
            "--config", type=Path, help="Store configuration (INI or JSON)"
        )
    gc.add_argument(
        "--dry-run", action="store_true", help="List candidates without deletion"
    )
    gc.add_argument(
        "--grace-days",
        type=int,
        default=14,
        help="Minimum age in days (default: 14)",
    )
    args = parser.parse_args()
    try:
        if args.command == "fsck":
            report = check_store(args.store, config=args.config)
        else:
            report = collect_store(
                args.store,
                config=args.config,
                dry_run=args.dry_run,
                grace_days=args.grace_days,
            )
    except MIGRATION_CLI_ERRORS as exc:
        parser.exit(1, f"{exc}\n")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return int(args.command == "fsck" and any(report.values()))


if __name__ == "__main__":
    raise SystemExit(main())
