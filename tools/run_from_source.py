#!/usr/bin/env python3
"""Run `p2g` from one clean, exact Git checkout despite ambient editable installs."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _select_source(repository: Path, *, expected_revision: str) -> dict[str, Any]:
    revision = _git(repository, "rev-parse", "--verify", "HEAD")
    if revision != expected_revision:
        raise RuntimeError(
            f"selected source revision {revision} differs from expected {expected_revision}"
        )
    if _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        "--ignore-submodules=none",
    ):
        raise RuntimeError("selected source checkout is not clean")

    source_root = (repository / "src").resolve()
    removed = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__module__ == "_pixel4dgs_editable"
    ]
    sys.meta_path[:] = [finder for finder in sys.meta_path if finder not in removed]
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    p2g = importlib.import_module("p2g")
    module_file = p2g.__file__
    if module_file is None:
        raise RuntimeError("p2g resolved as a namespace package without a source file")
    imported = Path(module_file).resolve()
    expected_package = source_root / "p2g"
    if expected_package not in imported.parents:
        raise RuntimeError(
            f"p2g resolved outside selected source: {imported} (expected below {expected_package})"
        )
    return {
        "schema_version": "p2g.source_admission.v1",
        "status": "PASS",
        "git_revision": revision,
        "repository": str(repository),
        "imported_package": str(imported),
        "removed_shadow_finder_count": len(removed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    try:
        receipt = _select_source(
            repository,
            expected_revision=options.expected_revision,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), file=sys.stderr)
    if options.probe_only:
        if options.arguments:
            parser.error("--probe-only does not accept p2g arguments")
        return 0
    if not options.arguments:
        parser.error("p2g arguments are required unless --probe-only is used")
    from p2g.cli import main as p2g_main

    sys.argv = ["p2g", *options.arguments]
    p2g_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
