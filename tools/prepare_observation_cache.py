#!/usr/bin/env python3
"""Build a tensor cache from an admitted observation manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from p2g.canonical import canonical_json_bytes
from p2g.errors import ContractError, OutputExistsError


def _path(value: str) -> Path:
    if not value or "\x00" in value or value.startswith(("~", "file://")):
        raise argparse.ArgumentTypeError(
            "paths must be non-empty filesystem paths without '~' or file:// expansion"
        )
    return Path(value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an append-only p2g.tensor_cache.v1 directory from a complete, "
            "audited p2g.observation_manifest.v2 and its RGB8 image files."
        )
    )
    parser.add_argument(
        "--observation-manifest",
        required=True,
        type=_path,
        help="public observation manifest JSON",
    )
    parser.add_argument(
        "--image-root",
        type=_path,
        help="image root; defaults to the observation manifest directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=_path,
        help="new append-only tensor-cache directory",
    )
    return parser


def build_tensor_cache(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load NumPy/Pillow-backed preparation only after argument parsing."""

    from p2g.training.prepare import build_tensor_cache as implementation

    return implementation(*args, **kwargs)


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        receipt = build_tensor_cache(
            arguments.output,
            observation_manifest=arguments.observation_manifest,
            image_root=arguments.image_root,
            progress=_progress,
        )
    except (ContractError, OutputExistsError, OSError) as exc:
        print(f"Tensor-cache preparation failed: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
