#!/usr/bin/env python3
"""Inspect or verify a portable AssetBundle without training-run inputs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json, write_new_json
from p2g.errors import ContractError, OutputExistsError
from p2g.schema import validate_payload


def load_asset_bundle(root: Path) -> Any:
    """Load Torch-backed asset code only when inspection actually begins."""

    from p2g.training.asset import load_asset_bundle as load_implementation

    return load_implementation(root)


def asset_summary(bundle: Any) -> dict[str, Any]:
    """Keep the standalone tool importable without importing Torch."""

    from p2g.training.asset import asset_summary as summarize_implementation

    return summarize_implementation(bundle)


def _path(value: str) -> Path:
    if not value or "\x00" in value or value.startswith(("~", "file://")):
        raise argparse.ArgumentTypeError(
            "paths must be non-empty filesystem paths without '~' or file:// expansion"
        )
    return Path(value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or hash-verify a portable AssetBundle. This tool never opens a "
            "training config, checkpoint, observation manifest, or GPU renderer."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="verify all bytes and print the asset summary")
    inspect.add_argument("asset", type=_path)
    verify = commands.add_parser(
        "verify", help="verify all bytes and publish an append-only verification receipt"
    )
    verify.add_argument("asset", type=_path)
    verify.add_argument("--output", type=_path, required=True)
    return parser


def _verification_receipt(bundle: Any, summary: dict[str, Any]) -> dict[str, Any]:
    root = Path(bundle.root).resolve()
    receipt: dict[str, Any] = {
        "schema_version": "p2g.asset_verification.v1",
        "status": "PASS",
        "asset": {
            "schema_version": summary["schema_version"],
            "bundle_id": summary["bundle_id"],
            "gaussian_count": summary["gaussian_count"],
            "tensor_count": summary["tensor_count"],
            "model_sha256": summary["model_sha256"],
            "equation_version": summary["equation_version"],
            "redistribution": summary["rights"]["redistribution"],
        },
        "files": {
            "manifest.json": sha256_file(root / "manifest.json"),
            "asset.json": sha256_file(root / "asset.json"),
            "model.safetensors": sha256_file(root / "model.safetensors"),
        },
        "claim_boundary": (
            "Every declared AssetBundle byte and semantic field was accepted; no render, "
            "source-data entitlement, visual-quality, or performance claim was made."
        ),
    }
    if receipt["asset"]["model_sha256"] != receipt["files"]["model.safetensors"]:
        raise ContractError("asset summary and model file digest disagree")
    receipt["logical_sha256"] = sha256_json(receipt)
    validate_payload("asset_verification", receipt)
    return receipt


def _inside(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        bundle = load_asset_bundle(arguments.asset)
        summary = asset_summary(bundle)
        payload = summary
        if arguments.command == "verify":
            output = arguments.output
            if output.suffix.casefold() != ".json":
                raise ContractError("asset verification output must use a .json filename")
            if _inside(output, Path(bundle.root).resolve()):
                raise ContractError("asset verification receipt must be outside the AssetBundle")
            payload = _verification_receipt(bundle, summary)
            write_new_json(output, payload)
    except (ContractError, OutputExistsError, ImportError, OSError) as exc:
        print(f"asset {arguments.command} failed: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
