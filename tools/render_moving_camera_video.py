#!/usr/bin/env python3
"""Render a portable AssetBundle along an explicit, hash-bound camera path."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from p2g.canonical import canonical_json_bytes
from p2g.errors import ContractError, OutputExistsError


def render_asset_video(
    asset: Path,
    *,
    camera_path_file: Path,
    output: Path,
    receipt: Path | None,
    device: str,
    crf: int,
) -> dict[str, Any]:
    """Load the ROCm/Torch renderer only after command-line validation."""

    from p2g.training.asset_render import render_asset_video as render_implementation

    return render_implementation(
        asset,
        camera_path_file=camera_path_file,
        output=output,
        receipt=receipt,
        device=device,
        crf=crf,
    )


def _path(value: str) -> Path:
    if not value or "\x00" in value or value.startswith(("~", "file://")):
        raise argparse.ArgumentTypeError(
            "paths must be non-empty filesystem paths without '~' or file:// expansion"
        )
    return Path(value).resolve()


def _crf(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("CRF must be an integer") from exc
    if not 0 <= parsed <= 51:
        raise argparse.ArgumentTypeError("CRF must be inside [0, 51]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render an AssetBundle without consulting a training run or source dataset. "
            "The camera-path JSON must name the exact bundle ID."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--asset", type=_path, required=True, help="verified AssetBundle root")
    parser.add_argument(
        "--camera-path",
        type=_path,
        required=True,
        help="p2g.camera_path.v1 JSON bound to the AssetBundle ID",
    )
    parser.add_argument("--output", type=_path, required=True, help="new .mp4 output path")
    parser.add_argument(
        "--receipt",
        type=_path,
        help="new render-receipt JSON path; defaults beside the video",
    )
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
        help="the v0 renderer admits only the ROCm Torch CUDA-compatibility device",
    )
    parser.add_argument("--crf", type=_crf, default=18, help="H.264 constant-rate factor")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            receipt = render_asset_video(
                arguments.asset,
                camera_path_file=arguments.camera_path,
                output=arguments.output,
                receipt=arguments.receipt,
                device=arguments.device,
                crf=arguments.crf,
            )
    except (ContractError, OutputExistsError, ImportError, OSError) as exc:
        print(f"asset video render failed: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(cast_receipt(receipt)).decode("utf-8"), end="")
    return 0


def cast_receipt(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("render_asset_video returned a non-object receipt")
    return cast(dict[str, Any], value)


if __name__ == "__main__":
    raise SystemExit(main())
