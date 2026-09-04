#!/usr/bin/env python3
"""Build or resume an explicit half-open sequence of RoMa proposal shards."""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from p2g.canonical import canonical_json_bytes
from p2g.errors import ContractError, OutputExistsError
from p2g.training.roma_point_sequence import build_roma_point_sequence


def _path(value: str) -> Path:
    if not value or "\x00" in value or value.startswith(("~", "file://")):
        raise argparse.ArgumentTypeError(
            "paths must be non-empty filesystem paths without '~' or file:// expansion"
        )
    return Path(value).resolve()


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = _nonnegative_integer(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_finite(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or resume append-only RoMa proposal shards from a public "
            "p2g.tensor_cache.v1 cache, then atomically publish the point inventory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tensor-cache",
        type=_path,
        required=True,
        help="directory containing tensor_cache.json and its bound NumPy arrays",
    )
    parser.add_argument(
        "--observation-manifest",
        type=_path,
        required=True,
        help="exact p2g.observation_manifest.v2 file bound by the tensor cache",
    )
    parser.add_argument(
        "--roma-indoor-weight",
        type=_path,
        required=True,
        help="local roma_indoor.pth whose size and SHA-256 match the public registry",
    )
    parser.add_argument(
        "--dinov2-weight",
        type=_path,
        required=True,
        help="local dinov2_vitl14_pretrain.pth matching the public registry",
    )
    parser.add_argument(
        "--environment-lock",
        type=_path,
        required=True,
        help="uv.lock binding the supported RoMa, Torch, and torchvision sources",
    )
    parser.add_argument(
        "--output",
        type=_path,
        required=True,
        help="sequence root; incomplete verified shards may be resumed in place",
    )
    parser.add_argument("--frame-start", type=_nonnegative_integer, default=0)
    parser.add_argument(
        "--frame-stop-exclusive",
        type=_positive_integer,
        default=60,
        help="exclusive upper bound of the requested frame interval",
    )
    parser.add_argument("--points-per-frame", type=_positive_integer, default=700_000)
    parser.add_argument("--nearest-cameras", type=_positive_integer, default=2)
    parser.add_argument("--seed", type=_nonnegative_integer, default=0)
    parser.add_argument("--world-bound", type=_positive_finite, default=1_000.0)
    return parser


def _emit_receipt(receipt: dict[str, Any]) -> None:
    print(canonical_json_bytes(receipt).decode("utf-8"), end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.frame_stop_exclusive <= arguments.frame_start:
        parser.error("--frame-stop-exclusive must be greater than --frame-start")
    frame_ids = tuple(range(arguments.frame_start, arguments.frame_stop_exclusive))
    try:
        # Per-frame progress belongs on stderr; stdout stays one JSON value.
        with contextlib.redirect_stdout(sys.stderr):
            receipt = build_roma_point_sequence(
                arguments.output,
                memmap_root=arguments.tensor_cache,
                observation_manifest=arguments.observation_manifest,
                roma_weight=arguments.roma_indoor_weight,
                dino_weight=arguments.dinov2_weight,
                environment_lock=arguments.environment_lock,
                frame_ids=frame_ids,
                num_points_per_frame=arguments.points_per_frame,
                nearest_cameras=arguments.nearest_cameras,
                seed=arguments.seed,
                world_bound=arguments.world_bound,
            )
    except (ContractError, OutputExistsError, OSError) as exc:
        print(f"RoMa sequence build failed: {exc}", file=sys.stderr)
        return 2
    _emit_receipt(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
