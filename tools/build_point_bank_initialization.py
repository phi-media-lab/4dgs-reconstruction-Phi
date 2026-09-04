#!/usr/bin/env python3
"""Build a hash-bound public Gaussian initialization from RoMa proposals."""

from __future__ import annotations

import argparse
import math
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


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _positive_finite(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = _finite_float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must lie in (0, 1]")
    return parsed


def _open_unit_interval(value: str) -> float:
    parsed = _finite_float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must lie in (0, 1)")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an append-only p2g.gaussian_initialization.v1 artifact from "
            "one complete, hash-bound RoMa proposal sequence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--proposal-sequence",
        type=_path,
        required=True,
        help="directory containing collection.json, frame receipts, and public PLY payloads",
    )
    parser.add_argument(
        "--tensor-cache",
        type=_path,
        required=True,
        help="matching p2g.tensor_cache.v1 root used to produce the proposals",
    )
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument("--num-gaussians", type=_positive_integer, default=500_000)
    parser.add_argument("--seed", type=_nonnegative_integer, default=0)
    parser.add_argument("--velocity-neighbors", type=_positive_integer, default=3)
    parser.add_argument("--scale-multiplier", type=_positive_finite, default=0.1)
    parser.add_argument(
        "--sampling-mode",
        choices=(
            "raw_candidate_uniform",
            "occupied_voxel_uniform",
            "triangulation_information_mixture",
            "paired_matcher_support_rank_mixture",
            "paired_multiview_consensus_rank_mixture",
        ),
        default="paired_multiview_consensus_rank_mixture",
    )
    parser.add_argument("--sampling-voxel-size", type=_positive_finite, default=0.02)
    parser.add_argument("--sampling-evidence-fraction", type=_unit_interval, default=0.5)
    parser.add_argument("--opacity", type=_open_unit_interval, default=0.5)
    parser.add_argument("--duration-seconds", type=_positive_finite, default=0.1)
    parser.add_argument(
        "--duration-min-seconds", type=_positive_finite, default=1.0 / 600.0
    )
    parser.add_argument("--duration-max-seconds", type=_positive_finite, default=1.0)
    parser.add_argument("--time-offset-seconds", type=_finite_float, default=0.0)
    return parser


def build_initialization(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Import Torch-backed assembly only after argument parsing."""

    from p2g.training.build_initialization import build_initialization as implementation

    return implementation(*args, **kwargs)


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not (
        arguments.duration_min_seconds
        < arguments.duration_seconds
        < arguments.duration_max_seconds
    ):
        parser.error(
            "duration bounds must satisfy --duration-min-seconds < "
            "--duration-seconds < --duration-max-seconds"
        )
    try:
        receipt = build_initialization(
            arguments.output,
            proposal_sequence=arguments.proposal_sequence,
            tensor_cache=arguments.tensor_cache,
            num_gaussians=arguments.num_gaussians,
            seed=arguments.seed,
            velocity_neighbors=arguments.velocity_neighbors,
            scale_multiplier=arguments.scale_multiplier,
            sampling_mode=arguments.sampling_mode,
            sampling_voxel_size=arguments.sampling_voxel_size,
            sampling_evidence_fraction=arguments.sampling_evidence_fraction,
            opacity=arguments.opacity,
            duration_seconds=arguments.duration_seconds,
            duration_min_seconds=arguments.duration_min_seconds,
            duration_max_seconds=arguments.duration_max_seconds,
            time_offset_seconds=arguments.time_offset_seconds,
            progress=_progress,
        )
    except (ContractError, OutputExistsError, OSError) as exc:
        print(f"Gaussian initialization build failed: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
