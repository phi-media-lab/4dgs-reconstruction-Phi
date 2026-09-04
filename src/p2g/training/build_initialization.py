"""Stable public entry point for first-party Gaussian initialization assembly."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from p2g.training.point_bank_initialization import (
    SamplingMode,
    build_point_bank_initialization,
)

__all__ = ["build_initialization"]


def build_initialization(
    output: Path,
    *,
    proposal_sequence: Path,
    tensor_cache: Path,
    num_gaussians: int = 500_000,
    seed: int = 0,
    velocity_neighbors: int = 3,
    scale_multiplier: float = 0.1,
    sampling_mode: SamplingMode = "paired_multiview_consensus_rank_mixture",
    sampling_voxel_size: float = 0.02,
    sampling_evidence_fraction: float = 0.5,
    opacity: float = 0.5,
    duration_seconds: float = 0.1,
    duration_min_seconds: float = 1.0 / 600.0,
    duration_max_seconds: float = 1.0,
    time_offset_seconds: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build the strict ``p2g.gaussian_initialization.v1`` artifact."""

    return build_point_bank_initialization(
        output,
        proposal_sequence=proposal_sequence,
        tensor_cache=tensor_cache,
        num_gaussians=num_gaussians,
        seed=seed,
        velocity_neighbors=velocity_neighbors,
        scale_multiplier=scale_multiplier,
        sampling_mode=sampling_mode,
        sampling_voxel_size=sampling_voxel_size,
        sampling_evidence_fraction=sampling_evidence_fraction,
        opacity=opacity,
        duration_seconds=duration_seconds,
        duration_min_seconds=duration_min_seconds,
        duration_max_seconds=duration_max_seconds,
        time_offset_seconds=time_offset_seconds,
        progress=progress,
    )
