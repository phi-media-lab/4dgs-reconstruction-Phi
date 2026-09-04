from __future__ import annotations

import numpy as np
import pytest
import torch

from p2g.errors import ContractError
from p2g.training.point_bank_initialization import (
    PointEvidence,
    canonical_initialization_tensor_sha256,
    sample_point_frame,
)


def _motion_fixture(count: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    xyz = np.stack((axis, axis**2, axis**3), axis=1).astype(np.float32)
    reference = xyz + np.asarray((0.01, -0.02, 0.03), dtype=np.float32)
    rgb = np.tile(np.asarray((64, 128, 192), dtype=np.uint8), (count, 1))
    return xyz, reference, rgb


def test_point_frame_sampling_is_deterministic_finite_and_explicit() -> None:
    xyz, reference, rgb = _motion_fixture(32)
    arguments = {
        "xyz": xyz,
        "rgb": rgb,
        "reference_xyz": reference,
        "center_time": 0.5,
        "delta_time": 1.0 / 60.0,
        "count": 16,
        "velocity_neighbors": 3,
    }

    first = sample_point_frame(**arguments, rng=np.random.default_rng(7))
    replay = sample_point_frame(**arguments, rng=np.random.default_rng(7))

    assert set(first) == {"means", "log_scales", "sh0", "center_times", "velocities"}
    assert all(torch.equal(first[name], replay[name]) for name in first)
    assert all(bool(value.isfinite().all()) for value in first.values())
    assert first["means"].shape == (16, 3)
    assert first["sh0"].shape == (16, 1, 3)
    assert bool((first["center_times"] > 0.49).all())
    assert bool((first["center_times"] < 0.51).all())


def test_voxel_uniform_sampling_balances_provider_density() -> None:
    dense = np.stack(
        (
            np.linspace(0.0, 0.009, 100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        ),
        axis=1,
    )
    xyz = np.concatenate((dense, np.asarray(((1.0, 1.0, 1.0),), dtype=np.float32)))
    reference = xyz + np.asarray((0.01, -0.02, 0.03), dtype=np.float32)
    rgb = np.tile(np.asarray((64, 128, 192), dtype=np.uint8), (len(xyz), 1))
    common = {
        "xyz": xyz,
        "rgb": rgb,
        "reference_xyz": reference,
        "center_time": 0.5,
        "delta_time": 1.0 / 60.0,
        "count": 2_000,
        "velocity_neighbors": 1,
        "sampling_voxel_size": 0.02,
    }

    raw = sample_point_frame(
        **common,
        rng=np.random.default_rng(11),
        sampling_mode="raw_candidate_uniform",
    )
    voxel = sample_point_frame(
        **common,
        rng=np.random.default_rng(11),
        sampling_mode="occupied_voxel_uniform",
    )

    assert int((raw["means"][:, 0] > 0.5).sum()) < 100
    assert 900 < int((voxel["means"][:, 0] > 0.5).sum()) < 1_100


def test_triangulation_information_is_a_soft_sampling_prior() -> None:
    low = np.stack(
        (
            np.linspace(0.0, 0.049, 50, dtype=np.float32),
            np.zeros(50, dtype=np.float32),
            np.zeros(50, dtype=np.float32),
        ),
        axis=1,
    )
    xyz = np.concatenate((low, low + np.asarray((1.0, 0.0, 0.0), dtype=np.float32)))
    reference = xyz + np.asarray((0.01, -0.02, 0.03), dtype=np.float32)
    rgb = np.tile(np.asarray((64, 128, 192), dtype=np.uint8), (len(xyz), 1))
    evidence = PointEvidence(
        angle_degrees=np.concatenate(
            (np.full(50, 1.0, dtype=np.float32), np.full(50, 30.0, dtype=np.float32))
        ),
        certainty=np.linspace(0.1, 1.0, 100, dtype=np.float32),
        pair_ordinal=np.concatenate(
            (np.zeros(50, dtype=np.int32), np.ones(50, dtype=np.int32))
        ),
    )

    sampled = sample_point_frame(
        xyz=xyz,
        rgb=rgb,
        reference_xyz=reference,
        center_time=0.5,
        delta_time=1.0 / 60.0,
        count=2_000,
        rng=np.random.default_rng(13),
        velocity_neighbors=1,
        sampling_mode="triangulation_information_mixture",
        sampling_evidence=evidence,
        sampling_evidence_fraction=0.5,
    )

    high_information = int((sampled["means"][:, 0] > 0.5).sum())
    assert 1_400 < high_information < 1_600
    assert 2_000 - high_information > 400


def test_multiview_consensus_preserves_paired_stream_and_favors_support() -> None:
    weak = np.stack(
        (
            np.linspace(0.0, 0.009, 100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        ),
        axis=1,
    )
    xyz = np.concatenate((weak, weak + np.asarray((1.0, 0.0, 0.0), dtype=np.float32)))
    reference = xyz + np.asarray((0.01, -0.02, 0.03), dtype=np.float32)
    rgb = np.tile(np.asarray((64, 128, 192), dtype=np.uint8), (len(xyz), 1))
    pair = np.concatenate((np.zeros(100, dtype=np.int32), np.repeat(np.arange(1, 5), 25)))
    source_camera = np.concatenate(
        (np.zeros(100, dtype=np.int32), np.repeat(np.asarray((0, 1, 2, 3)), 25))
    ).astype(np.int32)
    target_camera = np.concatenate(
        (np.ones(100, dtype=np.int32), np.repeat(np.asarray((2, 2, 3, 0)), 25))
    ).astype(np.int32)
    evidence = PointEvidence(
        angle_degrees=np.full(200, 15.0, dtype=np.float32),
        certainty=np.full(200, 0.75, dtype=np.float32),
        pair_ordinal=pair.astype(np.int32),
        source_camera=source_camera,
        target_camera=target_camera,
        ray_gap_world=np.full(200, 0.001, dtype=np.float32),
        source_reprojection_pixels=np.full(200, 0.1, dtype=np.float32),
        target_reprojection_pixels=np.full(200, 0.1, dtype=np.float32),
    )
    common = {
        "xyz": xyz,
        "rgb": rgb,
        "reference_xyz": reference,
        "center_time": 0.5,
        "delta_time": 1.0 / 60.0,
        "count": 4_000,
        "velocity_neighbors": 1,
    }
    raw = sample_point_frame(
        **common,
        rng=np.random.default_rng(19),
        sampling_mode="raw_candidate_uniform",
    )
    matcher = sample_point_frame(
        **common,
        rng=np.random.default_rng(19),
        sampling_mode="paired_matcher_support_rank_mixture",
        sampling_evidence=evidence,
        sampling_evidence_fraction=0.5,
        sampling_evidence_seed=123,
    )
    consensus = sample_point_frame(
        **common,
        rng=np.random.default_rng(19),
        sampling_mode="paired_multiview_consensus_rank_mixture",
        sampling_voxel_size=0.02,
        sampling_evidence=evidence,
        sampling_evidence_fraction=0.5,
        sampling_evidence_seed=123,
    )

    raw_matches_matcher = (raw["means"] == matcher["means"]).all(dim=1)
    raw_matches_consensus = (raw["means"] == consensus["means"]).all(dim=1)
    assert int((raw_matches_matcher & raw_matches_consensus).sum()) >= 2_000
    assert torch.equal(matcher["center_times"], raw["center_times"])
    assert torch.equal(consensus["center_times"], raw["center_times"])
    matcher_strong = int((matcher["means"][:, 0] > 0.5).sum())
    consensus_strong = int((consensus["means"][:, 0] > 0.5).sum())
    assert consensus_strong > matcher_strong + 150


def test_canonical_initialization_digest_is_order_independent_and_alias_free() -> None:
    count = 3
    tensors = {
        "means": torch.zeros((count, 3), dtype=torch.float32),
        "log_scales": torch.zeros((count, 3), dtype=torch.float32),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        "opacity_logits": torch.zeros((count, 1), dtype=torch.float32),
        "sh0": torch.zeros((count, 1, 3), dtype=torch.float32),
        "center_times": torch.zeros((count, 1), dtype=torch.float32),
        "duration_logits": torch.zeros((count, 1), dtype=torch.float32),
        "velocities": torch.zeros((count, 3), dtype=torch.float32),
        "runtime_ids": torch.arange(count, dtype=torch.int64),
    }

    expected = canonical_initialization_tensor_sha256(tensors)
    assert canonical_initialization_tensor_sha256(dict(reversed(tensors.items()))) == expected
    tensors["quats"] = tensors.pop("quaternions")
    with pytest.raises(ContractError, match="catalog mismatch"):
        canonical_initialization_tensor_sha256(tensors)
