from __future__ import annotations

import copy

import numpy as np
import pytest

from p2g.errors import ContractError
from p2g.training import roma_point_provider as provider


def _synthetic_frame() -> provider.TensorCacheFrame:
    rgb = np.zeros((3, 9, 11, 3), dtype=np.uint8)
    rgb[0, :, :, 0] = 90
    rgb[1, :, :, 1] = 120
    rgb[2, :, :, 2] = 150
    intrinsic = np.tile(
        np.asarray(
            ((12.0, 0.0, 5.5), (0.0, 12.0, 4.5), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        (3, 1, 1),
    )
    world_to_camera = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    world_to_camera[:, 0, 3] = np.asarray((2.0, 0.0, -2.0))
    return provider.TensorCacheFrame(
        frame_id=8,
        rgb=rgb,
        world_to_camera=world_to_camera,
        intrinsic=intrinsic,
        camera_timestamp_seconds=np.full(3, 0.4, dtype=np.float64),
        timestamp_seconds=0.4,
        camera_ids=("c0", "c1", "c2"),
        source_receipt={"schema": "project_owned_test_fixture.v1"},
    )


def test_canonical_digest_binds_frame_tensor_names_shapes_and_values() -> None:
    planes = {
        "z": np.asarray((1.0, 2.0), dtype=np.float32),
        "a": np.asarray((3, 4), dtype=np.int64),
    }
    reordered = {"a": planes["a"].copy(), "z": planes["z"].copy()}

    digest = provider.canonical_provenance_sha256(planes, frame_id=8)

    assert digest == provider.canonical_provenance_sha256(reordered, frame_id=8)
    assert digest != provider.canonical_provenance_sha256(reordered, frame_id=9)
    changed = {**reordered, "z": np.asarray((1.0, 3.0), dtype=np.float32)}
    assert digest != provider.canonical_provenance_sha256(changed, frame_id=8)


def test_nearest_graph_has_explicit_index_tie_break() -> None:
    frame = _synthetic_frame()

    graph = provider.nearest_camera_graph(frame.world_to_camera, neighbors=2)

    assert graph.tolist() == [[1, 2], [0, 2], [1, 0]]


def test_out_of_bounds_match_is_retained_but_not_admitted() -> None:
    frame = _synthetic_frame()
    source_pixels = np.asarray(((7.5, 4.5), (15.0, 4.5)), dtype=np.float64)
    target_pixels = np.asarray(((3.5, 4.5), (3.5, 4.5)), dtype=np.float64)
    matches = np.column_stack(
        (
            source_pixels[:, 0] * 2.0 / 11.0 - 1.0,
            source_pixels[:, 1] * 2.0 / 9.0 - 1.0,
            target_pixels[:, 0] * 2.0 / 11.0 - 1.0,
            target_pixels[:, 1] * 2.0 / 9.0 - 1.0,
        )
    )
    sampled = {
        "matches_normalized": matches,
        "raw_certainty": np.asarray((0.01, 0.99), dtype=np.float64),
        "selection_score": np.ones(2, dtype=np.float64),
        "dense_source_xy": np.asarray(((0, 0), (1, 0)), dtype=np.int64),
        "dense_source_valid": np.ones(2, dtype=np.bool_),
    }

    planes = provider.assemble_pair_provenance(
        frame,
        source_camera=0,
        target_camera=2,
        neighbor_rank=0,
        pair_ordinal=0,
        sampled=sampled,
    )

    assert planes["admitted"].tolist() == [True, False]
    assert planes["source_pixel_in_bounds"].tolist() == [True, False]
    assert planes["raw_certainty"].tolist() == pytest.approx([0.01, 0.99])
    assert planes["ply_row"].tolist() == [0, -1]


def test_registry_validation_rejects_weight_bundling_policy() -> None:
    registry, _ = provider.load_roma_provider_registry()
    changed = copy.deepcopy(registry)
    changed["policy"]["bundle_weights"] = True

    with pytest.raises(ContractError, match="fail-closed"):
        provider._validate_registry(changed)
