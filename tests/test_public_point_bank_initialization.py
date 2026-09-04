from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.numpy import save_file as save_numpy_file
from safetensors.torch import load_file

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json, write_new_json
from p2g.errors import ContractError, OutputExistsError
from p2g.schema import validate_payload
from p2g.training.config import InitializationConfig
from p2g.training.initialization import load_gaussian_init
from p2g.training.point_bank_initialization import (
    CANONICAL_TENSOR_SCHEMA,
    RECEIPT_SCHEMA,
    build_point_bank_initialization,
    canonical_initialization_tensor_sha256,
)
from p2g.training.roma_point_provider import canonical_provenance_sha256

ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools/build_point_bank_initialization.py"


def _write_cache(
    root: Path,
    *,
    observation_manifest_sha256: str = "1" * 64,
) -> tuple[Path, dict[str, Any], list[str]]:
    root.mkdir()
    frame_ids = [0, 1]
    camera_ids = ["left", "right"]
    rgb = np.zeros((2, 2, 3, 4, 3), dtype=np.uint8)
    rgb[0, :, :, :, 0] = 64
    rgb[1, :, :, :, 1] = 192
    intrinsic = np.tile(np.eye(3, dtype=np.float32), (2, 2, 1, 1))
    intrinsic[:, :, 0, 0] = 10.0
    intrinsic[:, :, 1, 1] = 10.0
    world_to_camera = np.tile(np.eye(4, dtype=np.float32), (2, 2, 1, 1))
    timestamp = np.asarray(((0.0, 0.0), (0.1, 0.1)), dtype=np.float64)
    arrays = {
        "rgb": rgb,
        "intrinsic": intrinsic,
        "world_to_camera": world_to_camera,
        "timestamp_seconds": timestamp,
    }
    records: dict[str, Any] = {}
    for name, value in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        records[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "dtype": value.dtype.name,
            "shape": list(value.shape),
            "order": "C",
        }
    manifest = {
        "schema_version": "p2g.tensor_cache.v1",
        "observation_manifest_sha256": observation_manifest_sha256,
        "camera_ids": camera_ids,
        "frame_ids": frame_ids,
        "arrays": records,
    }
    (root / "tensor_cache.json").write_bytes(canonical_json_bytes(manifest))
    rgb_hashes = [
        hashlib.sha256(memoryview(np.ascontiguousarray(rgb[index])).cast("B")).hexdigest()
        for index in range(2)
    ]
    return root, manifest, rgb_hashes


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    row_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    rows = np.empty(len(xyz), dtype=row_dtype)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rows["red"], rows["green"], rows["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(rows)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("xb") as stream:
        stream.write(header)
        rows.tofile(stream)


def _provenance_planes(count: int) -> dict[str, np.ndarray]:
    pair = np.repeat(np.arange(4, dtype=np.int32), count // 4)
    source = np.where(pair % 2 == 0, 0, 1).astype(np.int32)
    target = (1 - source).astype(np.int32)
    return {
        "admitted": np.ones(count, dtype=np.bool_),
        "ply_row": np.arange(count, dtype=np.int64),
        "triangulation_angle_degrees": np.linspace(5.0, 25.0, count, dtype=np.float32),
        "raw_certainty": np.linspace(0.1, 0.9, count, dtype=np.float32),
        "pair_ordinal": pair,
        "source_camera": source,
        "target_camera": target,
        "ray_gap_world": np.linspace(0.001, 0.008, count, dtype=np.float32),
        "source_reprojection_error_pixels": np.linspace(
            0.05, 0.2, count, dtype=np.float32
        ),
        "target_reprojection_error_pixels": np.linspace(
            0.06, 0.22, count, dtype=np.float32
        ),
    }


def _write_proposal_sequence(root: Path, cache: Path, rgb_hashes: list[str]) -> Path:
    root.mkdir()
    frames_root = root / "frames"
    points_root = root / "points"
    frames_root.mkdir()
    points_root.mkdir()
    cache_manifest_sha256 = sha256_file(cache / "tensor_cache.json")
    cache_manifest = json.loads((cache / "tensor_cache.json").read_text(encoding="utf-8"))
    observation_manifest_sha256 = cache_manifest["observation_manifest_sha256"]
    base = np.asarray(
        [(float(index), float(index % 2), 2.0) for index in range(8)], dtype=np.float32
    )
    colors = np.asarray(
        [(20 + index, 80 + index, 160 + index) for index in range(8)], dtype=np.uint8
    )
    rows: list[dict[str, Any]] = []
    for frame_id, timestamp in ((0, 0.0), (1, 0.1)):
        frame_root = frames_root / f"f{frame_id:06d}"
        frame_root.mkdir()
        ply_name = f"f{frame_id:06d}.ply"
        xyz = base + np.asarray((0.1 * frame_id, 0.0, 0.0), dtype=np.float32)
        ply_path = frame_root / ply_name
        _write_ply(ply_path, xyz, colors)
        os.link(ply_path, points_root / ply_name)
        planes = _provenance_planes(len(xyz))
        provenance_path = frame_root / "provenance.safetensors"
        save_numpy_file(
            planes,
            provenance_path,
            metadata={
                "schema": "p2g.roma_point_provenance.v1",
                "frame_id": str(frame_id),
                "row_semantics": "directed_pair_then_dense_source_linear_order",
            },
        )
        ply_sha256 = sha256_file(ply_path)
        provenance_sha256 = sha256_file(provenance_path)
        canonical_sha256 = canonical_provenance_sha256(planes, frame_id=frame_id)
        role_admission_unsigned = {
            "schema": "p2g.observation_role_admission.v1",
            "role": "train",
            "observation_manifest_sha256": observation_manifest_sha256,
            "frame_id": frame_id,
            "frame_timestamp_operator": (
                "arithmetic_mean_of_train_observation_timestamps_v1"
            ),
            "cache_camera_count": 2,
            "admitted_camera_ids": ["left", "right"],
            "admitted_observation_ids": [
                f"obs_left_{frame_id:06d}",
                f"obs_right_{frame_id:06d}",
            ],
            "excluded_camera_ids_by_role": {
                "diagnostic": [],
                "sealed": [],
                "free_view": [],
            },
        }
        role_admission = {
            **role_admission_unsigned,
            "logical_sha256": sha256_json(role_admission_unsigned),
        }
        receipt = {
            "schema": "p2g.roma_point_proposals.v1",
            "status": "COMPLETE",
            "frame": {
                "frame_id": frame_id,
                "timestamp_seconds": timestamp,
                "role": "train",
                "camera_ids": ["left", "right"],
            },
            "source": {
                "schema": "p2g.tensor_cache.v1",
                "manifest": {
                    "path": "tensor_cache.json",
                    "sha256": cache_manifest_sha256,
                    "observation_manifest_sha256": observation_manifest_sha256,
                },
                "frame_payload_sha256": {"rgb": rgb_hashes[frame_id]},
                "role_admission": role_admission,
            },
            "provider": {"name": "project_owned_test_provider", "revision": "v1"},
            "aggregate": {"sampled_count": len(xyz), "admitted_count": len(xyz)},
            "artifacts": {
                "point_ply": {
                    "path": ply_name,
                    "vertex_count": len(xyz),
                    "sha256": ply_sha256,
                },
                "provenance": {
                    "path": "provenance.safetensors",
                    "sha256": provenance_sha256,
                    "canonical_tensor_sha256": canonical_sha256,
                },
            },
        }
        write_new_json(frame_root / "receipt.json", receipt)
        rows.append(
            {
                "frame_id": frame_id,
                "frame_root": f"frames/f{frame_id:06d}",
                "point_ply": ply_name,
                "point_ply_sha256": ply_sha256,
                "provenance_sha256": provenance_sha256,
                "provenance_canonical_tensor_sha256": canonical_sha256,
                "sampled_count": len(xyz),
                "admitted_count": len(xyz),
                "source_rgb_sha256": rgb_hashes[frame_id],
                "role_admission_sha256": role_admission["logical_sha256"],
            }
        )
    collection = {
        "schema": "p2g.roma_point_proposal_sequence.v1",
        "status": "COMPLETE",
        "frame_ids": [0, 1],
        "frame_count": 2,
        "points_root": "points",
        "admitted_observation_role": "train",
        "observation_manifest_sha256": observation_manifest_sha256,
        "provider": {"name": "project_owned_test_provider", "revision": "v1"},
        "policy": {"fixture": "public_first_party_v1"},
        "aggregate": {
            "sampled_count": 16,
            "admitted_count": 16,
            "admitted_fraction": 1.0,
        },
        "frames": rows,
        "publication": "test_atomic_inventory_v1",
    }
    write_new_json(root / "collection.json", collection)
    return root


@pytest.fixture
def public_sources(tmp_path: Path) -> tuple[Path, Path]:
    cache, _, rgb_hashes = _write_cache(tmp_path / "tensor-cache")
    sequence = _write_proposal_sequence(tmp_path / "proposal-sequence", cache, rgb_hashes)
    return sequence, cache


def test_builder_emits_replayable_strict_trainer_asset(
    tmp_path: Path, public_sources: tuple[Path, Path]
) -> None:
    sequence, cache = public_sources
    first_output = tmp_path / "initialization-a"
    second_output = tmp_path / "initialization-b"
    arguments = {
        "proposal_sequence": sequence,
        "tensor_cache": cache,
        "num_gaussians": 11,
        "velocity_neighbors": 1,
        "sampling_mode": "paired_multiview_consensus_rank_mixture",
        "sampling_voxel_size": 0.5,
        "sampling_evidence_fraction": 0.5,
    }

    first = build_point_bank_initialization(first_output, **arguments)
    second = build_point_bank_initialization(second_output, **arguments)
    tensors = load_file(first_output / "initialization.safetensors", device="cpu")

    assert first == second
    assert first["schema"] == RECEIPT_SCHEMA
    assert first["population"] == {
        "frame_count": 2,
        "source_candidate_count": 16,
        "requested_gaussians": 11,
        "assembled_gaussians": 10,
        "discarded_budget_remainder": 1,
    }
    assert set(tensors) == {
        "means",
        "log_scales",
        "quaternions",
        "opacity_logits",
        "sh0",
        "center_times",
        "duration_logits",
        "velocities",
        "runtime_ids",
    }
    assert tensors["means"].shape == (10, 3)
    assert tensors["runtime_ids"].tolist() == list(range(10))
    assert first["tensor"]["canonical_digest_schema"] == CANONICAL_TENSOR_SCHEMA
    assert first["tensor"]["canonical_tensor_sha256"] == (
        canonical_initialization_tensor_sha256(tensors)
    )
    unsigned = {key: value for key, value in first.items() if key != "logical_sha256"}
    assert first["logical_sha256"] == sha256_json(unsigned)
    validate_payload("gaussian_initialization_receipt", first)
    assert str(tmp_path) not in json.dumps(first, sort_keys=True)
    assert (first_output / "initialization.safetensors").read_bytes() == (
        second_output / "initialization.safetensors"
    ).read_bytes()

    loaded = load_gaussian_init(
        InitializationConfig(path=(first_output / "initialization.safetensors").resolve())
    )
    sigma = loaded.duration_min_seconds + (
        loaded.duration_max_seconds - loaded.duration_min_seconds
    ) * torch.sigmoid(loaded.duration_logits)
    assert loaded.count == 10
    assert torch.allclose(sigma, torch.full_like(sigma, 0.1), rtol=0.0, atol=1.0e-6)
    assert bool(torch.allclose(loaded.velocities[:, 0], torch.ones(10), atol=1.0e-5))
    with safe_open(str(first_output / "initialization.safetensors"), framework="pt") as stream:
        metadata = stream.metadata() or {}
    assert metadata["schema_version"] == "p2g.gaussian_initialization.v1"
    assert metadata["proposal_sequence_sha256"] == sha256_file(sequence / "collection.json")
    assert metadata["tensor_cache_manifest_sha256"] == sha256_file(
        cache / "tensor_cache.json"
    )


def test_builder_fails_closed_on_policy_mismatch_tamper_and_overwrite(
    tmp_path: Path, public_sources: tuple[Path, Path]
) -> None:
    sequence, cache = public_sources
    output = tmp_path / "initialization"
    build_point_bank_initialization(
        output,
        proposal_sequence=sequence,
        tensor_cache=cache,
        num_gaussians=8,
        velocity_neighbors=2,
    )

    with pytest.raises(OutputExistsError, match="overwrite"):
        build_point_bank_initialization(
            output,
            proposal_sequence=sequence,
            tensor_cache=cache,
            num_gaussians=8,
            velocity_neighbors=2,
        )
    with pytest.raises(ContractError, match="differs from the run configuration"):
        load_gaussian_init(
            InitializationConfig(
                path=(output / "initialization.safetensors").resolve(),
                duration_min_seconds=0.02,
            )
        )

    timestamp_path = cache / "timestamp_seconds.npy"
    payload = bytearray(timestamp_path.read_bytes())
    payload[-1] ^= 1
    timestamp_path.write_bytes(payload)
    tampered_output = tmp_path / "tampered-output"
    with pytest.raises(ContractError, match="timestamp array SHA-256 mismatch"):
        build_point_bank_initialization(
            tampered_output,
            proposal_sequence=sequence,
            tensor_cache=cache,
            num_gaussians=8,
            velocity_neighbors=2,
        )
    assert not tampered_output.exists()


def test_builder_accepts_only_ulp_bounded_timestamp_reduction_roundoff(
    tmp_path: Path,
) -> None:
    cache, _, rgb_hashes = _write_cache(tmp_path / "cache")
    sequence = _write_proposal_sequence(tmp_path / "proposal-sequence", cache, rgb_hashes)
    receipt_path = sequence / "frames/f000001/receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["frame"]["timestamp_seconds"] = float(
        np.nextafter(np.float64(0.1), np.float64(np.inf))
    )
    receipt_path.write_bytes(canonical_json_bytes(receipt))

    build_point_bank_initialization(
        tmp_path / "roundoff-output",
        proposal_sequence=sequence,
        tensor_cache=cache,
        num_gaussians=8,
        velocity_neighbors=1,
    )

    receipt["frame"]["timestamp_seconds"] = 0.100000000001
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    rejected_output = tmp_path / "material-mismatch-output"
    with pytest.raises(ContractError, match="timestamps differ"):
        build_point_bank_initialization(
            rejected_output,
            proposal_sequence=sequence,
            tensor_cache=cache,
            num_gaussians=8,
            velocity_neighbors=1,
        )
    assert not rejected_output.exists()


def test_initializer_rejects_a_proposal_without_train_role_admission(
    tmp_path: Path, public_sources: tuple[Path, Path]
) -> None:
    sequence, cache = public_sources
    receipt_path = sequence / "frames/f000000/receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    del receipt["source"]["role_admission"]
    receipt_path.write_bytes(canonical_json_bytes(receipt))

    output = tmp_path / "rejected-initialization"
    with pytest.raises(ContractError, match="receipt disagrees"):
        build_point_bank_initialization(
            output,
            proposal_sequence=sequence,
            tensor_cache=cache,
            num_gaussians=8,
            velocity_neighbors=1,
        )
    assert not output.exists()


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_initialization_tool", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initialization_tool_help_is_lazy_and_uses_public_names() -> None:
    source_root = ROOT / "src"
    program = (
        "import importlib.util,json,sys;"
        "sys.meta_path[:]=[f for f in sys.meta_path "
        "if getattr(f,'__module__','')!='_pixel4dgs_editable'];"
        f"sys.path.insert(0,{str(source_root)!r});"
        f"p={str(TOOL)!r};"
        "s=importlib.util.spec_from_file_location('public_tool',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps({'torch_loaded':'torch' in sys.modules,"
        "'options':sorted(a.dest for a in m.build_parser()._actions)}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["torch_loaded"] is False
    assert "proposal_sequence" in result["options"]
    assert "tensor_cache" in result["options"]
    assert "points_root" not in result["options"]
    assert "memmap_root" not in result["options"]


def test_initialization_tool_dispatches_defaults_and_isolates_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_tool()
    observed: dict[str, Any] = {}

    def fake_builder(output: Path, **kwargs: Any) -> dict[str, Any]:
        observed.update({"output": output, **kwargs})
        kwargs["progress"]("FRAME [1/2] 000000")
        return {"schema": RECEIPT_SCHEMA, "status": "COMPLETE"}

    monkeypatch.setattr(module, "build_initialization", fake_builder)
    result = module.main(
        [
            "--proposal-sequence",
            str(tmp_path / "sequence"),
            "--tensor-cache",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert json.loads(captured.out) == {"schema": RECEIPT_SCHEMA, "status": "COMPLETE"}
    assert captured.err == "FRAME [1/2] 000000\n"
    assert observed["output"] == (tmp_path / "output").resolve()
    assert observed["proposal_sequence"] == (tmp_path / "sequence").resolve()
    assert observed["tensor_cache"] == (tmp_path / "cache").resolve()
    assert observed["num_gaussians"] == 500_000
    assert observed["sampling_mode"] == "paired_multiview_consensus_rank_mixture"
    assert observed["sampling_evidence_fraction"] == 0.5
