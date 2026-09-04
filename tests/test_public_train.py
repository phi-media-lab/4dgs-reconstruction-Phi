# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from p2g.canonical import canonical_json_bytes, sha256_file
from p2g.errors import ContractError
from p2g.training.config import (
    DataConfig,
    InitializationConfig,
    RunConfig,
    TensorMemmapConfig,
    TrainingConfig,
)
from p2g.training.dataset import SceneSampler
from p2g.training.initialization import load_gaussian_init
from p2g.training.model import DynamicGaussianModel
from p2g.training.optim import build_optimizers
from p2g.training.point_bank_initialization import build_point_bank_initialization
from p2g.training.train import (
    AssetPublication,
    _canonical_runtime_device,
    _NoRelocation,
    _NoScreenGuard,
    _reconcile_metrics_with_checkpoint,
    export_asset,
    run_training,
    verify_training_inputs,
)
from tests.test_public_point_bank_initialization import (
    _write_cache,
    _write_proposal_sequence,
)


def _manifest() -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for frame_id, timestamp in ((0, 0.0), (1, 0.1)):
        for camera_index, camera_id in enumerate(("left", "right")):
            observations.append(
                {
                    "observation_id": f"obs_{camera_id}_{frame_id:06d}",
                    "camera_id": camera_id,
                    "frame_id": frame_id,
                    "timestamp_seconds": timestamp,
                    "role": "train",
                    "image": {
                        "path": f"images/{camera_id}/{frame_id:06d}.png",
                        "sha256": f"{frame_id * 2 + camera_index + 1:064x}",
                        "width": 4,
                        "height": 3,
                        "color_space": "linear_rgb",
                        "encoding": {
                            "container": "png",
                            "channel_order": "RGB",
                            "bit_depth": 8,
                            "stored_range": "full",
                            "declared_transfer": None,
                            "declared_primaries": None,
                            "declared_matrix": None,
                            "canonical_decode_profile": "linear_passthrough_v1",
                        },
                    },
                    "camera": {
                        "model": "pinhole",
                        "pixel_domain": "undistorted",
                        "intrinsic": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]],
                        "world_to_camera": [
                            [1.0, 0.0, 0.0, -float(camera_index)],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "distortion": [],
                    },
                }
            )
    return {
        "schema_version": "p2g.observation_manifest.v2",
        "dataset_id": "public_training_fixture",
        "source": {
            "description": "project-owned analytic training fixture",
            "license": "CC0-1.0",
            "license_status": "declared",
            "root_sha256": "f" * 64,
        },
        "coordinate_conventions": {
            "handedness": "right",
            "extrinsic": "world_to_camera",
            "pixel_center": "half_pixel",
            "time_unit": "seconds",
            "photometric_space": "linear_rgb",
        },
        "sync": {
            "variant": "synthetic_exact_v1",
            "tolerance_seconds": 0.001,
            "per_camera_offset_seconds": {"left": 0.0, "right": 0.0},
        },
        "transforms": [],
        "observations": observations,
    }


def _bound_config(tmp_path: Path, *, iterations: int = 2) -> RunConfig:
    manifest_path = tmp_path / "observation_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(_manifest()))
    cache, _, rgb_hashes = _write_cache(
        tmp_path / "tensor-cache",
        observation_manifest_sha256=sha256_file(manifest_path),
    )
    sequence = _write_proposal_sequence(tmp_path / "proposal-sequence", cache, rgb_hashes)
    initialization = tmp_path / "initialization"
    build_point_bank_initialization(
        initialization,
        proposal_sequence=sequence,
        tensor_cache=cache,
        num_gaussians=8,
        velocity_neighbors=1,
    )
    return RunConfig(
        data=DataConfig(
            manifest=manifest_path.resolve(),
            tensor_memmap=TensorMemmapConfig(
                root=cache.resolve(),
                camera_ids=("left", "right"),
                frame_ids=(0, 1),
            ),
        ),
        initialization=InitializationConfig(
            path=(initialization / "initialization.safetensors").resolve()
        ),
        training=TrainingConfig(
            iterations=iterations,
            checkpoint_every=iterations,
            evaluate_every=iterations,
            log_every=iterations,
            sh_degree_interval=1,
        ),
    )


def test_training_input_binding_closes_manifest_cache_and_initialization(
    tmp_path: Path,
) -> None:
    config = _bound_config(tmp_path)

    binding = verify_training_inputs(config)

    assert binding["schema_version"] == "p2g.training_input_binding.v1"
    assert binding["optimization_roles"] == ["train"]
    assert binding["diagnostic_roles"] == ["diagnostic"]
    assert binding["sealed_roles_admitted"] == []
    assert binding["observation_manifest_sha256"] == sha256_file(config.data.manifest)
    assert binding["gaussian_initialization_sha256"] == sha256_file(
        config.initialization.path
    )

    config.data.manifest.write_bytes(canonical_json_bytes(_manifest() | {"dataset_id": "other"}))
    with pytest.raises(ContractError, match="another observation manifest"):
        verify_training_inputs(config)


def test_training_input_binding_rejects_receipt_substitution(tmp_path: Path) -> None:
    config = _bound_config(tmp_path)
    receipt_path = config.initialization.path.parent / "initialization.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["policy"]["seed"] += 1
    receipt_path.write_bytes(canonical_json_bytes(receipt))

    with pytest.raises(ContractError, match="logical hash"):
        verify_training_inputs(config)


def test_runtime_device_canonicalizes_unindexed_accelerator_alias() -> None:
    materialized = torch.device("cuda:0")

    assert _canonical_runtime_device("cuda", materialized) == materialized
    assert _canonical_runtime_device("cuda:0", materialized) == materialized
    with pytest.raises(ContractError, match="differs from the materialized model"):
        _canonical_runtime_device("cuda:1", materialized)
    with pytest.raises(ContractError, match="differs from the materialized model"):
        _canonical_runtime_device("cpu", materialized)


def test_sparse_metric_reconciliation_is_canonical_and_checkpoint_bounded(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 0, "next_step": 1},
        {"step": 9, "next_step": 10},
        {"step": 19, "next_step": 20},
    ]
    metrics.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))

    _reconcile_metrics_with_checkpoint(metrics, next_step=12)

    retained = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    assert [row["next_step"] for row in retained] == [1, 10]
    metrics.write_text('{"next_step":1,"step":0} \n', encoding="utf-8")
    with pytest.raises(ContractError, match="canonical and strictly step-ordered"):
        _reconcile_metrics_with_checkpoint(metrics, next_step=1)

    metrics.unlink()
    target = tmp_path / "outside-metrics.jsonl"
    target.write_bytes(canonical_json_bytes(rows[0]))
    metrics.symlink_to(target)
    with pytest.raises(ContractError, match="regular non-symlink"):
        _reconcile_metrics_with_checkpoint(metrics, next_step=1)


class _Batch:
    observation_id = "obs_left_000000"
    camera_id = "left"
    frame_id = 0
    role = "train"
    timestamp = torch.tensor(0.0)
    rgb = torch.full((2, 2, 3), 0.25)

    def to(self, _device: str) -> _Batch:
        return self


class _Scene:
    dataset_id = "public_training_fixture"
    train_indices = (0,)
    diagnostic_indices = (1,)
    eval_indices = diagnostic_indices
    sealed_indices = (2,)
    free_view_indices: tuple[int, ...] = ()
    excluded_indices: tuple[int, ...] = ()

    def load_batch(self, _index: int) -> _Batch:
        return _Batch()


class _Renderer:
    def render(
        self,
        model: DynamicGaussianModel,
        _batch: _Batch,
        *,
        sh_degree: int,
    ) -> SimpleNamespace:
        del sh_degree
        image = torch.sigmoid(model.sh0.mean()).expand(2, 2, 3)
        return SimpleNamespace(image=image, materialized=object(), aux={})


class _Loss:
    def __init__(self, _config: object, *, device: str) -> None:
        del device

    def __call__(self, **values: Any) -> dict[str, torch.Tensor]:
        return {"analytic": values["prediction"].mean()}


def test_cpu_orchestrator_publishes_hash_closed_result_and_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import p2g.training.train as training

    config = _bound_config(tmp_path)
    model = DynamicGaussianModel(
        load_gaussian_init(config.initialization),
        persistence=False,
    )
    optimizers = build_optimizers(
        model,
        config.optimizer,
        iterations=config.training.iterations,
        scene_extent=1.0,
    )
    scene = _Scene()
    renderer = _Renderer()
    runtime = {
        "renderer_abi": "p2g.gsplat_rocm.v1",
        "backend": "gsplat_rocm",
        "python_abi": "cp312",
        "torch": "2.10.0+rocm7.0",
        "hip": "7.0.0",
        "device": "CPU contract fixture",
        "architecture": "gfx942",
        "visible_device_count": 1,
        "gsplat_distribution_name": "amd-gsplat",
        "gsplat_distribution": "1.5.3+b01acd43",
        "gsplat_module": "1.5.3",
        "gsplat_source_revision": "b01acd43e3c7fa942f95fda0974e9125e4de7395",
    }

    monkeypatch.setattr(training, "LossFunction", _Loss)
    monkeypatch.setattr(
        training,
        "initialize_runtime",
        lambda _config, checkpoint_state: (
            scene,
            model,
            renderer,
            None,
            optimizers,
            SceneSampler((0,), seed=config.training.seed),
            0,
            runtime,
        ),
    )
    monkeypatch.setattr(
        training,
        "_create_population_controls",
        lambda _config, **_kwargs: (_NoRelocation(model.count), _NoScreenGuard()),
    )
    monkeypatch.setattr(
        training,
        "_gradient_norm_tensors",
        lambda *_args, **_kwargs: {"sh0": torch.tensor(1.0)},
    )

    def fake_evaluate(**_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": "p2g.evaluation.v1",
            "dataset_id": scene.dataset_id,
            "observation_count": 1,
            "mean": {"psnr": 12.0, "ssim": 0.5, "render_ms": 1.0},
            "observations": [],
        }

    monkeypatch.setattr(training, "evaluate_scene", fake_evaluate)
    run_dir = tmp_path / "run"

    result = run_training(config, run_dir=run_dir)

    assert result.completed_steps == 2
    assert result.receipt_path.is_file()
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "p2g.training_result.v1"
    assert receipt["status"] == "COMPLETE"
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["next_step"] for row in metrics] == [1, 2]
    assert all(row["observation_role"] == "train" for row in metrics)
    runtime_receipt = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    assert runtime_receipt["observation_roles"]["sealed_excluded"] == 1

    def fail_if_runtime_is_reinitialized(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a verified completed run must return before runtime initialization")

    monkeypatch.setattr(training, "initialize_runtime", fail_if_runtime_is_reinitialized)
    repeated = run_training(
        config,
        run_dir=run_dir,
        resume_checkpoint=result.final_checkpoint,
    )
    assert repeated == result

    publication = AssetPublication(
        output=tmp_path / "asset",
        producer_git_revision="a" * 40,
        asset_license="CC0-1.0",
        redistribution="allowed",
        provenance_summary="Project-owned analytic CPU contract fixture.",
    )
    with pytest.raises(ContractError, match="outside the immutable training run"):
        export_asset(run_dir, replace(publication, output=run_dir / "asset"))

    asset = export_asset(run_dir, publication)
    assert (asset / "manifest.json").is_file()
    metadata = json.loads((asset / "asset.json").read_text(encoding="utf-8"))
    assert metadata["training"]["final_step"] == 2
    assert metadata["rights"]["source_data_license"] == "CC0-1.0"


def test_public_train_source_has_no_reference_compatibility_or_forced_sync() -> None:
    source = (Path(__file__).parents[1] / "src/p2g/training/train.py").read_text(
        encoding="utf-8"
    )
    lowered = source.casefold()
    assert "freetime" not in lowered
    assert "ftgs" not in lowered
    assert "torch.cuda.synchronize" not in source
