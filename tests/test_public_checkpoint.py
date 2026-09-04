from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json
from p2g.errors import ContractError, OutputExistsError
from p2g.training.checkpoint import (
    CHECKPOINT_MANIFEST_SCHEMA,
    CHECKPOINT_SCHEMA,
    capture_rng_state,
    export_model,
    latest_checkpoint,
    load_exported_model,
    read_checkpoint,
    restore_rng_state,
    restore_training_state,
    save_checkpoint,
)
from p2g.training.config import (
    ColorCorrectionConfig,
    DataConfig,
    InitializationConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
)
from p2g.training.dataset import SceneSampler
from p2g.training.initialization import GaussianInit
from p2g.training.model import DynamicGaussianModel
from p2g.training.optim import OptimizerBundle, build_optimizers
from p2g.training.photometric import CameraColorCorrectors

ROOT = Path(__file__).parents[1]


def _config(tmp_path: Path, *, color_correction: bool = False) -> RunConfig:
    return RunConfig(
        data=DataConfig(manifest=(tmp_path / "observations.json").resolve()),
        initialization=InitializationConfig(
            path=(tmp_path / "initialization.safetensors").resolve()
        ),
        model=ModelConfig(persistence="learned", gate_logit_scale=1.25),
        color_correction=ColorCorrectionConfig(
            mode="per_camera_affine" if color_correction else "off",
            start=5,
        ),
        training=TrainingConfig(iterations=20, checkpoint_every=5, evaluate_every=5),
    )


def _model() -> DynamicGaussianModel:
    count = 2
    initialization = GaussianInit(
        means=torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.1, 2.2]], dtype=torch.float32),
        log_scales=torch.full((count, 3), math.log(0.1), dtype=torch.float32),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        opacity_logits=torch.zeros((count, 1)),
        sh0=torch.zeros((count, 1, 3)),
        sh_rest=torch.zeros((count, 15, 3)),
        center_times=torch.tensor([[0.0], [0.5]], dtype=torch.float32),
        duration_logits=torch.zeros((count, 1)),
        velocities=torch.tensor([[0.1, 0.0, 0.0]] * count),
        persistence_logits=torch.full((count, 1), -2.0),
        duration_min_seconds=torch.full((count, 1), 0.05),
        duration_max_seconds=torch.full((count, 1), 1.0),
        runtime_ids=torch.arange(10, 10 + count),
        source={"format": "synthetic_test"},
    )
    return DynamicGaussianModel(initialization, persistence=True, gate_logit_scale=1.25)


def _runtime(
    tmp_path: Path,
    *,
    color_correction: bool = False,
) -> tuple[
    RunConfig,
    DynamicGaussianModel,
    OptimizerBundle,
    SceneSampler,
    CameraColorCorrectors | None,
]:
    config = _config(tmp_path, color_correction=color_correction)
    model = _model()
    correctors = CameraColorCorrectors(("camera-a", "camera-b")) if color_correction else None
    optimizers = build_optimizers(
        model,
        config.optimizer,
        iterations=config.training.iterations,
        scene_extent=1.0,
        color_correctors=correctors,
        color_correction_lr=(config.color_correction.learning_rate if correctors else None),
    )
    sampler = SceneSampler((0, 1, 2), seed=config.training.seed)
    return config, model, optimizers, sampler, correctors


def _populate_optimizer(model: DynamicGaussianModel, optimizers: OptimizerBundle) -> None:
    objective = torch.stack([parameter.square().sum() for parameter in model.parameters()]).sum()
    objective.backward()
    optimizers.step()
    optimizers.zero_grad()


def _rebind_manifest(checkpoint: Path) -> None:
    files = []
    for name in ("config.toml", "metadata.json", "state.pt"):
        path = checkpoint / name
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "checkpoint_id": sha256_json(files),
        "files": files,
    }
    (checkpoint / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def test_checkpoint_is_atomic_hash_closed_and_weights_only_loadable(tmp_path: Path) -> None:
    config, model, optimizers, sampler, correctors = _runtime(tmp_path)
    run_dir = tmp_path / "run"

    checkpoint = save_checkpoint(
        run_dir,
        next_step=5,
        config=config,
        model=model,
        optimizers=optimizers,
        sampler=sampler,
        relocation_state={"visits": torch.tensor([1, 2], dtype=torch.int64)},
        color_correctors=correctors,
    )

    assert {path.name for path in checkpoint.iterdir()} == {
        "config.toml",
        "manifest.json",
        "metadata.json",
        "state.pt",
    }
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == CHECKPOINT_MANIFEST_SCHEMA
    assert manifest["checkpoint_id"] == sha256_json(manifest["files"])
    restricted = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    assert restricted["schema_version"] == CHECKPOINT_SCHEMA
    loaded_config, state, metadata = read_checkpoint(checkpoint)
    assert loaded_config == config
    assert state["next_step"] == 5
    assert metadata["trust_scope"] == "local_resume_only"
    assert metadata["redistributable"] is False
    with pytest.raises(OutputExistsError, match="overwrite"):
        save_checkpoint(
            run_dir,
            next_step=5,
            config=config,
            model=model,
            optimizers=optimizers,
            sampler=sampler,
        )


def test_resume_restores_model_optimizer_sampler_and_all_rngs(tmp_path: Path) -> None:
    config, model, optimizers, sampler, _ = _runtime(tmp_path)
    _populate_optimizer(model, optimizers)
    sampler.next_index()
    sampler.next_index()
    random.seed(17)
    np.random.seed(19)
    torch.manual_seed(23)
    checkpoint = save_checkpoint(
        tmp_path / "run",
        next_step=7,
        config=config,
        model=model,
        optimizers=optimizers,
        sampler=sampler,
    )
    expected_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    expected_samples = [sampler.next_index() for _ in range(8)]
    expected_model = {name: value.clone() for name, value in model.state_dict().items()}

    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    _, state, _ = read_checkpoint(checkpoint)
    _, restored_model, restored_optimizers, restored_sampler, _ = _runtime(tmp_path)
    next_step = restore_training_state(
        state,
        model=restored_model,
        optimizers=restored_optimizers,
        sampler=restored_sampler,
    )

    actual_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert next_step == 7
    assert actual_rng == pytest.approx(expected_rng)
    assert [restored_sampler.next_index() for _ in range(8)] == expected_samples
    assert all(
        torch.equal(value, restored_model.state_dict()[name])
        for name, value in expected_model.items()
    )
    assert any(optimizer.state for optimizer in restored_optimizers.optimizers.values())


def test_rng_state_uses_only_primitive_containers_and_tensors() -> None:
    state = capture_rng_state()
    assert isinstance(state["python"]["words"], list)
    assert isinstance(state["numpy"]["keys"], list)
    assert isinstance(state["torch_cpu"], torch.Tensor)

    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == pytest.approx(expected)

    with pytest.raises(ContractError, match="invalid fields or schema"):
        restore_rng_state(state | {"private": object()})


@pytest.mark.parametrize("mutation", ["payload", "extra", "manifest", "symlink"])
def test_checkpoint_rejects_transport_and_directory_tampering(
    tmp_path: Path, mutation: str
) -> None:
    config, model, optimizers, sampler, _ = _runtime(tmp_path)
    original = save_checkpoint(
        tmp_path / "run",
        next_step=5,
        config=config,
        model=model,
        optimizers=optimizers,
        sampler=sampler,
    )
    candidate = tmp_path / f"tampered-{mutation}"
    shutil.copytree(original, candidate)
    if mutation == "payload":
        with (candidate / "state.pt").open("ab") as stream:
            stream.write(b"tamper")
    elif mutation == "extra":
        (candidate / "undeclared.txt").write_text("extra", encoding="utf-8")
    elif mutation == "manifest":
        payload = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        (candidate / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        state_path = candidate / "state.pt"
        target = tmp_path / "outside.pt"
        shutil.copy2(state_path, target)
        state_path.unlink()
        state_path.symlink_to(target)

    with pytest.raises(ContractError):
        read_checkpoint(candidate)


def test_weights_only_state_rejects_unregistered_object_even_with_rebound_hash(
    tmp_path: Path,
) -> None:
    config, model, optimizers, sampler, _ = _runtime(tmp_path)
    checkpoint = save_checkpoint(
        tmp_path / "run",
        next_step=5,
        config=config,
        model=model,
        optimizers=optimizers,
        sampler=sampler,
    )
    unsafe_state = {
        "schema_version": CHECKPOINT_SCHEMA,
        "next_step": 5,
        "payload": np.arange(3),
    }
    torch.save(unsafe_state, checkpoint / "state.pt")
    _rebind_manifest(checkpoint)

    with pytest.raises(ContractError, match="restricted checkpoint state"):
        read_checkpoint(checkpoint)


def test_latest_checkpoint_uses_numeric_completed_step_and_fails_on_corrupt_newest(
    tmp_path: Path,
) -> None:
    config, model, optimizers, sampler, _ = _runtime(tmp_path)
    run_dir = tmp_path / "run"
    save_checkpoint(
        run_dir,
        next_step=2,
        config=config,
        model=model,
        optimizers=optimizers,
        sampler=sampler,
    )
    newest = save_checkpoint(
        run_dir,
        next_step=10,
        config=config,
        model=model,
        optimizers=optimizers,
        sampler=sampler,
    )
    (run_dir / "checkpoints" / "step_bad").mkdir()
    assert latest_checkpoint(run_dir) == newest

    corrupt = run_dir / "checkpoints" / "step_00000011"
    corrupt.mkdir()
    with pytest.raises(ContractError):
        latest_checkpoint(run_dir)


def test_model_export_roundtrips_and_binds_exact_tensor_bytes(tmp_path: Path) -> None:
    config, model, _, _, _ = _runtime(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    tensor_path, metadata_path = export_model(
        run_dir,
        model=model,
        config=config,
        final_step=20,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["tensor"] == {
        "file": "model.safetensors",
        "bytes": tensor_path.stat().st_size,
        "sha256": sha256_file(tensor_path),
    }
    assert metadata["artifact_role"] == "local_training_export"
    loaded, loaded_metadata = load_exported_model(tensor_path)
    assert loaded_metadata == metadata
    assert all(
        torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items()
    )
    with pytest.raises(OutputExistsError, match="overwrite"):
        export_model(run_dir, model=model, config=config, final_step=20)


def test_model_export_detects_tensor_tampering(tmp_path: Path) -> None:
    config, model, _, _, _ = _runtime(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tensor_path, _ = export_model(run_dir, model=model, config=config, final_step=20)
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)

    with pytest.raises(ContractError, match="tensor identity"):
        load_exported_model(tensor_path)


def test_color_correction_export_is_separately_hash_bound(tmp_path: Path) -> None:
    config, model, _, _, correctors = _runtime(tmp_path, color_correction=True)
    assert correctors is not None
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    export_model(
        run_dir,
        model=model,
        config=config,
        final_step=20,
        color_correctors=correctors,
    )

    metadata_path = run_dir / "color_correctors.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tensor_path = run_dir / "color_correctors.safetensors"
    assert metadata["tensor"]["sha256"] == sha256_file(tensor_path)
    assert metadata["camera_ids"] == ["camera-a", "camera-b"]
    assert metadata["applied_at_inference"] is False


def test_public_checkpoint_source_declares_the_local_trust_boundary() -> None:
    source = (ROOT / "src/p2g/training/checkpoint.py").read_text(encoding="utf-8").casefold()

    assert "weights_only=true" in source
    assert '"trust_scope": "local_resume_only"' in source
    assert "weights_only=false" not in source
    assert ("free" + "time") not in source
    assert ("ft" + "gs") not in source
