# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportUnknownMemberType=false

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json
from p2g.errors import ContractError, OutputExistsError
from p2g.training.config import PortableProfile, RunConfig, SceneInputs
from p2g.training.evaluate import evaluate_scene_selection
from p2g.training.sealed_evaluate import (
    _load_gate,
    evaluate_sealed_run,
    verify_sealed_receipt,
)

ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "tests/fixtures/sealed-evaluation/profile.toml"


def _gate_files(root: Path) -> Path:
    root.mkdir(parents=True)
    profile_path = root / "profile.toml"
    profile_path.write_bytes(PROFILE.read_bytes())
    profile_sha256 = sha256_file(profile_path)
    recipe: dict[str, Any] = {
        "schema_version": "p2g.sealed_evaluation_test_recipe.v1",
        "dataset": {"observation_manifest_sha256": "1" * 64},
        "training": {
            "profile_sha256": profile_sha256,
            "iterations": 30_000,
            "gaussian_count": 499_980,
        },
    }
    recipe["recipe_id"] = sha256_json(recipe)
    recipe_path = root / "recipe.json"
    recipe_path.write_bytes(canonical_json_bytes(recipe))
    gate: dict[str, Any] = {
        "schema_version": "p2g.sealed_quality_gate.v1",
        "protocol": {
            "recipe": {
                "file": recipe_path.name,
                "bytes": recipe_path.stat().st_size,
                "sha256": sha256_file(recipe_path),
                "recipe_id": recipe["recipe_id"],
            },
            "profile": {
                "file": profile_path.name,
                "bytes": profile_path.stat().st_size,
                "sha256": profile_sha256,
            },
            "dataset": {
                "dataset_id": "sealed-evaluation-fixture",
                "observation_manifest_sha256": "1" * 64,
                "diagnostic": {
                    "camera_ids": ["diagnostic-camera"],
                    "observation_count": 60,
                },
                "sealed": {
                    "camera_ids": ["sealed-camera"],
                    "observation_count": 60,
                },
            },
            "candidate": {
                "final_step": 30_000,
                "gaussian_count": 499_980,
                "max_sh_degree": 3,
            },
            "metrics": {
                "psnr_equation": "p2g.psnr_rgb_unit_range.v1",
                "ssim_equation": "p2g.gaussian_ssim_rgb_unit_range.v1",
                "ssim_padding": "valid",
                "aggregation": "arithmetic_mean_of_per_observation_scores_v1",
            },
        },
        "quality": {
            "diagnostic": {"minimum_mean": {"psnr": 28.0, "ssim": 0.9}},
            "sealed": {
                "anchor_mean": {"psnr": 25.0, "ssim": 0.85},
                "maximum_regression": {"psnr": 0.1, "ssim": 0.01},
                "minimum_mean": {"psnr": 24.9, "ssim": 0.84},
            },
        },
        "policy": {
            "sealed_use": "single_post_freeze_evaluation",
            "tuning_allowed": False,
            "checkpoint_selection_allowed": False,
            "write_once_output": True,
            "publish_failure_receipt": True,
        },
        "claim_boundary": "Test-only protocol with generated inputs and no external payload.",
    }
    gate["gate_id"] = sha256_json(gate)
    gate_path = root / "gate.json"
    gate_path.write_bytes(canonical_json_bytes(gate))
    return gate_path


class _Batch:
    def __init__(self, observation_id: str, role: str, value: float) -> None:
        self.observation_id = observation_id
        self.camera_id = "sealed-camera"
        self.frame_id = int(observation_id.rsplit("-", maxsplit=1)[1])
        self.role = role
        self.timestamp = torch.tensor(self.frame_id / 10.0)
        self.rgb = torch.full((12, 12, 3), value)

    def to(self, _device: str) -> _Batch:
        return self


class _SelectionScene:
    dataset_id = "selection-fixture"
    observations = tuple(range(3))

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def load_batch(self, index: int, *, access: str = "routine") -> _Batch:
        self.calls.append((index, access))
        return _Batch(f"sealed-{index}", "sealed", 0.25)


class _SelectionRenderer:
    def render(self, _model: object, batch: _Batch, *, sh_degree: int) -> Any:
        assert sh_degree == 3
        return SimpleNamespace(image=batch.rgb.clone())


def test_evaluation_selection_requires_explicit_sealed_access() -> None:
    scene = _SelectionScene()
    result = evaluate_scene_selection(
        model=object(),  # type: ignore[arg-type]
        renderer=_SelectionRenderer(),  # type: ignore[arg-type]
        scene=scene,  # type: ignore[arg-type]
        indices=(1, 2),
        access="sealed",
        device="cpu",
        sh_degree=3,
        output_dir=None,
        ssim_padding="valid",
    )

    assert scene.calls == [(1, "sealed"), (2, "sealed")]
    assert result["observation_count"] == 2
    assert result["mean"]["psnr"] == pytest.approx(120.0)
    assert result["mean"]["ssim"] == pytest.approx(1.0)
    with pytest.raises(ContractError, match="unique in-range"):
        evaluate_scene_selection(
            model=object(),  # type: ignore[arg-type]
            renderer=_SelectionRenderer(),  # type: ignore[arg-type]
            scene=scene,  # type: ignore[arg-type]
            indices=(1, 1),
            access="sealed",
            device="cpu",
            sh_degree=3,
            output_dir=None,
        )


def test_preregistered_gate_is_canonical_and_binds_recipe_profile_and_floors(
    tmp_path: Path,
) -> None:
    gate_path = _gate_files(tmp_path / "original")
    loaded = _load_gate(gate_path)
    gate = loaded.payload

    unsigned = dict(gate)
    assert unsigned.pop("gate_id") == sha256_json(unsigned)
    assert loaded.recipe["recipe_id"] == gate["protocol"]["recipe"]["recipe_id"]
    assert sha256_file(loaded.profile_path) == gate["protocol"]["profile"]["sha256"]
    sealed = gate["quality"]["sealed"]
    assert sealed["minimum_mean"]["psnr"] == pytest.approx(
        sealed["anchor_mean"]["psnr"] - sealed["maximum_regression"]["psnr"]
    )
    assert sealed["minimum_mean"]["ssim"] == pytest.approx(
        sealed["anchor_mean"]["ssim"] - sealed["maximum_regression"]["ssim"]
    )

    changed = json.loads(gate_path.read_text(encoding="utf-8"))
    changed["quality"]["diagnostic"]["minimum_mean"]["psnr"] -= 1.0
    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    changed_path = changed_root / gate_path.name
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    (changed_root / loaded.recipe_path.name).write_bytes(loaded.recipe_path.read_bytes())
    (changed_root / loaded.profile_path.name).write_bytes(loaded.profile_path.read_bytes())
    with pytest.raises(ContractError, match="canonical identity"):
        _load_gate(changed_path)


def _write_candidate_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _evaluation_rows(observations: list[Any], *, psnr: float, ssim: float) -> list[dict[str, Any]]:
    return [
        {
            "observation_id": item.observation_id,
            "camera_id": item.camera_id,
            "frame_id": item.frame_id,
            "timestamp_seconds": item.timestamp_seconds,
            "psnr": psnr,
            "ssim": ssim,
            "render_ms": 1.0,
        }
        for item in observations
    ]


def _fake_scene(tmp_path: Path) -> Any:
    target = tmp_path / "target.png"
    target.write_bytes(b"bound-target")
    target_sha = sha256_file(target)
    diagnostic = [
        SimpleNamespace(
            observation_id=f"diag-{index:06d}",
            camera_id="diagnostic-camera",
            frame_id=index,
            timestamp_seconds=index / 60.0,
            role="diagnostic",
            image_path=target,
            image_sha256=target_sha,
        )
        for index in range(60)
    ]
    sealed = [
        SimpleNamespace(
            observation_id=f"sealed-{index:06d}",
            camera_id="sealed-camera",
            frame_id=index,
            timestamp_seconds=index / 60.0,
            role="sealed",
            image_path=target,
            image_sha256=target_sha,
        )
        for index in range(60)
    ]
    return SimpleNamespace(
        dataset_id="sealed-evaluation-fixture",
        observations=tuple([*diagnostic, *sealed]),
        diagnostic_indices=tuple(range(60)),
        eval_indices=tuple(range(60)),
        sealed_indices=tuple(range(60, 120)),
    )


def _candidate_config(tmp_path: Path) -> RunConfig:
    profile = PortableProfile.load(PROFILE)
    return RunConfig.from_profile_inputs(
        profile,
        SceneInputs(
            manifest=(tmp_path / "observation_manifest.json").resolve(),
            initialization=(tmp_path / "initialization.safetensors").resolve(),
        ),
    )


def _prepare_candidate(tmp_path: Path, scene: Any) -> tuple[Path, RunConfig]:
    run = tmp_path / "run"
    config = _candidate_config(tmp_path)
    diagnostic_rows = _evaluation_rows(
        [scene.observations[index] for index in scene.diagnostic_indices],
        psnr=29.0,
        ssim=0.91,
    )
    diagnostic = {
        "schema_version": "p2g.evaluation.v1",
        "dataset_id": scene.dataset_id,
        "observation_count": 60,
        "mean": {"psnr": 29.0, "ssim": 0.91, "render_ms": 1.0},
        "observations": diagnostic_rows,
    }
    files = {
        run / "config.toml": b"bound config\n",
        run / "training.json": canonical_json_bytes({"status": "COMPLETE"}),
        run / "runtime.json": canonical_json_bytes({"runtime": "fixture"}),
        run / "model.safetensors": b"safe tensor fixture",
        run / "model.json": canonical_json_bytes(
            {"final_step": 30_000, "gaussian_count": 499_980, "max_sh_degree": 3}
        ),
        run / "renders/step_00030000/evaluation.json": canonical_json_bytes(diagnostic),
        config.data.manifest: b"bound manifest\n",
        config.initialization.path: b"bound initialization\n",
    }
    for path, payload in files.items():
        _write_candidate_file(path, payload)
    return run, config


@pytest.mark.parametrize(
    ("sealed_psnr", "sealed_ssim", "expected_status"),
    [(25.0, 0.85, "PASS"), (24.0, 0.80, "FAIL")],
)
def test_sealed_evaluator_publishes_write_once_receipt_and_verifies_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_psnr: float,
    sealed_ssim: float,
    expected_status: str,
) -> None:
    import p2g.training.sealed_evaluate as sealed_module

    gate_path = _gate_files(tmp_path / "protocol")
    scene = _fake_scene(tmp_path)
    run, config = _prepare_candidate(tmp_path, scene)
    model = SimpleNamespace(count=499_980, max_sh_degree=3)

    monkeypatch.setattr(
        sealed_module,
        "verify_completed_run",
        lambda _run: (config, {"completed_steps": 30_000}),
    )
    monkeypatch.setattr(
        sealed_module,
        "load_exported_run",
        lambda _run: (
            config,
            model,
            {"final_step": 30_000, "gaussian_count": 499_980, "max_sh_degree": 3},
            3,
        ),
    )
    monkeypatch.setattr(
        sealed_module,
        "PreparedScene",
        SimpleNamespace(load=lambda _config: scene),
    )
    monkeypatch.setattr(sealed_module, "_validate_recipe_and_scene", lambda *_args: None)

    class FakeRenderer:
        def __init__(self, _config: object) -> None:
            pass

        def validate_environment(self, device: str) -> dict[str, Any]:
            assert device == "cuda"
            return {"backend": "fixture", "visible_device_count": 1}

    monkeypatch.setattr(sealed_module, "GsplatRenderer", FakeRenderer)

    def fake_evaluate(**values: Any) -> dict[str, Any]:
        assert values["access"] == "sealed"
        assert tuple(values["indices"]) == scene.sealed_indices
        output = Path(values["output_dir"])
        selected = [scene.observations[index] for index in scene.sealed_indices]
        for item in selected:
            _write_candidate_file(
                output / f"{item.observation_id}.png",
                b"png-" + item.observation_id.encode(),
            )
        rows = _evaluation_rows(selected, psnr=sealed_psnr, ssim=sealed_ssim)
        return {
            "schema_version": "p2g.evaluation.v1",
            "dataset_id": scene.dataset_id,
            "observation_count": len(rows),
            "mean": {"psnr": sealed_psnr, "ssim": sealed_ssim, "render_ms": 1.0},
            "observations": rows,
        }

    monkeypatch.setattr(sealed_module, "evaluate_scene_selection", fake_evaluate)
    output = tmp_path / f"sealed-{expected_status.casefold()}"

    receipt = evaluate_sealed_run(run, gate_file=gate_path, output_dir=output)

    assert receipt["status"] == expected_status
    assert receipt["sealed"]["observation_count"] == 60
    assert len(receipt["sealed"]["observations"]) == 60
    assert (output / "receipt.json").read_bytes() == canonical_json_bytes(receipt)
    with pytest.raises(OutputExistsError, match="refusing to overwrite"):
        evaluate_sealed_run(run, gate_file=gate_path, output_dir=output)

    verification = verify_sealed_receipt(
        output,
        run_dir=run,
        gate_file=gate_path,
        expected_receipt_id=receipt["receipt_id"],
    )
    assert verification["status"] == "PASS"
    assert verification["evaluated_status"] == expected_status
    assert verification["render_count"] == 60
    with pytest.raises(ContractError, match="externally retained"):
        verify_sealed_receipt(
            output,
            run_dir=run,
            gate_file=gate_path,
            expected_receipt_id="0" * 64,
        )

    first_render = output / receipt["sealed"]["observations"][0]["render"]["file"]
    first_render.write_bytes(first_render.read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="bytes differ"):
        verify_sealed_receipt(
            output,
            run_dir=run,
            gate_file=gate_path,
            expected_receipt_id=receipt["receipt_id"],
        )
