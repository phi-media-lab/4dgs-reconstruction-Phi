from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from p2g.errors import ContractError, OutputExistsError
from p2g.training.asset import (
    AssetBundleSpec,
    asset_summary,
    load_asset_bundle,
    write_asset_bundle,
)
from p2g.training.config import RendererConfig
from p2g.training.initialization import GaussianInit
from p2g.training.model import DynamicGaussianModel


def _model() -> DynamicGaussianModel:
    count = 3
    initialization = GaussianInit(
        means=torch.tensor(
            [[0.0, 0.0, 2.0], [0.2, 0.1, 2.2], [-0.1, 0.3, 2.4]],
            dtype=torch.float32,
        ),
        log_scales=torch.full((count, 3), math.log(0.1), dtype=torch.float32),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        opacity_logits=torch.zeros((count, 1)),
        sh0=torch.zeros((count, 1, 3)),
        sh_rest=torch.zeros((count, 15, 3)),
        center_times=torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32),
        duration_logits=torch.zeros((count, 1)),
        velocities=torch.tensor([[0.1, 0.0, 0.0]] * count),
        persistence_logits=torch.full((count, 1), -2.0),
        duration_min_seconds=torch.full((count, 1), 0.05),
        duration_max_seconds=torch.full((count, 1), 1.0),
        runtime_ids=torch.arange(10, 10 + count),
        source={"format": "unit_test"},
    )
    return DynamicGaussianModel(
        initialization,
        persistence=True,
        gate_logit_scale=1.0,
    )


def _spec(**overrides: object) -> AssetBundleSpec:
    values: dict[str, object] = {
        "valid_time_start_seconds": 0.0,
        "valid_time_stop_seconds": 1.0,
        "reference_time_seconds": 0.0,
        "world_coordinate_convention": "right_handed_calibration_world",
        "world_unit": "calibration_unit",
        "calibration_scale": 1.0,
        "photometric_space": "linear_rgb",
        "default_sh_degree": 3,
        "final_step": 30_000,
        "source_bundle_digests": {"scene": "b" * 64, "initialization": "c" * 64},
        "producer_version": "0.1.0.dev0",
        "producer_git_revision": "a" * 40,
        "dependency_identities": {
            "amd-gsplat": "1.5.3+b01acd43",
            "torch": "2.10.0+rocm7.0",
        },
        "asset_license": "LicenseRef-Internal-Research-Only",
        "source_data_license": "LicenseRef-Test-Fixture",
        "redistribution": "restricted",
        "provenance_summary": "Synthetic model created by the Pixel4DGS unit test.",
    }
    values.update(overrides)
    return AssetBundleSpec(**values)  # type: ignore[arg-type]


def test_asset_bundle_roundtrips_without_training_inputs(tmp_path: Path) -> None:
    model = _model()
    destination = tmp_path / "portable-asset"
    write_asset_bundle(
        destination,
        model=model,
        spec=_spec(),
        renderer=RendererConfig(require_gfx942=True),
    )

    assert {path.name for path in destination.iterdir()} == {
        "asset.json",
        "manifest.json",
        "model.safetensors",
    }
    assert not any(path.suffix in {".pt", ".pkl", ".pickle"} for path in destination.iterdir())
    assert "/home/" not in (destination / "asset.json").read_text(encoding="utf-8")
    assert "/mnt/" not in (destination / "asset.json").read_text(encoding="utf-8")

    loaded = load_asset_bundle(destination)
    assert loaded.model.count == model.count
    assert loaded.model.persistence_enabled
    assert asset_summary(loaded)["status"] == "PASS"
    assert loaded.metadata["training"]["final_step"] == 30_000
    assert loaded.metadata["time"]["valid_interval"] == [0.0, 1.0]
    assert loaded.metadata["renderer"]["required_architecture"] == "gfx942"
    for name, value in model.state_dict().items():
        assert torch.equal(value.cpu(), loaded.model.state_dict()[name])


def test_asset_bundle_bytes_are_deterministic_and_never_overwritten(tmp_path: Path) -> None:
    model = _model()
    first = tmp_path / "first"
    second = tmp_path / "second"
    renderer = RendererConfig()
    write_asset_bundle(first, model=model, spec=_spec(), renderer=renderer)
    write_asset_bundle(second, model=model, spec=_spec(), renderer=renderer)
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "asset.json").read_bytes() == (second / "asset.json").read_bytes()
    assert (first / "model.safetensors").read_bytes() == (second / "model.safetensors").read_bytes()
    with pytest.raises(OutputExistsError, match="overwrite"):
        write_asset_bundle(first, model=model, spec=_spec(), renderer=renderer)


def test_asset_bundle_detects_tampering_and_undeclared_files(tmp_path: Path) -> None:
    model = _model()
    tampered = tmp_path / "tampered"
    write_asset_bundle(tampered, model=model, spec=_spec(), renderer=RendererConfig())
    metadata = json.loads((tampered / "asset.json").read_text(encoding="utf-8"))
    metadata["time"]["reference_time"] = 0.25
    (tampered / "asset.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ContractError, match="digest mismatch"):
        load_asset_bundle(tampered)

    extra = tmp_path / "extra"
    write_asset_bundle(extra, model=model, spec=_spec(), renderer=RendererConfig())
    (extra / "checkpoint.pt").touch()
    with pytest.raises(ContractError, match="undeclared"):
        load_asset_bundle(extra)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"valid_time_stop_seconds": 0.0}, "non-empty interval"),
        ({"producer_git_revision": "short"}, "full lowercase hash"),
        ({"source_bundle_digests": {"scene": "not-a-hash"}}, "SHA-256"),
        ({"calibration_scale": math.inf}, "positive and finite"),
    ],
)
def test_asset_bundle_rejects_invalid_semantics(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        write_asset_bundle(
            tmp_path / "invalid",
            model=_model(),
            spec=_spec(**overrides),
            renderer=RendererConfig(),
        )


def test_asset_bundle_rejects_machine_paths_in_metadata(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="machine path"):
        write_asset_bundle(
            tmp_path / "invalid-path",
            model=_model(),
            spec=_spec(dependency_identities={"renderer": "/mnt/internal/backend.whl"}),
            renderer=RendererConfig(),
        )


def test_asset_cli_inspects_and_verifies_without_a_training_run(tmp_path: Path) -> None:
    from p2g.cli import app

    destination = tmp_path / "portable"
    write_asset_bundle(
        destination,
        model=_model(),
        spec=_spec(),
        renderer=RendererConfig(),
    )
    runner = CliRunner()
    inspected = runner.invoke(app, ["asset", "inspect", str(destination)])
    assert inspected.exit_code == 0, inspected.output
    assert "p2g.asset_bundle.v1" in inspected.output
    assert "gaussian_count" in inspected.output

    report = tmp_path / "verification.json"
    verified = runner.invoke(
        app,
        ["asset", "verify", str(destination), "--output", str(report)],
    )
    assert verified.exit_code == 0, verified.output
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "PASS"
