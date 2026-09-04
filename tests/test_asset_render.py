from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from typer.testing import CliRunner

from p2g.canonical import sha256_file
from p2g.errors import ContractError
from p2g.training import asset_render
from p2g.training.asset import (
    AssetBundleSpec,
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
    return DynamicGaussianModel(initialization, persistence=True, gate_logit_scale=1.0)


def _spec() -> AssetBundleSpec:
    return AssetBundleSpec(
        valid_time_start_seconds=0.0,
        valid_time_stop_seconds=1.0,
        reference_time_seconds=0.0,
        world_coordinate_convention="right_handed_calibration_world",
        world_unit="calibration_unit",
        calibration_scale=1.0,
        photometric_space="linear_rgb",
        default_sh_degree=3,
        final_step=30_000,
        source_bundle_digests={"scene": "b" * 64},
        producer_version="0.1.0.dev0",
        producer_git_revision="a" * 40,
        dependency_identities={
            "amd-gsplat": (
                "1.5.3+b01acd43@b01acd43e3c7fa942f95fda0974e9125e4de7395"
            )
        },
        asset_license="LicenseRef-Internal-Research-Only",
        source_data_license="LicenseRef-Test-Fixture",
        redistribution="restricted",
        provenance_summary="Synthetic model created by the Pixel4DGS unit test.",
    )


def _camera_payload(bundle_id: str) -> dict[str, Any]:
    intrinsic = [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
    world_to_camera = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "schema_version": "p2g.camera_path.v1",
        "asset_bundle_id": bundle_id,
        "time_unit": "seconds",
        "camera_model": "pinhole",
        "pixel_domain": "pre-undistorted",
        "intrinsic_matrix": "3x3_pixel_center",
        "extrinsic_matrix": "4x4_world_to_camera",
        "camera_axes": "opencv_x_right_y_down_z_forward",
        "width": 3,
        "height": 3,
        "fps": 30,
        "frames": [
            {
                "timestamp_seconds": 0.0,
                "intrinsic": intrinsic,
                "world_to_camera": world_to_camera,
            },
            {
                "timestamp_seconds": 1.0,
                "intrinsic": intrinsic,
                "world_to_camera": world_to_camera,
            },
        ],
    }


def _trajectory_payload() -> dict[str, Any]:
    payload = _camera_payload("a" * 64)
    payload["schema_version"] = "p2g.camera_trajectory.v1"
    del payload["asset_bundle_id"]
    return payload


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_camera_path_binds_exact_bytes_and_asset(tmp_path: Path) -> None:
    payload = _camera_payload("a" * 64)
    path = tmp_path / "camera-path.json"
    _write_payload(path, payload)

    loaded = asset_render.load_camera_path(path)

    assert loaded.asset_bundle_id == "a" * 64
    assert loaded.source_sha256 == sha256_file(path)
    assert [frame.timestamp_seconds for frame in loaded.frames] == [0.0, 1.0]


def test_camera_path_rejects_unsupported_or_ambiguous_geometry(tmp_path: Path) -> None:
    skewed = _camera_payload("a" * 64)
    skewed["frames"][0]["intrinsic"][0][1] = 0.25
    skewed_path = tmp_path / "skewed.json"
    _write_payload(skewed_path, skewed)
    with pytest.raises(ContractError, match="skew unsupported"):
        asset_render.load_camera_path(skewed_path)

    left_handed = _camera_payload("a" * 64)
    left_handed["frames"][0]["world_to_camera"][0][0] = -1.0
    left_handed_path = tmp_path / "left-handed.json"
    _write_payload(left_handed_path, left_handed)
    with pytest.raises(ContractError, match="not right-handed"):
        asset_render.load_camera_path(left_handed_path)

    reversed_time = _camera_payload("a" * 64)
    reversed_time["frames"][0]["timestamp_seconds"] = 0.75
    reversed_time["frames"][1]["timestamp_seconds"] = 0.25
    reversed_path = tmp_path / "reversed.json"
    _write_payload(reversed_path, reversed_time)
    with pytest.raises(ContractError, match="nondecreasing"):
        asset_render.load_camera_path(reversed_path)


def test_camera_path_must_name_the_loaded_asset_and_valid_time(tmp_path: Path) -> None:
    destination = tmp_path / "asset"
    write_asset_bundle(
        destination,
        model=_model(),
        spec=_spec(),
        renderer=RendererConfig(require_gfx942=False),
    )
    bundle = load_asset_bundle(destination)
    wrong_path = tmp_path / "wrong-asset.json"
    _write_payload(wrong_path, _camera_payload("c" * 64))
    with pytest.raises(ContractError, match="different AssetBundle"):
        asset_render._validate_path_against_asset(
            asset_render.load_camera_path(wrong_path), bundle
        )

    outside = _camera_payload(bundle.manifest["bundle_id"])
    outside["frames"][1]["timestamp_seconds"] = 1.5
    outside_path = tmp_path / "outside-time.json"
    _write_payload(outside_path, outside)
    with pytest.raises(ContractError, match="outside asset interval"):
        asset_render._validate_path_against_asset(
            asset_render.load_camera_path(outside_path), bundle
        )


def test_asset_runtime_identity_binds_gsplat_version_and_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "asset"
    write_asset_bundle(
        destination,
        model=_model(),
        spec=_spec(),
        renderer=RendererConfig(require_gfx942=False),
    )
    bundle = load_asset_bundle(destination)
    runtime = {
        "gsplat_distribution": "1.5.3+b01acd43",
        "gsplat_source_revision": "b01acd43e3c7fa942f95fda0974e9125e4de7395",
        "torch": "test",
        "hip": "test",
    }
    monkeypatch.setattr(asset_render, "_distribution_version", lambda _: "test")

    asset_render._validate_runtime_against_asset(bundle, runtime)

    runtime["gsplat_source_revision"] = "c" * 40
    with pytest.raises(ContractError, match="dependency identity mismatch"):
        asset_render._validate_runtime_against_asset(bundle, runtime)


def test_camera_trajectory_is_bound_only_after_asset_publication(tmp_path: Path) -> None:
    asset = tmp_path / "asset"
    write_asset_bundle(
        asset,
        model=_model(),
        spec=_spec(),
        renderer=RendererConfig(require_gfx942=False),
    )
    bundle = load_asset_bundle(asset)
    trajectory_file = tmp_path / "trajectory.json"
    _write_payload(trajectory_file, _trajectory_payload())
    output = tmp_path / "camera-path.json"

    result = asset_render.bind_camera_trajectory(
        asset,
        trajectory_file=trajectory_file,
        output=output,
    )

    bound = asset_render.load_camera_path(output)
    assert bound.asset_bundle_id == bundle.manifest["bundle_id"]
    assert [frame.timestamp_seconds for frame in bound.frames] == [0.0, 1.0]
    assert result == {
        "schema_version": "p2g.camera_path_binding.v1",
        "status": "PASS",
        "asset_bundle_id": bundle.manifest["bundle_id"],
        "asset_model_sha256": bundle.metadata["model"]["sha256"],
        "trajectory_sha256": sha256_file(trajectory_file),
        "camera_path_sha256": sha256_file(output),
        "frame_count": 2,
        "fps": 30,
        "resolution": [3, 3],
        "time_interval_seconds": [0.0, 1.0],
        "claim_boundary": (
            "The new camera path is bound to this exact AssetBundle and admitted time "
            "interval; no render, visual-quality, or redistribution claim was made."
        ),
    }


def test_camera_trajectory_binding_rejects_invalid_publication(tmp_path: Path) -> None:
    asset = tmp_path / "asset"
    write_asset_bundle(
        asset,
        model=_model(),
        spec=_spec(),
        renderer=RendererConfig(require_gfx942=False),
    )
    trajectory = _trajectory_payload()
    trajectory["frames"][1]["timestamp_seconds"] = 1.5
    trajectory_file = tmp_path / "outside-time.json"
    _write_payload(trajectory_file, trajectory)
    output = tmp_path / "camera-path.json"

    with pytest.raises(ContractError, match="outside asset interval"):
        asset_render.bind_camera_trajectory(
            asset,
            trajectory_file=trajectory_file,
            output=output,
        )
    assert not output.exists()

    valid_file = tmp_path / "valid-trajectory.json"
    _write_payload(valid_file, _trajectory_payload())
    with pytest.raises(ContractError, match="outside the AssetBundle"):
        asset_render.bind_camera_trajectory(
            asset,
            trajectory_file=valid_file,
            output=asset / "camera-path.json",
        )


def test_camera_path_bind_cli_publishes_bound_path(tmp_path: Path) -> None:
    from p2g.cli import app

    asset = tmp_path / "asset"
    write_asset_bundle(
        asset,
        model=_model(),
        spec=_spec(),
        renderer=RendererConfig(require_gfx942=False),
    )
    trajectory_file = tmp_path / "trajectory.json"
    _write_payload(trajectory_file, _trajectory_payload())
    output = tmp_path / "bound.json"

    invoked = CliRunner().invoke(
        app,
        [
            "camera-path",
            "bind",
            str(asset),
            "--trajectory",
            str(trajectory_file),
            "--output",
            str(output),
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    report = json.loads(invoked.stdout)
    assert report["camera_path_sha256"] == sha256_file(output)
    assert asset_render.load_camera_path(output).asset_bundle_id == report["asset_bundle_id"]


def test_render_video_cli_consumes_only_asset_and_camera_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from p2g.cli import app

    destination = tmp_path / "asset"
    write_asset_bundle(
        destination,
        model=_model(),
        spec=_spec(),
        renderer=RendererConfig(require_gfx942=False),
    )
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    camera_path = tmp_path / "camera-path.json"
    _write_payload(camera_path, _camera_payload(manifest["bundle_id"]))

    class FakeRenderer:
        def __init__(self, config: RendererConfig) -> None:
            assert config.backend == "gsplat_rocm"

        def validate_environment(self, device: torch.device) -> dict[str, str]:
            assert device.type == "cpu"
            return {
                "torch": "test",
                "hip": "test",
                "gsplat_distribution": "1.5.3+b01acd43",
                "gsplat_module": "1.5.3",
                "gsplat_distribution_name": "amd-gsplat",
                "gsplat_source_revision": "b01acd43e3c7fa942f95fda0974e9125e4de7395",
                "renderer_abi": "p2g.gsplat_rocm.v1",
                "python_abi": "cp312",
                "visible_device_count": 1,
                "device": "test-device",
                "architecture": "gfx942",
                "backend": "gsplat_rocm",
            }

        def render(
            self,
            model: DynamicGaussianModel,
            batch: Any,
            *,
            sh_degree: int,
        ) -> SimpleNamespace:
            assert model.count == 3
            assert sh_degree == 3
            return SimpleNamespace(image=torch.zeros_like(batch.rgb))

    def fake_which(name: str) -> str:
        return f"/fake/{name}"

    def fake_subprocess_run(
        arguments: list[str],
        **_: Any,
    ) -> subprocess.CompletedProcess[str]:
        executable = Path(arguments[0]).name
        if arguments[1:] == ["-version"]:
            return subprocess.CompletedProcess(arguments, 0, f"{executable} version test\n", "")
        if executable == "ffprobe":
            probe = {
                "streams": [
                    {
                        "codec_name": "h264",
                        "width": 4,
                        "height": 4,
                        "pix_fmt": "yuv420p",
                        "avg_frame_rate": "30/1",
                        "nb_read_frames": "2",
                    }
                ]
            }
            return subprocess.CompletedProcess(arguments, 0, json.dumps(probe), "")
        Path(arguments[-1]).write_bytes(b"synthetic-h264")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def fake_write_render(path: Path, image: torch.Tensor, *, photometric_space: str) -> None:
        assert image.shape == (3, 3, 3)
        assert photometric_space == "linear_rgb"
        path.write_bytes(b"synthetic-png")

    monkeypatch.setattr(asset_render, "GsplatRenderer", FakeRenderer)
    monkeypatch.setattr(asset_render.shutil, "which", fake_which)
    monkeypatch.setattr(asset_render.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(asset_render, "write_render", fake_write_render)
    monkeypatch.setattr(
        asset_render,
        "_distribution_version",
        lambda name: {"safetensors": "test", "imageio": "test"}[name],
    )

    output = tmp_path / "preview.mp4"
    runner = CliRunner()
    invoked = runner.invoke(
        app,
        [
            "render-video",
            str(destination),
            "--camera-path",
            str(camera_path),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    assert invoked.exit_code == 0, invoked.output
    receipt = output.with_suffix(".render.json")
    report = json.loads(receipt.read_text(encoding="utf-8"))
    assert output.read_bytes() == b"synthetic-h264"
    assert report["asset_bundle_id"] == manifest["bundle_id"]
    assert report["camera_path_sha256"] == sha256_file(camera_path)
    assert report["encoder"]["verified_frame_count"] == 2
    assert report["consumer"]["renderer_abi"] == "p2g.asset_video_renderer.v1"
    assert report["rights"]["redistribution"] == "restricted"
    assert str(tmp_path) not in receipt.read_text(encoding="utf-8")
