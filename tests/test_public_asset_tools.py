# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from p2g.canonical import sha256_file, sha256_json
from p2g.errors import ContractError
from p2g.schema import validate_payload

ROOT = Path(__file__).parents[1]
RENDER_TOOL = ROOT / "tools/render_moving_camera_video.py"
VERIFY_TOOL = ROOT / "tools/verify_asset_bundle.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tool", [RENDER_TOOL, VERIFY_TOOL])
def test_asset_tool_import_does_not_load_torch(tool: Path) -> None:
    program = f"""
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location("standalone_asset_tool", {str(tool)!r})
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({{"torch_loaded": "torch" in sys.modules}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"torch_loaded": False}


def test_render_tool_dispatches_only_asset_and_camera_path_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load(RENDER_TOOL, "test_public_render_tool")
    observed: dict[str, Any] = {}

    def render(asset: Path, **kwargs: Any) -> dict[str, Any]:
        observed.update({"asset": asset, **kwargs})
        print("rendered 2/2")
        return {"schema_version": "p2g.asset_video_render.v1", "status": "PASS"}

    monkeypatch.setattr(module, "render_asset_video", render)
    result = module.main(
        [
            "--asset",
            str(tmp_path / "asset"),
            "--camera-path",
            str(tmp_path / "camera.json"),
            "--output",
            str(tmp_path / "preview.mp4"),
            "--receipt",
            str(tmp_path / "preview.json"),
            "--crf",
            "21",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert json.loads(captured.out) == {
        "schema_version": "p2g.asset_video_render.v1",
        "status": "PASS",
    }
    assert captured.err == "rendered 2/2\n"
    assert observed == {
        "asset": (tmp_path / "asset").resolve(),
        "camera_path_file": (tmp_path / "camera.json").resolve(),
        "output": (tmp_path / "preview.mp4").resolve(),
        "receipt": (tmp_path / "preview.json").resolve(),
        "device": "cuda",
        "crf": 21,
    }


@pytest.mark.parametrize("crf", ["-1", "52", "nan"])
def test_render_tool_rejects_invalid_encoding_policy(
    tmp_path: Path, crf: str, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load(RENDER_TOOL, f"test_public_render_invalid_{crf}")
    with pytest.raises(SystemExit) as raised:
        module.main(
            [
                "--asset",
                str(tmp_path / "asset"),
                "--camera-path",
                str(tmp_path / "camera.json"),
                "--output",
                str(tmp_path / "preview.mp4"),
                "--crf",
                crf,
            ]
        )
    assert raised.value.code == 2
    assert "CRF" in capsys.readouterr().err


def _fake_asset(root: Path) -> SimpleNamespace:
    root.mkdir()
    (root / "manifest.json").write_bytes(b"manifest\n")
    (root / "asset.json").write_bytes(b"metadata\n")
    (root / "model.safetensors").write_bytes(b"model\n")
    return SimpleNamespace(root=root)


def _summary(*, model_sha256: str = "b" * 64) -> dict[str, Any]:
    return {
        "schema_version": "p2g.asset_bundle.v1",
        "bundle_id": "a" * 64,
        "gaussian_count": 3,
        "tensor_count": 10,
        "model_sha256": model_sha256,
        "equation_version": "p2g.linear_motion_gaussian_gate.v1",
        "rights": {"redistribution": "restricted"},
        "status": "PASS",
    }


def test_asset_inspect_prints_summary_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load(VERIFY_TOOL, "test_public_asset_inspect")
    bundle = _fake_asset(tmp_path / "asset")

    def load(_: Path) -> Any:
        return bundle

    def summarize(_: Any) -> dict[str, Any]:
        return _summary()

    monkeypatch.setattr(module, "load_asset_bundle", load)
    monkeypatch.setattr(module, "asset_summary", summarize)

    result = module.main(["inspect", str(bundle.root)])

    assert result == 0
    assert json.loads(capsys.readouterr().out) == _summary()
    assert set(tmp_path.iterdir()) == {bundle.root}


def test_asset_verify_publishes_path_free_hash_closed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load(VERIFY_TOOL, "test_public_asset_verify")
    bundle = _fake_asset(tmp_path / "asset")
    model_sha256 = sha256_file(bundle.root / "model.safetensors")

    def load(_: Path) -> Any:
        return bundle

    def summarize(_: Any) -> dict[str, Any]:
        return _summary(model_sha256=model_sha256)

    monkeypatch.setattr(module, "load_asset_bundle", load)
    monkeypatch.setattr(module, "asset_summary", summarize)
    output = tmp_path / "verification.json"

    result = module.main(["verify", str(bundle.root), "--output", str(output)])
    captured = capsys.readouterr()
    receipt = json.loads(output.read_text())

    assert result == 0
    assert json.loads(captured.out) == receipt
    validate_payload("asset_verification", receipt)
    assert receipt["asset"] == {
        "schema_version": "p2g.asset_bundle.v1",
        "bundle_id": "a" * 64,
        "gaussian_count": 3,
        "tensor_count": 10,
        "model_sha256": model_sha256,
        "equation_version": "p2g.linear_motion_gaussian_gate.v1",
        "redistribution": "restricted",
    }
    assert receipt["files"] == {
        "manifest.json": sha256_file(bundle.root / "manifest.json"),
        "asset.json": sha256_file(bundle.root / "asset.json"),
        "model.safetensors": sha256_file(bundle.root / "model.safetensors"),
    }
    assert receipt["logical_sha256"] == sha256_json(
        {key: value for key, value in receipt.items() if key != "logical_sha256"}
    )
    assert str(tmp_path) not in output.read_text()


def test_verify_refuses_to_mutate_the_asset_or_overwrite_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load(VERIFY_TOOL, "test_public_asset_verify_boundary")
    bundle = _fake_asset(tmp_path / "asset")
    model_sha256 = sha256_file(bundle.root / "model.safetensors")

    def load(_: Path) -> Any:
        return bundle

    def summarize(_: Any) -> dict[str, Any]:
        return _summary(model_sha256=model_sha256)

    monkeypatch.setattr(module, "load_asset_bundle", load)
    monkeypatch.setattr(module, "asset_summary", summarize)

    inside = bundle.root / "verification.json"
    assert module.main(["verify", str(bundle.root), "--output", str(inside)]) == 2
    assert not inside.exists()
    assert "outside the AssetBundle" in capsys.readouterr().err

    outside = tmp_path / "verification.json"
    outside.write_text("owned by caller\n")
    assert module.main(["verify", str(bundle.root), "--output", str(outside)]) == 2
    assert outside.read_text() == "owned by caller\n"


def test_expected_asset_error_is_concise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load(VERIFY_TOOL, "test_public_asset_failure")

    def fail(_: Path) -> Any:
        raise ContractError("model digest differs")

    monkeypatch.setattr(module, "load_asset_bundle", fail)
    result = module.main(["inspect", str(tmp_path / "asset")])
    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "asset inspect failed: model digest differs\n"


def test_tools_have_no_training_run_or_private_adapter_inputs() -> None:
    sources = (RENDER_TOOL.read_text() + VERIFY_TOOL.read_text()).casefold()
    forbidden = (
        "runconfig",
        "preparedscene",
        "latest_checkpoint",
        "read_checkpoint",
        "run_dir",
        "freetime",
        "/mnt/",
        "/home/",
    )
    assert not any(token in sources for token in forbidden)
