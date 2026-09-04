from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from p2g.errors import ContractError

ROOT = Path(__file__).parents[1]
PROPOSAL_TOOL = ROOT / "tools/build_roma_point_proposals.py"
SEQUENCE_TOOL = ROOT / "tools/build_roma_point_sequence.py"


def _load_tool(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_paths(tmp_path: Path) -> list[str]:
    return [
        "--tensor-cache",
        str(tmp_path / "cache"),
        "--observation-manifest",
        str(tmp_path / "observations.json"),
        "--roma-indoor-weight",
        str(tmp_path / "roma.pth"),
        "--dinov2-weight",
        str(tmp_path / "dino.pth"),
        "--environment-lock",
        str(tmp_path / "uv.lock"),
        "--output",
        str(tmp_path / "output"),
    ]


@pytest.mark.parametrize("tool", [PROPOSAL_TOOL, SEQUENCE_TOOL])
def test_help_uses_public_names_and_never_imports_torch(tool: Path) -> None:
    source_root = ROOT / "src"
    program = (
        "import importlib.util,json,sys;"
        "sys.meta_path[:]=[f for f in sys.meta_path "
        "if getattr(f,'__module__','')!='_pixel4dgs_editable'];"
        f"sys.path.insert(0,{str(source_root)!r});"
        f"p={str(tool)!r};"
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
    assert "tensor_cache" in result["options"]
    assert "observation_manifest" in result["options"]
    assert "roma_indoor_weight" in result["options"]
    assert "memmap_root" not in result["options"]


def test_single_frame_tool_dispatches_exact_policy_and_keeps_stdout_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_tool(PROPOSAL_TOOL, "test_public_proposal_tool")
    observed: dict[str, Any] = {}

    def fake_builder(output: Path, **kwargs: Any) -> dict[str, Any]:
        observed.update({"output": output, **kwargs})
        print("provider progress")
        return {"status": "COMPLETE", "frame": {"frame_id": kwargs["frame_id"]}}

    monkeypatch.setattr(module, "build_roma_point_proposals", fake_builder)
    result = module.main(
        [
            *_required_paths(tmp_path),
            "--frame-id",
            "7",
            "--points-per-frame",
            "42",
            "--nearest-cameras",
            "1",
            "--seed",
            "9",
            "--world-bound",
            "25.5",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert json.loads(captured.out) == {"status": "COMPLETE", "frame": {"frame_id": 7}}
    assert captured.err == "provider progress\n"
    assert observed == {
        "output": (tmp_path / "output").resolve(),
        "memmap_root": (tmp_path / "cache").resolve(),
        "observation_manifest": (tmp_path / "observations.json").resolve(),
        "frame_id": 7,
        "roma_weight": (tmp_path / "roma.pth").resolve(),
        "dino_weight": (tmp_path / "dino.pth").resolve(),
        "environment_lock": (tmp_path / "uv.lock").resolve(),
        "num_points_per_frame": 42,
        "nearest_cameras": 1,
        "seed": 9,
        "world_bound": 25.5,
    }


def test_sequence_tool_uses_explicit_half_open_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_tool(SEQUENCE_TOOL, "test_public_sequence_tool")
    observed: dict[str, Any] = {}

    def fake_builder(output: Path, **kwargs: Any) -> dict[str, Any]:
        observed.update({"output": output, **kwargs})
        print("FRAME [1/3] 000004 built")
        return {"status": "COMPLETE", "frame_ids": list(kwargs["frame_ids"])}

    monkeypatch.setattr(module, "build_roma_point_sequence", fake_builder)
    result = module.main(
        [
            *_required_paths(tmp_path),
            "--frame-start",
            "4",
            "--frame-stop-exclusive",
            "7",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert json.loads(captured.out) == {"status": "COMPLETE", "frame_ids": [4, 5, 6]}
    assert captured.err == "FRAME [1/3] 000004 built\n"
    assert observed["frame_ids"] == (4, 5, 6)
    assert observed["num_points_per_frame"] == 700_000
    assert observed["nearest_cameras"] == 2


def test_sequence_tool_rejects_empty_or_reversed_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_tool(SEQUENCE_TOOL, "test_public_sequence_invalid")
    with pytest.raises(SystemExit) as raised:
        module.main(
            [
                *_required_paths(tmp_path),
                "--frame-start",
                "5",
                "--frame-stop-exclusive",
                "5",
            ]
        )
    assert raised.value.code == 2
    assert "must be greater than" in capsys.readouterr().err


def test_expected_contract_failure_is_concise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_tool(PROPOSAL_TOOL, "test_public_proposal_failure")

    def fail(*_: Any, **__: Any) -> dict[str, Any]:
        raise ContractError("registered weight digest differs")

    monkeypatch.setattr(module, "build_roma_point_proposals", fail)
    result = module.main([*_required_paths(tmp_path), "--frame-id", "0"])
    captured = capsys.readouterr()

    assert result == 2
    assert captured.out == ""
    assert captured.err == "RoMa proposal build failed: registered weight digest differs\n"


def test_tool_sources_exclude_private_adapters_and_downloaders() -> None:
    combined = (PROPOSAL_TOOL.read_text() + SEQUENCE_TOOL.read_text()).casefold()
    forbidden = (
        "torch.hub",
        "urlretrieve",
        "requests.get",
        "hair_train",
        "freetime",
        "/mnt/",
        "/home/",
    )
    assert not any(token in combined for token in forbidden)
