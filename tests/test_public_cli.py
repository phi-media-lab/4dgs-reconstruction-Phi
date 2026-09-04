from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rich.text import Text
from typer.testing import CliRunner

from p2g.cli import _stopped_process_specs, app
from p2g.errors import ContractError

ROOT = Path(__file__).parents[1]


def _help_output(
    runner: CliRunner,
    *command: str,
    terminal_width: int | None = None,
) -> str:
    arguments = [*command, "--help"]
    if terminal_width is None:
        result = runner.invoke(app, arguments)
    else:
        result = runner.invoke(app, arguments, terminal_width=terminal_width)
    assert result.exit_code == 0, result.output
    return Text.from_ansi(result.output).plain


def test_public_cli_exposes_only_the_supported_stage_and_auxiliary_commands() -> None:
    runner = CliRunner()
    output = _help_output(runner)

    for command in (
        "doctor",
        "run",
        "status",
        "prepare",
        "propose",
        "initialize",
        "train",
        "evaluate",
        "evaluate-sealed",
        "verify-sealed",
        "camera-path",
        "render-video",
        "asset",
        "data",
        "fixture",
    ):
        assert command in output
    for hidden_command in ("debug-only", "experimental-compat", "legacy-run"):
        assert hidden_command not in output.casefold()

    asset = _help_output(runner, "asset")
    assert "export" in asset
    assert "inspect" in asset
    assert "verify" in asset

    fixture = _help_output(runner, "fixture")
    assert "create" in fixture

    data = _help_output(runner, "data")
    assert "import-charge" in data
    assert "import-selfcap" in data

    camera_path = _help_output(runner, "camera-path")
    assert "bind" in camera_path

    camera_bind = _help_output(runner, "camera-path", "bind", terminal_width=180)
    assert "--trajectory" in camera_bind
    assert "--output" in camera_bind

    charge = _help_output(runner, "data", "import-charge", terminal_width=180)
    assert "--source-revision" in charge
    assert "--sealed-camera-count" in charge

    selfcap = _help_output(runner, "data", "import-selfcap", terminal_width=180)
    assert "--source-start-frame" in selfcap
    assert "--diagnostic-camera" in selfcap
    assert "--sealed-camera" in selfcap

    pipeline = _help_output(runner, "run", terminal_width=180)
    assert "--workspace" in pipeline
    assert "--stop-after" in pipeline

    doctor = _help_output(runner, "doctor", terminal_width=240)
    assert "--admission-mode" in doctor
    assert "--allow-stopped-proce" in doctor

    propose = _help_output(runner, "propose", terminal_width=180)
    assert "--observation-manife" in propose

    train = _help_output(runner, "train", terminal_width=180)
    assert "--resume-checkpoint" in train

    sealed = _help_output(runner, "evaluate-sealed", terminal_width=180)
    assert "--gate" in sealed
    assert "--output" in sealed

    sealed_verify = _help_output(runner, "verify-sealed", terminal_width=180)
    assert "--run-dir" in sealed_verify
    assert "--gate" in sealed_verify
    assert "--expected-receipt-id" in sealed_verify

    asset_export = _help_output(runner, "asset", "export", terminal_width=180)
    assert "--producer-git-revis" in asset_export
    assert "--redistribution" in asset_export


def test_module_help_is_lazy_and_does_not_import_torch() -> None:
    program = """
import json
import runpy
import sys
sys.argv = ["p2g", "--help"]
try:
    runpy.run_module("p2g", run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
print(json.dumps({"torch_loaded": "torch" in sys.modules}))
"""
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "HIP_VISIBLE_DEVICES": "-1",
            "PYTHONPATH": str(ROOT / "src"),
            "ROCR_VISIBLE_DEVICES": "-1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    final_line = completed.stdout.splitlines()[-1]
    assert json.loads(final_line) == {"torch_loaded": False}


def test_public_cli_source_has_no_internal_control_plane_imports() -> None:
    source = (ROOT / "src/p2g/cli.py").read_text(encoding="utf-8").casefold()
    forbidden = (
        "p2g.evidence",
        "p2g.ingest",
        "raster_v0",
        "raster_v1",
        "ufm_",
        "freetime",
        "/home/",
        "/mnt/",
    )
    assert not any(token in source for token in forbidden)


def test_stopped_process_cli_specs_bind_pid_and_start_time() -> None:
    assert _stopped_process_specs(["11:110", "7:70"]) == ((7, 70), (11, 110))
    with pytest.raises(ContractError, match="PID:STARTTIME_TICKS"):
        _stopped_process_specs(["11"])
    with pytest.raises(ContractError, match="duplicate PID"):
        _stopped_process_specs(["11:110", "11:111"])
