from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from p2g.cli import _stopped_process_specs, app
from p2g.errors import ContractError

ROOT = Path(__file__).parents[1]


def test_public_cli_exposes_only_the_supported_stage_and_auxiliary_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
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
        assert command in result.output
    for hidden_command in ("debug-only", "experimental-compat", "legacy-run"):
        assert hidden_command not in result.output.casefold()

    asset = runner.invoke(app, ["asset", "--help"])
    assert asset.exit_code == 0, asset.output
    assert "export" in asset.output
    assert "inspect" in asset.output
    assert "verify" in asset.output

    fixture = runner.invoke(app, ["fixture", "--help"])
    assert fixture.exit_code == 0, fixture.output
    assert "create" in fixture.output

    data = runner.invoke(app, ["data", "--help"])
    assert data.exit_code == 0, data.output
    assert "import-charge" in data.output
    assert "import-selfcap" in data.output

    camera_path = runner.invoke(app, ["camera-path", "--help"])
    assert camera_path.exit_code == 0, camera_path.output
    assert "bind" in camera_path.output

    camera_bind = runner.invoke(app, ["camera-path", "bind", "--help"], terminal_width=180)
    assert camera_bind.exit_code == 0, camera_bind.output
    assert "--trajectory" in camera_bind.output
    assert "--output" in camera_bind.output

    charge = runner.invoke(app, ["data", "import-charge", "--help"], terminal_width=180)
    assert charge.exit_code == 0, charge.output
    assert "--source-revision" in charge.output
    assert "--sealed-camera-count" in charge.output

    selfcap = runner.invoke(app, ["data", "import-selfcap", "--help"], terminal_width=180)
    assert selfcap.exit_code == 0, selfcap.output
    assert "--source-start-frame" in selfcap.output
    assert "--diagnostic-camera" in selfcap.output
    assert "--sealed-camera" in selfcap.output

    pipeline = runner.invoke(app, ["run", "--help"], terminal_width=180)
    assert pipeline.exit_code == 0, pipeline.output
    assert "--workspace" in pipeline.output
    assert "--stop-after" in pipeline.output

    doctor = runner.invoke(app, ["doctor", "--help"], terminal_width=240)
    assert doctor.exit_code == 0, doctor.output
    assert "--admission-mode" in doctor.output
    assert "--allow-stopped-proce" in doctor.output

    propose = runner.invoke(app, ["propose", "--help"], terminal_width=180)
    assert propose.exit_code == 0, propose.output
    assert "--observation-manife" in propose.output

    train = runner.invoke(app, ["train", "--help"], terminal_width=180)
    assert train.exit_code == 0, train.output
    assert "--resume-checkpoint" in train.output

    sealed = runner.invoke(app, ["evaluate-sealed", "--help"], terminal_width=180)
    assert sealed.exit_code == 0, sealed.output
    assert "--gate" in sealed.output
    assert "--output" in sealed.output

    sealed_verify = runner.invoke(app, ["verify-sealed", "--help"], terminal_width=180)
    assert sealed_verify.exit_code == 0, sealed_verify.output
    assert "--run-dir" in sealed_verify.output
    assert "--gate" in sealed_verify.output
    assert "--expected-receipt-id" in sealed_verify.output

    asset_export = runner.invoke(app, ["asset", "export", "--help"], terminal_width=180)
    assert asset_export.exit_code == 0, asset_export.output
    assert "--producer-git-revis" in asset_export.output
    assert "--redistribution" in asset_export.output


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
