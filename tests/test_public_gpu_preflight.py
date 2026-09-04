# pyright: reportPrivateUsage=false

from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from p2g.canonical import sha256_file, sha256_json
from p2g.gpu_preflight import (
    _fuser_stderr_is_benign,
    _snapshot_procfs,
    audit_mi300x_preflight,
    capture_mi300x_preflight,
)
from p2g.schema import validate_payload


def _process_payload(*pids: int) -> list[dict[str, Any]]:
    return [
        {
            "gpu": 0,
            "process_list": [
                {
                    "process_info": {
                        "pid": pid,
                        "name": f"worker-{pid}",
                        "memory_usage": {"vram_mem": {"value": 4096, "unit": "B"}},
                    }
                }
                for pid in pids
            ],
        }
    ]


def _metrics(*, gpu: float = 0.0, vram: float = 0.0) -> str:
    return f"device,GPU use (%),GPU Memory Allocated (VRAM%)\ncard0,{gpu},{vram}\n"


def _command_result(
    argv: list[str], *, returncode: int, stdout: str, stderr: str = ""
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    return (
        subprocess.CompletedProcess(argv, returncode, stdout, stderr),
        {
            "argv": argv,
            "returncode": returncode,
            "started_utc": "2026-09-03T00:00:00.000000+00:00",
            "finished_utc": "2026-09-03T00:00:00.100000+00:00",
            "elapsed_seconds": 0.1,
        },
    )


def _capture(monkeypatch: pytest.MonkeyPatch, *, busy: bool = False) -> dict[str, Any]:
    executable_text = shutil.which("true")
    assert executable_text is not None
    executable = Path(executable_text).resolve()

    def resolve(name: str) -> tuple[dict[str, Any], Path]:
        return {
            "name": name,
            "path": str(executable),
            "sha256": sha256_file(executable),
        }, executable

    def run(
        argv: list[str], *, timeout: int
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
        assert timeout == 20
        if "process" in argv:
            return _command_result(
                argv,
                returncode=0,
                stdout=json.dumps(_process_payload(707) if busy else _process_payload()),
            )
        if "/dev/kfd" in argv:
            return _command_result(
                argv,
                returncode=0 if busy else 1,
                stdout="707\n" if busy else "",
                stderr="/dev/kfd:\n",
            )
        return _command_result(argv, returncode=0, stdout=_metrics())

    monkeypatch.setattr("p2g.gpu_preflight._resolve_tool", resolve)
    monkeypatch.setattr("p2g.gpu_preflight._run_observed", run)

    def snapshot(pids: list[int]) -> list[dict[str, Any]]:
        return [] if not pids else [_procfs_row(pids[0])]

    monkeypatch.setattr("p2g.gpu_preflight._snapshot_procfs", snapshot)
    return capture_mi300x_preflight()


def _procfs_row(pid: int) -> dict[str, Any]:
    return {
        "pid": pid,
        "observed": True,
        "uid": 1000,
        "ppid": 1,
        "state": "S",
        "starttime_ticks": 123,
        "comm": "worker",
        "exe": "python",
        "cwd": None,
        "argv0": "python",
        "argc": 2,
        "cmdline_sha256": "1" * 64,
        "redaction": "argv arguments intentionally omitted; only argv0/hash/argc retained",
    }


def test_capture_is_schema_valid_hash_closed_and_replayable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _capture(monkeypatch)

    validate_payload("mi300x_preflight", receipt)
    unsigned = {key: value for key, value in receipt.items() if key != "logical_sha256"}
    assert receipt["logical_sha256"] == sha256_json(unsigned)
    assert receipt["status"] == "PASS"
    report = audit_mi300x_preflight(receipt)
    assert report.status == "PASS", report.to_dict()
    assert all(check.status == "PASS" for check in report.checks)


def test_shared_contention_is_valid_and_does_not_block_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _capture(monkeypatch, busy=True)
    report = audit_mi300x_preflight(receipt)

    assert receipt["status"] == "PASS"
    assert receipt["evaluation"]["warnings"] == ["FOREIGN_GPU_PROCESSES_PRESENT"]
    assert receipt["evaluation"]["contention_observed"] is True
    assert report.status == "PASS", report.to_dict()
    admission = next(check for check in report.checks if check.name == "admission_passed")
    assert admission.required is False
    assert admission.status == "PASS"


def test_audit_detects_logical_raw_and_replay_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _capture(monkeypatch)
    tampered = copy.deepcopy(receipt)
    tampered["raw"]["rocm_smi_metric_csv"] = _metrics(gpu=90.0)

    report = audit_mi300x_preflight(tampered)

    assert report.status == "FAIL"
    failed = {check.name for check in report.checks if check.status == "FAIL" and check.required}
    assert {"logical_identity", "deterministic_replay"} <= failed


def test_tool_identity_can_be_optional_only_for_offline_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _capture(monkeypatch)
    receipt["tools"]["amd_smi"]["sha256"] = "0" * 64
    receipt["logical_sha256"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "logical_sha256"}
    )

    assert audit_mi300x_preflight(receipt).status == "FAIL"
    offline = audit_mi300x_preflight(receipt, require_current_tools=False)
    assert offline.status == "PASS", offline.to_dict()


def test_procfs_snapshot_redacts_arguments_paths_and_working_directory(tmp_path: Path) -> None:
    process = tmp_path / "912"
    process.mkdir()
    tail = ["S", "42", *(["0"] * 17), "1234"]
    (process / "stat").write_text(f"912 (worker name) {' '.join(tail)}\n")
    (process / "status").write_text("Name:\tworker\nUid:\t1001\t1001\t1001\t1001\n")
    (process / "comm").write_text("worker\n")
    (process / "cmdline").write_bytes(b"/opt/private/python\0--token\0secret-value\0")
    os.symlink("/opt/private/python", process / "exe")
    os.symlink("/private/project", process / "cwd")

    [row] = _snapshot_procfs([912], proc_root=tmp_path)

    assert row["uid"] == 1001
    assert row["ppid"] == 42
    assert row["starttime_ticks"] == 1234
    assert row["argv0"] == "python"
    assert row["exe"] == "python"
    assert row["cwd"] is None
    assert row["argc"] == 3
    assert "secret" not in json.dumps(row)


def test_fuser_permission_diagnostic_cannot_be_mistaken_for_an_idle_device() -> None:
    assert _fuser_stderr_is_benign("")
    assert _fuser_stderr_is_benign("/dev/kfd:\n")
    assert _fuser_stderr_is_benign("/dev/kfd:           mm\n")
    assert _fuser_stderr_is_benign("/dev/kfd:  c  e  f  F  r  m\n")
    assert not _fuser_stderr_is_benign("/dev/kfd: Permission denied\n")
    assert not _fuser_stderr_is_benign("/dev/kfd: m; ignored diagnostic\n")


def test_import_has_no_torch_or_gpu_runtime_side_effect() -> None:
    source_root = Path(__file__).parents[1] / "src"
    program = (
        "import json,sys;"
        "sys.meta_path[:]=[f for f in sys.meta_path "
        "if getattr(f,'__module__','')!='_pixel4dgs_editable'];"
        f"sys.path.insert(0,{str(source_root)!r});"
        "import p2g.gpu_preflight;"
        "print(json.dumps({'torch':'torch' in sys.modules,'hip':'hip' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"torch": False, "hip": False}


def test_source_has_no_machine_specific_or_training_adapter_dependency() -> None:
    source = (Path(__file__).parents[1] / "src/p2g/gpu_preflight.py").read_text()
    assert not any(
        token in source.casefold() for token in ("/home/", "/mnt/", "freetime", "hair_train")
    )
    imports = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imports
