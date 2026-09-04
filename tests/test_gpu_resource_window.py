from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from p2g.errors import ContractError
from p2g.gpu_preflight import ProcessIdentity, StoppedProcessIdentity
from p2g.gpu_resource_window import (
    Mi300xResourceWindowConfig,
    evaluate_kfd_observation,
    run_in_mi300x_resource_window,
)
from p2g.schema import validate_payload


def _row(
    pid: int,
    *,
    ppid: int = 1,
    state: str = "T",
    starttime_ticks: int | None = None,
) -> dict[str, Any]:
    return {
        "pid": pid,
        "observed": True,
        "uid": 1000,
        "ppid": ppid,
        "state": state,
        "starttime_ticks": pid * 10 if starttime_ticks is None else starttime_ticks,
        "comm": "worker",
        "exe": "python",
        "cwd": None,
        "argv0": "python",
        "argc": 1,
        "cmdline_sha256": "1" * 64,
        "redaction": "argv arguments intentionally omitted; only argv0/hash/argc retained",
    }


def _config() -> Mi300xResourceWindowConfig:
    return Mi300xResourceWindowConfig(
        gpu_index=0,
        admission_mode="shared_quality",
        owner_process=ProcessIdentity(10, 100),
        allowed_stopped_processes=(StoppedProcessIdentity(20, 200),),
        poll_interval_seconds=0.01,
        maximum_observation_gap_seconds=1.0,
        command_timeout_seconds=1,
    )


def _exclusive_config() -> Mi300xResourceWindowConfig:
    return Mi300xResourceWindowConfig(
        gpu_index=0,
        admission_mode="exclusive_performance",
        owner_process=ProcessIdentity(10, 100),
        poll_interval_seconds=0.01,
        maximum_observation_gap_seconds=1.0,
        command_timeout_seconds=1,
    )


def test_observation_admits_only_owner_tree_and_exact_stopped_identity(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "10"
    owner.mkdir()
    (owner / "stat").write_text("10 (owner) S 1 " + "0 " * 17 + "100\n")
    child = tmp_path / "11"
    child.mkdir()
    (child / "stat").write_text("11 (child) R 10 " + "0 " * 17 + "110\n")

    result = evaluate_kfd_observation(
        pids=[10, 11, 20],
        process_metadata=[
            _row(10, state="R", starttime_ticks=100),
            _row(11, ppid=10, state="R", starttime_ticks=110),
            _row(20, starttime_ticks=200),
        ],
        config=_config(),
        proc_root=tmp_path,
    )

    assert result["status"] == "PASS"
    assert result["warnings"] == []
    assert result["contention_observed"] is False
    assert [row["classification"] for row in result["processes"]] == [
        "stage_owner",
        "owned_descendant",
        "allowed_stopped",
    ]


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (_row(20, state="R", starttime_ticks=200), "STOPPED_PROCESS_ALLOWANCE_MISMATCH"),
        (_row(20, state="T", starttime_ticks=201), "STOPPED_PROCESS_ALLOWANCE_MISMATCH"),
        (_row(30, state="R", starttime_ticks=300), "FOREIGN_KFD_PROCESS_ARRIVED"),
    ],
)
def test_shared_observation_records_resumed_reused_and_new_clients_as_contention(
    tmp_path: Path,
    row: dict[str, Any],
    reason: str,
) -> None:
    result = evaluate_kfd_observation(
        pids=[row["pid"]],
        process_metadata=[row],
        config=_config(),
        proc_root=tmp_path,
    )

    assert result["status"] == "PASS"
    assert result["reasons"] == []
    assert result["warnings"] == [reason]
    assert result["contention_observed"] is True


def test_exclusive_observation_rejects_foreign_client(tmp_path: Path) -> None:
    row = _row(30, state="R", starttime_ticks=300)
    result = evaluate_kfd_observation(
        pids=[30],
        process_metadata=[row],
        config=_exclusive_config(),
        proc_root=tmp_path,
    )

    assert result["status"] == "BUSY"
    assert result["reasons"] == ["FOREIGN_KFD_PROCESS_ARRIVED"]
    assert result["warnings"] == []
    assert result["contention_observed"] is True


def test_window_receipt_is_hash_closed_and_records_dynamic_violation(
    tmp_path: Path,
) -> None:
    observations = iter(
        [
            {
                "status": "PASS",
                "reasons": [],
                "warnings": [],
                "contention_observed": False,
                "processes": [],
            },
            {
                "status": "BUSY",
                "reasons": ["FOREIGN_KFD_PROCESS_ARRIVED"],
                "warnings": [],
                "contention_observed": True,
                "processes": [
                    {
                        "pid": 30,
                        "state": "R",
                        "starttime_ticks": 300,
                        "classification": "foreign",
                    }
                ],
            },
        ]
    )
    output = tmp_path / "window.json"

    with pytest.raises(ContractError, match="FOREIGN_KFD_PROCESS_ARRIVED"):
        run_in_mi300x_resource_window(
            lambda: "complete",
            output=output,
            stage="train",
            config=_exclusive_config(),
            sample=lambda: next(observations),
        )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    validate_payload("mi300x_resource_window", receipt)
    assert receipt["status"] == "BUSY"
    assert receipt["operation_status"] == "RETURNED"
    assert receipt["sample_count"] == 2
    assert len(receipt["transitions"]) == 2


def test_shared_window_completes_and_records_contention(tmp_path: Path) -> None:
    output = tmp_path / "window.json"
    observation = {
        "status": "PASS",
        "reasons": [],
        "warnings": ["FOREIGN_KFD_PROCESS_ARRIVED"],
        "contention_observed": True,
        "processes": [
            {
                "pid": 30,
                "state": "R",
                "starttime_ticks": 300,
                "classification": "foreign",
            }
        ],
    }

    assert run_in_mi300x_resource_window(
        lambda: "complete",
        output=output,
        stage="train",
        config=_config(),
        sample=lambda: observation,
    ) == "complete"

    receipt = json.loads(output.read_text(encoding="utf-8"))
    validate_payload("mi300x_resource_window", receipt)
    assert receipt["status"] == "PASS"
    assert receipt["warnings"] == ["FOREIGN_KFD_PROCESS_ARRIVED"]
    assert receipt["contention_observed"] is True


def test_window_preserves_receipt_when_operation_raises(tmp_path: Path) -> None:
    output = tmp_path / "window.json"

    def fail() -> None:
        raise RuntimeError("expected test failure")

    with pytest.raises(RuntimeError, match="expected test failure"):
        run_in_mi300x_resource_window(
            fail,
            output=output,
            stage="evaluate",
            config=_config(),
            sample=lambda: {
                "status": "PASS",
                "reasons": [],
                "warnings": [],
                "contention_observed": False,
                "processes": [],
            },
        )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["operation_status"] == "RAISED"
    assert receipt["sample_count"] == 2
