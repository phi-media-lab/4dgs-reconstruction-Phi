from __future__ import annotations

from typing import Any

import pytest

from p2g.errors import ContractError
from p2g.gpu_preflight import (
    Mi300xPreflightConfig,
    StoppedProcessIdentity,
    evaluate_mi300x_preflight,
    mi300x_preflight_config_from_dict,
)


def _process_payload(*pids: int) -> list[dict[str, Any]]:
    return [
        {
            "gpu": 0,
            "process_list": [
                {
                    "process_info": {
                        "pid": pid,
                        "name": f"worker-{pid}",
                        "memory_usage": {"vram_mem": {"value": pid * 1024, "unit": "B"}},
                    }
                }
                for pid in pids
            ],
        }
    ]


def _metrics(*, gpu: float = 0.0, vram: float = 0.0) -> str:
    return f"device,GPU use (%),GPU Memory Allocated (VRAM%)\ncard0,{gpu},{vram}\n"


def _procfs(pid: int, *, state: str = "T", starttime_ticks: int | None = None) -> dict[str, Any]:
    return {
        "pid": pid,
        "observed": True,
        "uid": 1000,
        "ppid": 1,
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


def test_config_is_canonical_and_rejects_ambient_policy() -> None:
    allowed = (StoppedProcessIdentity(7, 70), StoppedProcessIdentity(11, 110))
    config = Mi300xPreflightConfig(allowed_stopped_processes=allowed)
    payload = config.to_dict()

    assert mi300x_preflight_config_from_dict(payload) == config
    assert config.config_id() == Mi300xPreflightConfig(
        allowed_stopped_processes=allowed
    ).config_id()
    assert config.config_id().startswith("gpupreflight_")

    with pytest.raises(ContractError, match="ascending"):
        Mi300xPreflightConfig(
            allowed_stopped_processes=(
                StoppedProcessIdentity(11, 110),
                StoppedProcessIdentity(7, 70),
            )
        ).validate()
    with pytest.raises(ContractError, match="cannot allow"):
        Mi300xPreflightConfig(
            admission_mode="exclusive_performance",
            allowed_stopped_processes=allowed,
        ).validate()
    with pytest.raises(ContractError, match="ambient"):
        mi300x_preflight_config_from_dict({**payload, "implicit_override": True})
    with pytest.raises(ContractError, match=r"inside \[0, 100\]"):
        Mi300xPreflightConfig(maximum_gpu_use_percent=True).validate()


def test_clean_snapshot_passes_with_an_explicit_claim_boundary() -> None:
    result = evaluate_mi300x_preflight(
        process_payload=_process_payload(),
        metric_csv=_metrics(),
    )

    assert result["status"] == "PASS"
    assert result["reasons"] == []
    assert result["warnings"] == []
    assert result["contention_observed"] is False
    assert result["process_count"] == 0
    assert "not a scheduler reservation" in result["claim_boundary"]
    assert "hardware-identity proof" in result["claim_boundary"]


def test_capacity_reasons_and_shared_contention_are_independent() -> None:
    result = evaluate_mi300x_preflight(
        process_payload=_process_payload(101),
        metric_csv=_metrics(gpu=90.0, vram=8.0),
        kfd_fuser_stdout="101c 202\n",
        config=Mi300xPreflightConfig(
            maximum_gpu_use_percent=5.0,
            maximum_vram_percent=1.0,
        ),
    )

    assert result["status"] == "BUSY"
    assert result["reasons"] == ["GPU_USE_ABOVE_LIMIT", "VRAM_ABOVE_LIMIT"]
    assert result["warnings"] == ["FOREIGN_GPU_PROCESSES_PRESENT"]
    assert result["contention_observed"] is True
    assert result["process_count"] == 2
    assert result["processes"][0]["sources"] == ["amd_smi", "dev_kfd"]
    assert result["processes"][1]["sources"] == ["dev_kfd"]


def test_exact_stopped_processes_are_not_foreign_but_metrics_still_apply() -> None:
    config = Mi300xPreflightConfig(
        maximum_gpu_use_percent=5.0,
        allowed_stopped_processes=(
            StoppedProcessIdentity(101, 1010),
            StoppedProcessIdentity(202, 2020),
        )
    )
    clean = evaluate_mi300x_preflight(
        process_payload=_process_payload(101),
        metric_csv=_metrics(vram=1.0),
        kfd_fuser_stdout="101 202",
        process_metadata=[_procfs(101), _procfs(202)],
        config=config,
    )
    busy = evaluate_mi300x_preflight(
        process_payload=_process_payload(101),
        metric_csv=_metrics(gpu=6.0),
        kfd_fuser_stdout="101 202",
        process_metadata=[_procfs(101), _procfs(202)],
        config=config,
    )

    assert clean["status"] == "PASS"
    assert clean["foreign_process_count"] == 0
    assert clean["allowance_mismatch_count"] == 0
    assert all(
        row["admission"]["classification"] == "allowed_stopped"
        for row in clean["processes"]
    )
    assert busy["reasons"] == ["GPU_USE_ABOVE_LIMIT"]


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        [_procfs(101, state="R")],
        [_procfs(101, starttime_ticks=999)],
    ],
)
def test_allowed_pid_mismatch_is_recorded_as_shared_contention(
    metadata: list[dict[str, Any]],
) -> None:
    result = evaluate_mi300x_preflight(
        process_payload=_process_payload(101),
        metric_csv=_metrics(),
        process_metadata=metadata,
        config=Mi300xPreflightConfig(
            allowed_stopped_processes=(StoppedProcessIdentity(101, 1010),)
        ),
    )

    assert result["status"] == "PASS"
    assert result["reasons"] == []
    assert result["warnings"] == ["STOPPED_PROCESS_ALLOWANCE_MISMATCH"]
    assert result["contention_observed"] is True
    assert result["allowance_mismatch_count"] == 1
    assert result["processes"][0]["admission"]["classification"] == (
        "allowance_identity_mismatch"
    )


def test_exclusive_mode_rejects_foreign_processes() -> None:
    result = evaluate_mi300x_preflight(
        process_payload=_process_payload(101),
        metric_csv=_metrics(),
        process_metadata=[_procfs(101, state="R")],
        config=Mi300xPreflightConfig(admission_mode="exclusive_performance"),
    )

    assert result["status"] == "BUSY"
    assert result["reasons"] == ["FOREIGN_GPU_PROCESSES_PRESENT"]
    assert result["warnings"] == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"process_payload": {}}, "JSON array"),
        ({"process_payload": _process_payload(4, 4)}, "unique positive"),
        ({"kfd_fuser_stdout": "not-a-pid"}, "invalid PID token"),
        ({"kfd_fuser_stdout": "5 5"}, "unique and positive"),
        ({"metric_csv": _metrics(gpu=101.0)}, r"inside \[0, 100\]"),
        ({"metric_csv": _metrics() + "card0,0,0\n"}, "exactly one"),
    ],
)
def test_malformed_evidence_fails_closed(kwargs: dict[str, Any], message: str) -> None:
    arguments: dict[str, Any] = {
        "process_payload": _process_payload(),
        "metric_csv": _metrics(),
    }
    arguments.update(kwargs)
    with pytest.raises(ContractError, match=message):
        evaluate_mi300x_preflight(**arguments)
