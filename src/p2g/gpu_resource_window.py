# pyright: reportPrivateUsage=false

"""Continuous, CPU-only `/dev/kfd` guard for one MI300X pipeline stage."""

from __future__ import annotations

import math
import platform
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from p2g.canonical import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    write_new_bytes,
)
from p2g.errors import ContractError
from p2g.gpu_preflight import (
    AdmissionMode,
    ProcessIdentity,
    StoppedProcessIdentity,
    _fuser_stderr_is_benign,
    _parse_kfd_pids,
    _resolve_tool,
    _run_observed,
    _snapshot_procfs,
)
from p2g.schema import validate_payload

RESOURCE_WINDOW_OPERATOR = "mi300x_kfd_resource_window_guard_v1"
_RECEIPT_SCHEMA = "p2g.mi300x_resource_window.v1"
_ZERO_CHAIN = bytes(32)


class ResourceWindowViolation(ContractError):
    """The guarded stage encountered a hard resource-window violation."""


@dataclass(frozen=True, slots=True)
class Mi300xResourceWindowConfig:
    gpu_index: int
    admission_mode: AdmissionMode
    owner_process: ProcessIdentity
    allowed_stopped_processes: tuple[StoppedProcessIdentity, ...] = ()
    poll_interval_seconds: float = 2.0
    maximum_observation_gap_seconds: float = 15.0
    command_timeout_seconds: int = 20

    def validate(self) -> None:
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ContractError("resource-window gpu_index must be non-negative")
        if self.admission_mode not in {"shared_quality", "exclusive_performance"}:
            raise ContractError("resource-window admission_mode is invalid")
        self.owner_process.validate(label="resource-window owner")
        pids: list[int] = []
        for identity in self.allowed_stopped_processes:
            identity.validate()
            pids.append(identity.pid)
        if pids != sorted(set(pids)):
            raise ContractError("resource-window stopped-process PIDs must be unique and sorted")
        if self.owner_process.pid in pids:
            raise ContractError("resource-window owner cannot also be a stopped process")
        if self.admission_mode == "exclusive_performance" and pids:
            raise ContractError(
                "exclusive_performance resource windows cannot allow stopped clients"
            )
        for value, label in (
            (self.poll_interval_seconds, "poll_interval_seconds"),
            (self.maximum_observation_gap_seconds, "maximum_observation_gap_seconds"),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ContractError(f"resource-window {label} must be positive")
        if self.maximum_observation_gap_seconds <= self.poll_interval_seconds:
            raise ContractError(
                "resource-window maximum observation gap must exceed the poll interval"
            )
        if type(self.command_timeout_seconds) is not int or self.command_timeout_seconds <= 0:
            raise ContractError("resource-window command timeout must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "gpu_index": self.gpu_index,
            "admission_mode": self.admission_mode,
            "owner_process": self.owner_process.to_dict(),
            "allowed_stopped_processes": [
                identity.to_dict() for identity in self.allowed_stopped_processes
            ],
            "poll_interval_seconds": float(self.poll_interval_seconds),
            "maximum_observation_gap_seconds": float(
                self.maximum_observation_gap_seconds
            ),
            "command_timeout_seconds": self.command_timeout_seconds,
        }


def _procfs_row(pid: int, *, proc_root: Path) -> dict[str, Any]:
    [row] = _snapshot_procfs([pid], proc_root=proc_root)
    return row


def _owned_by(
    pid: int,
    *,
    owner: ProcessIdentity,
    first_row: Mapping[str, Any],
    proc_root: Path,
) -> bool:
    """Return true only when a live, non-reused ancestry chain reaches owner."""

    row = first_row
    current = pid
    seen: set[int] = set()
    for _ in range(64):
        if current in seen:
            return False
        seen.add(current)
        if current == owner.pid:
            return (
                row.get("observed") is True
                and row.get("starttime_ticks") == owner.starttime_ticks
            )
        parent = row.get("ppid")
        if type(parent) is not int or parent <= 0:
            return False
        current = parent
        row = _procfs_row(current, proc_root=proc_root)
    return False


def evaluate_kfd_observation(
    *,
    pids: list[int],
    process_metadata: list[dict[str, Any]],
    config: Mi300xResourceWindowConfig,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Classify one KFD inventory against the stage owner and stopped clients."""

    config.validate()
    if pids != sorted(set(pids)) or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ContractError("resource-window KFD PIDs must be unique, positive, and sorted")
    metadata: dict[int, dict[str, Any]] = {}
    for row in process_metadata:
        pid = row.get("pid")
        if type(pid) is not int or pid <= 0 or pid in metadata:
            raise ContractError("resource-window process metadata has invalid PIDs")
        metadata[pid] = row
    if set(metadata) != set(pids):
        raise ContractError("resource-window process metadata must exactly cover KFD PIDs")

    stopped = {identity.pid: identity for identity in config.allowed_stopped_processes}
    processes: list[dict[str, Any]] = []
    reasons: list[str] = []
    warnings: list[str] = []
    for pid in pids:
        row = metadata[pid]
        expected = stopped.get(pid)
        if _owned_by(
            pid,
            owner=config.owner_process,
            first_row=row,
            proc_root=proc_root,
        ):
            classification = (
                "stage_owner" if pid == config.owner_process.pid else "owned_descendant"
            )
        elif expected is not None and (
            row.get("observed") is True
            and row.get("state") == "T"
            and row.get("starttime_ticks") == expected.starttime_ticks
        ):
            classification = "allowed_stopped"
        elif expected is not None:
            classification = "stopped_identity_mismatch"
            target = (
                reasons
                if config.admission_mode == "exclusive_performance"
                else warnings
            )
            if "STOPPED_PROCESS_ALLOWANCE_MISMATCH" not in target:
                target.append("STOPPED_PROCESS_ALLOWANCE_MISMATCH")
        elif pid == config.owner_process.pid:
            classification = "owner_identity_mismatch"
            if "OWNER_PROCESS_IDENTITY_MISMATCH" not in reasons:
                reasons.append("OWNER_PROCESS_IDENTITY_MISMATCH")
        else:
            classification = "foreign"
            target = (
                reasons
                if config.admission_mode == "exclusive_performance"
                else warnings
            )
            if "FOREIGN_KFD_PROCESS_ARRIVED" not in target:
                target.append("FOREIGN_KFD_PROCESS_ARRIVED")
        processes.append(
            {
                "pid": pid,
                "state": row.get("state"),
                "starttime_ticks": row.get("starttime_ticks"),
                "classification": classification,
            }
        )
    return {
        "status": "PASS" if not reasons else "BUSY",
        "reasons": reasons,
        "warnings": warnings,
        "contention_observed": bool(
            warnings
            or {
                "FOREIGN_KFD_PROCESS_ARRIVED",
                "STOPPED_PROCESS_ALLOWANCE_MISMATCH",
            }.intersection(reasons)
        ),
        "processes": processes,
    }


def _capture_kfd_observation(
    *,
    config: Mi300xResourceWindowConfig,
    fuser_path: Path,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    completed, _ = _run_observed(
        [str(fuser_path), "/dev/kfd"], timeout=config.command_timeout_seconds
    )
    if completed.returncode not in (0, 1):
        raise ContractError(
            f"resource-window fuser /dev/kfd returned {completed.returncode}"
        )
    if not _fuser_stderr_is_benign(completed.stderr):
        raise ContractError("resource-window fuser reported a diagnostic")
    pids = _parse_kfd_pids(completed.stdout)
    expected_returncode = 0 if pids else 1
    if completed.returncode != expected_returncode:
        raise ContractError("resource-window fuser return code disagrees with its inventory")
    metadata = _snapshot_procfs(pids, proc_root=proc_root)
    return evaluate_kfd_observation(
        pids=pids,
        process_metadata=metadata,
        config=config,
        proc_root=proc_root,
    )


Sample = Callable[[], dict[str, Any]]


class _Monitor:
    def __init__(self, *, config: Mi300xResourceWindowConfig, sample: Sample) -> None:
        self.config = config
        self.sample = sample
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.started_ns = time.monotonic_ns()
        self.last_ns: int | None = None
        self.sample_count = 0
        self.maximum_gap = 0.0
        self.chain = _ZERO_CHAIN
        self.first: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None
        self.last_signature: bytes | None = None
        self.transitions: list[dict[str, Any]] = []
        self.violation_reasons: list[str] = []
        self.warning_reasons: list[str] = []

    def observe(self) -> None:
        observed_ns = time.monotonic_ns()
        try:
            evaluation = self.sample()
            if (
                set(evaluation)
                != {
                    "status",
                    "reasons",
                    "warnings",
                    "contention_observed",
                    "processes",
                }
                or evaluation["status"] not in {"PASS", "BUSY"}
                or not isinstance(evaluation["reasons"], list)
                or not isinstance(evaluation["warnings"], list)
                or type(evaluation["contention_observed"]) is not bool
                or not isinstance(evaluation["processes"], list)
            ):
                raise ContractError("resource-window sampler returned an invalid observation")
        except Exception as exc:  # fail closed while preserving the operation's finally path
            evaluation = {
                "status": "BUSY",
                "reasons": ["MONITOR_CAPTURE_FAILED"],
                "warnings": [],
                "contention_observed": False,
                "processes": [],
                "capture_error": f"{type(exc).__module__}.{type(exc).__qualname__}",
            }
        completed_ns = time.monotonic_ns()
        with self.lock:
            gap = (
                0.0
                if self.last_ns is None
                else (completed_ns - self.last_ns) / 1_000_000_000
            )
            self.maximum_gap = max(self.maximum_gap, gap)
            self.last_ns = completed_ns
            self.sample_count += 1
            observation = {
                "sample_index": self.sample_count,
                "observed_utc": datetime.now(UTC).isoformat(timespec="microseconds"),
                "elapsed_seconds": (observed_ns - self.started_ns) / 1_000_000_000,
                **evaluation,
            }
            encoded = canonical_json_bytes(observation)
            self.chain = bytes.fromhex(sha256_bytes(self.chain + encoded))
            signature = canonical_json_bytes(
                {
                    key: observation[key]
                    for key in (
                        "status",
                        "reasons",
                        "warnings",
                        "contention_observed",
                        "processes",
                    )
                }
            )
            if self.first is None:
                self.first = observation
            self.last = observation
            if signature != self.last_signature:
                self.transitions.append(observation)
                self.last_signature = signature
            for reason in cast(list[str], observation["reasons"]):
                if reason not in self.violation_reasons:
                    self.violation_reasons.append(reason)
            for warning in cast(list[str], observation["warnings"]):
                if warning not in self.warning_reasons:
                    self.warning_reasons.append(warning)

    def _loop(self) -> None:
        while not self.stop.wait(self.config.poll_interval_seconds):
            self.observe()

    def start(self) -> None:
        self.observe()
        self.thread = threading.Thread(
            target=self._loop,
            name="p2g-mi300x-resource-window",
            daemon=True,
        )
        self.thread.start()

    def finish(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=self.config.command_timeout_seconds + 1)
            if self.thread.is_alive():
                with self.lock:
                    if "MONITOR_THREAD_DID_NOT_STOP" not in self.violation_reasons:
                        self.violation_reasons.append("MONITOR_THREAD_DID_NOT_STOP")
        self.observe()
        with self.lock:
            if (
                self.maximum_gap > self.config.maximum_observation_gap_seconds
                and "OBSERVATION_GAP_ABOVE_LIMIT" not in self.violation_reasons
            ):
                self.violation_reasons.append("OBSERVATION_GAP_ABOVE_LIMIT")


def run_in_mi300x_resource_window[T](
    operation: Callable[[], T],
    *,
    output: Path,
    stage: Literal["propose", "train", "evaluate"],
    config: Mi300xResourceWindowConfig,
    sample: Sample | None = None,
) -> T:
    """Run an operation while polling KFD; write a receipt on every exit path."""

    config.validate()
    fuser_receipt: dict[str, Any]
    if sample is None:
        fuser_receipt, fuser_path = _resolve_tool("fuser")

        def effective_sample() -> dict[str, Any]:
            return _capture_kfd_observation(
                config=config,
                fuser_path=fuser_path,
            )

    else:
        fuser_receipt = {
            "name": "injected-test-sampler",
            "path": None,
            "sha256": None,
        }
        effective_sample = sample

    monitor = _Monitor(config=config, sample=effective_sample)
    created_utc = datetime.now(UTC).isoformat(timespec="seconds")
    operation_status = "NOT_STARTED"
    result: list[T] = []
    operation_error: BaseException | None = None
    monitor.start()
    if not monitor.violation_reasons:
        operation_status = "RAISED"
        try:
            result.append(operation())
            operation_status = "RETURNED"
        except BaseException as exc:
            operation_error = exc
        finally:
            monitor.finish()
    else:
        monitor.finish()

    reasons = list(monitor.violation_reasons)
    warnings = list(monitor.warning_reasons)
    contention_observed = bool(
        warnings
        or {
            "FOREIGN_KFD_PROCESS_ARRIVED",
            "STOPPED_PROCESS_ALLOWANCE_MISMATCH",
        }.intersection(reasons)
    )
    status = "PASS" if not reasons else "BUSY"
    hostname = platform.node()
    if not hostname:
        raise ContractError("resource-window hostname is empty")
    receipt: dict[str, Any] = {
        "schema_version": _RECEIPT_SCHEMA,
        "operator": RESOURCE_WINDOW_OPERATOR,
        "operator_source_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": created_utc,
        "closed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": hostname,
        "stage": stage,
        "config": config.to_dict(),
        "tool": fuser_receipt,
        "operation_status": operation_status,
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "contention_observed": contention_observed,
        "sample_count": monitor.sample_count,
        "maximum_observation_gap_seconds": monitor.maximum_gap,
        "observation_chain_sha256": monitor.chain.hex(),
        "first_observation": monitor.first,
        "last_observation": monitor.last,
        "transitions": monitor.transitions,
        "claim_boundary": (
            "This receipt samples /dev/kfd throughout one stage. shared_quality "
            "records external clients as contention without invalidating numerical "
            "quality work; its timings are inadmissible. exclusive_performance "
            "invalidates the stage on observed interference. Polling cannot replace "
            "a scheduler lease."
        ),
    }
    receipt["logical_sha256"] = sha256_json(receipt)
    validate_payload("mi300x_resource_window", receipt)
    write_new_bytes(output, canonical_json_bytes(receipt))

    if status != "PASS":
        violation = ResourceWindowViolation(
            f"MI300X resource window for {stage} was invalid: {', '.join(reasons)}"
        )
        if operation_error is not None:
            raise violation from operation_error
        raise violation
    if operation_error is not None:
        raise operation_error
    if len(result) != 1:  # pragma: no cover - guarded by operation status above
        raise AssertionError("resource-window operation returned no result")
    return result[0]


__all__ = [
    "RESOURCE_WINDOW_OPERATOR",
    "Mi300xResourceWindowConfig",
    "ResourceWindowViolation",
    "evaluate_kfd_observation",
    "run_in_mi300x_resource_window",
]
