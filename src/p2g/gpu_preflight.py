# pyright: reportUnnecessaryIsInstance=false

"""Replayable, non-reserving occupancy preflight for one MI300X device.

The module intentionally does not import Torch or open a GPU runtime.  It
captures three operating-system views (AMD SMI processes, ``/dev/kfd`` users,
and ROCm utilization), preserves the raw evidence, and derives a deterministic
verdict.  A passing snapshot is not a lease and cannot rule out a race after
the final command completes.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Literal, TypeGuard, cast

from p2g.audit import AuditReport
from p2g.canonical import content_id, sha256_bytes, sha256_file, sha256_json
from p2g.errors import ContractError
from p2g.schema import validate_payload

MI300X_PREFLIGHT_OPERATOR = "mi300x_resource_admission_preflight_v2"
_CONFIG_SCHEMA = "p2g.mi300x_preflight_config.v2"
_EVALUATION_SCHEMA = "p2g.mi300x_preflight_evaluation.v2"
_RECEIPT_SCHEMA = "p2g.mi300x_preflight.v2"
_PROCFS_REDACTION = "argv arguments intentionally omitted; only argv0/hash/argc retained"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _percentage(value: object, *, label: str) -> float:
    if not _is_number(value) or not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ContractError(f"{label} must be a finite number inside [0, 100]")
    return float(value)


AdmissionMode = Literal["shared_quality", "exclusive_performance"]


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """PID identity that remains stable across `/proc` PID reuse."""

    pid: int
    starttime_ticks: int

    def validate(self, *, label: str = "process") -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise ContractError(f"{label} PID must be a positive integer")
        if type(self.starttime_ticks) is not int or self.starttime_ticks <= 0:
            raise ContractError(f"{label} starttime_ticks must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return {"pid": self.pid, "starttime_ticks": self.starttime_ticks}


def _process_identity_from_dict(payload: object, *, label: str) -> ProcessIdentity:
    if not isinstance(payload, Mapping):
        raise ContractError(f"{label} identity has missing or ambient fields")
    raw = cast(Mapping[str, object], payload)
    if set(raw) != {"pid", "starttime_ticks"}:
        raise ContractError(f"{label} identity has missing or ambient fields")
    identity = ProcessIdentity(
        pid=cast(int, raw["pid"]),
        starttime_ticks=cast(int, raw["starttime_ticks"]),
    )
    identity.validate(label=label)
    return identity


@dataclass(frozen=True, slots=True)
class StoppedProcessIdentity:
    """One explicitly admitted, signal-stopped pre-existing GPU client."""

    pid: int
    starttime_ticks: int

    def validate(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise ContractError("stopped-process PID must be a positive integer")
        if type(self.starttime_ticks) is not int or self.starttime_ticks <= 0:
            raise ContractError("stopped-process starttime_ticks must be a positive integer")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return {
            "pid": self.pid,
            "starttime_ticks": self.starttime_ticks,
            "required_state": "T",
        }


def _stopped_process_identity_from_dict(payload: object) -> StoppedProcessIdentity:
    if not isinstance(payload, Mapping):
        raise ContractError("stopped-process identity has missing or ambient fields")
    raw = cast(Mapping[str, object], payload)
    if set(raw) != {
        "pid",
        "starttime_ticks",
        "required_state",
    }:
        raise ContractError("stopped-process identity has missing or ambient fields")
    if raw["required_state"] != "T":
        raise ContractError("stopped-process required_state must be T")
    identity = StoppedProcessIdentity(
        pid=cast(int, raw["pid"]),
        starttime_ticks=cast(int, raw["starttime_ticks"]),
    )
    identity.validate()
    return identity


@dataclass(frozen=True, slots=True)
class Mi300xPreflightConfig:
    """Policy for one replayable resource-admission observation."""

    gpu_index: int = 0
    maximum_gpu_use_percent: float = 100.0
    maximum_vram_percent: float = 80.0
    admission_mode: AdmissionMode = "shared_quality"
    owner_process: ProcessIdentity | None = None
    allowed_stopped_processes: tuple[StoppedProcessIdentity, ...] = ()
    command_timeout_seconds: int = 20

    def validate(self) -> None:
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ContractError("preflight gpu_index must be a non-negative integer")
        _percentage(self.maximum_gpu_use_percent, label="maximum_gpu_use_percent")
        _percentage(self.maximum_vram_percent, label="maximum_vram_percent")
        if self.admission_mode not in {"shared_quality", "exclusive_performance"}:
            raise ContractError("admission_mode must be shared_quality or exclusive_performance")
        if self.owner_process is not None:
            if not isinstance(self.owner_process, ProcessIdentity):
                raise ContractError("owner_process has an invalid identity")
            self.owner_process.validate(label="owner process")
        if type(self.command_timeout_seconds) is not int or self.command_timeout_seconds <= 0:
            raise ContractError("command_timeout_seconds must be a positive integer")
        if not isinstance(self.allowed_stopped_processes, tuple):
            raise ContractError("allowed_stopped_processes must be a tuple")
        for identity in self.allowed_stopped_processes:
            if not isinstance(identity, StoppedProcessIdentity):
                raise ContractError("allowed_stopped_processes contains an invalid identity")
            identity.validate()
        pids = tuple(identity.pid for identity in self.allowed_stopped_processes)
        if tuple(sorted(set(pids))) != pids:
            raise ContractError(
                "allowed_stopped_processes must have unique PIDs in ascending order"
            )
        if self.admission_mode == "exclusive_performance" and self.allowed_stopped_processes:
            raise ContractError(
                "exclusive_performance admission cannot allow stopped GPU clients"
            )
        if self.owner_process is not None and self.owner_process.pid in set(pids):
            raise ContractError("owner_process cannot also be an allowed stopped process")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "gpu_index": self.gpu_index,
            "maximum_gpu_use_percent": float(self.maximum_gpu_use_percent),
            "maximum_vram_percent": float(self.maximum_vram_percent),
            "admission_mode": self.admission_mode,
            "owner_process": (
                None if self.owner_process is None else self.owner_process.to_dict()
            ),
            "allowed_stopped_processes": [
                identity.to_dict() for identity in self.allowed_stopped_processes
            ],
            "command_timeout_seconds": self.command_timeout_seconds,
        }

    def config_id(self) -> str:
        return content_id(_CONFIG_SCHEMA, self.to_dict(), prefix="gpupreflight")


def mi300x_preflight_config_from_dict(payload: object) -> Mi300xPreflightConfig:
    """Decode exactly the public config fields; ambient policy is rejected."""

    expected = {
        "gpu_index",
        "maximum_gpu_use_percent",
        "maximum_vram_percent",
        "admission_mode",
        "owner_process",
        "allowed_stopped_processes",
        "command_timeout_seconds",
    }
    if not isinstance(payload, Mapping):
        raise ContractError("preflight config has missing or ambient fields")
    raw = cast(Mapping[str, Any], payload)
    if set(raw) != expected:
        raise ContractError("preflight config has missing or ambient fields")
    allowed = raw["allowed_stopped_processes"]
    if not isinstance(allowed, list):
        raise ContractError("preflight allowed_stopped_processes must be a JSON array")
    config = Mi300xPreflightConfig(
        gpu_index=cast(int, raw["gpu_index"]),
        maximum_gpu_use_percent=cast(float, raw["maximum_gpu_use_percent"]),
        maximum_vram_percent=cast(float, raw["maximum_vram_percent"]),
        admission_mode=cast(AdmissionMode, raw["admission_mode"]),
        owner_process=(
            None
            if raw["owner_process"] is None
            else _process_identity_from_dict(raw["owner_process"], label="owner process")
        ),
        allowed_stopped_processes=tuple(
            _stopped_process_identity_from_dict(item)
            for item in cast(list[Any], allowed)
        ),
        command_timeout_seconds=cast(int, raw["command_timeout_seconds"]),
    )
    config.validate()
    return config


def _parse_amd_processes(payload: Any, *, gpu_index: int) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ContractError("amd-smi process output must be a JSON array")
    selected: list[dict[str, Any]] = []
    for index, untyped in enumerate(cast(list[Any], payload)):
        if not isinstance(untyped, dict):
            raise ContractError(f"amd-smi GPU row {index} is malformed")
        row = cast(dict[str, Any], untyped)
        if type(row.get("gpu")) is not int:
            raise ContractError(f"amd-smi GPU row {index} is malformed")
        if row["gpu"] == gpu_index:
            selected.append(row)
    if len(selected) != 1:
        raise ContractError("amd-smi output must contain exactly one selected GPU row")
    raw_processes = selected[0].get("process_list")
    if not isinstance(raw_processes, list):
        raise ContractError("amd-smi selected GPU row has no process_list array")

    processes: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, untyped in enumerate(cast(list[Any], raw_processes)):
        if not isinstance(untyped, dict):
            raise ContractError(f"amd-smi process row {index} is malformed")
        process_row = cast(dict[str, Any], untyped)
        if not isinstance(process_row.get("process_info"), dict):
            raise ContractError(f"amd-smi process row {index} is malformed")
        info = cast(dict[str, Any], process_row["process_info"])
        pid = info.get("pid")
        name = info.get("name")
        if type(pid) is not int or pid <= 0 or pid in seen:
            raise ContractError("amd-smi process PIDs must be unique positive integers")
        if not isinstance(name, str) or not name:
            raise ContractError("amd-smi process name must be non-empty")
        seen.add(pid)
        vram_bytes: int | None = None
        memory = info.get("memory_usage")
        if memory is not None:
            if not isinstance(memory, dict):
                raise ContractError("amd-smi memory_usage must be an object when present")
            vram = cast(dict[str, Any], memory).get("vram_mem")
            if vram is not None:
                if not isinstance(vram, dict):
                    raise ContractError("amd-smi vram_mem must be an object when present")
                value = cast(dict[str, Any], vram).get("value")
                if type(value) is not int or value < 0:
                    raise ContractError("amd-smi VRAM bytes must be a non-negative integer")
                vram_bytes = value
        processes.append(
            {
                "pid": pid,
                "name": name,
                "vram_bytes": vram_bytes,
                "raw": info,
            }
        )
    return sorted(processes, key=lambda row: cast(int, row["pid"]))


def _parse_kfd_pids(stdout: str) -> list[int]:
    pids: list[int] = []
    for token in stdout.split():
        match = re.fullmatch(r"([0-9]+)[A-Za-z]*", token)
        if match is None:
            raise ContractError(f"fuser /dev/kfd emitted an invalid PID token: {token!r}")
        pid = int(match.group(1))
        if pid <= 0 or pid in pids:
            raise ContractError("fuser /dev/kfd PIDs must be unique and positive")
        pids.append(pid)
    return sorted(pids)


def _fuser_stderr_is_benign(stderr: str) -> bool:
    """Accept only silence or fuser's ordinary device/access annotation.

    GNU ``fuser`` writes the queried pathname and per-process access letters to
    stderr even when the command succeeds.  For example, two memory-mapped KFD
    clients can produce ``/dev/kfd:           mm`` while their PIDs are emitted
    on stdout.  Keep rejecting arbitrary diagnostics, but admit the documented
    access alphabet: current directory (``c``), executable (``e``), open file
    (``f``/``F``), root directory (``r``), and memory map/shared library
    (``m``).
    """

    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    ordinary = re.compile(r"/dev/kfd:(?:[ \t]+[cefFrm]+)*")
    return all(ordinary.fullmatch(line) is not None for line in lines)


def _metric_percentage(row: Mapping[str, str], field: str) -> float:
    try:
        raw = row[field]
        value = float(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"rocm-smi metric {field!r} is missing or invalid") from exc
    return _percentage(value, label=f"rocm-smi metric {field!r}")


def _parse_metric_csv(metric_csv: str, *, gpu_index: int) -> tuple[str, float, float]:
    if not metric_csv.strip():
        raise ContractError("rocm-smi metric output must be non-empty CSV text")
    try:
        rows = list(csv.DictReader(StringIO(metric_csv)))
    except csv.Error as exc:
        raise ContractError(f"rocm-smi metric CSV is malformed: {exc}") from exc
    device = f"card{gpu_index}"
    selected = [row for row in rows if row.get("device") == device]
    if len(selected) != 1:
        raise ContractError("rocm-smi metric CSV must contain exactly one selected card")
    row = selected[0]
    return (
        device,
        _metric_percentage(row, "GPU use (%)"),
        _metric_percentage(row, "GPU Memory Allocated (VRAM%)"),
    )


def _nullable_integer(value: object, *, label: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ContractError(f"{label} must be null or a non-negative integer")


def _nullable_text(value: object, *, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ContractError(f"{label} must be null or non-empty text")


def _validate_procfs_row(row: Mapping[str, Any]) -> int:
    expected = {
        "pid",
        "observed",
        "uid",
        "ppid",
        "state",
        "starttime_ticks",
        "comm",
        "exe",
        "cwd",
        "argv0",
        "argc",
        "cmdline_sha256",
        "redaction",
    }
    if set(row) != expected:
        raise ContractError("procfs process metadata has missing or ambient fields")
    pid = row["pid"]
    if type(pid) is not int or pid <= 0:
        raise ContractError("procfs process PID must be a positive integer")
    if type(row["observed"]) is not bool:
        raise ContractError("procfs observed flag must be boolean")
    for field in ("uid", "ppid", "starttime_ticks"):
        _nullable_integer(row[field], label=f"procfs {field}")
    for field in ("state", "comm", "exe", "cwd", "argv0"):
        _nullable_text(row[field], label=f"procfs {field}")
    if type(row["argc"]) is not int or row["argc"] < 0:
        raise ContractError("procfs argc must be a non-negative integer")
    digest = row["cmdline_sha256"]
    if digest is not None and (not isinstance(digest, str) or _SHA256.fullmatch(digest) is None):
        raise ContractError("procfs cmdline_sha256 must be null or lowercase SHA-256")
    if row["redaction"] != _PROCFS_REDACTION:
        raise ContractError("procfs redaction declaration is invalid")
    return pid


def _metadata_by_pid(process_metadata: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(process_metadata, list):
        raise ContractError("procfs process metadata must be a JSON array")
    result: dict[int, dict[str, Any]] = {}
    for untyped in cast(list[Any], process_metadata):
        if not isinstance(untyped, dict):
            raise ContractError("procfs process metadata rows must be objects")
        row = cast(dict[str, Any], untyped)
        pid = _validate_procfs_row(row)
        if pid in result:
            raise ContractError("procfs process metadata contains duplicate PIDs")
        result[pid] = row
    return result


def evaluate_mi300x_preflight(
    *,
    process_payload: Any,
    metric_csv: str,
    kfd_fuser_stdout: str = "",
    process_metadata: Any = None,
    config: Mi300xPreflightConfig | None = None,
) -> dict[str, Any]:
    """Derive a deterministic occupancy verdict from already captured text."""

    effective = config or Mi300xPreflightConfig()
    effective.validate()
    amd_processes = _parse_amd_processes(process_payload, gpu_index=effective.gpu_index)
    kfd_pids = _parse_kfd_pids(kfd_fuser_stdout)
    device, gpu_use, vram_percent = _parse_metric_csv(metric_csv, gpu_index=effective.gpu_index)
    amd_by_pid = {cast(int, row["pid"]): row for row in amd_processes}
    all_pids = sorted(set(amd_by_pid) | set(kfd_pids))
    metadata = _metadata_by_pid([] if process_metadata is None else process_metadata)
    ambient_metadata = sorted(set(metadata) - set(all_pids))
    if ambient_metadata:
        raise ContractError(
            f"procfs metadata names PIDs absent from device evidence: {ambient_metadata}"
        )

    processes: list[dict[str, Any]] = []
    for pid in all_pids:
        amd = amd_by_pid.get(pid)
        procfs = metadata.get(pid)
        sources = (["amd_smi"] if amd is not None else []) + (
            ["dev_kfd"] if pid in kfd_pids else []
        )
        fallback_name = cast(str | None, procfs.get("comm")) if procfs else None
        processes.append(
            {
                "pid": pid,
                "name": cast(str, amd["name"])
                if amd is not None
                else fallback_name or "<kfd-client>",
                "vram_bytes": amd["vram_bytes"] if amd is not None else None,
                "sources": sources,
                "procfs": procfs,
                "raw_amd_smi": amd["raw"] if amd is not None else None,
            }
        )

    allowances = {
        identity.pid: identity for identity in effective.allowed_stopped_processes
    }
    owner = effective.owner_process
    foreign: list[dict[str, Any]] = []
    allowance_mismatches: list[dict[str, Any]] = []
    owner_mismatches: list[dict[str, Any]] = []
    for row in processes:
        pid = cast(int, row["pid"])
        expected = allowances.get(pid)
        procfs = cast(dict[str, Any] | None, row["procfs"])
        if owner is not None and pid == owner.pid:
            if (
                procfs is not None
                and procfs["observed"] is True
                and procfs["starttime_ticks"] == owner.starttime_ticks
            ):
                classification = "stage_owner"
            else:
                classification = "owner_identity_mismatch"
                owner_mismatches.append(row)
        elif expected is None:
            classification = "foreign"
            foreign.append(row)
        elif (
            procfs is not None
            and procfs["observed"] is True
            and procfs["state"] == "T"
            and procfs["starttime_ticks"] == expected.starttime_ticks
        ):
            classification = "allowed_stopped"
        else:
            classification = "allowance_identity_mismatch"
            allowance_mismatches.append(row)
        row["admission"] = {
            "classification": classification,
            "expected_starttime_ticks": (
                owner.starttime_ticks
                if owner is not None and pid == owner.pid
                else None if expected is None else expected.starttime_ticks
            ),
            "required_state": (
                "T"
                if expected is not None and not (owner is not None and pid == owner.pid)
                else None
            ),
        }
    reasons: list[str] = []
    warnings: list[str] = []
    if gpu_use > effective.maximum_gpu_use_percent:
        reasons.append("GPU_USE_ABOVE_LIMIT")
    if vram_percent > effective.maximum_vram_percent:
        reasons.append("VRAM_ABOVE_LIMIT")
    process_findings: list[str] = []
    if foreign:
        process_findings.append("FOREIGN_GPU_PROCESSES_PRESENT")
    if allowance_mismatches:
        process_findings.append("STOPPED_PROCESS_ALLOWANCE_MISMATCH")
    if effective.admission_mode == "exclusive_performance":
        reasons.extend(process_findings)
    else:
        warnings.extend(process_findings)
    if owner_mismatches:
        reasons.append("OWNER_PROCESS_IDENTITY_MISMATCH")
    claim_class = (
        "shared_quality_observation"
        if effective.admission_mode == "shared_quality"
        else "exclusive_performance_precondition"
    )
    return {
        "schema_version": _EVALUATION_SCHEMA,
        "operator": MI300X_PREFLIGHT_OPERATOR,
        "config": effective.to_dict(),
        "config_id": effective.config_id(),
        "claim_class": claim_class,
        "status": "PASS" if not reasons else "BUSY",
        "reasons": reasons,
        "warnings": warnings,
        "contention_observed": bool(process_findings),
        "device": {
            "gpu_index": effective.gpu_index,
            "rocm_smi_device": device,
            "gpu_use_percent": gpu_use,
            "vram_percent": vram_percent,
        },
        "process_count": len(processes),
        "amd_smi_process_count": len(amd_processes),
        "kfd_process_count": len(kfd_pids),
        "foreign_process_count": len(foreign),
        "allowance_mismatch_count": len(allowance_mismatches),
        "owner_mismatch_count": len(owner_mismatches),
        "processes": processes,
        "claim_boundary": (
            "A PASS is one replayable amd-smi, /dev/kfd, and rocm-smi resource "
            "observation. shared_quality records external/resumed clients as "
            "contention and cannot support timing claims; exclusive_performance is "
            "only a precondition and still requires a scheduler lease, full-window "
            "guard, and postflight. A PASS is not a scheduler reservation; neither "
            "mode is a hardware-identity proof or a numerical result."
        ),
    }


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_link_basename(path: Path) -> str | None:
    try:
        target = os.readlink(path)
    except OSError:
        return None
    name = Path(target).name
    return name or None


def _snapshot_procfs(pids: list[int], *, proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    """Record useful identity while omitting argv arguments and working directories."""

    rows: list[dict[str, Any]] = []
    for pid in sorted(pids):
        root = proc_root / str(pid)
        try:
            cmdline = (root / "cmdline").read_bytes()
        except OSError:
            cmdline = b""
        argv = [part.decode("utf-8", errors="replace") for part in cmdline.split(b"\0") if part]
        stat_text = _read_text(root / "stat")
        state: str | None = None
        ppid: int | None = None
        starttime_ticks: int | None = None
        if stat_text is not None:
            _, separator, tail = stat_text.rpartition(") ")
            fields = tail.split() if separator else []
            if len(fields) >= 20:
                state = fields[0] or None
                try:
                    ppid = int(fields[1])
                    starttime_ticks = int(fields[19])
                except ValueError:
                    ppid = None
                    starttime_ticks = None
        uid: int | None = None
        status_text = _read_text(root / "status")
        if status_text is not None:
            for line in status_text.splitlines():
                if line.startswith("Uid:"):
                    try:
                        uid = int(line.split()[1])
                    except (IndexError, ValueError):
                        uid = None
                    break
        argv0 = Path(argv[0]).name if argv and Path(argv[0]).name else None
        rows.append(
            {
                "pid": pid,
                "observed": root.is_dir(),
                "uid": uid,
                "ppid": ppid,
                "state": state,
                "starttime_ticks": starttime_ticks,
                "comm": _read_text(root / "comm"),
                "exe": _read_link_basename(root / "exe"),
                "cwd": None,
                "argv0": argv0,
                "argc": len(argv),
                "cmdline_sha256": sha256_bytes(cmdline) if cmdline else None,
                "redaction": _PROCFS_REDACTION,
            }
        )
    return rows


def current_process_identity(*, proc_root: Path = Path("/proc")) -> ProcessIdentity:
    """Return the caller's PID identity without importing or opening a GPU runtime."""

    pid = os.getpid()
    [row] = _snapshot_procfs([pid], proc_root=proc_root)
    ticks = row["starttime_ticks"]
    if row["observed"] is not True or type(ticks) is not int or ticks <= 0:
        raise ContractError("cannot establish the current process start-time identity")
    return ProcessIdentity(pid=pid, starttime_ticks=ticks)


def _resolve_tool(name: str) -> tuple[dict[str, Any], Path]:
    candidate = shutil.which(name)
    if candidate is None:
        raise ContractError(f"required preflight tool is unavailable: {name}")
    try:
        path = Path(candidate).resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve preflight tool {name}: {exc}") from exc
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ContractError(f"preflight tool is not an executable regular file: {path}")
    return {"name": name, "path": str(path), "sha256": sha256_file(path)}, path


def _run_observed(
    argv: list[str], *, timeout: int
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    started_utc = datetime.now(UTC).isoformat(timespec="microseconds")
    started_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError(
            f"preflight command failed before a receipt was captured: {exc}"
        ) from exc
    elapsed = (time.monotonic_ns() - started_ns) / 1_000_000_000
    return completed, {
        "argv": argv,
        "returncode": completed.returncode,
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(timespec="microseconds"),
        "elapsed_seconds": elapsed,
    }


def _decode_process_json(stdout: str) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token}")

    try:
        return json.loads(stdout, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"amd-smi process output is not strict JSON: {exc}") from exc


def capture_mi300x_preflight(*, config: Mi300xPreflightConfig | None = None) -> dict[str, Any]:
    """Capture one non-reserving occupancy receipt without importing a GPU runtime."""

    effective = config or Mi300xPreflightConfig()
    effective.validate()
    amd_receipt, amd_smi = _resolve_tool("amd-smi")
    rocm_receipt, rocm_smi = _resolve_tool("rocm-smi")
    fuser_receipt, fuser = _resolve_tool("fuser")
    commands = {
        "process": [str(amd_smi), "process", "--gpu", str(effective.gpu_index), "--json"],
        "kfd": [str(fuser), "/dev/kfd"],
        "metric": [str(rocm_smi), "--showuse", "--showmemuse", "--csv"],
    }
    results: dict[str, subprocess.CompletedProcess[str]] = {}
    command_receipts: dict[str, dict[str, Any]] = {}
    for name in ("process", "kfd", "metric"):
        result, receipt = _run_observed(commands[name], timeout=effective.command_timeout_seconds)
        results[name] = result
        command_receipts[name] = receipt
    if results["process"].returncode != 0 or results["metric"].returncode != 0:
        raise ContractError(
            "preflight metric commands failed: "
            f"amd-smi={results['process'].returncode}, "
            f"rocm-smi={results['metric'].returncode}"
        )
    if results["kfd"].returncode not in (0, 1):
        raise ContractError(f"fuser /dev/kfd returned {results['kfd'].returncode}; expected 0 or 1")
    if not _fuser_stderr_is_benign(results["kfd"].stderr):
        raise ContractError("fuser /dev/kfd reported an access or execution diagnostic")

    process_payload = _decode_process_json(results["process"].stdout)
    amd_pids = [
        cast(int, row["pid"])
        for row in _parse_amd_processes(process_payload, gpu_index=effective.gpu_index)
    ]
    kfd_pids = _parse_kfd_pids(results["kfd"].stdout)
    expected_kfd_returncode = 0 if kfd_pids else 1
    if results["kfd"].returncode != expected_kfd_returncode:
        raise ContractError("fuser /dev/kfd return code disagrees with its PID inventory")
    process_metadata = _snapshot_procfs(sorted(set(amd_pids) | set(kfd_pids)))
    evaluation = evaluate_mi300x_preflight(
        process_payload=process_payload,
        metric_csv=results["metric"].stdout,
        kfd_fuser_stdout=results["kfd"].stdout,
        process_metadata=process_metadata,
        config=effective,
    )
    hostname = platform.node()
    if not hostname:
        raise ContractError("preflight hostname is empty")
    payload: dict[str, Any] = {
        "schema_version": _RECEIPT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": hostname,
        "operator": MI300X_PREFLIGHT_OPERATOR,
        "tools": {
            "amd_smi": amd_receipt,
            "rocm_smi": rocm_receipt,
            "fuser": fuser_receipt,
        },
        "commands": command_receipts,
        "raw": {
            "amd_smi_process": process_payload,
            "amd_smi_process_stdout": results["process"].stdout,
            "amd_smi_process_stderr": results["process"].stderr,
            "rocm_smi_metric_csv": results["metric"].stdout,
            "rocm_smi_metric_stderr": results["metric"].stderr,
            "kfd_fuser_stdout": results["kfd"].stdout,
            "kfd_fuser_stderr": results["kfd"].stderr,
            "process_metadata": process_metadata,
        },
        "evaluation": evaluation,
        "status": evaluation["status"],
        "claim_boundary": evaluation["claim_boundary"],
    }
    payload["logical_sha256"] = sha256_json(payload)
    validate_payload("mi300x_preflight", payload)
    return payload


def audit_mi300x_preflight(
    payload: dict[str, Any], *, require_current_tools: bool = True
) -> AuditReport:
    """Replay a receipt; distinguish evidence integrity from admission status."""

    report = AuditReport(subject="mi300x-resource-admission-preflight")
    try:
        validate_payload("mi300x_preflight", payload)
    except ContractError as exc:
        report.add("schema", False, detail=str(exc))
        return report
    report.add("schema", True, detail=payload["schema_version"])

    unsigned = {key: value for key, value in payload.items() if key != "logical_sha256"}
    computed = sha256_json(unsigned)
    report.add(
        "logical_identity",
        computed == payload["logical_sha256"],
        detail={"declared": payload["logical_sha256"], "computed": computed},
    )
    report.add(
        "audit_operator_identity",
        True,
        detail={
            "operator": "p2g.mi300x_preflight_audit.v1",
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )

    tools = cast(dict[str, dict[str, Any]], payload["tools"])
    tool_detail: dict[str, Any] = {}
    tools_match = True
    for key, receipt in tools.items():
        path = Path(cast(str, receipt["path"]))
        try:
            current = sha256_file(path) if path.is_file() else None
        except OSError:
            current = None
        matches = current == receipt["sha256"]
        tools_match = tools_match and matches
        tool_detail[key] = {
            "declared_sha256": receipt["sha256"],
            "current_sha256": current,
            "matches": matches,
        }
    report.add(
        "current_tool_identity",
        tools_match,
        required=require_current_tools,
        detail={"required_on_this_host": require_current_tools, "tools": tool_detail},
    )

    commands = cast(dict[str, dict[str, Any]], payload["commands"])
    evaluation = cast(dict[str, Any], payload["evaluation"])
    gpu_index = evaluation["config"]["gpu_index"]
    expected = {
        "process": [tools["amd_smi"]["path"], "process", "--gpu", str(gpu_index), "--json"],
        "kfd": [tools["fuser"]["path"], "/dev/kfd"],
        "metric": [tools["rocm_smi"]["path"], "--showuse", "--showmemuse", "--csv"],
    }
    command_detail: dict[str, Any] = {}
    commands_match = True
    for name, expected_argv in expected.items():
        returncode = commands[name]["returncode"]
        accepted = returncode in ((0, 1) if name == "kfd" else (0,))
        matches = commands[name]["argv"] == expected_argv and accepted
        commands_match = commands_match and matches
        command_detail[name] = {
            "argv_matches": commands[name]["argv"] == expected_argv,
            "returncode": returncode,
            "accepted_returncode": accepted,
        }
    report.add("command_identity", commands_match, detail=command_detail)

    raw = cast(dict[str, Any], payload["raw"])
    try:
        decoded = _decode_process_json(cast(str, raw["amd_smi_process_stdout"]))
        process_matches = decoded == raw["amd_smi_process"]
        fuser_stderr_benign = _fuser_stderr_is_benign(cast(str, raw["kfd_fuser_stderr"]))
        raw_matches = process_matches and fuser_stderr_benign
        raw_detail: Any = {
            "amd_smi_stdout_matches_decoded_payload": process_matches,
            "fuser_stderr_benign": fuser_stderr_benign,
        }
    except (ContractError, KeyError, TypeError) as exc:
        raw_matches = False
        raw_detail = str(exc)
    report.add("raw_capture_consistency", raw_matches, detail=raw_detail)

    try:
        config = mi300x_preflight_config_from_dict(cast(dict[str, Any], evaluation["config"]))
        replay = evaluate_mi300x_preflight(
            process_payload=raw["amd_smi_process"],
            metric_csv=cast(str, raw["rocm_smi_metric_csv"]),
            kfd_fuser_stdout=cast(str, raw["kfd_fuser_stdout"]),
            process_metadata=raw["process_metadata"],
            config=config,
        )
        kfd_returncode = 0 if _parse_kfd_pids(cast(str, raw["kfd_fuser_stdout"])) else 1
        replay_matches = (
            replay == evaluation
            and payload["status"] == replay["status"]
            and payload["claim_boundary"] == replay["claim_boundary"]
            and commands["kfd"]["returncode"] == kfd_returncode
        )
        replay_detail: Any = {
            "declared_status": payload["status"],
            "replayed_status": replay["status"],
            "config_id": replay["config_id"],
        }
    except (ContractError, KeyError, TypeError) as exc:
        replay_matches = False
        replay_detail = str(exc)
    report.add("deterministic_replay", replay_matches, detail=replay_detail)
    report.add(
        "admission_passed",
        payload["status"] == "PASS",
        required=False,
        detail={
            "status": payload["status"],
            "reasons": evaluation["reasons"],
            "warnings": evaluation["warnings"],
            "contention_observed": evaluation["contention_observed"],
            "lease": False,
        },
    )
    return report
