"""Digest-bound, fail-closed orchestration of the public Pixel4DGS stages."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json, write_new_bytes
from p2g.errors import ContractError

PIPELINE_PLAN_SCHEMA = "p2g.pipeline_plan.v3"
WORKSPACE_SCHEMA = "p2g.pipeline_workspace.v3"
STAGE_RECORD_SCHEMA = "p2g.pipeline_stage_record.v2"
COMPLETION_SCHEMA = "p2g.pipeline_completion.v3"
STATUS_SCHEMA = "p2g.pipeline_status.v2"
ORCHESTRATOR_VERSION = "p2g.public_pipeline_orchestrator.v3"

StageName = Literal[
    "prepare",
    "propose",
    "initialize",
    "train",
    "evaluate",
    "asset",
]
STAGE_ORDER: tuple[StageName, ...] = (
    "prepare",
    "propose",
    "initialize",
    "train",
    "evaluate",
    "asset",
)
GPU_STAGES: frozenset[StageName] = frozenset({"propose", "train", "evaluate"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PLAN_SNAPSHOT = re.compile(r"^(?P<attempt>[0-9]{6})-(?P<sha256>[0-9a-f]{64})\.toml$")

Progress = Callable[[str], None]


def _discard_progress(_message: str) -> None:
    return


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], *, context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ContractError(f"{context} contains unknown fields: {unknown}")


def _table(value: Any, *, context: str, required: bool = False) -> dict[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ContractError(f"{context} must be a TOML table")
    return cast(dict[str, Any], value)


def _integer(value: Any, *, name: str, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        suffix = f" in [{minimum}, {maximum}]" if maximum is not None else f" >= {minimum}"
        raise ContractError(f"{name} must be an integer{suffix}")
    return value


def _finite(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be a finite number")
    if minimum is not None and (
        result <= minimum if exclusive_minimum else result < minimum
    ):
        relation = ">" if exclusive_minimum else ">="
        raise ContractError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ContractError(f"{name} must be <= {maximum}")
    return result


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _path(value: Any, *, name: str, base: Path) -> Path:
    raw = _string(value, name=name)
    if raw.startswith(("~", "file://")):
        raise ContractError(f"{name} must not use home expansion or a file URI")
    candidate = Path(raw)
    unresolved = candidate if candidate.is_absolute() else base / candidate
    if unresolved.is_symlink():
        raise ContractError(f"{name} must not be a symlink")
    return unresolved.resolve()


def _regular_input(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{name} must be a regular non-symlink file")


def _directory_input(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"{name} must be a regular non-symlink directory")


@dataclass(frozen=True, slots=True)
class StoppedProcessOptions:
    pid: int
    starttime_ticks: int

    def validate(self) -> None:
        _integer(self.pid, name="preflight allowed stopped-process PID", minimum=1)
        _integer(
            self.starttime_ticks,
            name="preflight allowed stopped-process starttime_ticks",
            minimum=1,
        )


@dataclass(frozen=True, slots=True)
class PreflightOptions:
    gpu_index: int = 0
    maximum_gpu_use_percent: float = 100.0
    maximum_vram_percent: float = 80.0
    admission_mode: Literal["shared_quality", "exclusive_performance"] = "shared_quality"
    allowed_stopped_processes: tuple[StoppedProcessOptions, ...] = ()
    command_timeout_seconds: int = 20

    def validate(self) -> None:
        _integer(self.gpu_index, name="preflight.gpu_index", minimum=0)
        _finite(
            self.maximum_gpu_use_percent,
            name="preflight.maximum_gpu_use_percent",
            minimum=0.0,
            maximum=100.0,
        )
        _finite(
            self.maximum_vram_percent,
            name="preflight.maximum_vram_percent",
            minimum=0.0,
            maximum=100.0,
        )
        if self.admission_mode not in {"shared_quality", "exclusive_performance"}:
            raise ContractError(
                "preflight.admission_mode must be shared_quality or exclusive_performance"
        )
        for identity in self.allowed_stopped_processes:
            identity.validate()
        pids = tuple(identity.pid for identity in self.allowed_stopped_processes)
        if tuple(sorted(set(pids))) != pids:
            raise ContractError(
                "preflight.allowed_stopped_processes must have unique ascending PIDs"
            )
        if self.admission_mode == "exclusive_performance" and pids:
            raise ContractError(
                "exclusive_performance admission cannot allow stopped GPU clients"
            )
        _integer(
            self.command_timeout_seconds,
            name="preflight.command_timeout_seconds",
            minimum=1,
        )


@dataclass(frozen=True, slots=True)
class ProposalOptions:
    frame_start: int = 0
    frame_stop_exclusive: int = 60
    points_per_frame: int = 700_000
    nearest_cameras: int = 2
    seed: int = 0
    world_bound: float = 1_000.0

    def validate(self) -> None:
        start = _integer(self.frame_start, name="proposal.frame_start", minimum=0)
        stop = _integer(
            self.frame_stop_exclusive,
            name="proposal.frame_stop_exclusive",
            minimum=1,
        )
        if stop <= start:
            raise ContractError("proposal.frame_stop_exclusive must exceed frame_start")
        _integer(self.points_per_frame, name="proposal.points_per_frame", minimum=1)
        _integer(self.nearest_cameras, name="proposal.nearest_cameras", minimum=1)
        _integer(self.seed, name="proposal.seed", minimum=0)
        _finite(
            self.world_bound,
            name="proposal.world_bound",
            minimum=0.0,
            exclusive_minimum=True,
        )


@dataclass(frozen=True, slots=True)
class InitializationOptions:
    num_gaussians: int = 500_000
    seed: int = 0
    velocity_neighbors: int = 3
    scale_multiplier: float = 0.1
    sampling_mode: str = "paired_multiview_consensus_rank_mixture"
    sampling_voxel_size: float = 0.02
    sampling_evidence_fraction: float = 0.5
    opacity: float = 0.5
    duration_seconds: float = 0.1
    duration_min_seconds: float = 1.0 / 600.0
    duration_max_seconds: float = 1.0
    time_offset_seconds: float = 0.0

    def validate(self) -> None:
        _integer(self.num_gaussians, name="initialization.num_gaussians", minimum=1)
        _integer(self.seed, name="initialization.seed", minimum=0)
        _integer(
            self.velocity_neighbors,
            name="initialization.velocity_neighbors",
            minimum=1,
        )
        _finite(
            self.scale_multiplier,
            name="initialization.scale_multiplier",
            minimum=0.0,
            exclusive_minimum=True,
        )
        modes = {
            "raw_candidate_uniform",
            "occupied_voxel_uniform",
            "triangulation_information_mixture",
            "paired_matcher_support_rank_mixture",
            "paired_multiview_consensus_rank_mixture",
        }
        if self.sampling_mode not in modes:
            raise ContractError("initialization.sampling_mode is unsupported")
        _finite(
            self.sampling_voxel_size,
            name="initialization.sampling_voxel_size",
            minimum=0.0,
            exclusive_minimum=True,
        )
        evidence = _finite(
            self.sampling_evidence_fraction,
            name="initialization.sampling_evidence_fraction",
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
        )
        opacity = _finite(
            self.opacity,
            name="initialization.opacity",
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
        )
        if evidence > 1.0 or opacity >= 1.0:
            raise ContractError("initialization fractions must lie in their open bounds")
        duration = _finite(self.duration_seconds, name="initialization.duration_seconds")
        duration_min = _finite(
            self.duration_min_seconds,
            name="initialization.duration_min_seconds",
            minimum=0.0,
            exclusive_minimum=True,
        )
        duration_max = _finite(
            self.duration_max_seconds,
            name="initialization.duration_max_seconds",
            minimum=0.0,
            exclusive_minimum=True,
        )
        _finite(self.time_offset_seconds, name="initialization.time_offset_seconds")
        if not duration_min < duration < duration_max:
            raise ContractError(
                "initialization duration must satisfy minimum < value < maximum"
            )


@dataclass(frozen=True, slots=True)
class AssetOptions:
    producer_git_revision: str
    asset_license: str
    redistribution: str
    provenance_summary: str
    world_unit: str = "calibration_unit"
    calibration_scale: float = 1.0
    default_sh_degree: int | None = None

    def validate(self) -> None:
        revision = _string(
            self.producer_git_revision,
            name="asset.producer_git_revision",
        )
        if _GIT_REVISION.fullmatch(revision) is None:
            raise ContractError("asset.producer_git_revision must be a full lowercase hash")
        _string(self.asset_license, name="asset.asset_license")
        if self.redistribution not in {"allowed", "restricted", "review_required"}:
            raise ContractError("asset.redistribution is invalid")
        _string(self.provenance_summary, name="asset.provenance_summary")
        _string(self.world_unit, name="asset.world_unit")
        _finite(
            self.calibration_scale,
            name="asset.calibration_scale",
            minimum=0.0,
            exclusive_minimum=True,
        )
        if self.default_sh_degree is not None:
            _integer(
                self.default_sh_degree,
                name="asset.default_sh_degree",
                minimum=0,
                maximum=3,
            )


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    source_path: Path
    source_bytes: int
    source_sha256: str
    source_git_revision: str
    profile: Path
    observation_manifest: Path
    image_root: Path | None
    roma_indoor_weight: Path
    dinov2_weight: Path
    environment_lock: Path
    preflight: PreflightOptions
    proposal: ProposalOptions
    initialization: InitializationOptions
    asset: AssetOptions

    def validate(self) -> None:
        _regular_input(self.source_path, name="pipeline plan")
        _regular_input(self.profile, name="profile")
        _regular_input(self.observation_manifest, name="observation_manifest")
        if self.image_root is not None:
            _directory_input(self.image_root, name="image_root")
        _regular_input(self.roma_indoor_weight, name="roma_indoor_weight")
        _regular_input(self.dinov2_weight, name="dinov2_weight")
        _regular_input(self.environment_lock, name="environment_lock")
        if self.source_bytes <= 0 or _SHA256.fullmatch(self.source_sha256) is None:
            raise ContractError("pipeline plan source identity is invalid")
        if _GIT_REVISION.fullmatch(self.source_git_revision) is None:
            raise ContractError("source_git_revision must be a full lowercase Git hash")
        self.preflight.validate()
        self.proposal.validate()
        self.initialization.validate()
        self.asset.validate()
        if self.asset.producer_git_revision != self.source_git_revision:
            raise ContractError(
                "asset.producer_git_revision must equal source_git_revision"
            )

    @classmethod
    def load(cls, path: Path) -> Self:
        unresolved = path.expanduser()
        if unresolved.is_symlink():
            raise ContractError("pipeline plan must be a regular non-symlink file")
        resolved = unresolved.resolve()
        _regular_input(resolved, name="pipeline plan")
        try:
            before = resolved.stat()
            payload = resolved.read_bytes()
            after = resolved.stat()
            decoded: Any = tomllib.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ContractError(f"invalid pipeline plan: {exc}") from exc
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise ContractError("pipeline plan changed while it was read")
        if not isinstance(decoded, dict):  # pragma: no cover - tomllib invariant
            raise ContractError("pipeline plan must be a TOML table")
        raw = cast(dict[str, Any], decoded)
        allowed = {
            "schema_version",
            "source_git_revision",
            "profile",
            "observation_manifest",
            "image_root",
            "roma_indoor_weight",
            "dinov2_weight",
            "environment_lock",
            "preflight",
            "proposal",
            "initialization",
            "asset",
        }
        _reject_unknown(raw, allowed, context="pipeline plan")
        if raw.get("schema_version") != PIPELINE_PLAN_SCHEMA:
            raise ContractError(f"pipeline plan must declare {PIPELINE_PLAN_SCHEMA}")
        base = resolved.parent

        preflight_raw = _table(raw.get("preflight"), context="preflight")
        _reject_unknown(
            preflight_raw,
            {
                "gpu_index",
                "maximum_gpu_use_percent",
                "maximum_vram_percent",
                "admission_mode",
                "allowed_stopped_processes",
                "command_timeout_seconds",
            },
            context="preflight",
        )
        allowed_processes_raw = preflight_raw.get("allowed_stopped_processes", [])
        if not isinstance(allowed_processes_raw, list):
            raise ContractError(
                "preflight.allowed_stopped_processes must be a TOML array of tables"
            )
        allowed_processes: list[StoppedProcessOptions] = []
        for index, item in enumerate(cast(list[Any], allowed_processes_raw)):
            if not isinstance(item, dict):
                raise ContractError(
                    "preflight.allowed_stopped_processes must be a TOML array of tables"
                )
            entry = cast(dict[str, Any], item)
            _reject_unknown(
                entry,
                {"pid", "starttime_ticks"},
                context=f"preflight.allowed_stopped_processes[{index}]",
            )
            if set(entry) != {"pid", "starttime_ticks"}:
                raise ContractError(
                    f"preflight.allowed_stopped_processes[{index}] is incomplete"
                )
            allowed_processes.append(StoppedProcessOptions(**entry))
        preflight = PreflightOptions(
            gpu_index=preflight_raw.get("gpu_index", 0),
            maximum_gpu_use_percent=preflight_raw.get("maximum_gpu_use_percent", 100.0),
            maximum_vram_percent=preflight_raw.get("maximum_vram_percent", 80.0),
            admission_mode=preflight_raw.get("admission_mode", "shared_quality"),
            allowed_stopped_processes=tuple(allowed_processes),
            command_timeout_seconds=preflight_raw.get("command_timeout_seconds", 20),
        )

        proposal_raw = _table(raw.get("proposal"), context="proposal")
        _reject_unknown(
            proposal_raw,
            {
                "frame_start",
                "frame_stop_exclusive",
                "points_per_frame",
                "nearest_cameras",
                "seed",
                "world_bound",
            },
            context="proposal",
        )
        proposal = ProposalOptions(**proposal_raw)

        initialization_raw = _table(raw.get("initialization"), context="initialization")
        _reject_unknown(
            initialization_raw,
            {
                "num_gaussians",
                "seed",
                "velocity_neighbors",
                "scale_multiplier",
                "sampling_mode",
                "sampling_voxel_size",
                "sampling_evidence_fraction",
                "opacity",
                "duration_seconds",
                "duration_min_seconds",
                "duration_max_seconds",
                "time_offset_seconds",
            },
            context="initialization",
        )
        initialization = InitializationOptions(**initialization_raw)

        asset_raw = _table(raw.get("asset"), context="asset", required=True)
        _reject_unknown(
            asset_raw,
            {
                "producer_git_revision",
                "asset_license",
                "redistribution",
                "provenance_summary",
                "world_unit",
                "calibration_scale",
                "default_sh_degree",
            },
            context="asset",
        )
        for required in (
            "producer_git_revision",
            "asset_license",
            "redistribution",
            "provenance_summary",
        ):
            if required not in asset_raw:
                raise ContractError(f"asset.{required} is required")
        asset = AssetOptions(**asset_raw)

        plan = cls(
            source_path=resolved,
            source_bytes=len(payload),
            source_sha256=hashlib.sha256(payload).hexdigest(),
            source_git_revision=_string(
                raw.get("source_git_revision"), name="source_git_revision"
            ),
            profile=_path(raw.get("profile"), name="profile", base=base),
            observation_manifest=_path(
                raw.get("observation_manifest"),
                name="observation_manifest",
                base=base,
            ),
            image_root=(
                None
                if raw.get("image_root") is None
                else _path(raw["image_root"], name="image_root", base=base)
            ),
            roma_indoor_weight=_path(
                raw.get("roma_indoor_weight"),
                name="roma_indoor_weight",
                base=base,
            ),
            dinov2_weight=_path(
                raw.get("dinov2_weight"), name="dinov2_weight", base=base
            ),
            environment_lock=_path(
                raw.get("environment_lock"), name="environment_lock", base=base
            ),
            preflight=preflight,
            proposal=proposal,
            initialization=initialization,
            asset=asset,
        )
        plan.validate()
        return plan


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    root: Path

    @property
    def header(self) -> Path:
        return self.root / "workspace.json"

    @property
    def records(self) -> Path:
        return self.root / "stages"

    @property
    def plans(self) -> Path:
        return self.root / "plans"

    @property
    def preflights(self) -> Path:
        return self.root / "preflight"

    @property
    def resource_windows(self) -> Path:
        return self.root / "resource-window"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def quarantine(self) -> Path:
        return self.artifacts / "quarantine"

    @property
    def tensor_cache(self) -> Path:
        return self.artifacts / "tensor-cache"

    @property
    def proposals(self) -> Path:
        return self.artifacts / "proposals"

    @property
    def initialization(self) -> Path:
        return self.artifacts / "initialization"

    @property
    def run_config(self) -> Path:
        return self.artifacts / "resolved-run.toml"

    @property
    def run(self) -> Path:
        return self.artifacts / "run"

    @property
    def evaluation(self) -> Path:
        return self.artifacts / "evaluation"

    @property
    def asset(self) -> Path:
        return self.artifacts / "asset"

    @property
    def completion(self) -> Path:
        return self.root / "pipeline.json"

    def record(self, stage: StageName) -> Path:
        return self.records / f"{STAGE_ORDER.index(stage):02d}-{stage}.json"

    def receipt(self, stage: StageName) -> Path:
        return {
            "prepare": self.tensor_cache / "tensor_cache.json",
            "propose": self.proposals / "collection.json",
            "initialize": self.initialization / "initialization.json",
            "train": self.run / "training.json",
            "evaluate": self.evaluation / "evaluation.json",
            "asset": self.asset / "manifest.json",
        }[stage]


class PipelineBackend(Protocol):
    def preflight(self, options: PreflightOptions) -> dict[str, Any]: ...

    def execute_guarded(
        self,
        stage: StageName,
        *,
        plan: PipelinePlan,
        paths: PipelinePaths,
        progress: Progress,
        resource_window: Path,
    ) -> Path: ...

    def execute(
        self,
        stage: StageName,
        *,
        plan: PipelinePlan,
        paths: PipelinePaths,
        progress: Progress,
    ) -> Path: ...


def _read_json_object(path: Path, *, label: str, canonical: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be a regular non-symlink file")
    try:
        payload = path.read_bytes()
        decoded: Any = json.loads(payload)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ContractError(f"{label} must be a JSON object")
    result = cast(dict[str, Any], decoded)
    if canonical and canonical_json_bytes(result) != payload:
        raise ContractError(f"{label} must use canonical JSON encoding")
    return result


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _ensure_directory(path: Path, *, label: str) -> None:
    if _exists(path):
        if path.is_symlink() or not path.is_dir():
            raise ContractError(f"{label} must be a regular non-symlink directory")
        return
    try:
        path.mkdir(parents=True)
    except FileExistsError as exc:
        raise ContractError(f"{label} appeared with an unsafe type") from exc


def _file_identity(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be a regular non-symlink file")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ContractError(f"{label} must be a regular file")
    digest = sha256_file(path)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ContractError(f"{label} changed while its digest was computed")
    return {"bytes": before.st_size, "sha256": digest}


def _workspace_header() -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": WORKSPACE_SCHEMA,
        "orchestrator": ORCHESTRATOR_VERSION,
        "stage_order": list(STAGE_ORDER),
        "layout": "stage_scoped_plan_history_and_fixed_outputs_v2",
        "plan_policy": "append_only_snapshots_stage_scoped_requests_v1",
        "claim_boundary": (
            "Each completed stage binds only its semantic request and upstream receipts. "
            "Plan history is append-only; release qualification still requires one revision."
        ),
    }
    return {**unsigned, "logical_sha256": sha256_json(unsigned)}


def _validate_header(header: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "orchestrator",
        "stage_order",
        "layout",
        "plan_policy",
        "claim_boundary",
        "logical_sha256",
    }
    if set(header) != expected:
        raise ContractError("workspace header fields are invalid")
    unsigned = dict(header)
    logical = unsigned.pop("logical_sha256")
    if (
        header.get("schema_version") != WORKSPACE_SCHEMA
        or header.get("orchestrator") != ORCHESTRATOR_VERSION
        or header.get("stage_order") != list(STAGE_ORDER)
        or not isinstance(logical, str)
        or logical != sha256_json(unsigned)
    ):
        raise ContractError("workspace header identity is invalid")
    if (
        header.get("layout") != "stage_scoped_plan_history_and_fixed_outputs_v2"
        or header.get("plan_policy")
        != "append_only_snapshots_stage_scoped_requests_v1"
    ):
        raise ContractError("workspace stage-scoped plan policy is invalid")


def _plan_snapshots(paths: PipelinePaths) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for candidate in paths.plans.iterdir():
        if candidate.is_symlink() or not candidate.is_file():
            raise ContractError("plan-history directory contains an unsafe entry")
        match = _PLAN_SNAPSHOT.fullmatch(candidate.name)
        if match is None:
            raise ContractError("plan-history filename is invalid")
        identity = _file_identity(candidate, label="pipeline-plan snapshot")
        if identity["bytes"] <= 0 or identity["sha256"] != match.group("sha256"):
            raise ContractError("pipeline-plan snapshot identity is invalid")
        snapshots.append(
            {
                "attempt": int(match.group("attempt")),
                "path": f"plans/{candidate.name}",
                **identity,
            }
        )
    snapshots.sort(key=lambda item: cast(int, item["attempt"]))
    if [item["attempt"] for item in snapshots] != list(range(1, len(snapshots) + 1)):
        raise ContractError("pipeline-plan snapshot sequence is not contiguous")
    return snapshots


def _register_plan(paths: PipelinePaths, plan: PipelinePlan) -> dict[str, Any]:
    payload = plan.source_path.read_bytes()
    if (
        len(payload) != plan.source_bytes
        or hashlib.sha256(payload).hexdigest() != plan.source_sha256
    ):
        raise ContractError("pipeline plan changed after validation")
    snapshots = _plan_snapshots(paths)
    if snapshots and snapshots[-1]["sha256"] == plan.source_sha256:
        return snapshots[-1]
    if _exists(paths.completion):
        completion = _read_json_object(
            paths.completion,
            label="pipeline completion seal",
            canonical=True,
        )
        if completion.get("terminal_plan_sha256") != plan.source_sha256:
            raise ContractError("a completed workspace cannot adopt a different plan")
    attempt = len(snapshots) + 1
    destination = paths.plans / f"{attempt:06d}-{plan.source_sha256}.toml"
    write_new_bytes(destination, payload)
    return {
        "attempt": attempt,
        "path": f"plans/{destination.name}",
        **_file_identity(destination, label="pipeline-plan snapshot"),
    }


def _open_workspace(plan: PipelinePlan, workspace: Path) -> PipelinePaths:
    unresolved = workspace.expanduser()
    if unresolved.is_symlink():
        raise ContractError("pipeline workspace must not be a symlink")
    root = unresolved.resolve()
    if _exists(root):
        if root.is_symlink() or not root.is_dir():
            raise ContractError("pipeline workspace must be a regular directory")
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            root.mkdir()
        except FileExistsError as exc:
            raise ContractError("pipeline workspace appeared with an unsafe type") from exc
    paths = PipelinePaths(root)
    expected = _workspace_header()
    if _exists(paths.header):
        existing = _read_json_object(paths.header, label="workspace header", canonical=True)
        _validate_header(existing)
        if existing != expected:
            raise ContractError("workspace header differs from this orchestrator")
    else:
        existing_entries = list(root.iterdir())
        if existing_entries:
            raise ContractError("workspace without a header must be empty")
        write_new_bytes(paths.header, canonical_json_bytes(expected))
    _ensure_directory(paths.records, label="workspace stage-record directory")
    _ensure_directory(paths.plans, label="workspace plan-history directory")
    _ensure_directory(paths.preflights, label="workspace preflight directory")
    _ensure_directory(paths.resource_windows, label="workspace resource-window directory")
    _ensure_directory(paths.artifacts, label="workspace artifact directory")
    _register_plan(paths, plan)
    return paths


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError("pipeline artifact escaped the workspace") from exc
    return relative.as_posix()


def _receipt_identity(path: Path, *, expected: Path, root: Path) -> dict[str, Any]:
    if path.resolve() != expected.resolve():
        raise ContractError("stage backend returned an unexpected receipt path")
    payload = _read_json_object(path, label="stage receipt")
    identity = _file_identity(path, label="stage receipt")
    schema = payload.get("schema_version", payload.get("schema"))
    if not isinstance(schema, str) or not schema:
        raise ContractError("stage receipt has no schema identity")
    return {
        "path": _relative(path, root),
        **identity,
        "schema": schema,
        "status": payload.get("status"),
    }


def _validate_stage_record(
    record: dict[str, Any], *, stage: StageName, paths: PipelinePaths
) -> None:
    expected_fields = {
        "schema_version",
        "status",
        "stage",
        "ordinal",
        "request",
        "input_sha256",
        "receipt",
        "logical_sha256",
    }
    if stage in GPU_STAGES:
        expected_fields.add("resource_window")
    if set(record) != expected_fields:
        raise ContractError(f"{stage} stage record fields are invalid")
    unsigned = dict(record)
    logical = unsigned.pop("logical_sha256")
    request = record.get("request")
    if (
        record.get("schema_version") != STAGE_RECORD_SCHEMA
        or record.get("status") != "COMPLETE"
        or record.get("stage") != stage
        or record.get("ordinal") != STAGE_ORDER.index(stage)
        or not isinstance(request, dict)
        or record.get("input_sha256") != sha256_json(request)
        or not isinstance(logical, str)
        or logical != sha256_json(unsigned)
    ):
        raise ContractError(f"{stage} stage record identity is invalid")
    raw_receipt: object = record.get("receipt")
    if not isinstance(raw_receipt, dict):
        raise ContractError(f"{stage} receipt identity is invalid")
    receipt = cast(dict[str, Any], raw_receipt)
    expected_receipt = paths.receipt(stage)
    expected_relative = _relative(expected_receipt, paths.root)
    if receipt.get("path") != expected_relative:
        raise ContractError(f"{stage} receipt path is invalid")
    actual = _receipt_identity(
        expected_receipt,
        expected=expected_receipt,
        root=paths.root,
    )
    if actual != receipt:
        raise ContractError(f"{stage} receipt changed after completion")


def _load_stage_record(paths: PipelinePaths, stage: StageName) -> dict[str, Any] | None:
    record_path = paths.record(stage)
    if not _exists(record_path):
        return None
    record = _read_json_object(record_path, label=f"{stage} stage record", canonical=True)
    _validate_stage_record(record, stage=stage, paths=paths)
    return record


def _validate_file_identity_fields(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} identity must be an object")
    identity = cast(dict[str, Any], value)
    if (
        set(identity) != {"bytes", "sha256"}
        or type(identity.get("bytes")) is not int
        or cast(int, identity["bytes"]) < 0
        or not isinstance(identity.get("sha256"), str)
        or _SHA256.fullmatch(cast(str, identity["sha256"])) is None
    ):
        raise ContractError(f"{label} identity is invalid")
    return identity


def _validate_stage_link(
    record: dict[str, Any],
    *,
    stage: StageName,
    prior: Mapping[StageName, dict[str, Any]],
    paths: PipelinePaths,
) -> None:
    request = cast(dict[str, Any], record["request"])
    expected_fields = {
        "schema_version",
        "stage",
        "parameters",
        "direct_inputs",
        "implementation",
        "upstream",
    }
    if stage in GPU_STAGES:
        expected_fields.add("preflight")
    if (
        set(request) != expected_fields
        or request.get("schema_version") != "p2g.pipeline_stage_request.v2"
        or request.get("stage") != stage
        or not isinstance(request.get("parameters"), dict)
    ):
        raise ContractError(f"{stage} stage request is invalid")

    raw_direct_inputs: object = request.get("direct_inputs")
    if not isinstance(raw_direct_inputs, dict):
        raise ContractError(f"{stage} direct-input catalog is invalid")
    direct_inputs = cast(dict[object, object], raw_direct_inputs)
    for label, identity in direct_inputs.items():
        if not isinstance(label, str) or not label:
            raise ContractError(f"{stage} direct-input label is invalid")
        _validate_file_identity_fields(identity, label=f"{stage} direct input {label}")

    raw_implementation: object = request.get("implementation")
    if not isinstance(raw_implementation, dict):
        raise ContractError(f"{stage} implementation identity is invalid")
    implementation = cast(dict[str, Any], raw_implementation)
    if set(implementation) != {
        "operator",
        "declared_git_revision",
        "entry_modules",
        "files",
        "logical_sha256",
    }:
        raise ContractError(f"{stage} implementation identity fields are invalid")
    logical = implementation.get("logical_sha256")
    unsigned_implementation = dict(implementation)
    unsigned_implementation.pop("logical_sha256")
    entries = implementation.get("entry_modules")
    files = implementation.get("files")
    if (
        implementation.get("operator")
        != "p2g.recursive_stage_python_source_closure.v1"
        or not isinstance(implementation.get("declared_git_revision"), str)
        or _GIT_REVISION.fullmatch(
            cast(str, implementation["declared_git_revision"])
        )
        is None
        or entries != list(_STAGE_ENTRY_MODULES[stage])
        or not isinstance(files, list)
        or not isinstance(logical, str)
        or logical != sha256_json(unsigned_implementation)
    ):
        raise ContractError(f"{stage} implementation identity is invalid")
    prior_path = ""
    for untyped_file in cast(list[Any], files):
        if not isinstance(untyped_file, dict):
            raise ContractError(f"{stage} implementation file identity is invalid")
        implementation_file = cast(dict[str, Any], untyped_file)
        if set(implementation_file) != {"path", "bytes", "sha256"}:
            raise ContractError(f"{stage} implementation file fields are invalid")
        source_path = implementation_file.get("path")
        if (
            not isinstance(source_path, str)
            or not source_path.startswith("p2g/")
            or source_path <= prior_path
            or ".." in Path(source_path).parts
        ):
            raise ContractError(f"{stage} implementation source path is invalid")
        _validate_file_identity_fields(
            {
                "bytes": implementation_file.get("bytes"),
                "sha256": implementation_file.get("sha256"),
            },
            label=f"{stage} implementation file",
        )
        prior_path = source_path

    expected_upstream = {
        name: cast(str, prior[name]["logical_sha256"])
        for name in STAGE_ORDER[: STAGE_ORDER.index(stage)]
    }
    if request.get("upstream") != expected_upstream:
        raise ContractError(f"{stage} upstream stage binding is invalid")

    if stage not in GPU_STAGES:
        return
    raw_preflight: object = request.get("preflight")
    if not isinstance(raw_preflight, dict):
        raise ContractError(f"{stage} preflight identity is invalid")
    preflight = cast(dict[str, Any], raw_preflight)
    if set(preflight) != {"path", "bytes", "sha256"}:
        raise ContractError(f"{stage} preflight identity fields are invalid")
    relative = preflight.get("path")
    expected_prefix = f"preflight/{STAGE_ORDER.index(stage):02d}-{stage}-"
    if (
        not isinstance(relative, str)
        or not relative.startswith(expected_prefix)
        or re.fullmatch(r"[0-9]{6}\.json", relative[len(expected_prefix) :]) is None
    ):
        raise ContractError(f"{stage} preflight path is invalid")
    identity = _validate_file_identity_fields(
        {"bytes": preflight.get("bytes"), "sha256": preflight.get("sha256")},
        label=f"{stage} preflight",
    )
    candidate = (paths.root / relative).resolve()
    if _relative(candidate, paths.root) != relative:
        raise ContractError(f"{stage} preflight escaped the workspace")
    actual = _file_identity(candidate, label=f"{stage} preflight receipt")
    if actual != identity:
        raise ContractError(f"{stage} preflight receipt changed after completion")
    receipt = _read_json_object(
        candidate,
        label=f"{stage} preflight receipt",
        canonical=True,
    )
    if receipt.get("status") != "PASS":
        raise ContractError(f"{stage} completed without a passing preflight")
    raw_window: object = record.get("resource_window")
    if not isinstance(raw_window, dict):
        raise ContractError(f"{stage} resource-window identity is invalid")
    window = cast(dict[str, Any], raw_window)
    if set(window) != {"path", "bytes", "sha256"}:
        raise ContractError(f"{stage} resource-window identity fields are invalid")
    window_relative = window.get("path")
    window_prefix = f"resource-window/{STAGE_ORDER.index(stage):02d}-{stage}-"
    if (
        not isinstance(window_relative, str)
        or not window_relative.startswith(window_prefix)
        or re.fullmatch(
            r"[0-9]{6}\.json", window_relative[len(window_prefix) :]
        )
        is None
    ):
        raise ContractError(f"{stage} resource-window path is invalid")
    window_identity = _validate_file_identity_fields(
        {"bytes": window.get("bytes"), "sha256": window.get("sha256")},
        label=f"{stage} resource window",
    )
    window_candidate = (paths.root / window_relative).resolve()
    if _relative(window_candidate, paths.root) != window_relative:
        raise ContractError(f"{stage} resource-window receipt escaped the workspace")
    actual_window = _file_identity(
        window_candidate, label=f"{stage} resource-window receipt"
    )
    if actual_window != window_identity:
        raise ContractError(f"{stage} resource-window receipt changed after completion")
    window_receipt = _read_json_object(
        window_candidate,
        label=f"{stage} resource-window receipt",
        canonical=True,
    )
    if (
        window_receipt.get("status") != "PASS"
        or window_receipt.get("operation_status") != "RETURNED"
    ):
        raise ContractError(f"{stage} completed without a passing resource window")


def _stage_parameters(plan: PipelinePlan, stage: StageName) -> dict[str, Any]:
    if stage == "prepare":
        result = {
            "image_root": "manifest_parent" if plan.image_root is None else "explicit"
        }
    elif stage == "propose":
        result = asdict(plan.proposal)
    elif stage == "initialize":
        result = asdict(plan.initialization)
    elif stage == "train":
        result = {"configuration": "profile_plus_workspace_artifacts_v1"}
    elif stage == "evaluate":
        result = {"source": "completed_exported_run"}
    elif stage == "asset":
        result = {
            "producer_git_revision": plan.asset.producer_git_revision,
            "asset_license": plan.asset.asset_license,
            "redistribution": plan.asset.redistribution,
            "provenance_summary_sha256": hashlib.sha256(
                plan.asset.provenance_summary.encode("utf-8")
            ).hexdigest(),
            "world_unit": plan.asset.world_unit,
            "calibration_scale": plan.asset.calibration_scale,
            "default_sh_degree": plan.asset.default_sh_degree,
        }
    else:  # pragma: no cover - StageName exhaustiveness
        raise AssertionError(f"unhandled pipeline stage: {stage}")
    if stage in GPU_STAGES:
        result["admission_mode"] = plan.preflight.admission_mode
    return result


def _direct_inputs(plan: PipelinePlan, stage: StageName) -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {}
    if stage in {"prepare", "propose", "train"}:
        paths["observation_manifest"] = plan.observation_manifest
    if stage == "propose":
        paths.update(
            {
                "roma_indoor_weight": plan.roma_indoor_weight,
                "dinov2_weight": plan.dinov2_weight,
                "environment_lock": plan.environment_lock,
            }
        )
    if stage == "train":
        paths["profile"] = plan.profile
    return {
        label: _file_identity(path, label=f"pipeline input {label}")
        for label, path in sorted(paths.items())
    }


_STAGE_ENTRY_MODULES: dict[StageName, tuple[str, ...]] = {
    "prepare": ("p2g.training.prepare",),
    "propose": ("p2g.training.roma_point_sequence",),
    "initialize": ("p2g.training.build_initialization",),
    "train": ("p2g.training.train",),
    "evaluate": ("p2g.training.evaluate",),
    "asset": ("p2g.training.asset", "p2g.training.train"),
}


def _module_source(module: str, *, package_root: Path) -> Path | None:
    if module == "p2g":
        return package_root / "__init__.py"
    if not module.startswith("p2g."):
        return None
    relative = module.split(".")[1:]
    module_file = package_root.joinpath(*relative).with_suffix(".py")
    if module_file.is_file() and not module_file.is_symlink():
        return module_file
    package_file = package_root.joinpath(*relative, "__init__.py")
    if package_file.is_file() and not package_file.is_symlink():
        return package_file
    return None


def _stage_implementation(plan: PipelinePlan, stage: StageName) -> dict[str, Any]:
    """Hash the recursive in-package Python source closure for one stage."""

    package_root = Path(__file__).resolve().parent
    pending = list(_STAGE_ENTRY_MODULES[stage])
    discovered: dict[str, Path] = {}
    while pending:
        module = pending.pop()
        if module in discovered:
            continue
        source = _module_source(module, package_root=package_root)
        if source is None:
            raise ContractError(f"cannot resolve stage implementation module {module}")
        discovered[module] = source
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ContractError(
                f"cannot inspect stage implementation module {module}: {exc}"
            ) from exc
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names if alias.name.startswith("p2g"))
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.startswith("p2g")
            ):
                imports.add(node.module)
                for alias in node.names:
                    candidate = f"{node.module}.{alias.name}"
                    if _module_source(candidate, package_root=package_root) is not None:
                        imports.add(candidate)
        for imported in imports:
            components = imported.split(".")
            for length in range(1, len(components) + 1):
                candidate = ".".join(components[:length])
                if _module_source(candidate, package_root=package_root) is not None:
                    pending.append(candidate)

    files: list[dict[str, Any]] = []
    for source in sorted(set(discovered.values())):
        identity = _file_identity(source, label=f"{stage} implementation source")
        files.append(
            {
                "path": f"p2g/{source.relative_to(package_root).as_posix()}",
                **identity,
            }
        )
    unsigned: dict[str, Any] = {
        "operator": "p2g.recursive_stage_python_source_closure.v1",
        "declared_git_revision": plan.source_git_revision,
        "entry_modules": list(_STAGE_ENTRY_MODULES[stage]),
        "files": files,
    }
    return {**unsigned, "logical_sha256": sha256_json(unsigned)}


def _stage_request(
    plan: PipelinePlan,
    stage: StageName,
    prior: Mapping[StageName, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "p2g.pipeline_stage_request.v2",
        "stage": stage,
        "parameters": _stage_parameters(plan, stage),
        "direct_inputs": _direct_inputs(plan, stage),
        "implementation": _stage_implementation(plan, stage),
        "upstream": {
            name: cast(str, prior[name]["logical_sha256"])
            for name in STAGE_ORDER[: STAGE_ORDER.index(stage)]
        },
    }


def _semantic_stage_request(request: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(request)
    result.pop("preflight", None)
    raw_implementation = result.get("implementation")
    if isinstance(raw_implementation, dict):
        implementation = dict(cast(dict[str, Any], raw_implementation))
        implementation.pop("declared_git_revision", None)
        implementation.pop("logical_sha256", None)
        result["implementation"] = implementation
    return result


def _preflight_attempt_indices(paths: PipelinePaths) -> dict[StageName, list[int]]:
    indices: dict[StageName, list[int]] = {stage: [] for stage in GPU_STAGES}
    for candidate in paths.preflights.iterdir():
        if candidate.is_symlink() or not candidate.is_file() or candidate.suffix != ".json":
            raise ContractError("preflight directory contains an unsafe entry")
        _read_json_object(candidate, label="preflight attempt", canonical=True)
        name = candidate.name
        matched: StageName | None = None
        raw_index: str | None = None
        for candidate_stage in GPU_STAGES:
            prefix = f"{STAGE_ORDER.index(candidate_stage):02d}-{candidate_stage}-"
            if name.startswith(prefix) and name.endswith(".json"):
                matched = candidate_stage
                raw_index = name[len(prefix) : -len(".json")]
                break
        if matched is None or raw_index is None or len(raw_index) != 6 or not raw_index.isdigit():
            raise ContractError("preflight attempt filename is invalid")
        indices[matched].append(int(raw_index))
    for stage, attempts in indices.items():
        attempts.sort()
        if attempts != list(range(1, len(attempts) + 1)):
            raise ContractError(f"{stage} preflight attempt sequence is not contiguous")
    return indices


def _next_preflight_path(paths: PipelinePaths, stage: StageName) -> Path:
    ordinal = STAGE_ORDER.index(stage)
    prefix = f"{ordinal:02d}-{stage}-"
    indices = _preflight_attempt_indices(paths)[stage]
    return paths.preflights / f"{prefix}{len(indices) + 1:06d}.json"


def _resource_window_attempt_indices(paths: PipelinePaths) -> dict[StageName, list[int]]:
    indices: dict[StageName, list[int]] = {stage: [] for stage in GPU_STAGES}
    for candidate in paths.resource_windows.iterdir():
        if candidate.is_symlink() or not candidate.is_file() or candidate.suffix != ".json":
            raise ContractError("resource-window directory contains an unsafe entry")
        _read_json_object(candidate, label="resource-window attempt", canonical=True)
        matched: StageName | None = None
        raw_index: str | None = None
        for candidate_stage in GPU_STAGES:
            prefix = f"{STAGE_ORDER.index(candidate_stage):02d}-{candidate_stage}-"
            if candidate.name.startswith(prefix) and candidate.name.endswith(".json"):
                matched = candidate_stage
                raw_index = candidate.name[len(prefix) : -len(".json")]
                break
        if (
            matched is None
            or raw_index is None
            or len(raw_index) != 6
            or not raw_index.isdigit()
        ):
            raise ContractError("resource-window attempt filename is invalid")
        indices[matched].append(int(raw_index))
    for stage, attempts in indices.items():
        attempts.sort()
        if attempts != list(range(1, len(attempts) + 1)):
            raise ContractError(f"{stage} resource-window attempt sequence is not contiguous")
    return indices


def _next_resource_window_path(paths: PipelinePaths, stage: StageName) -> Path:
    ordinal = STAGE_ORDER.index(stage)
    prefix = f"{ordinal:02d}-{stage}-"
    indices = _resource_window_attempt_indices(paths)[stage]
    return paths.resource_windows / f"{prefix}{len(indices) + 1:06d}.json"


def _resource_window_identity(path: Path, *, paths: PipelinePaths) -> dict[str, Any]:
    receipt = _read_json_object(path, label="resource-window receipt", canonical=True)
    if receipt.get("status") != "PASS" or receipt.get("operation_status") != "RETURNED":
        raise ContractError("GPU stage did not close a passing resource window")
    schema = receipt.get("schema_version")
    if not isinstance(schema, str) or not schema:
        raise ContractError("resource-window receipt has no schema identity")
    return {
        "path": _relative(path, paths.root),
        **_file_identity(path, label="resource-window receipt"),
    }


def _capture_preflight(
    backend: PipelineBackend,
    *,
    plan: PipelinePlan,
    paths: PipelinePaths,
    stage: StageName,
) -> dict[str, Any]:
    receipt = backend.preflight(plan.preflight)
    destination = _next_preflight_path(paths, stage)
    write_new_bytes(destination, canonical_json_bytes(receipt))
    identity = {
        "path": _relative(destination, paths.root),
        **_file_identity(destination, label="preflight receipt"),
    }
    if receipt.get("status") != "PASS":
        raise ContractError(
            f"MI300X preflight for {stage} did not pass; receipt {identity['path']} was retained"
        )
    return identity


def _publish_stage_record(
    *,
    paths: PipelinePaths,
    stage: StageName,
    request: dict[str, Any],
    receipt_path: Path,
    resource_window_path: Path | None = None,
) -> dict[str, Any]:
    receipt = _receipt_identity(
        receipt_path,
        expected=paths.receipt(stage),
        root=paths.root,
    )
    unsigned: dict[str, Any] = {
        "schema_version": STAGE_RECORD_SCHEMA,
        "status": "COMPLETE",
        "stage": stage,
        "ordinal": STAGE_ORDER.index(stage),
        "request": request,
        "input_sha256": sha256_json(request),
        "receipt": receipt,
    }
    if stage in GPU_STAGES:
        if resource_window_path is None:
            raise ContractError(f"{stage} has no resource-window receipt")
        unsigned["resource_window"] = _resource_window_identity(
            resource_window_path,
            paths=paths,
        )
    elif resource_window_path is not None:
        raise ContractError(f"{stage} unexpectedly has a resource-window receipt")
    record = {**unsigned, "logical_sha256": sha256_json(unsigned)}
    write_new_bytes(paths.record(stage), canonical_json_bytes(record))
    return record


def _completion_payload(
    records: Mapping[StageName, dict[str, Any]],
    *,
    terminal_plan_sha256: str,
    terminal_source_git_revision: str,
) -> dict[str, Any]:
    stage_source_revisions = {
        stage: cast(
            str,
            cast(
                dict[str, Any],
                cast(dict[str, Any], records[stage]["request"])["implementation"],
            )["declared_git_revision"],
        )
        for stage in STAGE_ORDER
    }
    single_revision = set(stage_source_revisions.values()) == {
        terminal_source_git_revision
    }
    unsigned: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "orchestrator": ORCHESTRATOR_VERSION,
        "terminal_plan_sha256": terminal_plan_sha256,
        "terminal_source_git_revision": terminal_source_git_revision,
        "stage_source_revisions": stage_source_revisions,
        "revision_consistency": (
            "SINGLE_REVISION" if single_revision else "MIXED_REVISION"
        ),
        "stage_records": {
            stage: cast(str, records[stage]["logical_sha256"]) for stage in STAGE_ORDER
        },
        "final_outputs": {
            "asset_manifest_sha256": cast(dict[str, Any], records["asset"]["receipt"])[
                "sha256"
            ],
        },
        "claim_boundary": (
            "The configured pipeline completed with hash-bound receipts; release, data-rights, "
            "scene-quality, and performance claims require their separate gates."
        ),
    }
    return {**unsigned, "logical_sha256": sha256_json(unsigned)}


def pipeline_status(workspace: Path) -> dict[str, Any]:
    unresolved = workspace.expanduser()
    if unresolved.is_symlink():
        raise ContractError("pipeline workspace must not be a symlink")
    root = unresolved.resolve()
    _directory_input(root, name="pipeline workspace")
    paths = PipelinePaths(root)
    header = _read_json_object(paths.header, label="workspace header", canonical=True)
    _validate_header(header)
    _directory_input(paths.records, name="workspace stage-record directory")
    _directory_input(paths.plans, name="workspace plan-history directory")
    _directory_input(paths.preflights, name="workspace preflight directory")
    _directory_input(paths.resource_windows, name="workspace resource-window directory")
    _directory_input(paths.artifacts, name="workspace artifact directory")
    snapshots = _plan_snapshots(paths)
    if not snapshots:
        raise ContractError("workspace contains no pipeline-plan snapshot")
    latest_plan_sha256 = cast(str, snapshots[-1]["sha256"])
    expected_record_names = {paths.record(stage).name for stage in STAGE_ORDER}
    for path in paths.records.iterdir():
        if path.is_symlink() or not path.is_file() or path.name not in expected_record_names:
            raise ContractError("stage-record directory contains an unsafe entry")
    records: dict[StageName, dict[str, Any]] = {}
    gap_seen = False
    stages: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        record = _load_stage_record(paths, stage)
        if record is None:
            gap_seen = True
            stages.append({"name": stage, "status": "PENDING"})
            continue
        if gap_seen:
            raise ContractError("pipeline stage records are out of order")
        _validate_stage_link(
            record,
            stage=stage,
            prior=records,
            paths=paths,
        )
        records[stage] = record
        stages.append(
            {
                "name": stage,
                "status": "COMPLETE",
                "logical_sha256": record["logical_sha256"],
                "receipt_sha256": cast(dict[str, Any], record["receipt"])["sha256"],
            }
        )
    preflight_count = sum(
        len(attempts) for attempts in _preflight_attempt_indices(paths).values()
    )
    resource_window_count = sum(
        len(attempts) for attempts in _resource_window_attempt_indices(paths).values()
    )
    completed = len(records)
    completion_logical: str | None = None
    completion_exists = _exists(paths.completion)
    if completion_exists:
        if completed != len(STAGE_ORDER):
            raise ContractError("pipeline completion seal exists before all stages")
        actual = _read_json_object(
            paths.completion,
            label="pipeline completion seal",
            canonical=True,
        )
        terminal_plan_sha256 = actual.get("terminal_plan_sha256")
        terminal_source_git_revision = actual.get("terminal_source_git_revision")
        if (
            not isinstance(terminal_plan_sha256, str)
            or terminal_plan_sha256 not in {item["sha256"] for item in snapshots}
            or not isinstance(terminal_source_git_revision, str)
            or _GIT_REVISION.fullmatch(terminal_source_git_revision) is None
        ):
            raise ContractError("pipeline completion names an unknown terminal plan")
        terminal_snapshot = next(
            item for item in snapshots if item["sha256"] == terminal_plan_sha256
        )
        terminal_plan_path = paths.root / cast(str, terminal_snapshot["path"])
        try:
            terminal_plan = tomllib.loads(
                terminal_plan_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ContractError(f"cannot decode terminal pipeline-plan snapshot: {exc}") from exc
        if terminal_plan.get("source_git_revision") != terminal_source_git_revision:
            raise ContractError("pipeline completion source revision differs from its plan")
        expected = _completion_payload(
            records,
            terminal_plan_sha256=terminal_plan_sha256,
            terminal_source_git_revision=terminal_source_git_revision,
        )
        if actual != expected:
            raise ContractError("pipeline completion seal is invalid")
        completion_logical = cast(str, actual["logical_sha256"])
        status = "COMPLETE"
    elif completed == len(STAGE_ORDER):
        status = "READY_TO_SEAL"
    else:
        status = "PARTIAL"
    next_stage = None if completed == len(STAGE_ORDER) else STAGE_ORDER[completed]
    return {
        "schema_version": STATUS_SCHEMA,
        "status": status,
        "latest_plan_sha256": latest_plan_sha256,
        "plan_snapshot_count": len(snapshots),
        "completed_stage_count": completed,
        "stage_count": len(STAGE_ORDER),
        "next_stage": next_stage,
        "preflight_attempt_count": preflight_count,
        "resource_window_attempt_count": resource_window_count,
        "completion_logical_sha256": completion_logical,
        "stages": stages,
    }


def _write_or_verify_bytes(path: Path, payload: bytes, *, label: str) -> None:
    if _exists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ContractError(f"existing {label} differs from the planned bytes")
        return
    write_new_bytes(path, payload)


def _stage_output_directory(paths: PipelinePaths, stage: StageName) -> Path:
    return {
        "prepare": paths.tensor_cache,
        "propose": paths.proposals,
        "initialize": paths.initialization,
        "train": paths.run,
        "evaluate": paths.evaluation,
        "asset": paths.asset,
    }[stage]


def _quarantine_stage_output(
    paths: PipelinePaths,
    stage: StageName,
    *,
    reason: str,
    progress: Progress,
) -> Path | None:
    source = _stage_output_directory(paths, stage)
    if not _exists(source):
        return None
    from p2g.quarantine import quarantine_stage_directory

    receipt = quarantine_stage_directory(
        source,
        workspace_root=paths.root,
        quarantine_root=paths.quarantine,
        ordinal=STAGE_ORDER.index(stage),
        stage=stage,
        reason=reason,
    )
    progress(f"QUARANTINE {stage}: {_relative(receipt, paths.root)}")
    return receipt


class DefaultPipelineBackend:
    """Lazy adapter that calls the existing public stage implementations."""

    def preflight(self, options: PreflightOptions) -> dict[str, Any]:
        from p2g.gpu_preflight import (
            Mi300xPreflightConfig,
            StoppedProcessIdentity,
            capture_mi300x_preflight,
            current_process_identity,
        )

        config = Mi300xPreflightConfig(
            gpu_index=options.gpu_index,
            maximum_gpu_use_percent=options.maximum_gpu_use_percent,
            maximum_vram_percent=options.maximum_vram_percent,
            admission_mode=options.admission_mode,
            owner_process=current_process_identity(),
            allowed_stopped_processes=tuple(
                StoppedProcessIdentity(
                    pid=identity.pid,
                    starttime_ticks=identity.starttime_ticks,
                )
                for identity in options.allowed_stopped_processes
            ),
            command_timeout_seconds=options.command_timeout_seconds,
        )
        return capture_mi300x_preflight(config=config)

    def execute_guarded(
        self,
        stage: StageName,
        *,
        plan: PipelinePlan,
        paths: PipelinePaths,
        progress: Progress,
        resource_window: Path,
    ) -> Path:
        from p2g.gpu_preflight import (
            ProcessIdentity,
            StoppedProcessIdentity,
            current_process_identity,
        )
        from p2g.gpu_resource_window import (
            Mi300xResourceWindowConfig,
            run_in_mi300x_resource_window,
        )

        if stage not in GPU_STAGES:
            raise ContractError("a resource window may wrap only a GPU stage")
        options = plan.preflight
        owner: ProcessIdentity = current_process_identity()
        guard = Mi300xResourceWindowConfig(
            gpu_index=options.gpu_index,
            admission_mode=options.admission_mode,
            owner_process=owner,
            allowed_stopped_processes=tuple(
                StoppedProcessIdentity(
                    pid=identity.pid,
                    starttime_ticks=identity.starttime_ticks,
                )
                for identity in options.allowed_stopped_processes
            ),
            command_timeout_seconds=options.command_timeout_seconds,
        )
        return run_in_mi300x_resource_window(
            lambda: self.execute(
                stage,
                plan=plan,
                paths=paths,
                progress=progress,
            ),
            output=resource_window,
            stage=cast(Any, stage),
            config=guard,
        )

    def _resolved_config(self, plan: PipelinePlan, paths: PipelinePaths) -> Any:
        from p2g.training.config import (
            PortableProfile,
            RunConfig,
            SceneInputs,
            TensorMemmapConfig,
        )

        cache = _read_json_object(
            paths.tensor_cache / "tensor_cache.json",
            label="tensor-cache manifest",
        )
        raw_camera_ids: object = cache.get("camera_ids")
        raw_frame_ids: object = cache.get("frame_ids")
        if (
            not isinstance(raw_camera_ids, list)
            or not all(isinstance(item, str) for item in cast(list[object], raw_camera_ids))
            or not isinstance(raw_frame_ids, list)
            or not all(type(item) is int for item in cast(list[object], raw_frame_ids))
        ):
            raise ContractError("tensor-cache axes are invalid")
        camera_ids = cast(list[str], raw_camera_ids)
        frame_ids = cast(list[int], raw_frame_ids)
        profile = PortableProfile.load(plan.profile)
        initialization = plan.initialization
        if (
            profile.initialization.time_offset_seconds != initialization.time_offset_seconds
            or profile.initialization.duration_min_seconds
            != initialization.duration_min_seconds
            or profile.initialization.duration_max_seconds
            != initialization.duration_max_seconds
        ):
            raise ContractError(
                "profile temporal initialization policy differs from the pipeline builder"
            )
        inputs = SceneInputs(
            manifest=plan.observation_manifest,
            initialization=paths.initialization / "initialization.safetensors",
            image_root=(
                plan.observation_manifest.parent if plan.image_root is None else plan.image_root
            ),
            tensor_memmap=TensorMemmapConfig(
                root=paths.tensor_cache,
                camera_ids=tuple(camera_ids),
                frame_ids=tuple(frame_ids),
                verify_transport_sha256=True,
            ),
        )
        config = RunConfig.from_profile_inputs(profile, inputs)
        _write_or_verify_bytes(
            paths.run_config,
            config.to_toml_bytes(),
            label="resolved run configuration",
        )
        return config

    def execute(
        self,
        stage: StageName,
        *,
        plan: PipelinePlan,
        paths: PipelinePaths,
        progress: Progress,
    ) -> Path:
        receipt = paths.receipt(stage)
        if stage == "prepare":
            if receipt.is_file() and not receipt.is_symlink():
                _read_json_object(receipt, label="tensor-cache manifest")
                return receipt
            if _exists(paths.tensor_cache):
                _quarantine_stage_output(
                    paths,
                    stage,
                    reason="INCOMPLETE_NON_RESUMABLE_OUTPUT",
                    progress=progress,
                )
            from p2g.training.prepare import build_tensor_cache

            build_tensor_cache(
                paths.tensor_cache,
                observation_manifest=plan.observation_manifest,
                image_root=plan.image_root,
                progress=progress,
            )
            return receipt

        if stage == "propose":
            if receipt.is_file() and not receipt.is_symlink():
                collection = _read_json_object(receipt, label="proposal collection")
                if collection.get("status") != "COMPLETE":
                    raise ContractError("proposal collection is not complete")
                return receipt
            from p2g.training.roma_point_sequence import build_roma_point_sequence

            with contextlib.redirect_stdout(sys.stderr):
                build_roma_point_sequence(
                    paths.proposals,
                    memmap_root=paths.tensor_cache,
                    observation_manifest=plan.observation_manifest,
                    roma_weight=plan.roma_indoor_weight,
                    dino_weight=plan.dinov2_weight,
                    environment_lock=plan.environment_lock,
                    frame_ids=tuple(
                        range(
                            plan.proposal.frame_start,
                            plan.proposal.frame_stop_exclusive,
                        )
                    ),
                    num_points_per_frame=plan.proposal.points_per_frame,
                    nearest_cameras=plan.proposal.nearest_cameras,
                    seed=plan.proposal.seed,
                    world_bound=plan.proposal.world_bound,
                )
            return receipt

        if stage == "initialize":
            if receipt.is_file() and not receipt.is_symlink():
                result = _read_json_object(receipt, label="initialization receipt")
                if result.get("status") != "COMPLETE" or result.get("trainer_eligible") is not True:
                    raise ContractError("Gaussian initialization is not trainer eligible")
                return receipt
            if _exists(paths.initialization):
                _quarantine_stage_output(
                    paths,
                    stage,
                    reason="INCOMPLETE_NON_RESUMABLE_OUTPUT",
                    progress=progress,
                )
            from p2g.training.build_initialization import build_initialization

            options = plan.initialization
            build_initialization(
                paths.initialization,
                proposal_sequence=paths.proposals,
                tensor_cache=paths.tensor_cache,
                num_gaussians=options.num_gaussians,
                seed=options.seed,
                velocity_neighbors=options.velocity_neighbors,
                scale_multiplier=options.scale_multiplier,
                sampling_mode=cast(Any, options.sampling_mode),
                sampling_voxel_size=options.sampling_voxel_size,
                sampling_evidence_fraction=options.sampling_evidence_fraction,
                opacity=options.opacity,
                duration_seconds=options.duration_seconds,
                duration_min_seconds=options.duration_min_seconds,
                duration_max_seconds=options.duration_max_seconds,
                time_offset_seconds=options.time_offset_seconds,
                progress=progress,
            )
            return receipt

        if stage == "train":
            config = self._resolved_config(plan, paths)
            from p2g.training.checkpoint import latest_checkpoint
            from p2g.training.train import run_training

            checkpoint: Path | None = None
            if _exists(paths.run):
                if paths.run.is_symlink() or not paths.run.is_dir():
                    raise ContractError("training run has an unsafe type")
                try:
                    checkpoint = latest_checkpoint(paths.run)
                except ContractError:
                    _quarantine_stage_output(
                        paths,
                        stage,
                        reason="NO_SAFE_TRAINING_CHECKPOINT",
                        progress=progress,
                    )
                    checkpoint = None
            with contextlib.redirect_stdout(sys.stderr):
                run_training(config, run_dir=paths.run, resume_checkpoint=checkpoint)
            return receipt

        if stage == "evaluate":
            if receipt.is_file() and not receipt.is_symlink():
                evaluation = _read_json_object(receipt, label="evaluation receipt")
                if evaluation.get("schema_version") != "p2g.evaluation.v1":
                    raise ContractError("evaluation receipt schema is invalid")
                return receipt
            if _exists(paths.evaluation):
                _quarantine_stage_output(
                    paths,
                    stage,
                    reason="INCOMPLETE_NON_RESUMABLE_OUTPUT",
                    progress=progress,
                )
            from p2g.training.evaluate import evaluate_exported_run

            evaluate_exported_run(paths.run, output_dir=paths.evaluation)
            return receipt

        if stage == "asset":
            if receipt.is_file() and not receipt.is_symlink():
                from p2g.training.asset import load_asset_bundle

                load_asset_bundle(paths.asset)
                return receipt
            if _exists(paths.asset):
                _quarantine_stage_output(
                    paths,
                    stage,
                    reason="INCOMPLETE_NON_RESUMABLE_OUTPUT",
                    progress=progress,
                )
            from p2g.training.train import AssetPublication, export_asset

            options = plan.asset
            export_asset(
                paths.run,
                AssetPublication(
                    output=paths.asset,
                    producer_git_revision=options.producer_git_revision,
                    asset_license=options.asset_license,
                    redistribution=cast(Any, options.redistribution),
                    provenance_summary=options.provenance_summary,
                    world_unit=options.world_unit,
                    calibration_scale=options.calibration_scale,
                    default_sh_degree=options.default_sh_degree,
                ),
            )
            return receipt

        raise AssertionError(f"unhandled pipeline stage: {stage}")


def run_pipeline(
    plan_path: Path,
    *,
    workspace: Path,
    stop_after: StageName | None = None,
    backend: PipelineBackend | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run or resume the fixed public pipeline without duplicating stage logic."""

    plan = PipelinePlan.load(plan_path)
    if stop_after is not None and stop_after not in STAGE_ORDER:
        raise ContractError(f"unsupported stop-after stage: {stop_after}")
    paths = _open_workspace(plan, workspace)
    selected_backend = DefaultPipelineBackend() if backend is None else backend
    report: Progress = _discard_progress if progress is None else progress

    pipeline_status(paths.root)
    target_index = len(STAGE_ORDER) - 1 if stop_after is None else STAGE_ORDER.index(stop_after)
    records: dict[StageName, dict[str, Any]] = {}
    for index, stage in enumerate(STAGE_ORDER):
        existing = _load_stage_record(paths, stage)
        if existing is not None:
            expected_request = _stage_request(plan, stage, records)
            actual_request = cast(dict[str, Any], existing["request"])
            if _semantic_stage_request(actual_request) != _semantic_stage_request(
                expected_request
            ):
                raise ContractError(
                    f"completed {stage} stage is incompatible with the active plan; "
                    "use a new workspace or retain the prior stage-scoped inputs"
                )
            records[stage] = existing
            report(f"SKIP {stage}: verified completion record")
        elif index <= target_index:
            request = _stage_request(plan, stage, records)
            if stage in GPU_STAGES:
                report(f"PREFLIGHT {stage}: capture current MI300X occupancy")
                request["preflight"] = _capture_preflight(
                    selected_backend,
                    plan=plan,
                    paths=paths,
                    stage=stage,
                )
            report(f"RUN {stage}")
            resource_window_path: Path | None = None
            if stage in GPU_STAGES:
                resource_window_path = _next_resource_window_path(paths, stage)
                try:
                    receipt_path = selected_backend.execute_guarded(
                        stage,
                        plan=plan,
                        paths=paths,
                        progress=report,
                        resource_window=resource_window_path,
                    )
                except Exception as exc:
                    from p2g.gpu_resource_window import ResourceWindowViolation

                    if isinstance(exc, ResourceWindowViolation):
                        window_receipt = _read_json_object(
                            resource_window_path,
                            label=f"{stage} failed resource-window receipt",
                            canonical=True,
                        )
                        if window_receipt.get("operation_status") != "NOT_STARTED":
                            _quarantine_stage_output(
                                paths,
                                stage,
                                reason="RESOURCE_WINDOW_INVALIDATED",
                                progress=report,
                            )
                    raise
            else:
                receipt_path = selected_backend.execute(
                    stage,
                    plan=plan,
                    paths=paths,
                    progress=report,
                )
            records[stage] = _publish_stage_record(
                paths=paths,
                stage=stage,
                request=request,
                receipt_path=receipt_path,
                resource_window_path=resource_window_path,
            )
            report(f"COMPLETE {stage}")
        if index == target_index:
            break

    if len(records) == len(STAGE_ORDER):
        completion = _completion_payload(
            records,
            terminal_plan_sha256=plan.source_sha256,
            terminal_source_git_revision=plan.source_git_revision,
        )
        if _exists(paths.completion):
            existing_completion = _read_json_object(
                paths.completion,
                label="pipeline completion seal",
                canonical=True,
            )
            if existing_completion != completion:
                raise ContractError("existing pipeline completion seal differs")
        else:
            write_new_bytes(paths.completion, canonical_json_bytes(completion))
    return pipeline_status(paths.root)


__all__ = [
    "ORCHESTRATOR_VERSION",
    "PIPELINE_PLAN_SCHEMA",
    "STAGE_ORDER",
    "AssetOptions",
    "DefaultPipelineBackend",
    "InitializationOptions",
    "PipelineBackend",
    "PipelinePaths",
    "PipelinePlan",
    "PreflightOptions",
    "ProposalOptions",
    "StageName",
    "StoppedProcessOptions",
    "pipeline_status",
    "run_pipeline",
]
