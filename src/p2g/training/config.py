# pyright: reportUnnecessaryIsInstance=false

"""Portable, fail-closed configuration for the public training pipeline.

User configuration is split deliberately:

* :class:`PortableProfile` contains only algorithm and runtime policy;
* :class:`SceneInputs` contains paths to one scene and initialization; and
* :class:`RunConfig` is the resolved, self-contained record saved with a run.

The parser uses only the Python standard library, rejects unknown fields, and
never admits evaluation-only observations into the optimization role.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Self, cast

from p2g.canonical import write_new_bytes
from p2g.errors import ContractError

PROFILE_SCHEMA = "p2g.portable_profile.v1"
SCENE_INPUTS_SCHEMA = "p2g.scene_inputs.v1"
RESOLVED_RUN_SCHEMA = "p2g.resolved_run.v1"

ROLE_NAMES = frozenset({"train", "diagnostic", "sealed", "free_view"})
OPTIMIZER_PARAMETER_NAMES = frozenset(
    {
        "means",
        "log_scales",
        "quaternions",
        "opacity_logits",
        "sh0",
        "sh_rest",
        "center_times",
        "duration_logits",
        "velocities",
        "persistence_logits",
    }
)

_BARE_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any, *, name: str) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")
    return float(value)


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" and at least {minimum}" if minimum is not None else ""
        raise ContractError(f"{name} must be an integer{suffix}")
    return value


def _boolean(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{name} must be boolean")
    return value


def _path(value: Any, *, name: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ContractError(f"{name} must be a non-empty filesystem path")
    if value.startswith(("~", "file://")):
        raise ContractError(f"{name} must not use home expansion or a file URI")
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _require_absolute_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ContractError(f"{name} must be a resolved absolute path")
    return value


def _read_toml(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = path.expanduser().resolve()
    try:
        value: Any = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ContractError(f"invalid TOML configuration {resolved}: {exc}") from exc
    if not isinstance(value, dict):  # pragma: no cover - tomllib always returns a dict
        raise ContractError(f"TOML configuration must be a table: {resolved}")
    return cast(dict[str, Any], value), resolved


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], *, context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ContractError(f"{context} contains unknown fields: {unknown}")


def _table(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{context} must be a TOML table")
    return cast(dict[str, Any], value)


def _construct[T](
    cls: type[T],
    raw: Mapping[str, Any],
    *,
    context: str,
    tuple_fields: frozenset[str] = frozenset(),
    path_fields: frozenset[str] = frozenset(),
    base_dir: Path | None = None,
) -> T:
    allowed = {item.name for item in fields(cast(Any, cls)) if item.init}
    _reject_unknown(raw, allowed, context=context)
    values = dict(raw)
    for name in tuple_fields & set(values):
        value = values[name]
        if isinstance(value, list):
            values[name] = tuple(cast(list[Any], value))
        elif not isinstance(value, tuple):
            raise ContractError(f"{context}.{name} must be a TOML array")
    for name in path_fields & set(values):
        if base_dir is None:  # pragma: no cover - caller invariant
            raise AssertionError("path conversion needs a base directory")
        value = values[name]
        if value is not None:
            values[name] = _path(value, name=f"{context}.{name}", base_dir=base_dir)
    try:
        return cls(**values)
    except TypeError as exc:
        raise ContractError(f"invalid {context}: {exc}") from exc


def _toml_key(value: str) -> str:
    return value if _BARE_TOML_KEY.fullmatch(value) else json.dumps(value, ensure_ascii=False)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, Path):
        return json.dumps(value.as_posix(), ensure_ascii=False)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("TOML serialization rejects NaN and infinity")
        return repr(0.0 if value == 0.0 else value)
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[Any], value)
        return "[" + ", ".join(_toml_scalar(item) for item in items) + "]"
    raise ContractError(f"unsupported TOML configuration value: {type(value)!r}")


def _emit_toml_table(
    value: Mapping[str, Any],
    *,
    prefix: tuple[str, ...],
    lines: list[str],
) -> None:
    if prefix:
        if lines and lines[-1]:
            lines.append("")
        lines.append("[" + ".".join(_toml_key(item) for item in prefix) + "]")
    for key in sorted(value):
        item = value[key]
        if item is not None and not isinstance(item, Mapping):
            lines.append(f"{_toml_key(key)} = {_toml_scalar(item)}")
    for key in sorted(value):
        item = value[key]
        if isinstance(item, Mapping):
            _emit_toml_table(cast(Mapping[str, Any], item), prefix=(*prefix, key), lines=lines)


def _toml_bytes(value: Any) -> bytes:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise ContractError("TOML configuration root must be a dataclass instance")
    built: dict[str, Any] = asdict(value)
    lines: list[str] = []
    _emit_toml_table(built, prefix=(), lines=lines)
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _save_new(path: Path, value: Any) -> None:
    write_new_bytes(path, _toml_bytes(value))


@dataclass(frozen=True, slots=True)
class TensorMemmapConfig:
    """Optional tensor-backed SceneBundle storage, with an explicit inventory."""

    root: Path
    camera_ids: tuple[str, ...]
    frame_ids: tuple[int, ...]
    verify_transport_sha256: bool = True

    def validate(self) -> None:
        _require_absolute_path(self.root, name="tensor_memmap.root")
        if (
            not isinstance(self.camera_ids, tuple)
            or not self.camera_ids
            or any(
                not isinstance(camera_id, str) or not _CAMERA_ID.fullmatch(camera_id)
                for camera_id in self.camera_ids
            )
            or len(set(self.camera_ids)) != len(self.camera_ids)
        ):
            raise ContractError("tensor_memmap.camera_ids must be non-empty and unique IDs")
        if (
            not isinstance(self.frame_ids, tuple)
            or not self.frame_ids
            or any(type(frame_id) is not int or frame_id < 0 for frame_id in self.frame_ids)
            or tuple(sorted(self.frame_ids)) != self.frame_ids
            or len(set(self.frame_ids)) != len(self.frame_ids)
        ):
            raise ContractError(
                "tensor_memmap.frame_ids must be unique, non-negative, and ascending"
            )
        _boolean(self.verify_transport_sha256, name="tensor_memmap.verify_transport_sha256")


@dataclass(frozen=True, slots=True)
class DataPolicyConfig:
    downscale: int = 1
    train_roles: tuple[str, ...] = ("train",)
    eval_roles: tuple[str, ...] = ("diagnostic",)
    max_train_observations: int | None = None
    max_eval_observations: int | None = None
    image_cache_size: int = 8

    def validate(self) -> None:
        _integer(self.downscale, name="data.downscale", minimum=1)
        _integer(self.image_cache_size, name="data.image_cache_size", minimum=0)
        for name, value in (
            ("data.max_train_observations", self.max_train_observations),
            ("data.max_eval_observations", self.max_eval_observations),
        ):
            if value is not None:
                _integer(value, name=name, minimum=1)
        if (
            not isinstance(self.train_roles, tuple)
            or not self.train_roles
            or any(not isinstance(role, str) or role != "train" for role in self.train_roles)
        ):
            raise ContractError("data.train_roles must contain only the train role")
        if self.eval_roles != ("diagnostic",):
            raise ContractError("data.eval_roles is fixed to the diagnostic role")
        if len(set(self.train_roles)) != len(self.train_roles) or len(set(self.eval_roles)) != len(
            self.eval_roles
        ):
            raise ContractError("data role selections must not contain duplicates")
        if set(self.train_roles) & set(self.eval_roles):  # defensive if roles expand later
            raise ContractError("training and evaluation roles must be disjoint")
        if (set(self.train_roles) | set(self.eval_roles)) - ROLE_NAMES:
            raise ContractError("data roles contain an unknown role")


@dataclass(frozen=True, slots=True)
class DataConfig(DataPolicyConfig):
    manifest: Path = Path("observation_manifest.json")
    image_root: Path | None = None
    tensor_memmap: TensorMemmapConfig | None = None

    def validate(self) -> None:
        DataPolicyConfig.validate(self)
        _require_absolute_path(self.manifest, name="data.manifest")
        if self.image_root is not None:
            _require_absolute_path(self.image_root, name="data.image_root")
        if self.tensor_memmap is not None:
            self.tensor_memmap.validate()


@dataclass(frozen=True, slots=True)
class InitializationPolicyConfig:
    format: Literal["p2g_safetensors"] = "p2g_safetensors"
    sh_degree: int = 3
    time_offset_seconds: float = 0.0
    duration_min_seconds: float = 1.0 / 600.0
    duration_max_seconds: float = 1.0
    persistence_initial_logit: float = -6.0

    def validate(self) -> None:
        if self.format != "p2g_safetensors":
            raise ContractError("initialization.format must be p2g_safetensors")
        degree = _integer(self.sh_degree, name="initialization.sh_degree", minimum=0)
        if degree > 3:
            raise ContractError("initialization.sh_degree must be in [0, 3]")
        _finite(self.time_offset_seconds, name="initialization.time_offset_seconds")
        duration_min = _finite(
            self.duration_min_seconds, name="initialization.duration_min_seconds"
        )
        duration_max = _finite(
            self.duration_max_seconds, name="initialization.duration_max_seconds"
        )
        if not 0.0 < duration_min < duration_max:
            raise ContractError("initialization duration bounds must satisfy 0 < min < max")
        _finite(
            self.persistence_initial_logit,
            name="initialization.persistence_initial_logit",
        )


@dataclass(frozen=True, slots=True)
class InitializationConfig(InitializationPolicyConfig):
    path: Path = Path("initialization.safetensors")

    def validate(self) -> None:
        InitializationPolicyConfig.validate(self)
        _require_absolute_path(self.path, name="initialization.path")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    persistence: Literal["off", "learned"] = "off"
    gate_logit_scale: float = 1.0

    def validate(self) -> None:
        if self.persistence not in {"off", "learned"}:
            raise ContractError("model.persistence must be off or learned")
        if _finite(self.gate_logit_scale, name="model.gate_logit_scale") <= 0.0:
            raise ContractError("model.gate_logit_scale must be positive")


@dataclass(frozen=True, slots=True)
class RendererConfig:
    backend: Literal["gsplat_rocm"] = "gsplat_rocm"
    near_plane: float = 0.01
    far_plane: float = 1.0e10
    eps2d: float = 0.3
    radius_clip: float = 0.0
    tile_size: int = 8
    packed: bool = True
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    clamp_rgb: bool = True
    require_gfx942: bool = True

    def validate(self) -> None:
        if self.backend != "gsplat_rocm":
            raise ContractError("renderer.backend must be gsplat_rocm")
        near = _finite(self.near_plane, name="renderer.near_plane")
        far = _finite(self.far_plane, name="renderer.far_plane")
        if near <= 0.0 or near >= far:
            raise ContractError("renderer near/far planes are invalid")
        if _finite(self.eps2d, name="renderer.eps2d") <= 0.0:
            raise ContractError("renderer.eps2d must be positive")
        if _finite(self.radius_clip, name="renderer.radius_clip") < 0.0:
            raise ContractError("renderer.radius_clip must be non-negative")
        if _integer(self.tile_size, name="renderer.tile_size", minimum=1) != 8:
            raise ContractError("the registered AMD gsplat backend requires tile_size=8")
        if _boolean(self.packed, name="renderer.packed") is not True:
            raise ContractError("the registered AMD gsplat backend requires packed=true")
        if (
            not isinstance(self.background, tuple)
            or len(self.background) != 3
            or any(
                not 0.0 <= _finite(value, name="renderer.background") <= 1.0
                for value in self.background
            )
        ):
            raise ContractError("renderer.background must contain three values in [0, 1]")
        _boolean(self.clamp_rgb, name="renderer.clamp_rgb")
        _boolean(self.require_gfx942, name="renderer.require_gfx942")


@dataclass(frozen=True, slots=True)
class LossConfig:
    l1: float = 0.8
    ssim: float = 0.2
    ssim_padding: Literal["same", "valid"] = "same"
    ssim_backend: Literal["torch", "fused"] = "torch"
    lpips: float = 0.0
    opacity: float = 1.0e-3
    scale: float = 1.0e-4
    persistence: float = 0.0
    gate: float = 0.0

    def validate(self) -> None:
        if self.ssim_padding not in {"same", "valid"}:
            raise ContractError("loss.ssim_padding must be same or valid")
        if self.ssim_backend not in {"torch", "fused"}:
            raise ContractError("loss.ssim_backend must be torch or fused")
        reconstruction = 0.0
        for name in ("l1", "ssim", "lpips", "opacity", "scale", "persistence", "gate"):
            value = _finite(getattr(self, name), name=f"loss.{name}")
            if value < 0.0:
                raise ContractError("loss weights must be non-negative")
            if name in {"l1", "ssim", "lpips"}:
                reconstruction += value
        if reconstruction <= 0.0:
            raise ContractError("at least one reconstruction loss must be enabled")


@dataclass(frozen=True, slots=True)
class ColorCorrectionConfig:
    mode: Literal["off", "per_camera_affine"] = "off"
    start: int = 15_000
    learning_rate: float = 1.0e-3
    regularization: float = 5.0e-2

    def validate(self, *, iterations: int) -> None:
        if self.mode not in {"off", "per_camera_affine"}:
            raise ContractError("color_correction.mode is unsupported")
        _integer(self.start, name="color_correction.start", minimum=0)
        if self.mode != "off" and self.start > iterations:
            raise ContractError("color_correction.start is outside the training interval")
        if _finite(self.learning_rate, name="color_correction.learning_rate") <= 0.0:
            raise ContractError("color_correction.learning_rate must be positive")
        if _finite(self.regularization, name="color_correction.regularization") < 0.0:
            raise ContractError("color_correction.regularization must be non-negative")


def _default_lrs() -> dict[str, float]:
    return {
        "means": 1.6e-4,
        "log_scales": 5.0e-3,
        "quaternions": 1.0e-3,
        "opacity_logits": 5.0e-2,
        "sh0": 2.5e-3,
        "sh_rest": 1.25e-4,
        "center_times": 1.6e-4,
        "duration_logits": 5.0e-3,
        "velocities": 1.6e-3,
        "persistence_logits": 1.0e-3,
    }


def _default_lr_final_factors() -> dict[str, float]:
    return {"means": 0.01, "velocities": 0.01}


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    lrs: dict[str, float] = field(default_factory=_default_lrs)
    lr_final_factors: dict[str, float] = field(default_factory=_default_lr_final_factors)
    eps: float = 1.0e-15
    beta1: float = 0.9
    beta2: float = 0.999

    def validate(self) -> None:
        if not isinstance(self.lrs, dict) or set(self.lrs) != set(OPTIMIZER_PARAMETER_NAMES):
            raise ContractError("optimizer.lrs must name every public model parameter exactly")
        if not isinstance(self.lr_final_factors, dict) or set(self.lr_final_factors) - set(
            self.lrs
        ):
            raise ContractError("every LR schedule must name a configured optimizer")
        for name, value in self.lrs.items():
            if _finite(value, name=f"optimizer.lrs.{name}") <= 0.0:
                raise ContractError("optimizer learning rates must be positive")
        for name, value in self.lr_final_factors.items():
            factor = _finite(value, name=f"optimizer.lr_final_factors.{name}")
            if not 0.0 < factor <= 1.0:
                raise ContractError("optimizer final LR factors must be in (0, 1]")
        if _finite(self.eps, name="optimizer.eps") <= 0.0:
            raise ContractError("optimizer.eps must be positive")
        beta1 = _finite(self.beta1, name="optimizer.beta1")
        beta2 = _finite(self.beta2, name="optimizer.beta2")
        if not 0.0 <= beta1 < beta2 < 1.0:
            raise ContractError("optimizer betas must satisfy 0 <= beta1 < beta2 < 1")


@dataclass(frozen=True, slots=True)
class RelocationConfig:
    mode: Literal["off", "fixed_budget_relocation_v1"] = "off"
    start: int = 600
    stop: int = 25_000
    every: int = 100
    opacity_threshold: float = 5.0e-3

    def validate(self, *, iterations: int) -> None:
        if self.mode not in {"off", "fixed_budget_relocation_v1"}:
            raise ContractError("training.relocation.mode is not a public mode")
        _integer(self.start, name="training.relocation.start", minimum=0)
        _integer(self.stop, name="training.relocation.stop", minimum=1)
        _integer(self.every, name="training.relocation.every", minimum=1)
        threshold = _finite(self.opacity_threshold, name="training.relocation.opacity_threshold")
        if not 0.0 < threshold < 1.0:
            raise ContractError("relocation.opacity_threshold must be in (0, 1)")
        if self.mode != "off" and not 0 <= self.start < self.stop <= iterations:
            raise ContractError("relocation interval is outside the training interval")


@dataclass(frozen=True, slots=True)
class ScreenInfluenceGuardConfig:
    mode: Literal["off", "formation_alpha_projection"] = "off"
    frame_ids: tuple[int, ...] = (0, 30, 59)
    start: int = 100
    stop: int = 30_000
    every: int = 100
    near_plane_multiple: float = 10.0
    tile_coverage_minimum: float = 0.9
    solo_alpha_mean_maximum: float = 0.25
    combined_alpha_mean_maximum: float = 0.05
    alpha_fraction_threshold: float = 0.1
    alpha_tolerance: float = 1.0e-4
    bisection_iterations: int = 16
    maximum_candidates_per_observation: int = 64

    def validate(self, *, iterations: int) -> None:
        if self.mode not in {"off", "formation_alpha_projection"}:
            raise ContractError("training.screen_guard.mode is unsupported")
        if (
            not isinstance(self.frame_ids, tuple)
            or not self.frame_ids
            or any(type(frame_id) is not int or frame_id < 0 for frame_id in self.frame_ids)
            or len(set(self.frame_ids)) != len(self.frame_ids)
        ):
            raise ContractError("screen_guard.frame_ids must be non-empty unique frame IDs")
        _integer(self.start, name="training.screen_guard.start", minimum=1)
        _integer(self.stop, name="training.screen_guard.stop", minimum=1)
        _integer(self.every, name="training.screen_guard.every", minimum=1)
        if self.mode != "off":
            if not 0 < self.start <= self.stop <= iterations:
                raise ContractError("screen_guard interval is outside the training interval")
            if self.start % self.every != 0:
                raise ContractError("screen_guard.every must divide screen_guard.start")
        positive = (
            ("near_plane_multiple", self.near_plane_multiple),
            ("tile_coverage_minimum", self.tile_coverage_minimum),
            ("solo_alpha_mean_maximum", self.solo_alpha_mean_maximum),
            ("combined_alpha_mean_maximum", self.combined_alpha_mean_maximum),
            ("alpha_fraction_threshold", self.alpha_fraction_threshold),
            ("alpha_tolerance", self.alpha_tolerance),
        )
        values = {
            name: _finite(value, name=f"training.screen_guard.{name}") for name, value in positive
        }
        if values["near_plane_multiple"] <= 0.0:
            raise ContractError("screen_guard.near_plane_multiple must be positive")
        for name in (
            "tile_coverage_minimum",
            "solo_alpha_mean_maximum",
            "combined_alpha_mean_maximum",
            "alpha_fraction_threshold",
        ):
            if not 0.0 < values[name] <= 1.0:
                raise ContractError(f"screen_guard.{name} must be in (0, 1]")
        if not 0.0 < values["alpha_tolerance"] < values["solo_alpha_mean_maximum"]:
            raise ContractError("screen_guard.alpha_tolerance must be positive and bounded")
        iterations_count = _integer(
            self.bisection_iterations,
            name="training.screen_guard.bisection_iterations",
            minimum=1,
        )
        if iterations_count > 32:
            raise ContractError("screen_guard.bisection_iterations must be in [1, 32]")
        _integer(
            self.maximum_candidates_per_observation,
            name="training.screen_guard.maximum_candidates_per_observation",
            minimum=1,
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    iterations: int = 30_000
    seed: int = 42
    sampling: Literal["shuffled_epoch", "frame_camera_with_replacement"] = "shuffled_epoch"
    device: Literal["cuda"] = "cuda"
    checkpoint_every: int = 1_000
    evaluate_every: int = 1_000
    log_every: int = 10
    sh_degree_interval: int = 1_000
    deterministic: bool = True
    relocation: RelocationConfig = field(default_factory=RelocationConfig)
    screen_guard: ScreenInfluenceGuardConfig = field(default_factory=ScreenInfluenceGuardConfig)

    def validate(self) -> None:
        iterations = _integer(self.iterations, name="training.iterations", minimum=1)
        _integer(self.seed, name="training.seed", minimum=0)
        if self.sampling not in {"shuffled_epoch", "frame_camera_with_replacement"}:
            raise ContractError("training.sampling is not a public sampling policy")
        if self.device != "cuda":
            raise ContractError("the public MI300X profile requires training.device=cuda")
        for name in ("checkpoint_every", "evaluate_every", "log_every", "sh_degree_interval"):
            _integer(getattr(self, name), name=f"training.{name}", minimum=1)
        _boolean(self.deterministic, name="training.deterministic")
        self.relocation.validate(iterations=iterations)
        self.screen_guard.validate(iterations=iterations)


@dataclass(frozen=True, slots=True)
class PortableProfile:
    schema_version: str = field(default=PROFILE_SCHEMA, init=False)
    data: DataPolicyConfig = field(default_factory=DataPolicyConfig)
    initialization: InitializationPolicyConfig = field(default_factory=InitializationPolicyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    color_correction: ColorCorrectionConfig = field(default_factory=ColorCorrectionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.schema_version != PROFILE_SCHEMA:
            raise ContractError("unsupported portable profile schema")
        self.data.validate()
        self.initialization.validate()
        self.model.validate()
        self.renderer.validate()
        self.loss.validate()
        self.optimizer.validate()
        self.training.validate()
        self.color_correction.validate(iterations=self.training.iterations)

    @classmethod
    def load(cls, path: Path) -> Self:
        raw, _ = _read_toml(path)
        return cast(Self, _profile_from_mapping(raw))

    def to_toml_bytes(self) -> bytes:
        self.validate()
        return _toml_bytes(self)

    def save(self, path: Path) -> None:
        self.validate()
        _save_new(path, self)


@dataclass(frozen=True, slots=True)
class SceneInputs:
    manifest: Path
    initialization: Path
    schema_version: str = field(default=SCENE_INPUTS_SCHEMA, init=False)
    image_root: Path | None = None
    tensor_memmap: TensorMemmapConfig | None = None

    def validate(self) -> None:
        if self.schema_version != SCENE_INPUTS_SCHEMA:
            raise ContractError("unsupported scene-input schema")
        _require_absolute_path(self.manifest, name="scene manifest")
        _require_absolute_path(self.initialization, name="scene initialization")
        if self.image_root is not None:
            _require_absolute_path(self.image_root, name="scene image_root")
        if self.tensor_memmap is not None:
            self.tensor_memmap.validate()

    @classmethod
    def load(cls, path: Path) -> Self:
        raw, resolved = _read_toml(path)
        allowed = {
            "schema_version",
            "manifest",
            "initialization",
            "image_root",
            "tensor_memmap",
        }
        _reject_unknown(raw, allowed, context="scene inputs")
        if raw.get("schema_version") != SCENE_INPUTS_SCHEMA:
            raise ContractError(f"scene inputs must declare {SCENE_INPUTS_SCHEMA}")
        tensor_raw = raw.get("tensor_memmap")
        tensor_memmap = None
        if tensor_raw is not None:
            tensor_memmap = _construct(
                TensorMemmapConfig,
                _table(tensor_raw, context="tensor_memmap"),
                context="tensor_memmap",
                tuple_fields=frozenset({"camera_ids", "frame_ids"}),
                path_fields=frozenset({"root"}),
                base_dir=resolved.parent,
            )
        inputs = _construct(
            cls,
            {
                key: value
                for key, value in raw.items()
                if key not in {"schema_version", "tensor_memmap"}
            }
            | {"tensor_memmap": tensor_memmap},
            context="scene inputs",
            path_fields=frozenset({"manifest", "initialization", "image_root"}),
            base_dir=resolved.parent,
        )
        inputs.validate()
        return inputs

    def to_toml_bytes(self) -> bytes:
        self.validate()
        return _toml_bytes(self)

    def save(self, path: Path) -> None:
        self.validate()
        _save_new(path, self)


@dataclass(frozen=True, slots=True)
class RunConfig:
    data: DataConfig
    initialization: InitializationConfig
    schema_version: str = field(default=RESOLVED_RUN_SCHEMA, init=False)
    model: ModelConfig = field(default_factory=ModelConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    color_correction: ColorCorrectionConfig = field(default_factory=ColorCorrectionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.schema_version != RESOLVED_RUN_SCHEMA:
            raise ContractError("unsupported resolved-run schema")
        self.data.validate()
        self.initialization.validate()
        self.model.validate()
        self.renderer.validate()
        self.loss.validate()
        self.optimizer.validate()
        self.training.validate()
        self.color_correction.validate(iterations=self.training.iterations)

    @classmethod
    def from_profile_inputs(cls, profile: PortableProfile, inputs: SceneInputs) -> Self:
        profile.validate()
        inputs.validate()
        data = DataConfig(
            downscale=profile.data.downscale,
            train_roles=profile.data.train_roles,
            eval_roles=profile.data.eval_roles,
            max_train_observations=profile.data.max_train_observations,
            max_eval_observations=profile.data.max_eval_observations,
            image_cache_size=profile.data.image_cache_size,
            manifest=inputs.manifest,
            image_root=inputs.image_root,
            tensor_memmap=inputs.tensor_memmap,
        )
        initialization = InitializationConfig(
            format=profile.initialization.format,
            sh_degree=profile.initialization.sh_degree,
            time_offset_seconds=profile.initialization.time_offset_seconds,
            duration_min_seconds=profile.initialization.duration_min_seconds,
            duration_max_seconds=profile.initialization.duration_max_seconds,
            persistence_initial_logit=profile.initialization.persistence_initial_logit,
            path=inputs.initialization,
        )
        config = cls(
            data=data,
            initialization=initialization,
            model=profile.model,
            renderer=profile.renderer,
            loss=profile.loss,
            color_correction=profile.color_correction,
            optimizer=profile.optimizer,
            training=profile.training,
        )
        config.validate()
        return config

    @classmethod
    def from_files(cls, *, profile: Path, scene: Path) -> Self:
        return cls.from_profile_inputs(PortableProfile.load(profile), SceneInputs.load(scene))

    @classmethod
    def load(cls, path: Path) -> Self:
        raw, resolved = _read_toml(path)
        allowed = {
            "schema_version",
            "data",
            "initialization",
            "model",
            "renderer",
            "loss",
            "color_correction",
            "optimizer",
            "training",
        }
        _reject_unknown(raw, allowed, context="resolved run config")
        if raw.get("schema_version") != RESOLVED_RUN_SCHEMA:
            raise ContractError(f"resolved run config must declare {RESOLVED_RUN_SCHEMA}")

        data_raw = dict(_table(raw.get("data"), context="data"))
        tensor_raw = data_raw.pop("tensor_memmap", None)
        tensor_memmap = None
        if tensor_raw is not None:
            tensor_memmap = _construct(
                TensorMemmapConfig,
                _table(tensor_raw, context="data.tensor_memmap"),
                context="data.tensor_memmap",
                tuple_fields=frozenset({"camera_ids", "frame_ids"}),
                path_fields=frozenset({"root"}),
                base_dir=resolved.parent,
            )
        data = _construct(
            DataConfig,
            {**data_raw, "tensor_memmap": tensor_memmap},
            context="data",
            tuple_fields=frozenset({"train_roles", "eval_roles"}),
            path_fields=frozenset({"manifest", "image_root"}),
            base_dir=resolved.parent,
        )
        initialization = _construct(
            InitializationConfig,
            _table(raw.get("initialization"), context="initialization"),
            context="initialization",
            path_fields=frozenset({"path"}),
            base_dir=resolved.parent,
        )

        profile_payload: dict[str, Any] = {
            "schema_version": PROFILE_SCHEMA,
            "model": raw.get("model", {}),
            "renderer": raw.get("renderer", {}),
            "loss": raw.get("loss", {}),
            "color_correction": raw.get("color_correction", {}),
            "optimizer": raw.get("optimizer", {}),
            "training": raw.get("training", {}),
            "data": {
                name: getattr(data, name)
                for name in (
                    "downscale",
                    "train_roles",
                    "eval_roles",
                    "max_train_observations",
                    "max_eval_observations",
                    "image_cache_size",
                )
                if getattr(data, name) is not None
            },
            "initialization": {
                name: getattr(initialization, name)
                for name in (
                    "format",
                    "sh_degree",
                    "time_offset_seconds",
                    "duration_min_seconds",
                    "duration_max_seconds",
                    "persistence_initial_logit",
                )
            },
        }
        # Reuse the same fail-closed section parser without publishing a temporary file.
        profile = _profile_from_mapping(profile_payload)
        config = cls(
            data=data,
            initialization=initialization,
            model=profile.model,
            renderer=profile.renderer,
            loss=profile.loss,
            color_correction=profile.color_correction,
            optimizer=profile.optimizer,
            training=profile.training,
        )
        config.validate()
        return config

    def to_toml_bytes(self) -> bytes:
        self.validate()
        return _toml_bytes(self)

    def save(self, path: Path) -> None:
        self.validate()
        _save_new(path, self)


def _profile_from_mapping(raw: Mapping[str, Any]) -> PortableProfile:
    """Parse an already decoded profile using the public loader's exact rules."""

    allowed = {
        "schema_version",
        "data",
        "initialization",
        "model",
        "renderer",
        "loss",
        "color_correction",
        "optimizer",
        "training",
    }
    _reject_unknown(raw, allowed, context="portable profile")
    if raw.get("schema_version") != PROFILE_SCHEMA:
        raise ContractError(f"portable profile must declare {PROFILE_SCHEMA}")

    temporary = PortableProfile(
        data=_construct(
            DataPolicyConfig,
            _table(raw.get("data", {}), context="data"),
            context="data",
            tuple_fields=frozenset({"train_roles", "eval_roles"}),
        ),
        initialization=_construct(
            InitializationPolicyConfig,
            _table(raw.get("initialization", {}), context="initialization"),
            context="initialization",
        ),
        model=_construct(
            ModelConfig,
            _table(raw.get("model", {}), context="model"),
            context="model",
        ),
        renderer=_construct(
            RendererConfig,
            _table(raw.get("renderer", {}), context="renderer"),
            context="renderer",
            tuple_fields=frozenset({"background"}),
        ),
        loss=_construct(
            LossConfig,
            _table(raw.get("loss", {}), context="loss"),
            context="loss",
        ),
        color_correction=_construct(
            ColorCorrectionConfig,
            _table(raw.get("color_correction", {}), context="color_correction"),
            context="color_correction",
        ),
        optimizer=_construct(
            OptimizerConfig,
            _table(raw.get("optimizer", {}), context="optimizer"),
            context="optimizer",
        ),
        training=_training_from_mapping(_table(raw.get("training", {}), context="training")),
    )
    temporary.validate()
    return temporary


def _training_from_mapping(raw: Mapping[str, Any]) -> TrainingConfig:
    values = dict(raw)
    relocation = _construct(
        RelocationConfig,
        _table(values.pop("relocation", {}), context="training.relocation"),
        context="training.relocation",
    )
    screen_guard = _construct(
        ScreenInfluenceGuardConfig,
        _table(values.pop("screen_guard", {}), context="training.screen_guard"),
        context="training.screen_guard",
        tuple_fields=frozenset({"frame_ids"}),
    )
    return _construct(
        TrainingConfig,
        {**values, "relocation": relocation, "screen_guard": screen_guard},
        context="training",
    )
