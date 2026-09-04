# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

"""White-box, fixed-capacity Gaussian relocation.

Relocation recycles low-opacity parameter slots without changing tensor
capacity.  Source selection, alpha splitting, projected-footprint correction,
optimizer-state invalidation, and checkpoint state are explicit in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from p2g.errors import ContractError
from p2g.training.config import RelocationConfig
from p2g.training.model import DynamicGaussianModel
from p2g.training.optim import OptimizerBundle

RELOCATION_STATE_SCHEMA = "p2g.fixed_budget_relocation_state.v1"
RELOCATION_EVENT_SCHEMA = "p2g.fixed_budget_relocation_event.v1"
RELOCATION_ROLE_NONE = 0
RELOCATION_ROLE_DESTINATION = 1
RELOCATION_ROLE_SOURCE = 2
MAX_SPLIT_MULTIPLICITY = 64


def _validate_split_inputs(
    opacities: Tensor,
    scales: Tensor,
    multiplicities: Tensor,
) -> None:
    if not opacities.is_floating_point() or opacities.ndim != 1:
        raise ContractError("split opacities must be one floating-point vector")
    if (
        not scales.is_floating_point()
        or tuple(scales.shape) != (int(opacities.shape[0]), 3)
    ):
        raise ContractError("split scales must have shape [N,3]")
    if multiplicities.is_floating_point() or multiplicities.is_complex():
        raise ContractError("split multiplicities must use an integer dtype")
    if tuple(multiplicities.shape) != tuple(opacities.shape):
        raise ContractError("split multiplicities must have shape [N]")
    if not (opacities.device == scales.device == multiplicities.device):
        raise ContractError("split tensors must share one device")
    if opacities.numel() == 0:
        return
    if not bool(torch.isfinite(opacities).all() & torch.isfinite(scales).all()):
        raise FloatingPointError("split inputs contain non-finite values")
    if not bool(((opacities > 0.0) & (opacities < 1.0)).all()):
        raise ContractError("split opacities must lie strictly inside (0,1)")
    if not bool((scales > 0.0).all()):
        raise ContractError("split scales must be positive")
    minimum = int(multiplicities.min())
    maximum = int(multiplicities.max())
    if minimum < 1 or maximum > MAX_SPLIT_MULTIPLICITY:
        raise ContractError(
            f"split multiplicities must be in [1,{MAX_SPLIT_MULTIPLICITY}]"
        )


def _composited_profile_integral(piece_opacity: Tensor, multiplicities: Tensor) -> Tensor:
    """Return the dimensionless integral of coincident composited 2D profiles."""

    log_transmission = torch.log1p(-piece_opacity)
    coefficient = torch.zeros_like(piece_opacity)
    maximum = int(multiplicities.max()) if multiplicities.numel() else 0
    for order in range(1, maximum + 1):
        term = -torch.expm1(log_transmission * order) / order
        coefficient = coefficient + torch.where(
            multiplicities >= order,
            term,
            torch.zeros_like(term),
        )
    return coefficient


def split_projected_alpha_mass(
    opacities: Tensor,
    scales: Tensor,
    multiplicities: Tensor,
) -> tuple[Tensor, Tensor]:
    """Split coincident Gaussians while preserving center alpha and 2D alpha mass.

    For source alpha ``a`` and ``r`` coincident pieces, each piece receives

    ``a_piece = 1 - (1 - a) ** (1 / r)``.

    This makes the center composite exactly ``a``.  Expanding the ideal
    front-to-back composite of ``r`` identical 2D Gaussian profiles gives the
    dimensionless integral

    ``D = sum(j=1..r) (1 - (1 - a_piece) ** j) / j``.

    Uniformly multiplying all three source scales by ``sqrt(a / D)`` therefore
    preserves the projected alpha integral for every locally linear view,
    because a projected covariance area scales by the square of that factor.
    Float64 is used only for this infrequent control-plane calculation.
    """

    _validate_split_inputs(opacities, scales, multiplicities)
    if opacities.numel() == 0:
        return opacities.clone(), scales.clone()
    alpha = opacities.to(torch.float64)
    physical_scales = scales.to(torch.float64)
    ratios = multiplicities.to(torch.int64)
    piece_alpha = -torch.expm1(torch.log1p(-alpha) / ratios)
    integral = _composited_profile_integral(piece_alpha, ratios)
    scale_factor = torch.sqrt(alpha / integral)
    piece_scales = physical_scales * scale_factor.unsqueeze(-1)
    if not bool(torch.isfinite(piece_alpha).all() & torch.isfinite(piece_scales).all()):
        raise FloatingPointError("split equation produced non-finite parameters")
    if not bool((piece_scales > 0.0).all()):
        raise FloatingPointError("split equation produced non-positive scales")
    return piece_alpha.to(opacities.dtype), piece_scales.to(scales.dtype)


def _clone_capacities(opacities: Tensor, *, opacity_threshold: float) -> Tensor:
    """Maximum clones whose split opacity remains above the dead threshold."""

    guarded_threshold = min(
        opacity_threshold + 8.0 * torch.finfo(torch.float32).eps,
        1.0 - torch.finfo(torch.float32).eps,
    )
    threshold = torch.as_tensor(
        guarded_threshold,
        dtype=torch.float64,
        device=opacities.device,
    )
    alpha = opacities.to(torch.float64).clamp(
        min=torch.finfo(torch.float64).tiny,
        max=1.0 - torch.finfo(torch.float64).eps,
    )
    maximum_pieces = torch.floor(
        torch.log1p(-alpha) / torch.log1p(-threshold)
    ).to(torch.int64)
    maximum_pieces.clamp_(min=1, max=MAX_SPLIT_MULTIPLICITY)
    return maximum_pieces - 1


def _split_persistent_fraction(
    source_opacity: Tensor,
    source_persistence: Tensor,
    piece_opacity: Tensor,
    multiplicities: Tensor,
) -> Tensor:
    """Preserve the far-time composite alpha of a learned persistent component."""

    far_alpha = source_opacity * source_persistence
    piece_far_alpha = -torch.expm1(
        torch.log1p(-far_alpha) / multiplicities.to(torch.float64)
    )
    persistence = piece_far_alpha / piece_opacity
    if not bool(torch.isfinite(persistence).all()):
        raise FloatingPointError("persistent split equation produced non-finite values")
    tolerance = 32.0 * torch.finfo(torch.float64).eps
    if not bool(((persistence >= -tolerance) & (persistence <= 1.0 + tolerance)).all()):
        raise FloatingPointError("persistent split equation left the physical interval")
    return persistence.clamp(0.0, 1.0)


def _sample_with_capacities(
    scores: Tensor,
    capacities: Tensor,
    *,
    count: int,
) -> tuple[Tensor, int]:
    if scores.ndim != 1 or tuple(capacities.shape) != tuple(scores.shape):
        raise ContractError("source scores and capacities must have shape [N]")
    if (
        not scores.is_floating_point()
        or capacities.is_floating_point()
        or capacities.is_complex()
    ):
        raise ContractError("source scores must be floating point and capacities integral")
    if scores.device != capacities.device:
        raise ContractError("source scores and capacities must share one device")
    if type(count) is not int or count < 0:
        raise ContractError("requested relocation count must be a non-negative integer")
    if not bool(torch.isfinite(scores).all() & (scores >= 0.0).all()):
        raise FloatingPointError("source sampling scores are invalid")
    if not bool((capacities >= 0).all()):
        raise ContractError("source capacities must be non-negative")
    if bool(((capacities > 0) & (scores <= 0.0)).any()):
        raise FloatingPointError("positive source capacity must have positive utility")

    target = min(count, int(capacities.sum()))
    if target == 0:
        return torch.empty((0,), dtype=torch.int64, device=scores.device), 0

    # Allocate breadth-first through clone-capacity strata.  Every eligible
    # source gets at most one clone in a stratum; a source can receive a second
    # clone only after the first stratum has been filled for every source that
    # can accept one.  This bounds control flow by MAX_SPLIT_MULTIPLICITY - 1
    # and keeps split multiplicities low.  Utility weights decide only the
    # final partial stratum, without replacement.
    capacity64 = capacities.to(torch.int64)
    histogram = torch.bincount(capacity64)
    available_by_stratum = torch.flip(
        torch.cumsum(torch.flip(histogram[1:], dims=(0,)), dim=0),
        dims=(0,),
    )
    remaining = target
    full_strata = 0
    for available in available_by_stratum.tolist():
        available_count = int(available)
        if remaining < available_count:
            break
        remaining -= available_count
        full_strata += 1
        if remaining == 0:
            break

    selected_counts = torch.minimum(
        capacity64,
        torch.full_like(capacity64, full_strata),
    )
    allocation_strata = full_strata
    if remaining:
        frontier = capacity64 > full_strata
        frontier_weights = torch.where(frontier, scores, torch.zeros_like(scores))
        draws = torch.multinomial(
            frontier_weights,
            remaining,
            replacement=False,
        )
        selected_counts.index_add_(0, draws, torch.ones_like(draws))
        allocation_strata += 1

    sampled = torch.repeat_interleave(
        torch.arange(scores.numel(), dtype=torch.int64, device=scores.device),
        selected_counts,
    )
    if sampled.numel() > 1:
        sampled = sampled[torch.randperm(sampled.numel(), device=sampled.device)]
    return sampled, allocation_strata


def _optimizer_for_parameter(
    optimizers: OptimizerBundle,
    *,
    name: str,
    parameter: Tensor,
) -> torch.optim.Optimizer | None:
    if not parameter.requires_grad:
        return None
    if name not in optimizers:
        raise ContractError(f"relocation has no optimizer for trainable plane: {name}")
    optimizer = optimizers[name]
    owned = any(
        candidate is parameter
        for group in optimizer.param_groups
        for candidate in group["params"]
    )
    if not owned:
        raise ContractError(f"relocation optimizer does not own model plane: {name}")
    return optimizer


def _zero_parameter_rows(
    optimizer: torch.optim.Optimizer,
    parameter: Tensor,
    rows: Tensor,
    *,
    gaussian_count: int,
) -> None:
    if rows.numel() == 0:
        return
    _validate_optimizer_rows(
        optimizer,
        parameter,
        gaussian_count=gaussian_count,
    )
    gradient = parameter.grad
    if gradient is not None:
        gradient.index_fill_(0, rows, 0)
    for value in optimizer.state.get(parameter, {}).values():
        if isinstance(value, Tensor) and value.ndim > 0:
            value.index_fill_(0, rows, 0)


def _validate_optimizer_rows(
    optimizer: torch.optim.Optimizer,
    parameter: Tensor,
    *,
    gaussian_count: int,
) -> None:
    gradient = parameter.grad
    if gradient is not None and (
        tuple(gradient.shape) != tuple(parameter.shape)
        or gradient.device != parameter.device
    ):
        raise ContractError("Gaussian parameter gradient has invalid row state")
    for value in optimizer.state.get(parameter, {}).values():
        if isinstance(value, Tensor) and value.ndim > 0 and (
            tuple(value.shape) != tuple(parameter.shape)
            or value.shape[0] != gaussian_count
            or value.device != parameter.device
        ):
            raise ContractError("Gaussian optimizer tensor has invalid row state")


def _relocation_indices(
    model: DynamicGaussianModel,
    dead_indices: Tensor,
    sampled_sources: Tensor,
) -> None:
    if dead_indices.ndim != 1 or sampled_sources.ndim != 1:
        raise ContractError("relocation indices must be one-dimensional")
    if tuple(dead_indices.shape) != tuple(sampled_sources.shape):
        raise ContractError("every relocation destination must have one source")
    if dead_indices.dtype != torch.int64 or sampled_sources.dtype != torch.int64:
        raise ContractError("relocation indices must use torch.int64")
    if dead_indices.device != model.means.device or sampled_sources.device != model.means.device:
        raise ContractError("relocation indices and model must share one device")
    if dead_indices.numel() == 0:
        return
    if (
        int(dead_indices.min()) < 0
        or int(dead_indices.max()) >= model.count
        or int(sampled_sources.min()) < 0
        or int(sampled_sources.max()) >= model.count
    ):
        raise ContractError("relocation index is outside the fixed population")
    if torch.unique(dead_indices).numel() != dead_indices.numel():
        raise ContractError("relocation destinations must be unique")
    if bool(torch.isin(dead_indices, sampled_sources).any()):
        raise ContractError("relocation sources and destinations must be disjoint")


@torch.no_grad()
def relocate_fixed_budget(
    model: DynamicGaussianModel,
    optimizers: OptimizerBundle,
    *,
    dead_indices: Tensor,
    sampled_sources: Tensor,
) -> dict[str, int | float]:
    """Recycle fixed slots and invalidate only optimizer rows made stale."""

    _relocation_indices(model, dead_indices, sampled_sources)
    relocated = int(dead_indices.numel())
    if relocated == 0:
        return {
            "relocated": 0,
            "unique_sources": 0,
            "maximum_multiplicity": 1,
            "center_alpha_max_error": 0.0,
            "projected_alpha_mass_relative_max_error": 0.0,
            "persistent_alpha_max_error": 0.0,
            "temporal_profile_alpha_max_error": 0.0,
            "maximum_scale_factor": 1.0,
        }

    unique_sources, inverse, clone_counts = torch.unique(
        sampled_sources,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    multiplicities = clone_counts + 1
    if int(multiplicities.max()) > MAX_SPLIT_MULTIPLICITY:
        raise ContractError("relocation source exceeds the public split-multiplicity bound")

    source_logits = model.opacity_logits[unique_sources, 0].to(torch.float64)
    source_opacity = torch.sigmoid(source_logits).clamp(
        min=torch.finfo(torch.float64).tiny,
        max=1.0 - torch.finfo(torch.float64).eps,
    )
    source_scales = torch.exp(model.log_scales[unique_sources].to(torch.float64))
    piece_opacity, piece_scales = split_projected_alpha_mass(
        source_opacity,
        source_scales,
        multiplicities,
    )
    piece_logits = torch.logit(piece_opacity).to(model.opacity_logits.dtype)
    piece_log_scales = torch.log(piece_scales).to(model.log_scales.dtype)
    realized_piece_opacity = torch.sigmoid(piece_logits.to(torch.float64))
    realized_piece_scales = torch.exp(piece_log_scales.to(torch.float64))
    if model.persistence_enabled:
        gate_scale = model.gate_logit_scale.detach().to(torch.float64)
        source_persistence = torch.sigmoid(
            gate_scale * model.persistence_logits[unique_sources, 0].to(torch.float64)
        ).clamp(
            min=torch.finfo(torch.float64).tiny,
            max=1.0 - torch.finfo(torch.float64).eps,
        )
        piece_persistence = _split_persistent_fraction(
            source_opacity,
            source_persistence,
            piece_opacity.to(torch.float64),
            multiplicities,
        )
        representable_persistence = piece_persistence.clamp(
            min=torch.finfo(model.persistence_logits.dtype).tiny,
            max=1.0 - torch.finfo(model.persistence_logits.dtype).eps,
        )
        piece_persistence_logits = (
            torch.logit(representable_persistence) / gate_scale
        ).to(model.persistence_logits.dtype)
        realized_piece_persistence = torch.sigmoid(
            gate_scale * piece_persistence_logits.to(torch.float64)
        )
    else:
        source_persistence = torch.zeros_like(source_opacity)
        piece_persistence_logits = None
        realized_piece_persistence = torch.zeros_like(source_opacity)
    finite_outputs = torch.isfinite(piece_logits).all() & torch.isfinite(
        piece_log_scales
    ).all()
    if piece_persistence_logits is not None:
        finite_outputs &= torch.isfinite(piece_persistence_logits).all()
    if not bool(finite_outputs):
        raise FloatingPointError("relocation produced non-finite trainable parameters")

    parameters = dict(model.named_parameters())
    destination_rows = torch.unique(dead_indices, sorted=True)
    source_and_destination_rows = torch.unique(
        torch.cat((unique_sources, destination_rows)),
        sorted=True,
    )
    source_mutated_planes = {"opacity_logits", "log_scales"}
    if model.persistence_enabled:
        source_mutated_planes.add("persistence_logits")
    reset_plan: list[tuple[torch.optim.Optimizer, Tensor, Tensor]] = []
    for name, parameter in parameters.items():
        if (
            parameter.ndim == 0
            or parameter.shape[0] != model.count
            or parameter.device != model.means.device
        ):
            raise ContractError(f"model plane has no fixed-population row axis: {name}")
        optimizer = _optimizer_for_parameter(
            optimizers,
            name=name,
            parameter=parameter,
        )
        if optimizer is not None:
            rows = (
                source_and_destination_rows
                if name in source_mutated_planes
                else destination_rows
            )
            _validate_optimizer_rows(
                optimizer,
                parameter,
                gaussian_count=model.count,
            )
            reset_plan.append((optimizer, parameter, rows))
    for buffer in (model.duration_min_seconds, model.duration_max_seconds):
        if (
            buffer.ndim == 0
            or buffer.shape[0] != model.count
            or buffer.device != model.means.device
        ):
            raise ContractError("duration bound has no fixed-population row axis")

    # All model and optimizer topology checks finish before the first write.
    # A malformed caller therefore cannot leave a partially relocated model.
    runtime_ids_before = model.runtime_ids.clone()
    for parameter in parameters.values():
        parameter[dead_indices] = parameter[sampled_sources].clone()
    for buffer in (model.duration_min_seconds, model.duration_max_seconds):
        buffer[dead_indices] = buffer[sampled_sources].clone()

    model.opacity_logits[unique_sources, 0] = piece_logits
    model.opacity_logits[dead_indices, 0] = piece_logits[inverse]
    model.log_scales[unique_sources] = piece_log_scales
    model.log_scales[dead_indices] = piece_log_scales[inverse]
    if piece_persistence_logits is not None:
        model.persistence_logits[unique_sources, 0] = piece_persistence_logits
        model.persistence_logits[dead_indices, 0] = piece_persistence_logits[inverse]
    if not torch.equal(model.runtime_ids, runtime_ids_before):
        raise ContractError("relocation changed stable runtime slot identities")

    for optimizer, parameter, rows in reset_plan:
        _zero_parameter_rows(
            optimizer,
            parameter,
            rows,
            gaussian_count=model.count,
        )

    alpha64 = realized_piece_opacity
    ratios64 = multiplicities.to(torch.float64)
    combined = -torch.expm1(ratios64 * torch.log1p(-alpha64))
    center_error = torch.abs(combined - source_opacity)
    coefficient = _composited_profile_integral(
        alpha64,
        multiplicities,
    )
    scale_factor = realized_piece_scales[:, 0] / source_scales[:, 0]
    mass_relative_error = torch.abs(
        coefficient * scale_factor.square() / source_opacity - 1.0
    )
    far_source_alpha = source_opacity * source_persistence
    far_piece_alpha = alpha64 * realized_piece_persistence
    far_combined_alpha = -torch.expm1(
        ratios64 * torch.log1p(-far_piece_alpha)
    )
    transient = torch.linspace(
        0.0,
        1.0,
        17,
        dtype=torch.float64,
        device=model.means.device,
    ).reshape(1, -1)
    source_activation = source_persistence.unsqueeze(1) + (
        1.0 - source_persistence.unsqueeze(1)
    ) * transient
    piece_activation = realized_piece_persistence.unsqueeze(1) + (
        1.0 - realized_piece_persistence.unsqueeze(1)
    ) * transient
    source_temporal_alpha = source_opacity.unsqueeze(1) * source_activation
    piece_temporal_alpha = alpha64.unsqueeze(1) * piece_activation
    combined_temporal_alpha = -torch.expm1(
        multiplicities.to(torch.float64).unsqueeze(1)
        * torch.log1p(-piece_temporal_alpha)
    )
    temporal_profile_error = torch.abs(combined_temporal_alpha - source_temporal_alpha)
    return {
        "relocated": relocated,
        "unique_sources": int(unique_sources.numel()),
        "maximum_multiplicity": int(multiplicities.max()),
        "center_alpha_max_error": float(center_error.max()),
        "projected_alpha_mass_relative_max_error": float(mass_relative_error.max()),
        "persistent_alpha_max_error": float(
            torch.abs(far_combined_alpha - far_source_alpha).max()
        ),
        "temporal_profile_alpha_max_error": float(temporal_profile_error.max()),
        "maximum_scale_factor": float(scale_factor.max()),
    }


def _exact_non_negative_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"checkpoint relocation {name} must be a non-negative integer")
    return value


@dataclass(slots=True)
class RelocationController:
    config: RelocationConfig
    gaussian_count: int
    device: torch.device
    gradient_sum: Tensor = field(init=False)
    visibility_count: Tensor = field(init=False)
    last_relocation_step: Tensor = field(init=False)
    last_relocation_role: Tensor = field(init=False)
    window_steps: int = field(init=False, default=0)
    scheduled_events: int = field(init=False, default=0)
    applied_events: int = field(init=False, default=0)
    total_relocated: int = field(init=False, default=0)
    last_event_step: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if type(self.gaussian_count) is not int or self.gaussian_count <= 0:
            raise ContractError("relocation Gaussian count must be a positive integer")
        if self.config.mode not in {"off", "fixed_budget_relocation_v1"}:
            raise ContractError("relocation controller received an unsupported public mode")
        self.gradient_sum = torch.zeros(
            self.gaussian_count,
            dtype=torch.float32,
            device=self.device,
        )
        self.visibility_count = torch.zeros(
            self.gaussian_count,
            dtype=torch.int64,
            device=self.device,
        )
        self.last_relocation_step = torch.full(
            (self.gaussian_count,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.last_relocation_role = torch.full(
            (self.gaussian_count,),
            RELOCATION_ROLE_NONE,
            dtype=torch.uint8,
            device=self.device,
        )

    @classmethod
    def create(
        cls,
        config: RelocationConfig,
        *,
        gaussian_count: int,
        device: str | torch.device,
    ) -> RelocationController:
        try:
            resolved_device = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ContractError("relocation device is invalid") from exc
        return cls(
            config=config,
            gaussian_count=gaussian_count,
            device=resolved_device,
        )

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    def _policy(self) -> dict[str, int | float | str]:
        return {
            "mode": self.config.mode,
            "start": self.config.start,
            "stop": self.config.stop,
            "every": self.config.every,
            "opacity_threshold": self.config.opacity_threshold,
            "maximum_split_multiplicity": MAX_SPLIT_MULTIPLICITY,
        }

    def _scheduled(self, completed_step: int) -> bool:
        return (
            self.enabled
            and self.config.start <= completed_step < self.config.stop
            and (completed_step - self.config.start) % self.config.every == 0
        )

    def lineage(self, gaussian_ids: Tensor) -> tuple[Tensor, Tensor]:
        if gaussian_ids.dtype != torch.int64 or gaussian_ids.ndim != 1:
            raise ContractError("relocation lineage IDs must be one int64 vector")
        if gaussian_ids.device != self.device:
            raise ContractError("relocation lineage IDs and controller must share one device")
        if gaussian_ids.numel() and (
            int(gaussian_ids.min()) < 0 or int(gaussian_ids.max()) >= self.gaussian_count
        ):
            raise ContractError("relocation lineage ID is outside the fixed population")
        return (
            self.last_relocation_step[gaussian_ids],
            self.last_relocation_role[gaussian_ids],
        )

    @torch.no_grad()
    def _record_lineage(
        self,
        completed_step: int,
        *,
        destinations: Tensor,
        sources: Tensor,
    ) -> None:
        unique_sources = torch.unique(sources, sorted=True)
        self.last_relocation_step[destinations] = completed_step
        self.last_relocation_role[destinations] = RELOCATION_ROLE_DESTINATION
        self.last_relocation_step[unique_sources] = completed_step
        self.last_relocation_role[unique_sources] = RELOCATION_ROLE_SOURCE

    @torch.no_grad()
    def accumulate(self, aux: dict[str, Any]) -> None:
        if not self.enabled:
            return
        means2d = aux.get("means2d")
        gaussian_ids = aux.get("gaussian_ids")
        if not isinstance(means2d, Tensor) or not isinstance(gaussian_ids, Tensor):
            raise ContractError("relocation requires packed means2d and gaussian_ids metadata")
        if means2d.device != self.device or gaussian_ids.device != self.device:
            raise ContractError("relocation metadata and controller must share one device")
        gradient = means2d.grad
        if gradient is None:
            raise ContractError("relocation requires retained screen-space mean gradients")
        if (
            not gradient.is_floating_point()
            or tuple(gradient.shape) != (int(means2d.shape[0]), 2)
        ):
            raise ContractError("screen-space mean gradients must have shape [M,2]")
        if gaussian_ids.dtype != torch.int64 or tuple(gaussian_ids.shape) != (
            int(means2d.shape[0]),
        ):
            raise ContractError("gaussian_ids must map every packed screen-space gradient")
        if gaussian_ids.numel() and (
            int(gaussian_ids.min()) < 0
            or int(gaussian_ids.max()) >= self.gaussian_count
        ):
            raise ContractError("packed gaussian_ids contains an out-of-range row")
        width = aux.get("width")
        height = aux.get("height")
        camera_count = aux.get("n_cameras")
        if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
            raise ContractError("relocation requires positive integer raster dimensions")
        if type(camera_count) is not int or camera_count <= 0:
            raise ContractError("relocation requires a positive integer camera count")

        screen_gradient = gradient.detach()
        x = screen_gradient[:, 0] * (width * camera_count / 2.0)
        y = screen_gradient[:, 1] * (height * camera_count / 2.0)
        magnitudes = torch.sqrt(x.square() + y.square()).to(torch.float32)
        if not bool(torch.isfinite(magnitudes).all()):
            raise FloatingPointError("relocation received non-finite screen-space gradients")
        self.gradient_sum.index_add_(0, gaussian_ids, magnitudes)
        self.visibility_count.index_add_(0, gaussian_ids, torch.ones_like(gaussian_ids))
        self.window_steps += 1

    @torch.no_grad()
    def maybe_apply(
        self,
        completed_step: int,
        *,
        model: DynamicGaussianModel,
        optimizers: OptimizerBundle,
    ) -> dict[str, Any] | None:
        if type(completed_step) is not int or completed_step <= 0:
            raise ContractError("completed relocation step must be a positive integer")
        if not self._scheduled(completed_step):
            return None
        if model.count != self.gaussian_count or model.means.device != self.device:
            raise ContractError("relocation controller and model population disagree")
        if not bool(
            torch.isfinite(model.opacity_logits).all()
            & torch.isfinite(self.gradient_sum).all()
        ):
            raise FloatingPointError("relocation state or model opacity is non-finite")

        self.scheduled_events += 1
        logits64 = model.opacity_logits[:, 0].detach().to(torch.float64)
        opacity = torch.sigmoid(logits64).clamp(
            min=torch.finfo(torch.float64).tiny,
            max=1.0 - torch.finfo(torch.float64).eps,
        )
        dead_mask = opacity <= self.config.opacity_threshold
        dead_indices = torch.nonzero(dead_mask, as_tuple=True)[0]
        denominator = self.visibility_count.clamp_min(1).to(torch.float32)
        average_gradient = self.gradient_sum / denominator
        capacities = _clone_capacities(
            opacity,
            opacity_threshold=self.config.opacity_threshold,
        )
        source_eligible = (
            (~dead_mask)
            & (self.visibility_count > 0)
            & (average_gradient > 0.0)
            & (capacities > 0)
        )
        scores = torch.zeros_like(opacity)
        if bool(source_eligible.any()):
            maximum_gradient = average_gradient[source_eligible].max().to(torch.float64)
            scores[source_eligible] = (
                average_gradient[source_eligible].to(torch.float64) / maximum_gradient
            ) * opacity[source_eligible]
        admitted_capacities = torch.where(
            source_eligible,
            capacities,
            torch.zeros_like(capacities),
        )
        sampled_sources, allocation_strata = _sample_with_capacities(
            scores,
            admitted_capacities,
            count=int(dead_indices.numel()),
        )
        if sampled_sources.numel() < dead_indices.numel():
            order = torch.argsort(opacity[dead_indices], stable=True)
            dead_indices = dead_indices[order[: sampled_sources.numel()]]
        details = relocate_fixed_budget(
            model,
            optimizers,
            dead_indices=dead_indices,
            sampled_sources=sampled_sources,
        )
        relocated = int(details["relocated"])
        if relocated:
            self._record_lineage(
                completed_step,
                destinations=dead_indices,
                sources=sampled_sources,
            )
            self.applied_events += 1
            self.total_relocated += relocated
        self.last_event_step = completed_step
        event: dict[str, Any] = {
            "schema_version": RELOCATION_EVENT_SCHEMA,
            "mode": self.config.mode,
            "completed_step": completed_step,
            "status": "APPLIED" if relocated else "NOOP",
            "dead_candidates": int(dead_mask.sum()),
            "deferred_dead_candidates": int(dead_mask.sum()) - relocated,
            "eligible_sources": int(source_eligible.sum()),
            "source_clone_capacity": int(admitted_capacities.sum()),
            "window_steps": self.window_steps,
            "visible_gaussians": int((self.visibility_count > 0).sum()),
            "allocation_strata": allocation_strata,
            "source_allocation": "breadth_first_capacity_strata_v1",
            "source_score": "opacity_times_mean_pixel_position_gradient_v1",
            "split_equation": "center_alpha_and_projected_alpha_integral_v1",
            **details,
        }
        self.gradient_sum.zero_()
        self.visibility_count.zero_()
        self.window_steps = 0
        return event

    def state_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "schema_version": RELOCATION_STATE_SCHEMA,
            "policy": self._policy(),
            "gaussian_count": self.gaussian_count,
            "gradient_sum": self.gradient_sum.detach().cpu(),
            "visibility_count": self.visibility_count.detach().cpu(),
            "window_steps": self.window_steps,
            "scheduled_events": self.scheduled_events,
            "applied_events": self.applied_events,
            "total_relocated": self.total_relocated,
            "last_event_step": self.last_event_step,
            "last_relocation_step": self.last_relocation_step.detach().cpu(),
            "last_relocation_role": self.last_relocation_role.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any], *, require_state: bool) -> None:
        if not self.enabled:
            if state or require_state:
                raise ContractError("disabled relocation must have empty checkpoint state")
            return
        if not state:
            if require_state:
                raise ContractError("checkpoint is missing required relocation state")
            return
        expected_fields = {
            "schema_version",
            "policy",
            "gaussian_count",
            "gradient_sum",
            "visibility_count",
            "window_steps",
            "scheduled_events",
            "applied_events",
            "total_relocated",
            "last_event_step",
            "last_relocation_step",
            "last_relocation_role",
        }
        if set(state) != expected_fields or state.get("schema_version") != RELOCATION_STATE_SCHEMA:
            raise ContractError("checkpoint relocation state has invalid fields or schema")
        if state.get("policy") != self._policy():
            raise ContractError("checkpoint relocation state has a different policy")
        if type(state.get("gaussian_count")) is not int or state["gaussian_count"] != (
            self.gaussian_count
        ):
            raise ContractError("checkpoint relocation state has a different population")

        gradient_sum = state["gradient_sum"]
        visibility_count = state["visibility_count"]
        lineage_step = state["last_relocation_step"]
        lineage_role = state["last_relocation_role"]
        if (
            not isinstance(gradient_sum, Tensor)
            or gradient_sum.dtype != torch.float32
            or tuple(gradient_sum.shape) != (self.gaussian_count,)
            or not bool(torch.isfinite(gradient_sum).all() & (gradient_sum >= 0.0).all())
        ):
            raise ContractError("checkpoint relocation gradient accumulator is invalid")
        if (
            not isinstance(visibility_count, Tensor)
            or visibility_count.dtype != torch.int64
            or tuple(visibility_count.shape) != (self.gaussian_count,)
            or bool((visibility_count < 0).any())
        ):
            raise ContractError("checkpoint relocation visibility accumulator is invalid")
        if (
            not isinstance(lineage_step, Tensor)
            or lineage_step.dtype != torch.int32
            or tuple(lineage_step.shape) != (self.gaussian_count,)
            or bool((lineage_step < -1).any())
        ):
            raise ContractError("checkpoint relocation lineage steps are invalid")
        if (
            not isinstance(lineage_role, Tensor)
            or lineage_role.dtype != torch.uint8
            or tuple(lineage_role.shape) != (self.gaussian_count,)
            or bool((lineage_role > RELOCATION_ROLE_SOURCE).any())
            or not torch.equal(lineage_role == RELOCATION_ROLE_NONE, lineage_step == -1)
        ):
            raise ContractError("checkpoint relocation lineage roles are invalid")

        window_steps = _exact_non_negative_integer(state["window_steps"], name="window_steps")
        scheduled_events = _exact_non_negative_integer(
            state["scheduled_events"], name="scheduled_events"
        )
        applied_events = _exact_non_negative_integer(
            state["applied_events"], name="applied_events"
        )
        total_relocated = _exact_non_negative_integer(
            state["total_relocated"], name="total_relocated"
        )
        if applied_events > scheduled_events or total_relocated < applied_events:
            raise ContractError("checkpoint relocation event counters are inconsistent")
        raw_last_event = state["last_event_step"]
        if raw_last_event is not None and (
            type(raw_last_event) is not int or raw_last_event <= 0
        ):
            raise ContractError("checkpoint relocation last event step is invalid")
        if scheduled_events == 0 and raw_last_event is not None:
            raise ContractError("checkpoint relocation last event has no scheduled event")
        if scheduled_events > 0 and raw_last_event is None:
            raise ContractError("checkpoint relocation scheduled events have no last step")
        if raw_last_event is not None and (
            not self._scheduled(raw_last_event)
            or bool((lineage_step > raw_last_event).any())
        ):
            raise ContractError("checkpoint relocation event chronology is invalid")

        self.gradient_sum.copy_(gradient_sum.to(self.device))
        self.visibility_count.copy_(visibility_count.to(self.device))
        self.last_relocation_step.copy_(lineage_step.to(self.device))
        self.last_relocation_role.copy_(lineage_role.to(self.device))
        self.window_steps = window_steps
        self.scheduled_events = scheduled_events
        self.applied_events = applied_events
        self.total_relocated = total_relocated
        self.last_event_step = raw_last_event


__all__ = [
    "MAX_SPLIT_MULTIPLICITY",
    "RELOCATION_ROLE_DESTINATION",
    "RELOCATION_ROLE_NONE",
    "RELOCATION_ROLE_SOURCE",
    "RelocationController",
    "relocate_fixed_budget",
    "split_projected_alpha_mass",
]
