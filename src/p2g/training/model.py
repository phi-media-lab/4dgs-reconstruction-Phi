"""Independent explicit continuous-time 4D Gaussian parameterization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from p2g.errors import ContractError
from p2g.training.initialization import GaussianInit

MODEL_EQUATION_VERSION = "p2g.linear_motion_gaussian_gate.v1"


def _state_values_are_tensors(state: Mapping[str, object]) -> bool:
    return all(isinstance(value, Tensor) for value in state.values())


@dataclass(frozen=True, slots=True)
class MaterializedGaussians:
    """Physical Gaussian planes at one scalar time, ready for rasterization."""

    means: Tensor
    quaternions: Tensor
    scales: Tensor
    opacities: Tensor
    colors: Tensor
    temporal_activation: Tensor
    temporal_sigma: Tensor
    time_delta: Tensor


class DynamicGaussianModel(nn.Module):
    """Struct-of-arrays trainable state with vectorized scalar-time evaluation."""

    duration_min_seconds: Tensor
    duration_max_seconds: Tensor
    runtime_ids: Tensor
    gate_logit_scale: Tensor

    def __init__(
        self,
        initialization: GaussianInit,
        *,
        persistence: bool,
        gate_logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        initialization.validate()
        if type(persistence) is not bool:
            raise ContractError("persistence must be a boolean")
        if (
            isinstance(gate_logit_scale, bool)
            or not math.isfinite(gate_logit_scale)
            or gate_logit_scale <= 0.0
        ):
            raise ContractError("gate_logit_scale must be positive and finite")
        self.means = nn.Parameter(initialization.means.clone())
        self.log_scales = nn.Parameter(initialization.log_scales.clone())
        self.quaternions = nn.Parameter(initialization.quaternions.clone())
        self.opacity_logits = nn.Parameter(initialization.opacity_logits.clone())
        self.sh0 = nn.Parameter(initialization.sh0.clone())
        self.sh_rest = nn.Parameter(initialization.sh_rest.clone())
        self.center_times = nn.Parameter(initialization.center_times.clone())
        self.duration_logits = nn.Parameter(initialization.duration_logits.clone())
        self.velocities = nn.Parameter(initialization.velocities.clone())
        self.persistence_logits = nn.Parameter(
            initialization.persistence_logits.clone(), requires_grad=persistence
        )
        self.register_buffer(
            "duration_min_seconds",
            initialization.duration_min_seconds.clone(),
            persistent=True,
        )
        self.register_buffer(
            "duration_max_seconds",
            initialization.duration_max_seconds.clone(),
            persistent=True,
        )
        self.register_buffer("runtime_ids", initialization.runtime_ids.clone(), persistent=True)
        self.register_buffer(
            "gate_logit_scale",
            torch.tensor(gate_logit_scale, dtype=torch.float32),
            persistent=True,
        )
        self.persistence_enabled = persistence

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def max_sh_degree(self) -> int:
        coefficient_count = int(self.sh_rest.shape[1]) + 1
        root = math.isqrt(coefficient_count)
        if root * root != coefficient_count:
            raise ContractError("model SH coefficient count must be a perfect square")
        return root - 1

    def gate(self) -> Tensor:
        """Return the persistent mixture fraction in physical [0, 1] space."""

        return torch.sigmoid(self.gate_logit_scale * self.persistence_logits)

    def materialize(self, timestamp: float | Tensor, *, sh_degree: int) -> MaterializedGaussians:
        """Evaluate every Gaussian at one time without per-Gaussian Python work."""

        if type(sh_degree) is not int or not 0 <= sh_degree <= self.max_sh_degree:
            raise ContractError(
                f"active SH degree {sh_degree} is outside [0, {self.max_sh_degree}]"
            )
        try:
            time = torch.as_tensor(timestamp, dtype=self.means.dtype, device=self.means.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ContractError("materialization timestamp must be one numeric scalar") from exc
        if time.numel() != 1:
            raise ContractError("materialization timestamp must be one finite scalar")
        # Scene admission already proves timestamp finiteness. Repeating a Python
        # truth-value read on a HIP tensor here would synchronize every render.
        if time.device.type == "cpu" and not bool(torch.isfinite(time).all()):
            raise ContractError("materialization timestamp must be one finite scalar")
        delta = time.reshape(1, 1) - self.center_times
        duration_fraction = torch.sigmoid(self.duration_logits)
        sigma = self.duration_min_seconds + (
            self.duration_max_seconds - self.duration_min_seconds
        ) * duration_fraction
        normalized_time = delta / sigma
        transient = torch.exp(-0.5 * normalized_time.square())
        if self.persistence_enabled:
            persistent = self.gate()
            activation = persistent + (1.0 - persistent) * transient
        else:
            activation = transient
        coefficient_count = (sh_degree + 1) ** 2 - 1
        colors = torch.cat((self.sh0, self.sh_rest[:, :coefficient_count]), dim=1)
        return MaterializedGaussians(
            means=self.means + self.velocities * delta,
            quaternions=F.normalize(self.quaternions, dim=1, eps=1.0e-12),
            scales=torch.exp(self.log_scales),
            opacities=(torch.sigmoid(self.opacity_logits) * activation).squeeze(-1),
            colors=colors,
            temporal_activation=activation,
            temporal_sigma=sigma,
            time_delta=delta,
        )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    @classmethod
    def from_checkpoint_state(
        cls,
        state: Mapping[str, Tensor],
        *,
        persistence: bool,
        gate_logit_scale: float | None = None,
    ) -> DynamicGaussianModel:
        """Reconstruct from an exact tensor-only state and verify external policy."""

        required = {
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
            "duration_min_seconds",
            "duration_max_seconds",
            "runtime_ids",
            "gate_logit_scale",
        }
        if set(state) != required:
            missing = sorted(required - set(state))
            unknown = sorted(set(state) - required)
            raise ContractError(
                f"checkpoint model tensor catalog mismatch: missing={missing}, unknown={unknown}"
            )
        if not _state_values_are_tensors(state):
            raise ContractError("checkpoint model state must contain only tensors")
        stored_scale = state["gate_logit_scale"]
        if stored_scale.numel() != 1 or not bool(torch.isfinite(stored_scale).all()):
            raise ContractError("checkpoint gate_logit_scale must be one finite scalar")
        resolved_scale = float(stored_scale.detach().cpu().item())
        if resolved_scale <= 0.0:
            raise ContractError("checkpoint gate_logit_scale must be positive")
        if gate_logit_scale is not None and not math.isclose(
            resolved_scale,
            gate_logit_scale,
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        ):
            raise ContractError("checkpoint gate_logit_scale differs from the run config")

        def cpu(name: str) -> Tensor:
            return state[name].detach().to(device="cpu").contiguous()

        initialization = GaussianInit(
            means=cpu("means"),
            log_scales=cpu("log_scales"),
            quaternions=cpu("quaternions"),
            opacity_logits=cpu("opacity_logits"),
            sh0=cpu("sh0"),
            sh_rest=cpu("sh_rest"),
            center_times=cpu("center_times"),
            duration_logits=cpu("duration_logits"),
            velocities=cpu("velocities"),
            persistence_logits=cpu("persistence_logits"),
            duration_min_seconds=cpu("duration_min_seconds"),
            duration_max_seconds=cpu("duration_max_seconds"),
            runtime_ids=cpu("runtime_ids"),
            source={"format": "checkpoint"},
        )
        model = cls(
            initialization,
            persistence=persistence,
            gate_logit_scale=resolved_scale,
        )
        canonical_state = {name: cpu(name) for name in required}
        model.load_state_dict(canonical_state, strict=True)
        return model
