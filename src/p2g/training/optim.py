from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch

from p2g.errors import ContractError
from p2g.training.config import OptimizerConfig
from p2g.training.model import DynamicGaussianModel
from p2g.training.photometric import CameraColorCorrectors


class OptimizerBundle(Mapping[str, torch.optim.Optimizer]):
    def __init__(
        self,
        optimizers: dict[str, torch.optim.Optimizer],
        schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    ) -> None:
        self.optimizers = optimizers
        self.schedulers = schedulers

    def __getitem__(self, name: str) -> torch.optim.Optimizer:
        return self.optimizers[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step_schedulers(self) -> None:
        for scheduler in self.schedulers.values():
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": {name: value.state_dict() for name, value in self.optimizers.items()},
            "schedulers": {name: value.state_dict() for name, value in self.schedulers.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state.get("optimizers", {})) != set(self.optimizers):
            raise ContractError("checkpoint optimizer set does not match this model")
        if set(state.get("schedulers", {})) != set(self.schedulers):
            raise ContractError("checkpoint scheduler set does not match this run")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(state["schedulers"][name])


def build_optimizers(
    model: DynamicGaussianModel,
    config: OptimizerConfig,
    *,
    iterations: int,
    scene_extent: float,
    color_correctors: CameraColorCorrectors | None = None,
    color_correction_lr: float | None = None,
) -> OptimizerBundle:
    parameters = dict(model.named_parameters())
    unknown = set(config.lrs) - set(parameters)
    if unknown:
        raise ContractError(f"optimizer config names unknown model parameters: {sorted(unknown)}")
    optimizers: dict[str, torch.optim.Optimizer] = {}
    for name, learning_rate in config.lrs.items():
        parameter = parameters[name]
        if not parameter.requires_grad:
            continue
        effective_lr = (
            learning_rate * scene_extent if name in {"means", "velocities"} else learning_rate
        )
        optimizers[name] = torch.optim.Adam(
            [{"params": [parameter], "lr": effective_lr, "name": name}],
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    if color_correctors is not None:
        if color_correction_lr is None or color_correction_lr <= 0.0:
            raise ContractError("enabled color correction requires a positive learning rate")
        optimizers["color_correctors"] = torch.optim.Adam(
            [
                {
                    "params": list(color_correctors.parameters()),
                    "lr": color_correction_lr,
                    "name": "color_correctors",
                }
            ],
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {
        name: torch.optim.lr_scheduler.ExponentialLR(
            optimizers[name], gamma=final_factor ** (1.0 / iterations)
        )
        for name, final_factor in config.lr_final_factors.items()
        if name in optimizers
    }
    return OptimizerBundle(optimizers, schedulers)
