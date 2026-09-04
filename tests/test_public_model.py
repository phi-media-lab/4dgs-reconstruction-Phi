from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from p2g.errors import ContractError
from p2g.training.config import InitializationConfig
from p2g.training.initialization import (
    INITIALIZATION_SCHEMA,
    GaussianInit,
    load_gaussian_init,
    load_p2g_safetensors,
)
from p2g.training.model import DynamicGaussianModel

ROOT = Path(__file__).parents[1]


def _initialization(*, count: int = 2, sh_degree: int = 3) -> GaussianInit:
    rest_count = (sh_degree + 1) ** 2 - 1
    return GaussianInit(
        means=torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.5, 2.0]])[:count].contiguous(),
        log_scales=torch.zeros((count, 3), dtype=torch.float32),
        quaternions=torch.tensor([[2.0, 0.0, 0.0, 0.0]] * count),
        opacity_logits=torch.zeros((count, 1), dtype=torch.float32),
        sh0=torch.arange(count * 3, dtype=torch.float32).reshape(count, 1, 3),
        sh_rest=torch.zeros((count, rest_count, 3), dtype=torch.float32),
        center_times=torch.tensor([[0.25], [0.75]], dtype=torch.float32)[:count].contiguous(),
        duration_logits=torch.zeros((count, 1), dtype=torch.float32),
        velocities=torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])[:count].contiguous(),
        persistence_logits=torch.zeros((count, 1), dtype=torch.float32),
        duration_min_seconds=torch.full((count, 1), 0.1, dtype=torch.float32),
        duration_max_seconds=torch.full((count, 1), 0.9, dtype=torch.float32),
        runtime_ids=torch.arange(100, 100 + count, dtype=torch.int64),
        source={"format": "synthetic_test"},
    )


def _file_tensors(*, count: int = 2, sh_degree: int | None = None) -> dict[str, torch.Tensor]:
    initialization = _initialization(count=count, sh_degree=sh_degree or 0)
    tensors = {
        "means": initialization.means,
        "log_scales": initialization.log_scales,
        "quaternions": initialization.quaternions,
        "opacity_logits": initialization.opacity_logits,
        "sh0": initialization.sh0,
        "center_times": initialization.center_times,
        "duration_logits": initialization.duration_logits,
        "velocities": initialization.velocities,
        "runtime_ids": initialization.runtime_ids,
    }
    if sh_degree is not None:
        tensors["sh_rest"] = initialization.sh_rest
    return tensors


def _save(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    schema: str = INITIALIZATION_SCHEMA,
) -> None:
    save_file(tensors, str(path), metadata={"schema_version": schema})


def test_gaussian_initialization_enforces_physical_and_storage_invariants() -> None:
    initialization = _initialization()
    initialization.validate()
    assert initialization.count == 2
    assert initialization.max_sh_degree == 3

    with pytest.raises(ContractError, match="unique"):
        replace(initialization, runtime_ids=torch.tensor([1, 1], dtype=torch.int64)).validate()
    with pytest.raises(ContractError, match="zero quaternion"):
        replace(initialization, quaternions=torch.zeros((2, 4))).validate()
    with pytest.raises(ContractError, match="0 < min < max"):
        replace(
            initialization,
            duration_min_seconds=torch.ones((2, 1)),
            duration_max_seconds=torch.ones((2, 1)),
        ).validate()
    with pytest.raises(ContractError, match="perfect square"):
        replace(initialization, sh_rest=torch.zeros((2, 2, 3))).validate()
    with pytest.raises(ContractError, match="machine path"):
        replace(initialization, source={"format": "file:///private/init"}).validate()


def test_public_safetensors_loader_resolves_policy_without_aliases(tmp_path: Path) -> None:
    path = tmp_path / "initialization.safetensors"
    _save(path, _file_tensors())
    config = InitializationConfig(
        path=path,
        sh_degree=2,
        time_offset_seconds=0.125,
        duration_min_seconds=0.02,
        duration_max_seconds=0.8,
        persistence_initial_logit=-4.0,
    )

    loaded = load_gaussian_init(config)

    assert loaded.sh_rest.shape == (2, 8, 3)
    assert torch.count_nonzero(loaded.sh_rest) == 0
    assert torch.equal(loaded.center_times, _initialization(sh_degree=0).center_times - 0.125)
    assert torch.equal(loaded.duration_min_seconds, torch.full((2, 1), 0.02))
    assert torch.equal(loaded.duration_max_seconds, torch.full((2, 1), 0.8))
    assert torch.equal(loaded.persistence_logits, torch.full((2, 1), -4.0))
    assert loaded.source == {
        "format": "p2g_safetensors",
        "schema_version": INITIALIZATION_SCHEMA,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_public_safetensors_loader_preserves_exact_higher_order_sh(tmp_path: Path) -> None:
    path = tmp_path / "with-sh.safetensors"
    tensors = _file_tensors(sh_degree=2)
    tensors["sh_rest"] = torch.arange(48, dtype=torch.float32).reshape(2, 8, 3)
    _save(path, tensors)

    loaded = load_p2g_safetensors(path, InitializationConfig(path=path, sh_degree=2))

    assert torch.equal(loaded.sh_rest, tensors["sh_rest"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing"),
        ("unknown", "unknown"),
        ("alias", "missing"),
        ("wrong_dtype", "CPU torch.float32"),
    ],
)
def test_public_safetensors_loader_rejects_ambiguous_catalogs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    tensors = _file_tensors()
    if mutation == "missing":
        del tensors["velocities"]
    elif mutation == "unknown":
        tensors["private_plane"] = torch.zeros((2, 1))
    elif mutation == "alias":
        tensors["quats"] = tensors.pop("quaternions")
    else:
        tensors["means"] = tensors["means"].to(torch.float64)
    path = tmp_path / f"{mutation}.safetensors"
    _save(path, tensors)

    with pytest.raises(ContractError, match=message):
        load_gaussian_init(InitializationConfig(path=path))


def test_public_safetensors_loader_rejects_wrong_schema_and_unsafe_path(tmp_path: Path) -> None:
    wrong_schema = tmp_path / "wrong.safetensors"
    _save(wrong_schema, _file_tensors(), schema="private.schema")
    with pytest.raises(ContractError, match=INITIALIZATION_SCHEMA):
        load_gaussian_init(InitializationConfig(path=wrong_schema))

    target = tmp_path / "target.safetensors"
    link = tmp_path / "link.safetensors"
    _save(target, _file_tensors())
    link.symlink_to(target)
    with pytest.raises(ContractError, match="non-symlink"):
        load_gaussian_init(InitializationConfig(path=link))

    pickle_path = tmp_path / "legacy.pt"
    pickle_path.write_bytes(b"not a safe tensor file")
    with pytest.raises(ContractError, match=r"\.safetensors suffix"):
        load_p2g_safetensors(
            pickle_path,
            replace(InitializationConfig(path=pickle_path), format="p2g_safetensors"),
        )


def test_materialization_matches_the_declared_continuous_time_equations() -> None:
    initialization = _initialization(count=1, sh_degree=3)
    initialization.log_scales[:] = torch.log(torch.tensor([[0.5, 1.0, 2.0]]))
    model = DynamicGaussianModel(initialization, persistence=True, gate_logit_scale=2.0)

    materialized = model.materialize(0.75, sh_degree=0)

    transient = math.exp(-0.5)
    activation = 0.5 + 0.5 * transient
    assert torch.allclose(materialized.means, torch.tensor([[2.0, 2.0, 2.5]]))
    assert torch.allclose(materialized.quaternions, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(materialized.scales, torch.tensor([[0.5, 1.0, 2.0]]))
    assert torch.allclose(materialized.temporal_sigma, torch.tensor([[0.5]]))
    assert torch.allclose(materialized.temporal_activation, torch.tensor([[activation]]))
    assert torch.allclose(materialized.opacities, torch.tensor([0.5 * activation]))
    assert torch.equal(materialized.colors, initialization.sh0)
    assert torch.equal(materialized.time_delta, torch.tensor([[0.5]]))


def test_materialization_has_gradients_for_every_enabled_parameter_plane() -> None:
    model = DynamicGaussianModel(_initialization(), persistence=True)
    materialized = model.materialize(torch.tensor(0.5), sh_degree=3)
    objective = sum(
        value.sum()
        for value in (
            materialized.means,
            materialized.quaternions,
            materialized.scales,
            materialized.opacities,
            materialized.colors,
            materialized.temporal_sigma,
        )
    )
    objective.backward()

    assert model.trainable_parameter_names() == (
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
    )
    assert all(parameter.grad is not None for parameter in model.parameters())

    transient_only = DynamicGaussianModel(_initialization(), persistence=False)
    assert not transient_only.persistence_logits.requires_grad
    assert "persistence_logits" not in transient_only.trainable_parameter_names()


def test_checkpoint_state_is_exact_and_policy_bound() -> None:
    model = DynamicGaussianModel(_initialization(), persistence=True, gate_logit_scale=1.25)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    restored = DynamicGaussianModel.from_checkpoint_state(
        state,
        persistence=True,
        gate_logit_scale=1.25,
    )

    assert restored.persistence_enabled
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in state.items())
    with pytest.raises(ContractError, match="unknown"):
        DynamicGaussianModel.from_checkpoint_state(
            state | {"extra": torch.tensor(1.0)}, persistence=True
        )
    with pytest.raises(ContractError, match="differs from the run config"):
        DynamicGaussianModel.from_checkpoint_state(
            state,
            persistence=True,
            gate_logit_scale=2.0,
        )


@pytest.mark.parametrize(
    ("timestamp", "degree"),
    [
        (torch.tensor([0.0, 1.0]), 0),
        (math.nan, 0),
        (0.0, 4),
        (0.0, True),
    ],
)
def test_materialization_rejects_non_scalar_time_and_invalid_degree(
    timestamp: Any, degree: Any
) -> None:
    model = DynamicGaussianModel(_initialization(), persistence=False)

    with pytest.raises(ContractError):
        model.materialize(timestamp, sh_degree=degree)


def test_public_initialization_and_model_sources_have_no_reference_adapter() -> None:
    combined = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8").casefold()
        for relative in (
            "src/p2g/training/initialization.py",
            "src/p2g/training/model.py",
        )
    )
    forbidden = (
        "free" + "time",
        "ft" + "gs",
        "torch.load(",
        "weights_" + "only",
        "sys.path",
        "compat" + "ibility",
    )
    assert not any(token in combined for token in forbidden)
