# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from p2g.errors import ContractError
from p2g.training.config import OptimizerConfig, RelocationConfig, ScreenInfluenceGuardConfig
from p2g.training.initialization import GaussianInit
from p2g.training.model import DynamicGaussianModel
from p2g.training.optim import OptimizerBundle, build_optimizers
from p2g.training.photometric import CameraColorCorrectors
from p2g.training.relocation import (
    MAX_SPLIT_MULTIPLICITY,
    RELOCATION_ROLE_DESTINATION,
    RELOCATION_ROLE_NONE,
    RELOCATION_ROLE_SOURCE,
    RelocationController,
    _sample_with_capacities,
    relocate_fixed_budget,
    split_projected_alpha_mass,
)
from p2g.training.screen_guard import project_base_opacity_rows, screen_candidate_mask


def _initialization(count: int = 5) -> GaussianInit:
    index = torch.arange(count, dtype=torch.float32).reshape(count, 1)
    return GaussianInit(
        means=torch.cat((0.1 + index, 0.2 + index, 2.0 + index), dim=1),
        log_scales=torch.log(
            torch.cat((0.1 + index * 0.01, 0.2 + index * 0.01, 0.3 + index * 0.01), dim=1)
        ),
        quaternions=torch.tensor([[1.0, 0.1, 0.2, 0.3]] * count),
        opacity_logits=torch.linspace(-1.0, 1.0, count).reshape(count, 1),
        sh0=torch.arange(count * 3, dtype=torch.float32).reshape(count, 1, 3) / 10.0,
        sh_rest=torch.arange(count * 45, dtype=torch.float32).reshape(count, 15, 3)
        / 100.0,
        center_times=index / max(count - 1, 1),
        duration_logits=-0.5 + index / 10.0,
        velocities=torch.cat((index / 10.0, index / 20.0, -index / 30.0), dim=1),
        persistence_logits=-2.0 + index / 10.0,
        duration_min_seconds=0.02 + index / 1000.0,
        duration_max_seconds=0.8 + index / 100.0,
        runtime_ids=torch.arange(100, 100 + count, dtype=torch.int64),
        source={"format": "project_owned_analytic_fixture"},
    )


def _runtime(
    count: int = 5,
    *,
    camera_count: int | None = None,
) -> tuple[DynamicGaussianModel, OptimizerBundle, CameraColorCorrectors | None]:
    model = DynamicGaussianModel(_initialization(count), persistence=True)
    correctors = (
        None
        if camera_count is None
        else CameraColorCorrectors(tuple(f"camera-{index}" for index in range(camera_count)))
    )
    optimizers = build_optimizers(
        model,
        OptimizerConfig(),
        iterations=20,
        scene_extent=1.0,
        color_correctors=correctors,
        color_correction_lr=1.0e-3 if correctors is not None else None,
    )
    return model, optimizers, correctors


def _populate_adam(
    model: DynamicGaussianModel,
    optimizers: OptimizerBundle,
    correctors: CameraColorCorrectors | None = None,
) -> None:
    objective = torch.stack(
        [parameter.square().sum() for parameter in model.parameters()]
    ).sum()
    if correctors is not None:
        objective = objective + sum(
            (parameter + 1.0).square().sum() for parameter in correctors.parameters()
        )
    objective.backward()
    optimizers.step()
    optimizers.zero_grad(set_to_none=True)


def _set_opacities(model: DynamicGaussianModel, values: list[float]) -> None:
    with torch.no_grad():
        model.opacity_logits[:, 0] = torch.logit(torch.tensor(values, dtype=torch.float32))


def test_split_equation_preserves_center_alpha_and_projected_integral() -> None:
    opacities = torch.tensor([0.2, 0.7, 0.9], dtype=torch.float32)
    scales = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.25, 0.8], [1.0, 0.5, 0.125]],
        dtype=torch.float32,
    )
    multiplicities = torch.tensor([1, 2, 5], dtype=torch.int64)

    piece_alpha, piece_scales = split_projected_alpha_mass(
        opacities,
        scales,
        multiplicities,
    )

    combined = 1.0 - torch.pow(
        1.0 - piece_alpha.to(torch.float64),
        multiplicities,
    )
    assert torch.allclose(combined, opacities.to(torch.float64), atol=1.0e-7)
    for row, multiplicity in enumerate(multiplicities.tolist()):
        alpha = float(piece_alpha[row])
        coefficient = sum(
            ((-1.0) ** (order + 1))
            * math.comb(multiplicity, order)
            * alpha**order
            / order
            for order in range(1, multiplicity + 1)
        )
        scale_factor = float(piece_scales[row, 0] / scales[row, 0])
        assert coefficient * scale_factor**2 == pytest.approx(
            float(opacities[row]),
            abs=2.0e-7,
        )
    assert torch.equal(piece_alpha[:1], opacities[:1])
    assert torch.all(piece_scales <= scales)


@pytest.mark.parametrize(
    ("opacities", "scales", "multiplicities"),
    [
        (torch.tensor([0.0]), torch.ones((1, 3)), torch.ones(1, dtype=torch.int64)),
        (torch.tensor([0.5]), torch.zeros((1, 3)), torch.ones(1, dtype=torch.int64)),
        (
            torch.tensor([0.5]),
            torch.ones((1, 3)),
            torch.tensor([MAX_SPLIT_MULTIPLICITY + 1]),
        ),
    ],
)
def test_split_equation_rejects_non_physical_inputs(
    opacities: torch.Tensor,
    scales: torch.Tensor,
    multiplicities: torch.Tensor,
) -> None:
    with pytest.raises((ContractError, FloatingPointError)):
        split_projected_alpha_mass(opacities, scales, multiplicities)


def test_capacity_sampler_spreads_sources_before_increasing_multiplicity() -> None:
    torch.manual_seed(5)
    sampled, allocation_strata = _sample_with_capacities(
        torch.tensor([100.0, 1.0, 1.0]),
        torch.tensor([3, 3, 3], dtype=torch.int64),
        count=5,
    )
    counts = torch.bincount(sampled, minlength=3)

    assert int(counts.sum()) == 5
    assert int(counts.min()) == 1
    assert int(counts.max()) == 2
    assert allocation_strata == 2


def test_relocation_keeps_capacity_and_runtime_ids_and_resets_precise_adam_rows() -> None:
    model, optimizers, _ = _runtime()
    _set_opacities(model, [0.8, 0.4, 0.001, 0.002, 0.3])
    _populate_adam(model, optimizers)
    runtime_ids = model.runtime_ids.clone()
    source_mean = model.means[0].detach().clone()
    source_center_time = model.center_times[0].detach().clone()
    source_duration_min = model.duration_min_seconds[0].detach().clone()
    original_opacity = float(torch.sigmoid(model.opacity_logits[0, 0]).detach())
    original_persistence = float(model.gate()[0, 0].detach())
    means_moment = optimizers["means"].state[model.means]["exp_avg"].clone()
    opacity_moment = optimizers["opacity_logits"].state[model.opacity_logits][
        "exp_avg"
    ].clone()
    persistence_moment = optimizers["persistence_logits"].state[
        model.persistence_logits
    ]["exp_avg"].clone()

    details = relocate_fixed_budget(
        model,
        optimizers,
        dead_indices=torch.tensor([2, 3], dtype=torch.int64),
        sampled_sources=torch.tensor([0, 0], dtype=torch.int64),
    )

    assert model.count == 5
    assert torch.equal(model.runtime_ids, runtime_ids)
    assert torch.equal(model.means[2], source_mean)
    assert torch.equal(model.means[3], source_mean)
    assert torch.equal(model.center_times[2], source_center_time)
    assert torch.equal(model.duration_min_seconds[3], source_duration_min)
    split_opacity = torch.sigmoid(model.opacity_logits[[0, 2, 3], 0])
    assert torch.allclose(split_opacity, split_opacity[:1].expand_as(split_opacity))
    assert 1.0 - (1.0 - float(split_opacity[0].detach())) ** 3 == pytest.approx(
        original_opacity,
        abs=2.0e-6,
    )
    assert details["relocated"] == 2
    assert details["maximum_multiplicity"] == 3
    assert details["center_alpha_max_error"] < 1.0e-6
    assert details["projected_alpha_mass_relative_max_error"] < 1.0e-6
    piece_persistence = model.gate()[[0, 2, 3], 0]
    assert torch.allclose(
        piece_persistence,
        piece_persistence[:1].expand_as(piece_persistence),
    )
    far_composite = 1.0 - (
        1.0 - float(split_opacity[0].detach() * piece_persistence[0].detach())
    ) ** 3
    assert far_composite == pytest.approx(
        original_opacity * original_persistence,
        abs=2.0e-6,
    )
    assert details["persistent_alpha_max_error"] < 1.0e-6
    assert details["temporal_profile_alpha_max_error"] >= 0.0

    updated_means_moment = optimizers["means"].state[model.means]["exp_avg"]
    updated_opacity_moment = optimizers["opacity_logits"].state[model.opacity_logits][
        "exp_avg"
    ]
    assert torch.equal(updated_means_moment[0], means_moment[0])
    assert torch.count_nonzero(updated_means_moment[[2, 3]]) == 0
    assert torch.equal(updated_means_moment[1], means_moment[1])
    assert torch.count_nonzero(updated_opacity_moment[[0, 2, 3]]) == 0
    assert torch.equal(updated_opacity_moment[1], opacity_moment[1])
    updated_persistence_moment = optimizers["persistence_logits"].state[
        model.persistence_logits
    ]["exp_avg"]
    assert torch.count_nonzero(updated_persistence_moment[[0, 2, 3]]) == 0
    assert torch.equal(updated_persistence_moment[1], persistence_moment[1])


def test_relocation_preflights_optimizer_topology_before_model_mutation() -> None:
    model, optimizers, _ = _runtime()
    _set_opacities(model, [0.8, 0.4, 0.001, 0.002, 0.3])
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizers.optimizers.pop("velocities")

    with pytest.raises(ContractError, match="no optimizer"):
        relocate_fixed_budget(
            model,
            optimizers,
            dead_indices=torch.tensor([2], dtype=torch.int64),
            sampled_sources=torch.tensor([0], dtype=torch.int64),
        )

    assert set(model.state_dict()) == set(before)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_relocation_never_resets_an_unrelated_equal_length_camera_optimizer() -> None:
    model, optimizers, correctors = _runtime(camera_count=5)
    assert correctors is not None
    _set_opacities(model, [0.8, 0.4, 0.001, 0.2, 0.3])
    _populate_adam(model, optimizers, correctors)
    color_state = {
        (id(parameter), name): value.clone()
        for parameter, state in optimizers["color_correctors"].state.items()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }

    relocate_fixed_budget(
        model,
        optimizers,
        dead_indices=torch.tensor([2], dtype=torch.int64),
        sampled_sources=torch.tensor([0], dtype=torch.int64),
    )

    for parameter, state in optimizers["color_correctors"].state.items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, color_state[(id(parameter), name)])


def test_screen_guard_candidate_rule_is_an_explicit_depth_and_tile_intersection() -> None:
    config = ScreenInfluenceGuardConfig()
    selected = screen_candidate_mask(
        torch.tensor([0.02, 0.02, 0.2, float("nan")]),
        torch.tensor([90, 89, 100, 100], dtype=torch.int32),
        total_tiles=100,
        near_plane=0.01,
        near_plane_multiple=config.near_plane_multiple,
        tile_coverage_minimum=config.tile_coverage_minimum,
    )
    assert selected.tolist() == [True, False, False, False]


def test_screen_guard_projection_changes_only_named_opacity_rows_and_moments() -> None:
    model, optimizers, _ = _runtime()
    _set_opacities(model, [0.8, 0.6, 0.4, 0.2, 0.1])
    _populate_adam(model, optimizers)
    opacity_before = float(torch.sigmoid(model.opacity_logits[2, 0]).detach())
    opacity_moment = optimizers["opacity_logits"].state[model.opacity_logits][
        "exp_avg"
    ].clone()
    means_moment = optimizers["means"].state[model.means]["exp_avg"].clone()

    event = project_base_opacity_rows(
        model,
        optimizers,
        gaussian_ids=torch.tensor([2], dtype=torch.int64),
        opacity_scales=torch.tensor([0.25]),
    )

    assert event["runtime_slots"] == [102]
    assert float(torch.sigmoid(model.opacity_logits[2, 0]).detach()) == pytest.approx(
        opacity_before * 0.25,
        abs=1.0e-6,
    )
    updated_opacity_moment = optimizers["opacity_logits"].state[model.opacity_logits][
        "exp_avg"
    ]
    assert torch.count_nonzero(updated_opacity_moment[2]) == 0
    assert torch.equal(updated_opacity_moment[0], opacity_moment[0])
    assert torch.equal(optimizers["means"].state[model.means]["exp_avg"], means_moment)


def test_controller_accumulates_pixel_gradient_and_applies_start_anchored_schedule() -> None:
    model, optimizers, _ = _runtime(count=3)
    _set_opacities(model, [0.8, 0.001, 0.4])
    config = RelocationConfig(
        mode="fixed_budget_relocation_v1",
        start=2,
        stop=7,
        every=2,
        opacity_threshold=0.01,
    )
    controller = RelocationController.create(config, gaussian_count=3, device="cpu")
    means2d = torch.zeros((3, 2), requires_grad=True)
    means2d.grad = torch.tensor([[3.0, 4.0], [0.0, 2.0], [0.0, 6.0]])
    controller.accumulate(
        {
            "means2d": means2d,
            "gaussian_ids": torch.tensor([0, 2, 0], dtype=torch.int64),
            "width": 10,
            "height": 20,
            "n_cameras": 1,
        }
    )
    expected_source_sum = math.sqrt(15.0**2 + 40.0**2) + 60.0
    assert float(controller.gradient_sum[0]) == pytest.approx(expected_source_sum)
    assert float(controller.gradient_sum[2]) == pytest.approx(20.0)
    assert controller.maybe_apply(1, model=model, optimizers=optimizers) is None

    torch.manual_seed(9)
    event = controller.maybe_apply(2, model=model, optimizers=optimizers)

    assert event is not None
    assert event["schema_version"] == "p2g.fixed_budget_relocation_event.v1"
    assert event["status"] == "APPLIED"
    assert event["dead_candidates"] == 1
    assert event["relocated"] == 1
    assert event["window_steps"] == 1
    assert event["allocation_strata"] == 1
    assert event["source_allocation"] == "breadth_first_capacity_strata_v1"
    assert event["split_equation"] == "center_alpha_and_projected_alpha_integral_v1"
    assert torch.count_nonzero(controller.gradient_sum) == 0
    assert torch.count_nonzero(controller.visibility_count) == 0
    steps, roles = controller.lineage(torch.tensor([0, 1, 2], dtype=torch.int64))
    assert steps.tolist() == [2, 2, -1]
    assert roles.tolist() == [
        RELOCATION_ROLE_SOURCE,
        RELOCATION_ROLE_DESTINATION,
        RELOCATION_ROLE_NONE,
    ]
    assert controller._scheduled(4)
    assert controller._scheduled(6)
    assert not controller._scheduled(7)


def test_controller_reports_noop_when_no_live_visible_source_has_capacity() -> None:
    model, optimizers, _ = _runtime(count=3)
    _set_opacities(model, [0.001, 0.002, 0.003])
    controller = RelocationController.create(
        RelocationConfig(
            mode="fixed_budget_relocation_v1",
            start=1,
            stop=3,
            every=1,
            opacity_threshold=0.01,
        ),
        gaussian_count=3,
        device="cpu",
    )
    means2d = torch.zeros((3, 2), requires_grad=True)
    means2d.grad = torch.ones((3, 2))
    controller.accumulate(
        {
            "means2d": means2d,
            "gaussian_ids": torch.arange(3, dtype=torch.int64),
            "width": 4,
            "height": 4,
            "n_cameras": 1,
        }
    )

    event = controller.maybe_apply(1, model=model, optimizers=optimizers)

    assert event is not None
    assert event["status"] == "NOOP"
    assert event["dead_candidates"] == 3
    assert event["deferred_dead_candidates"] == 3
    assert event["eligible_sources"] == 0
    assert event["source_clone_capacity"] == 0
    assert controller.scheduled_events == 1
    assert controller.applied_events == 0


def test_controller_defers_dead_slots_instead_of_splitting_below_threshold() -> None:
    model, optimizers, _ = _runtime(count=5)
    _set_opacities(model, [0.03, 0.001, 0.002, 0.003, 0.004])
    controller = RelocationController.create(
        RelocationConfig(
            mode="fixed_budget_relocation_v1",
            start=1,
            stop=2,
            every=1,
            opacity_threshold=0.01,
        ),
        gaussian_count=5,
        device="cpu",
    )
    means2d = torch.zeros((1, 2), requires_grad=True)
    means2d.grad = torch.tensor([[1.0, 1.0]])
    controller.accumulate(
        {
            "means2d": means2d,
            "gaussian_ids": torch.tensor([0], dtype=torch.int64),
            "width": 8,
            "height": 8,
            "n_cameras": 1,
        }
    )

    event = controller.maybe_apply(1, model=model, optimizers=optimizers)

    assert event is not None
    assert event["source_clone_capacity"] == 2
    assert event["relocated"] == 2
    assert event["deferred_dead_candidates"] == 2
    split_rows = controller.last_relocation_step == 1
    assert int(split_rows.sum()) == 3
    assert torch.all(torch.sigmoid(model.opacity_logits[split_rows, 0]) > 0.01)


def test_controller_state_roundtrip_is_policy_bound_and_lineage_closed() -> None:
    config = RelocationConfig(
        mode="fixed_budget_relocation_v1",
        start=2,
        stop=8,
        every=2,
        opacity_threshold=0.01,
    )
    controller = RelocationController.create(config, gaussian_count=3, device="cpu")
    controller._record_lineage(
        4,
        destinations=torch.tensor([1], dtype=torch.int64),
        sources=torch.tensor([0], dtype=torch.int64),
    )
    controller.scheduled_events = 2
    controller.applied_events = 1
    controller.total_relocated = 1
    controller.last_event_step = 4
    controller.window_steps = 3
    controller.gradient_sum[:] = torch.tensor([1.0, 2.0, 3.0])
    controller.visibility_count[:] = torch.tensor([1, 2, 3])
    state = controller.state_dict()

    restored = RelocationController.create(config, gaussian_count=3, device="cpu")
    restored.load_state_dict(state, require_state=True)

    assert torch.equal(restored.gradient_sum, controller.gradient_sum)
    assert torch.equal(restored.visibility_count, controller.visibility_count)
    assert torch.equal(restored.last_relocation_step, controller.last_relocation_step)
    assert torch.equal(restored.last_relocation_role, controller.last_relocation_role)
    assert restored.window_steps == 3
    assert restored.last_event_step == 4

    changed_policy = RelocationController.create(
        RelocationConfig(
            mode="fixed_budget_relocation_v1",
            start=2,
            stop=8,
            every=1,
            opacity_threshold=0.01,
        ),
        gaussian_count=3,
        device="cpu",
    )
    with pytest.raises(ContractError, match="different policy"):
        changed_policy.load_state_dict(state, require_state=True)
    tampered: dict[str, Any] = dict(state)
    tampered["last_relocation_role"] = torch.tensor([9, 0, 0], dtype=torch.uint8)
    with pytest.raises(ContractError, match="lineage roles"):
        restored.load_state_dict(tampered, require_state=True)


def test_relocation_rejects_aliasing_or_out_of_population_indices() -> None:
    model, optimizers, _ = _runtime(count=3)
    with pytest.raises(ContractError, match="disjoint"):
        relocate_fixed_budget(
            model,
            optimizers,
            dead_indices=torch.tensor([1], dtype=torch.int64),
            sampled_sources=torch.tensor([1], dtype=torch.int64),
        )
    with pytest.raises(ContractError, match="outside"):
        relocate_fixed_budget(
            model,
            optimizers,
            dead_indices=torch.tensor([3], dtype=torch.int64),
            sampled_sources=torch.tensor([0], dtype=torch.int64),
        )
