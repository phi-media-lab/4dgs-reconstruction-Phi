from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from p2g.errors import ContractError
from p2g.training import losses as loss_module
from p2g.training.config import LossConfig
from p2g.training.initialization import GaussianInit
from p2g.training.losses import LOSS_TERM_NAMES, LossFunction, psnr, structural_similarity
from p2g.training.model import DynamicGaussianModel

ROOT = Path(__file__).parents[1]


def _model(*, persistence: bool = True) -> DynamicGaussianModel:
    count = 3
    initialization = GaussianInit(
        means=torch.tensor(
            [[0.0, 0.0, 2.0], [0.2, 0.1, 2.2], [-0.1, 0.3, 2.4]],
            dtype=torch.float32,
        ),
        log_scales=torch.full((count, 3), math.log(0.1), dtype=torch.float32),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        opacity_logits=torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32),
        sh0=torch.zeros((count, 1, 3)),
        sh_rest=torch.zeros((count, 15, 3)),
        center_times=torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32),
        duration_logits=torch.zeros((count, 1)),
        velocities=torch.tensor([[0.1, 0.0, 0.0]] * count),
        persistence_logits=torch.tensor([[-2.0], [0.0], [2.0]], dtype=torch.float32),
        duration_min_seconds=torch.full((count, 1), 0.05),
        duration_max_seconds=torch.full((count, 1), 1.0),
        runtime_ids=torch.arange(10, 10 + count),
        source={"format": "synthetic_test"},
    )
    return DynamicGaussianModel(initialization, persistence=persistence, gate_logit_scale=1.25)


def test_ssim_identity_constant_equation_and_prediction_gradient() -> None:
    identical = torch.linspace(0.0, 1.0, 13 * 15 * 3, dtype=torch.float32).reshape(13, 15, 3)
    assert structural_similarity(identical, identical, padding="same") == pytest.approx(
        1.0, abs=2.0e-6
    )
    assert structural_similarity(identical, identical, padding="valid") == pytest.approx(
        1.0, abs=2.0e-6
    )

    prediction = torch.full((13, 15, 3), 0.2, dtype=torch.float32, requires_grad=True)
    target = torch.full_like(prediction, 0.6)
    observed = structural_similarity(prediction, target, padding="valid")
    expected = (2.0 * 0.2 * 0.6 + 0.01**2) / (0.2**2 + 0.6**2 + 0.01**2)
    assert float(observed.detach()) == pytest.approx(expected, abs=1.0e-4)
    observed.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert bool(torch.any(prediction.grad != 0.0))


def test_ssim_uses_largest_odd_window_for_small_images() -> None:
    image = torch.rand((4, 6, 3), generator=torch.Generator().manual_seed(7))
    same = structural_similarity(image, image, padding="same")
    valid = structural_similarity(image, image, padding="valid")

    assert same == pytest.approx(1.0, abs=2.0e-6)
    assert valid == pytest.approx(1.0, abs=2.0e-6)


@pytest.mark.parametrize(
    ("prediction", "target", "message"),
    [
        (torch.zeros((3, 4)), torch.zeros((3, 4)), "HWC RGB"),
        (torch.zeros((3, 4, 3)), torch.zeros((3, 5, 3)), "shapes"),
        (
            torch.zeros((3, 4, 3), dtype=torch.float64),
            torch.zeros((3, 4, 3), dtype=torch.float64),
            "float32",
        ),
        (torch.zeros((0, 4, 3)), torch.zeros((0, 4, 3)), "non-empty"),
    ],
)
def test_image_metrics_reject_inputs_outside_the_public_profile(
    prediction: torch.Tensor,
    target: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        structural_similarity(prediction, target)
    with pytest.raises(ContractError, match=message):
        psnr(prediction, target)


def test_ssim_rejects_unknown_padding() -> None:
    image = torch.zeros((12, 12, 3), dtype=torch.float32)
    with pytest.raises(ContractError, match="padding"):
        structural_similarity(image, image, padding="reflect")


def test_psnr_has_explicit_unit_range_and_exact_match_floor() -> None:
    black = torch.zeros((4, 5, 3), dtype=torch.float32)
    white = torch.ones_like(black)

    assert float(psnr(black, black)) == pytest.approx(120.0)
    assert float(psnr(black, white)) == pytest.approx(0.0)


def test_loss_catalog_matches_explicit_weighted_equations_and_gradients() -> None:
    model = _model()
    materialized = model.materialize(0.25, sh_degree=0)
    target = torch.full((16, 17, 3), 0.6, dtype=torch.float32)
    prediction = torch.sigmoid(model.sh0.mean(dim=(0, 1))).expand_as(target)
    color_regularization = 0.01 * model.center_times.square().mean()
    config = LossConfig(
        l1=0.8,
        ssim=0.2,
        opacity=1.0e-3,
        scale=1.0e-4,
        persistence=0.0,
        gate=1.0e-3,
    )

    terms = LossFunction(config)(
        model=model,
        materialized=materialized,
        prediction=prediction,
        target=target,
        color_correction_regularization=color_regularization,
    )

    assert tuple(terms) == LOSS_TERM_NAMES
    assert all(value.ndim == 0 and torch.isfinite(value) for value in terms.values())
    expected_l1 = 0.8 * torch.mean(torch.abs(prediction - target))
    assert float(terms["l1"].detach()) == pytest.approx(float(expected_l1.detach()))
    expected_opacity = 1.0e-3 * torch.mean(
        torch.sigmoid(model.opacity_logits) * materialized.temporal_activation.detach()
    )
    assert float(terms["opacity"].detach()) == pytest.approx(float(expected_opacity.detach()))
    expected_scale = 1.0e-4 * materialized.scales.mean()
    expected_gate = 1.0e-3 * (1.0 - model.gate()).mean()
    assert float(terms["scale"].detach()) == pytest.approx(float(expected_scale.detach()))
    assert float(terms["gate"].detach()) == pytest.approx(float(expected_gate.detach()))
    assert terms["lpips"] == 0.0
    assert terms["persistence"] == 0.0
    assert torch.equal(terms["color_correction"], color_regularization)

    torch.stack(tuple(terms.values())).sum().backward()
    for parameter in (
        model.sh0,
        model.opacity_logits,
        model.log_scales,
        model.persistence_logits,
        model.center_times,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_reference_kernel_is_cached_per_execution_layout() -> None:
    function = LossFunction(LossConfig(l1=0.8, ssim=0.2))
    model = _model()
    materialized = model.materialize(0.25, sh_degree=0)
    image = torch.rand((16, 17, 3), generator=torch.Generator().manual_seed(11))

    for _ in range(2):
        function(model=model, materialized=materialized, prediction=image, target=image)

    assert len(function._kernel_cache) == 1


def test_disabled_fused_term_does_not_import_a_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded() -> Any:
        raise AssertionError("disabled fused provider was loaded")

    monkeypatch.setattr(loss_module, "_load_fused_ssim", fail_if_loaded)
    function = LossFunction(LossConfig(l1=1.0, ssim=0.0, ssim_backend="fused"))

    assert function.provider_identity["ssim_backend"] == "disabled"


def test_disabled_lpips_term_does_not_import_a_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(_device: object) -> Any:
        raise AssertionError("disabled LPIPS provider was loaded")

    monkeypatch.setattr(loss_module, "_load_lpips", fail_if_loaded)
    function = LossFunction(LossConfig(l1=1.0, ssim=0.0, lpips=0.0))

    assert function.provider_identity["lpips_backend"] == "disabled"


def test_fused_backend_is_explicit_and_has_no_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], str]] = []

    def fake_fused(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        padding: str,
    ) -> torch.Tensor:
        calls.append((tuple(prediction.shape), tuple(target.shape), padding))
        assert prediction.is_contiguous() and target.is_contiguous()
        return prediction.new_tensor(0.75)

    monkeypatch.setattr(
        loss_module,
        "_load_fused_ssim",
        lambda: (
            fake_fused,
            {
                "backend": "fused",
                "distribution": "fused-ssim",
                "version": "1.0.0",
                "source_revision": "a" * 40,
            },
        ),
    )
    function = LossFunction(
        LossConfig(l1=0.0, ssim=1.0, ssim_backend="fused", opacity=0.0, scale=0.0),
        device="cuda",
    )
    model = _model()
    image = torch.zeros((12, 14, 3), dtype=torch.float32)
    terms = function(
        model=model,
        materialized=model.materialize(0.0, sh_degree=0),
        prediction=image,
        target=image,
    )

    assert terms["ssim"] == pytest.approx(0.25)
    assert calls == [((1, 3, 12, 14), (1, 3, 12, 14), "same")]
    assert function.provider_identity["ssim_distribution"] == "fused-ssim"


def test_fused_and_lpips_fail_closed_without_registered_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ContractError, match="explicit CUDA/HIP"):
        LossFunction(LossConfig(ssim_backend="fused"))

    def missing_provider() -> Any:
        raise ContractError("registered provider missing")

    monkeypatch.setattr(loss_module, "_load_fused_ssim", missing_provider)
    with pytest.raises(ContractError, match="registered provider missing"):
        LossFunction(LossConfig(ssim_backend="fused"), device="cuda")
    with pytest.raises(ContractError, match="explicit execution device"):
        LossFunction(LossConfig(l1=0.9, ssim=0.0, lpips=0.1))

    def missing_lpips(_device: object) -> Any:
        raise ContractError("registered LPIPS provider missing")

    monkeypatch.setattr(loss_module, "_load_lpips", missing_lpips)
    with pytest.raises(ContractError, match="registered LPIPS provider missing"):
        LossFunction(LossConfig(l1=0.9, ssim=0.0, lpips=0.1), device="cpu")


def test_lpips_reproduces_frozen_clamp_layout_weight_and_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    resets: list[None] = []

    class FakeLpips:
        def __call__(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            calls.append((prediction.detach().clone(), target.detach().clone()))
            return (prediction - target).square().mean().reshape(1)

        def reset(self) -> None:
            resets.append(None)

    identity = {
        "backend": "torchmetrics_alex",
        "distribution": "torchmetrics",
        "version": "1.9.0",
        "input_contract": loss_module.LPIPS_INPUT_CONTRACT,
    }
    monkeypatch.setattr(loss_module, "_load_lpips", lambda _device: (FakeLpips(), identity))
    function = LossFunction(
        LossConfig(
            l1=0.0,
            ssim=0.0,
            lpips=0.01,
            opacity=0.0,
            scale=0.0,
        ),
        device="cpu",
    )
    model = _model()
    prediction = torch.linspace(-0.25, 1.25, 4 * 5 * 3).reshape(4, 5, 3)
    prediction.requires_grad_(True)
    target = torch.linspace(1.2, -0.2, 4 * 5 * 3).reshape(4, 5, 3)

    terms = function(
        model=model,
        materialized=model.materialize(0.0, sh_degree=0),
        prediction=prediction,
        target=target,
    )

    expected_prediction = prediction.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)
    expected_target = target.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)
    assert len(calls) == 1
    assert torch.equal(calls[0][0], expected_prediction.detach())
    assert torch.equal(calls[0][1], expected_target)
    assert not calls[0][0].is_contiguous() and not calls[0][1].is_contiguous()
    assert resets == [None]
    assert float(terms["lpips"].detach()) == pytest.approx(
        0.01 * float((expected_prediction - expected_target).square().mean().detach())
    )
    assert function.provider_identity["lpips_distribution"] == "torchmetrics"
    assert function.provider_identity["lpips_input_contract"] == (
        "clamp_rgb_unit_range_then_nchw_view_with_normalize_false_v1"
    )
    terms["lpips"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert bool(torch.any(prediction.grad != 0.0))


def test_lpips_registry_pins_code_runtime_weights_and_no_download_policy() -> None:
    registry = json.loads(
        (ROOT / "src/p2g/registries/lpips_alex_v1.json").read_text(encoding="utf-8")
    )

    assert registry["schema_version"] == "p2g.lpips_alex_provider_registry.v1"
    assert registry["provider"]["distribution"] == "torchmetrics"
    assert registry["provider"]["distribution_version"] == "1.9.0"
    assert registry["runtime"]["torch"] == "2.10.0+rocm7.0"
    assert registry["runtime"]["torchvision"] == "0.25.0+rocm7.0"
    assert registry["provider"]["files"]["linear_weights"] == {
        "path": "torchmetrics/functional/image/lpips_models/alex.pth",
        "bytes": 6009,
        "sha256": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    }
    assert registry["external_weights"]["alexnet_features"]["sha256"] == (
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
    )
    assert registry["metric"] == {
        "net_type": "alex",
        "reduction": "mean",
        "normalize": False,
        "input_contract": "clamp_rgb_unit_range_then_nchw_view_with_normalize_false_v1",
    }
    assert registry["policy"] == {
        "automatic_download": False,
        "bundle_weights": False,
        "require_local_hash_match": True,
        "record_provider_identity": True,
    }


def test_persistence_penalty_cannot_silently_target_a_disabled_plane() -> None:
    model = _model(persistence=False)
    image = torch.zeros((12, 12, 3), dtype=torch.float32)
    function = LossFunction(LossConfig(gate=1.0e-3))

    with pytest.raises(ContractError, match="requires learned persistence"):
        function(
            model=model,
            materialized=model.materialize(0.0, sh_degree=0),
            prediction=image,
            target=image,
        )


def test_public_loss_source_has_no_unpinned_network_fetch_or_reference_adapter() -> None:
    source = (ROOT / "src/p2g/training/losses.py").read_text(encoding="utf-8").casefold()

    assert "lpips_alex_v1.json" in source
    assert "learnedperceptualimagepatchsimilarity" in source
    assert "load_state_dict_from_url(" not in source
    assert "urlopen(" not in source
    assert "requests." not in source
    assert "automatic download is disabled" in source
    assert ("free" + "time") not in source
    assert ("ft" + "gs") not in source
    assert "except exception" not in source
