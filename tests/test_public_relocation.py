# pyright: reportUnknownMemberType=false

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
import torch

from p2g.training.relocation import (
    MAX_SPLIT_MULTIPLICITY,
    split_projected_alpha_mass,
)

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("source_alpha", "multiplicity"),
    [(0.01, 2), (0.2, 5), (0.8, 3), (0.99, MAX_SPLIT_MULTIPLICITY)],
)
def test_split_matches_an_independent_scalar_compositing_oracle(
    source_alpha: float,
    multiplicity: int,
) -> None:
    scales = torch.tensor([[0.2, 0.5, 1.25]], dtype=torch.float64)
    piece_alpha, piece_scales = split_projected_alpha_mass(
        torch.tensor([source_alpha], dtype=torch.float64),
        scales,
        torch.tensor([multiplicity], dtype=torch.int64),
    )
    alpha = float(piece_alpha[0])
    center = 1.0 - (1.0 - alpha) ** multiplicity
    with localcontext() as context:
        context.prec = 100
        decimal_alpha = Decimal(str(alpha))
        projected_integral = float(
            sum(
                Decimal((-1) ** (order + 1))
                * Decimal(math.comb(multiplicity, order))
                * decimal_alpha**order
                / Decimal(order)
                for order in range(1, multiplicity + 1)
            )
        )
    scale_factor = float(piece_scales[0, 1] / scales[0, 1])

    assert center == pytest.approx(source_alpha, abs=2.0e-13)
    assert projected_integral * scale_factor**2 == pytest.approx(
        source_alpha,
        abs=2.0e-12,
    )


def test_public_relocation_and_its_core_tests_have_independent_vocabulary() -> None:
    combined = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8").casefold()
        for relative in (
            "src/p2g/training/relocation.py",
            "tests/test_training_core.py",
        )
    )
    forbidden = (
        "free" + "time",
        "ft" + "gs",
        "gsplat." + "relocation",
        "importlib",
        "sys.path",
        "compat" + "ibility",
    )
    assert not any(token in combined for token in forbidden)
    assert "center_alpha_and_projected_alpha_integral_v1" in combined
    assert "opacity_times_mean_pixel_position_gradient_v1" in combined


def test_relocation_document_exposes_the_4d_approximation_and_release_boundary() -> None:
    contract = (ROOT / "docs/RELOCATION_CONTRACT.md").read_text(encoding="utf-8")

    assert "far-time persistent" in contract
    assert "does not claim full temporal" in contract
    assert "equivalence" in contract
    assert "maximum interior alpha residual" in contract
    assert "fresh-source MI300X gates" in contract
