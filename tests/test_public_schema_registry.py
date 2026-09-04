from __future__ import annotations

import pytest

from p2g.errors import ContractError
from p2g.schema import SCHEMAS, load_schema

EXPECTED_SCHEMAS = {
    "asset_bundle",
    "asset_verification",
    "asset_video_render",
    "camera_path",
    "camera_trajectory",
    "gaussian_initialization_receipt",
    "ingest_receipt",
    "initialization_appearance",
    "initialization_proposal",
    "mi300x_preflight",
    "mi300x_resource_window",
    "observation",
    "role_inventory",
    "sample_plan",
    "sealed_evaluation_receipt",
    "sealed_quality_gate",
    "source_bindings",
    "stage_quarantine",
    "tensor_cache",
}


def test_public_schema_registry_is_exact_and_every_resource_is_valid() -> None:
    assert set(SCHEMAS) == EXPECTED_SCHEMAS
    assert len(set(SCHEMAS.values())) == len(SCHEMAS)
    for name in sorted(SCHEMAS):
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert isinstance(schema.get("$id"), str)


@pytest.mark.parametrize("name", ["observation_v1", "ufm_weight", "raster_v1_contract_plan"])
def test_internal_and_legacy_schema_names_are_not_exposed(name: str) -> None:
    with pytest.raises(ContractError, match="unknown public schema"):
        load_schema(name)
