"""JSON-schema registry for the supported public Pixel4DGS artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from importlib import resources
from typing import Any, cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from p2g.errors import ContractError

SCHEMAS: dict[str, str] = {
    "asset_bundle": "asset_bundle.v1.schema.json",
    "asset_verification": "asset_verification.v1.schema.json",
    "asset_video_render": "asset_video_render.v1.schema.json",
    "camera_path": "camera_path.v1.schema.json",
    "camera_trajectory": "camera_trajectory.v1.schema.json",
    "gaussian_initialization_receipt": "gaussian_initialization_receipt.v1.schema.json",
    "ingest_receipt": "ingest_receipt.schema.json",
    "initialization_appearance": "initialization_appearance_manifest.schema.json",
    "initialization_proposal": "initialization_proposal_manifest.schema.json",
    "mi300x_preflight": "mi300x_preflight.schema.json",
    "mi300x_resource_window": "mi300x_resource_window.v1.schema.json",
    "observation": "observation_manifest.v2.schema.json",
    "role_inventory": "role_inventory.schema.json",
    "sample_plan": "sample_plan.schema.json",
    "sealed_evaluation_receipt": "sealed_evaluation_receipt.v1.schema.json",
    "sealed_quality_gate": "sealed_quality_gate.v1.schema.json",
    "source_bindings": "source_bindings.schema.json",
    "stage_quarantine": "stage_quarantine.v1.schema.json",
    "tensor_cache": "tensor_cache.v1.schema.json",
}


def load_schema(name: str) -> dict[str, Any]:
    """Load one named, package-owned public artifact schema."""

    try:
        filename = SCHEMAS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCHEMAS))
        raise ContractError(f"unknown public schema {name!r}; available: {available}") from exc
    resource = resources.files("p2g.schemas").joinpath(filename)
    try:
        payload: Any = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load public schema {name!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"public schema {name!r} must be a JSON object")
    schema = cast(dict[str, Any], payload)
    validator_class: Any = Draft202012Validator
    validator_class.check_schema(schema)
    return schema


def validate_payload(name: str, payload: Any) -> None:
    """Validate a decoded artifact and report stable, path-addressed errors."""

    validator = Draft202012Validator(load_schema(name), format_checker=FormatChecker())
    validator_api: Any = validator
    error_stream = cast(Iterable[ValidationError], validator_api.iter_errors(payload))
    errors = sorted(error_stream, key=lambda error: list(error.absolute_path))
    if not errors:
        return
    messages: list[str] = []
    for error in errors[:20]:
        location = "/".join(str(item) for item in error.absolute_path) or "<root>"
        messages.append(f"{location}: {error.message}")
    if len(errors) > 20:
        messages.append(f"... {len(errors) - 20} additional schema errors")
    raise ContractError(f"{name} schema validation failed: " + " | ".join(messages))
