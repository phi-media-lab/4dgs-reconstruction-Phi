from __future__ import annotations

import json
import zlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from p2g.audit import audit_observation_manifest
from p2g.cli import app
from p2g.errors import ContractError, OutputExistsError
from p2g.synthetic_fixture import (
    _stored_zlib_stream,
    create_synthetic_multiview_fixture,
)
from p2g.training.prepare import build_tensor_cache


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_fixture_is_deterministic_path_free_and_preparable(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    receipt = create_synthetic_multiview_fixture(first)
    second_receipt = create_synthetic_multiview_fixture(second)

    assert receipt == second_receipt
    assert _files(first) == _files(second)
    assert receipt["configuration"] == {
        "camera_count": 3,
        "frame_count": 3,
        "height": 24,
        "train_frame_count": 2,
        "width": 32,
    }
    assert receipt["logical_sha256"] == (
        "a5f6eb17c6c1e9626b856541b9eae05cbe8c63e1e17d44da484a59718af6f2c5"
    )
    assert len(receipt["images"]) == 9
    assert receipt["rights"] == {
        "contains_third_party_payload": False,
        "license": "Apache-2.0",
        "publication_status": "source_fixture_redistributable",
    }
    manifest_path = first / "observation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["license"] == "Apache-2.0"
    assert manifest["source"]["license_status"] == "declared"
    assert {item["role"] for item in manifest["observations"]} == {
        "train",
        "diagnostic",
    }
    audit = audit_observation_manifest(manifest, base_dir=first, verify_files=True)
    assert audit.status == "PASS", audit.to_dict()

    cache = tmp_path / "cache"
    cache_receipt = build_tensor_cache(cache, observation_manifest=manifest_path)
    assert cache_receipt["camera_ids"] == ["cam000", "cam001", "cam002"]
    assert cache_receipt["frame_ids"] == [0, 1, 2]
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (first / "fixture.json", manifest_path, cache / "tensor_cache.json")
    )
    assert str(tmp_path) not in combined


@pytest.mark.parametrize("size", [0, 1, 65_535, 65_536, 131_071])
def test_stored_zlib_stream_round_trips_fixed_block_boundaries(size: int) -> None:
    payload = bytes(index % 251 for index in range(size))

    encoded = _stored_zlib_stream(payload)

    assert encoded[:2] == b"\x78\x01"
    assert zlib.decompress(encoded) == payload


def test_fixture_rejects_overwrite_and_unbounded_dimensions(tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    create_synthetic_multiview_fixture(output)
    before = _files(output)

    with pytest.raises(OutputExistsError, match="refusing to overwrite"):
        create_synthetic_multiview_fixture(output)
    assert _files(output) == before
    with pytest.raises(ContractError, match="bounded smoke-test profile"):
        create_synthetic_multiview_fixture(tmp_path / "too-large", width=257)
    assert not (tmp_path / "too-large").exists()


def test_fixture_cli_is_lazy_and_emits_the_receipt(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "fixture"
    result = runner.invoke(app, ["fixture", "create", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["schema_version"] == (
        "p2g.synthetic_multiview_fixture.v1"
    )
    assert (output / "fixture.json").is_file()

    repeated = runner.invoke(app, ["fixture", "create", "--output", str(output)])
    assert repeated.exit_code == 2
    assert "refusing to overwrite" in repeated.stderr
