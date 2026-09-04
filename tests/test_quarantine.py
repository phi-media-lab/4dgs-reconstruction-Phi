from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2g.canonical import sha256_file
from p2g.errors import ContractError
from p2g.quarantine import quarantine_stage_directory
from p2g.schema import validate_payload


def test_quarantine_atomically_preserves_and_inventories_stage_bytes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "artifacts" / "evaluation"
    nested = source / "renders"
    nested.mkdir(parents=True)
    payload = nested / "000001.png"
    payload.write_bytes(b"partial-render")
    quarantine = workspace / "artifacts" / "quarantine"

    receipt_path = quarantine_stage_directory(
        source,
        workspace_root=workspace,
        quarantine_root=quarantine,
        ordinal=4,
        stage="evaluate",
        reason="INCOMPLETE_NON_RESUMABLE_OUTPUT",
    )

    assert not source.exists()
    assert receipt_path == quarantine / "04-evaluate-000001" / "quarantine.json"
    preserved = receipt_path.parent / "payload" / "renders" / "000001.png"
    assert preserved.read_bytes() == b"partial-render"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_payload("stage_quarantine", receipt)
    assert receipt["files"] == [
        {
            "path": "renders/000001.png",
            "mode": "0644",
            "bytes": len(b"partial-render"),
            "sha256": sha256_file(preserved),
        }
    ]
    assert receipt["summary"]["total_bytes"] == len(b"partial-render")


def test_quarantine_sequences_attempts_per_stage_without_colliding(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    quarantine = workspace / "artifacts" / "quarantine"
    for stage, ordinal in (("prepare", 0), ("evaluate", 4), ("evaluate", 4)):
        source = workspace / "artifacts" / stage
        source.mkdir(parents=True)
        (source / "partial").write_text(stage, encoding="utf-8")
        receipt = quarantine_stage_directory(
            source,
            workspace_root=workspace,
            quarantine_root=quarantine,
            ordinal=ordinal,
            stage=stage,
            reason="INCOMPLETE_NON_RESUMABLE_OUTPUT",
        )
        assert receipt.is_file()

    assert sorted(path.name for path in quarantine.iterdir()) == [
        "00-prepare-000001",
        "04-evaluate-000001",
        "04-evaluate-000002",
    ]


def test_quarantine_fails_closed_on_symlink_and_interrupted_transaction(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "artifacts" / "asset"
    source.mkdir(parents=True)
    (source / "escape").symlink_to(tmp_path / "outside")

    with pytest.raises(ContractError, match="symlink"):
        quarantine_stage_directory(
            source,
            workspace_root=workspace,
            quarantine_root=workspace / "artifacts" / "quarantine",
            ordinal=5,
            stage="asset",
            reason="INCOMPLETE_NON_RESUMABLE_OUTPUT",
        )

    (source / "escape").unlink()
    pending = workspace / "artifacts" / "quarantine" / ".pending-old"
    pending.mkdir(parents=True)
    with pytest.raises(ContractError, match="interrupted pending"):
        quarantine_stage_directory(
            source,
            workspace_root=workspace,
            quarantine_root=workspace / "artifacts" / "quarantine",
            ordinal=5,
            stage="asset",
            reason="INCOMPLETE_NON_RESUMABLE_OUTPUT",
        )
