# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

"""Resumable publication of a complete first-party RoMa point sequence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from p2g.canonical import sha256_file, sha256_json, write_new_json
from p2g.errors import ContractError, OutputExistsError
from p2g.training.roma_point_provider import (
    ROMA_POINT_PROVIDER_SCHEMA,
    RomaIndoorPairSampler,
    build_roma_point_proposals,
    build_train_role_admission,
    load_observation_authority,
)

ROMA_POINT_SEQUENCE_SCHEMA = "p2g.roma_point_proposal_sequence.v1"


def _read_frame_receipt(frame_root: Path, *, frame_id: int) -> dict[str, Any]:
    receipt_path = frame_root / "receipt.json"
    try:
        receipt = cast(dict[str, Any], json.loads(receipt_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"incomplete RoMa frame proposal {frame_root}: {exc}") from exc
    ply = receipt.get("artifacts", {}).get("point_ply", {})
    provenance = receipt.get("artifacts", {}).get("provenance", {})
    ply_path = frame_root / str(ply.get("path", ""))
    provenance_path = frame_root / str(provenance.get("path", ""))
    if (
        receipt.get("schema") != ROMA_POINT_PROVIDER_SCHEMA
        or receipt.get("status") != "COMPLETE"
        or receipt.get("frame", {}).get("frame_id") != frame_id
        or not ply_path.is_file()
        or not provenance_path.is_file()
        or sha256_file(ply_path) != ply.get("sha256")
        or sha256_file(provenance_path) != provenance.get("sha256")
        or not isinstance(provenance.get("canonical_tensor_sha256"), str)
        or len(provenance["canonical_tensor_sha256"]) != 64
        or provenance.get("canonical_digest_schema")
        != "p2g.roma_point_provenance_canonical_digest.v1"
    ):
        raise ContractError(f"RoMa frame proposal failed replay verification: {frame_root}")
    return receipt


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_roma_point_sequence(
    output: Path,
    *,
    memmap_root: Path,
    observation_manifest: Path,
    roma_weight: Path,
    dino_weight: Path,
    environment_lock: Path,
    frame_ids: tuple[int, ...] = tuple(range(60)),
    num_points_per_frame: int = 700_000,
    nearest_cameras: int = 2,
    seed: int = 0,
    world_bound: float = 1_000.0,
) -> dict[str, Any]:
    """Build or resume frame shards, then atomically publish a PLY point root."""

    destination = output.expanduser().resolve()
    collection_path = destination / "collection.json"
    if collection_path.exists() or collection_path.is_symlink():
        raise OutputExistsError(f"refusing to overwrite complete RoMa sequence: {destination}")
    if (
        not frame_ids
        or len(set(frame_ids)) != len(frame_ids)
        or tuple(sorted(frame_ids)) != frame_ids
        or min(frame_ids) < 0
    ):
        raise ContractError("RoMa sequence frame inventory must be unique and ascending")
    authority = load_observation_authority(observation_manifest)
    if any(frame_id not in authority.frame_ids for frame_id in frame_ids):
        raise ContractError("RoMa sequence requests a frame outside the observation manifest")
    destination.mkdir(parents=True, exist_ok=True)
    frames_root = destination / "frames"
    frames_root.mkdir(exist_ok=True)
    points_root = destination / "points"
    if points_root.is_symlink() or (points_root.exists() and not points_root.is_dir()):
        raise ContractError("partial RoMa sequence has an invalid points-root entry")

    sampler = RomaIndoorPairSampler(
        roma_weight=roma_weight,
        dino_weight=dino_weight,
        environment_lock=environment_lock,
    )
    started = time.monotonic()
    frame_receipts: list[dict[str, Any]] = []
    for position, frame_id in enumerate(frame_ids):
        frame_root = frames_root / f"f{frame_id:06d}"
        if frame_root.exists() or frame_root.is_symlink():
            receipt = _read_frame_receipt(frame_root, frame_id=frame_id)
            _, expected_admission = build_train_role_admission(
                authority, frame_id=frame_id
            )
            if (
                sha256_json(receipt["provider"]) != sha256_json(sampler.identity)
                or receipt.get("frame", {}).get("role") != "train"
                or receipt.get("frame", {}).get("camera_ids")
                != expected_admission["admitted_camera_ids"]
                or receipt.get("source", {}).get("role_admission") != expected_admission
                or receipt["policy"]["num_points_per_frame_requested"] != num_points_per_frame
                or receipt["policy"]["nearest_cameras"] != nearest_cameras
                or receipt["policy"]["global_seed"] != seed
                or receipt["policy"]["world_coordinate_absolute_bound"] != world_bound
            ):
                raise ContractError(
                    f"existing RoMa frame uses a different frozen provider policy: {frame_root}"
                )
            action = "verified existing"
        else:
            receipt = build_roma_point_proposals(
                frame_root,
                memmap_root=memmap_root,
                observation_manifest=authority.path,
                frame_id=frame_id,
                roma_weight=roma_weight,
                dino_weight=dino_weight,
                environment_lock=environment_lock,
                num_points_per_frame=num_points_per_frame,
                nearest_cameras=nearest_cameras,
                seed=seed,
                world_bound=world_bound,
                sampler=sampler,
            )
            action = "built"
        frame_receipts.append(receipt)
        print(
            f"FRAME [{position + 1}/{len(frame_ids)}] {frame_id:06d} {action}: "
            f"{receipt['aggregate']['admitted_count']:,} proposals",
            flush=True,
        )

    frame_inventory: list[dict[str, Any]] = []
    for frame_id, receipt in zip(frame_ids, frame_receipts, strict=True):
        frame_root = frames_root / f"f{frame_id:06d}"
        frame_inventory.append(
            {
                "frame_id": frame_id,
                "frame_root": str(frame_root.relative_to(destination)),
                "point_ply": f"f{frame_id:06d}.ply",
                "point_ply_sha256": receipt["artifacts"]["point_ply"]["sha256"],
                "provenance_sha256": receipt["artifacts"]["provenance"]["sha256"],
                "provenance_canonical_tensor_sha256": receipt["artifacts"]["provenance"][
                    "canonical_tensor_sha256"
                ],
                "sampled_count": receipt["aggregate"]["sampled_count"],
                "admitted_count": receipt["aggregate"]["admitted_count"],
                "source_rgb_sha256": receipt["source"]["frame_payload_sha256"]["rgb"],
                "role_admission_sha256": receipt["source"]["role_admission"][
                    "logical_sha256"
                ],
            }
        )
    if points_root.exists():
        for row in frame_inventory:
            point_path = points_root / row["point_ply"]
            if not point_path.is_file() or sha256_file(point_path) != row["point_ply_sha256"]:
                raise ContractError("interrupted RoMa points root failed replay verification")
    else:
        stage = Path(tempfile.mkdtemp(prefix=".points.", dir=destination))
        try:
            for frame_id, receipt in zip(frame_ids, frame_receipts, strict=True):
                frame_root = frames_root / f"f{frame_id:06d}"
                source = frame_root / receipt["artifacts"]["point_ply"]["path"]
                os.link(source, stage / f"f{frame_id:06d}.ply")
            _fsync_directory(stage)
            os.rename(stage, points_root)
            _fsync_directory(destination)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    total_sampled = sum(int(row["sampled_count"]) for row in frame_inventory)
    total_admitted = sum(int(row["admitted_count"]) for row in frame_inventory)
    collection: dict[str, Any] = {
        "schema": ROMA_POINT_SEQUENCE_SCHEMA,
        "status": "COMPLETE",
        "frame_ids": list(frame_ids),
        "frame_count": len(frame_ids),
        "points_root": "points",
        "admitted_observation_role": "train",
        "observation_manifest_sha256": authority.sha256,
        "provider": sampler.identity,
        "policy": {
            "num_points_per_frame_requested": num_points_per_frame,
            "nearest_cameras": nearest_cameras,
            "global_seed": seed,
            "world_coordinate_absolute_bound": world_bound,
        },
        "aggregate": {
            "sampled_count": total_sampled,
            "admitted_count": total_admitted,
            "admitted_fraction": total_admitted / total_sampled,
        },
        "frames": frame_inventory,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "publication": "verified_frame_shards_plus_atomic_hardlinked_points_root_v1",
    }
    write_new_json(collection_path, collection)
    _fsync_directory(destination)
    return collection
