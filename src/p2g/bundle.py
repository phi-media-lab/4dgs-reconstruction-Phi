from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path
from typing import Any, cast

from p2g.audit import AuditReport
from p2g.canonical import read_json, sha256_file, sha256_json
from p2g.errors import ContractError
from p2g.identity import source_bindings_logical_sha256
from p2g.schema import validate_payload


def _safe_relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or "\\" in value:
        raise ContractError(f"checksum path must be relative POSIX syntax: {value!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ContractError(f"checksum path escapes bundle: {value!r}")
    return resolved


def _checksum_file_failures(checksum_path: Path) -> list[dict[str, Any]]:
    root = checksum_path.parent.resolve()
    failures: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [{"path": str(checksum_path), "reason": str(exc)}]
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("  ", 1)
        if len(fields) != 2:
            failures.append({"line": line_number, "reason": "malformed checksum line"})
            continue
        digest, relative = fields
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            failures.append({"line": line_number, "reason": "invalid SHA-256"})
            continue
        if relative in expected_paths:
            failures.append({"line": line_number, "reason": "duplicate path", "path": relative})
            continue
        expected_paths.add(relative)
        try:
            path = _safe_relative(root, relative)
        except ContractError as exc:
            failures.append({"line": line_number, "reason": str(exc)})
            continue
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != digest:
            failures.append(
                {
                    "path": relative,
                    "reason": "sha256_mismatch",
                    "expected": digest,
                    "actual": actual,
                }
            )

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual_paths != expected_paths:
        failures.append(
            {
                "reason": "checksum_inventory_mismatch",
                "unlisted": sorted(actual_paths - expected_paths),
                "listed_but_absent": sorted(expected_paths - actual_paths),
            }
        )
    return failures


def _verify_ingest_semantics(root: Path, report: AuditReport) -> None:
    try:
        receipt = cast(dict[str, Any], read_json(root / "receipt.json"))
        validate_payload("ingest_receipt", receipt)
    except (OSError, ContractError, TypeError, ValueError) as exc:
        report.add("ingest_receipt_schema", False, detail=str(exc))
        return
    report.add("ingest_receipt_schema", True, detail=receipt["schema_version"])

    try:
        runtime_identity = cast(dict[str, Any], read_json(root / "runtime_identity.json"))
        runtime_hash = sha256_file(root / "runtime_identity.json")
        runtime_git = cast(dict[str, Any], runtime_identity["git"])
        runtime_ok = (
            runtime_hash == receipt["runtime_identity_sha256"]
            and runtime_identity["status"] == receipt["runtime_identity_status"]
            and runtime_git.get("head") == receipt["git_head"]
            and runtime_git.get("worktree_sha256") == receipt["git_worktree_sha256"]
            and runtime_git.get("dirty") == receipt["git_dirty"]
        )
        runtime_detail: Any = {
            "computed_sha256": runtime_hash,
            "receipt_sha256": receipt["runtime_identity_sha256"],
            "status": runtime_identity["status"],
            "git_head": runtime_git.get("head"),
            "git_dirty": runtime_git.get("dirty"),
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        runtime_ok = False
        runtime_detail = str(exc)
    report.add("runtime_and_code_identity", runtime_ok, detail=runtime_detail)

    try:
        manifest = cast(dict[str, Any], read_json(root / "observation_manifest.json"))
        validate_payload("observation", manifest)
        manifest_hash = sha256_json(manifest)
        manifest_ok = manifest_hash == receipt["observation_manifest_logical_sha256"]
    except (OSError, ContractError, TypeError, ValueError) as exc:
        report.add("observation_manifest_identity", False, detail=str(exc))
        return
    report.add(
        "observation_manifest_identity",
        manifest_ok,
        detail={
            "expected": receipt["observation_manifest_logical_sha256"],
            "actual": manifest_hash,
        },
    )

    try:
        role_inventory = cast(dict[str, Any], read_json(root / "role_inventory.json"))
        validate_payload("role_inventory", role_inventory)
        unsigned_roles = dict(role_inventory)
        declared_roles_hash = cast(str, unsigned_roles.pop("logical_sha256"))
        computed_roles_hash = sha256_json(unsigned_roles)
        roles_ok = (
            computed_roles_hash == declared_roles_hash
            and computed_roles_hash == receipt["role_inventory_logical_sha256"]
            and role_inventory["dataset_id"] == manifest["dataset_id"]
        )
    except (OSError, ContractError, KeyError, TypeError, ValueError) as exc:
        report.add("role_inventory_identity", False, detail=str(exc))
        return
    report.add(
        "role_inventory_identity",
        roles_ok,
        detail={
            "computed": computed_roles_hash,
            "declared": declared_roles_hash,
            "receipt": receipt["role_inventory_logical_sha256"],
        },
    )

    try:
        source = cast(dict[str, Any], read_json(root / "source_bindings.json"))
        validate_payload("source_bindings", source)
        source_hash = source_bindings_logical_sha256(
            dataset_id=cast(str, source["dataset_id"]),
            bindings=cast(list[dict[str, Any]], source["bindings"]),
        )
        source_ok = (
            source_hash == source["logical_sha256"]
            and source_hash == receipt["source_bindings_logical_sha256"]
            and source["verification_complete"] == receipt["source_verification_complete"]
        )
    except (OSError, ContractError, TypeError, ValueError) as exc:
        report.add("source_bindings_identity", False, detail=str(exc))
    else:
        report.add(
            "source_bindings_identity",
            source_ok,
            detail={
                "computed": source_hash,
                "declared": source["logical_sha256"],
                "receipt": receipt["source_bindings_logical_sha256"],
            },
        )

    try:
        sample_plan = cast(dict[str, Any], read_json(root / "sample_plan/sample_plan.json"))
        validate_payload("sample_plan", sample_plan)
        unsigned_plan = dict(sample_plan)
        declared_plan_hash = cast(str, unsigned_plan.pop("logical_sha256"))
        computed_plan_hash = sha256_json(unsigned_plan)
        plan_ok = (
            computed_plan_hash == declared_plan_hash
            and computed_plan_hash == receipt["sample_plan_logical_sha256"]
            and sample_plan["observation_manifest_sha256"] == manifest_hash
        )
    except (OSError, ContractError, KeyError, TypeError, ValueError) as exc:
        report.add("sample_plan_identity", False, detail=str(exc))
        return
    report.add(
        "sample_plan_identity",
        plan_ok,
        detail={
            "computed": computed_plan_hash,
            "declared": declared_plan_hash,
            "receipt": receipt["sample_plan_logical_sha256"],
        },
    )

    observations = cast(list[dict[str, Any]], manifest["observations"])
    observation_by_id = {item["observation_id"]: item for item in observations}
    role_counts = Counter(item["role"] for item in observations)
    role_cameras: dict[str, set[str]] = {}
    for item in observations:
        role_cameras.setdefault(item["role"], set()).add(item["camera_id"])
    inventory_roles = cast(dict[str, dict[str, Any]], role_inventory["roles"])
    inventory_ok = all(
        declared["observation_count"] == role_counts[role]
        and declared["camera_count"] == len(role_cameras.get(role, set()))
        and declared["cameras"] == sorted(role_cameras.get(role, set()))
        for role, declared in inventory_roles.items()
    )
    sample_entries = cast(list[dict[str, Any]], sample_plan["entries"])
    sample_leaks = [
        entry["sample_id"]
        for entry in sample_entries
        if entry["observation_id"] not in observation_by_id
        or observation_by_id[entry["observation_id"]]["role"] != "train"
    ]
    report.add(
        "role_inventory_and_sample_isolation",
        dict(sorted(role_counts.items())) == receipt["role_counts"]
        and inventory_ok
        and not sample_leaks,
        detail={
            "manifest_role_counts": dict(sorted(role_counts.items())),
            "receipt_role_counts": receipt["role_counts"],
            "inventory_consistent": inventory_ok,
            "non_train_or_unknown_samples": sample_leaks,
        },
    )

    try:
        pa_parquet: Any = importlib.import_module("pyarrow.parquet")
        table: Any = pa_parquet.read_table(
            root / "observations.parquet", columns=["observation_id", "role"]
        )
        parquet_ids = cast(list[str], table.column("observation_id").to_pylist())
        parquet_roles = cast(list[str], table.column("role").to_pylist())
        parquet_metadata = cast(dict[bytes, bytes] | None, table.schema.metadata) or {}
        parquet_ok = (
            table.num_rows == len(observations)
            and parquet_ids == [item["observation_id"] for item in observations]
            and parquet_roles == [item["role"] for item in observations]
            and parquet_metadata.get(b"p2g.schema_version") == b"p2g.observations.parquet.v1"
            and parquet_metadata.get(b"p2g.observation_manifest_sha256") == manifest_hash.encode()
        )
        parquet_detail: Any = {
            "rows": table.num_rows,
            "manifest_observations": len(observations),
            "schema_version": (
                parquet_metadata.get(b"p2g.schema_version", b"").decode(errors="replace")
            ),
        }
    except Exception as exc:
        parquet_ok = False
        parquet_detail = str(exc)
    report.add("observation_parquet_identity", parquet_ok, detail=parquet_detail)

    try:
        pa_parquet = importlib.import_module("pyarrow.parquet")
        sample_table: Any = pa_parquet.read_table(
            root / "sample_plan/sample_plan.parquet",
            columns=["sample_id", "observation_id"],
        )
        sample_metadata = cast(dict[bytes, bytes] | None, sample_table.schema.metadata) or {}
        sample_parquet_ok = (
            sample_table.num_rows == len(sample_entries)
            and cast(list[str], sample_table.column("sample_id").to_pylist())
            == [entry["sample_id"] for entry in sample_entries]
            and cast(list[str], sample_table.column("observation_id").to_pylist())
            == [entry["observation_id"] for entry in sample_entries]
            and sample_metadata.get(b"p2g.schema_version") == b"p2g.sample_plan.parquet.v1"
            and sample_metadata.get(b"p2g.logical_sha256") == computed_plan_hash.encode()
            and sample_metadata.get(b"p2g.observation_manifest_sha256") == manifest_hash.encode()
        )
        sample_parquet_detail: Any = {
            "rows": sample_table.num_rows,
            "plan_entries": len(sample_entries),
        }
    except Exception as exc:
        sample_parquet_ok = False
        sample_parquet_detail = str(exc)
    report.add(
        "sample_plan_parquet_identity",
        sample_parquet_ok,
        detail=sample_parquet_detail,
    )

    try:
        source_audit = cast(dict[str, Any], read_json(root / "source_audit.json"))
        observation_audit = cast(dict[str, Any], read_json(root / "observation_audit.json"))
        derived_pass = (
            source_audit["status"] == observation_audit["status"] == "PASS"
            and receipt["runtime_identity_status"] == "PASS"
            and (
                not receipt["strict_runtime_requested"]
                or (
                    receipt["git_dirty"] is False
                    and receipt["source_verification_complete"] is True
                    and receipt["raw_video_hashes_verified"] is True
                )
            )
        )
        audit_ok = (
            source_audit["status"] == receipt["source_audit_status"]
            and observation_audit["status"] == receipt["observation_audit_status"]
            and receipt["status"] == ("PASS" if derived_pass else "FAIL")
        )
        audit_detail: Any = {
            "receipt": receipt["status"],
            "source": source_audit["status"],
            "observation": observation_audit["status"],
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        audit_ok = False
        audit_detail = str(exc)
    report.add("audit_receipt_consistency", audit_ok, detail=audit_detail)
    report.add(
        "ingest_gate_verdict",
        receipt["status"] == "PASS",
        detail={
            "status": receipt["status"],
            "strict_runtime_requested": receipt["strict_runtime_requested"],
        },
    )


def verify_bundle(root: Path) -> AuditReport:
    root = root.resolve()
    report = AuditReport(subject=str(root))
    if not root.is_dir():
        report.add("bundle_directory", False, detail=f"not a directory: {root}")
        return report
    checksum_files = sorted(root.rglob("SHA256SUMS"))
    checksum_failures = [
        {"checksum_file": str(path.relative_to(root)), "failures": failures}
        for path in checksum_files
        if (failures := _checksum_file_failures(path))
    ]
    report.add(
        "checksum_tree",
        bool(checksum_files) and not checksum_failures,
        detail={
            "checksum_files": [str(path.relative_to(root)) for path in checksum_files],
            "failures": checksum_failures,
        },
    )

    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        try:
            receipt = cast(dict[str, Any], read_json(receipt_path))
            schema_version = receipt.get("schema_version")
        except (OSError, TypeError, ValueError) as exc:
            report.add("recognized_bundle_semantics", False, detail=str(exc))
            return report
        if schema_version == "p2g.ingest_receipt.v1":
            report.add("recognized_bundle_semantics", True, detail=schema_version)
            _verify_ingest_semantics(root, report)
        else:
            report.add(
                "recognized_bundle_semantics",
                False,
                detail=f"unsupported receipt schema: {schema_version}",
            )
    else:
        report.add(
            "recognized_bundle_semantics",
            False,
            detail="transport hashes only; receipt.json is absent",
            required=False,
        )
    return report
