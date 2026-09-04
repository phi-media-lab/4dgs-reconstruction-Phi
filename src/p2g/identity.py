from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from p2g.canonical import canonical_json_bytes, sha256_bytes, sha256_file


def source_bindings_logical_sha256(*, dataset_id: str, bindings: list[dict[str, Any]]) -> str:
    """Hash source content independently from its verification location and time."""

    identity_payload = {
        "schema_version": "p2g.source_identity.v1",
        "dataset_id": dataset_id,
        "bindings": [
            {
                "binding_id": item["binding_id"],
                "kind": item["kind"],
                "size_bytes": item["size_bytes"],
                "content_sha256": item["expected_sha256"] or item["actual_sha256"],
            }
            for item in sorted(bindings, key=lambda item: str(item["binding_id"]))
        ],
    }
    return sha256_bytes(canonical_json_bytes(identity_payload))


def _command(name: str, *arguments: str) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        return {"name": name, "path": None, "returncode": None, "first_line": None}
    try:
        result = subprocess.run(
            [path, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        combined = (result.stdout + "\n" + result.stderr).strip()
        return {
            "name": name,
            "path": path,
            "returncode": result.returncode,
            "first_line": combined.splitlines()[0] if combined else "",
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "name": name,
            "path": path,
            "returncode": None,
            "first_line": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _distribution_versions() -> dict[str, str | None]:
    names = [
        "pixel4dgs",
        "torch",
        "torchvision",
        "numpy",
        "pyarrow",
        "safetensors",
        "zarr",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_identity(repo: Path | None) -> dict[str, Any]:
    if repo is None:
        return {"root": None, "head": None, "dirty": None}
    try:
        requested_root = repo.resolve()
        git_prefix = ["git", "-c", f"safe.directory={requested_root}"]
        root = subprocess.run(
            [*git_prefix, "-C", str(requested_root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        resolved_root = Path(root).resolve()
        git_prefix = ["git", "-c", f"safe.directory={resolved_root}"]
        head = subprocess.run(
            [*git_prefix, "-C", str(resolved_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            [
                *git_prefix,
                "-C",
                str(resolved_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        diff = subprocess.run(
            [*git_prefix, "-C", str(resolved_root), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        staged = subprocess.run(
            [*git_prefix, "-C", str(resolved_root), "diff", "--cached", "--binary", "HEAD"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        untracked_output = subprocess.run(
            [
                *git_prefix,
                "-C",
                str(resolved_root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        untracked: list[dict[str, Any]] = []
        for encoded_path in untracked_output.split(b"\0"):
            if not encoded_path:
                continue
            relative = encoded_path.decode("utf-8", errors="strict")
            path = resolved_root / relative
            untracked.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
            )
        worktree_payload = {
            "head": head,
            "diff_sha256": sha256_bytes(diff),
            "staged_diff_sha256": sha256_bytes(staged),
            "untracked": sorted(untracked, key=lambda item: item["path"]),
        }
        return {
            "root": str(resolved_root),
            "head": head,
            "dirty": bool(status),
            "status": status.splitlines(),
            "worktree": worktree_payload,
            "worktree_sha256": sha256_bytes(canonical_json_bytes(worktree_payload)),
        }
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {
            "root": str(repo),
            "head": None,
            "dirty": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_runtime_identity(
    *,
    expect_arch: str | None = None,
    expect_wavefront: int = 64,
    expect_python: str = "3.12.3",
    expect_torch: str = "2.10.0+rocm7.0",
    expect_hip: str = "7.0.51831",
    expect_device_substring: str = "MI300X",
    strict: bool = False,
    repo: Path | None = None,
    require_clean_git: bool = False,
    identity_files: list[Path] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any, required: bool = True) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "required": required,
                "detail": detail,
            }
        )

    actual_python = platform.python_version()
    check(
        "expected_python",
        actual_python == expect_python,
        {"expected": expect_python, "actual": actual_python},
        required=strict,
    )

    torch_info: dict[str, Any] = {"imported": False}
    selected_device: dict[str, Any] | None = None
    try:
        import torch

        torch_info = {
            "imported": True,
            "version": torch.__version__,
            "git_version": getattr(torch.version, "git_version", None),
            "hip_version": getattr(torch.version, "hip", None),
            "module_file": getattr(torch, "__file__", None),
            "accelerator_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        check(
            "expected_torch",
            torch.__version__ == expect_torch,
            {"expected": expect_torch, "actual": torch.__version__},
            required=strict,
        )
        check(
            "expected_hip",
            torch.version.hip == expect_hip,
            {"expected": expect_hip, "actual": torch.version.hip},
            required=strict,
        )
        if torch.cuda.is_available():
            # Torch's ROCm device-property return type is not exposed in its
            # type stubs. Isolate that uncertainty at this inspection boundary.
            cuda_api: Any = torch.cuda
            properties: Any = cuda_api.get_device_properties(0)
            arch = getattr(properties, "gcnArchName", None)
            if arch is None:
                arch = getattr(properties, "gcn_arch_name", None)
            selected_device = {
                "index": 0,
                "name": torch.cuda.get_device_name(0),
                "arch": arch,
                "wavefront": getattr(properties, "warp_size", None),
                "total_memory_bytes": getattr(properties, "total_memory", None),
            }
            check(
                "expected_device",
                expect_device_substring in selected_device["name"],
                {"expected_substring": expect_device_substring, "actual": selected_device["name"]},
                required=strict,
            )
            check(
                "single_visible_device",
                torch_info["device_count"] == 1,
                {"expected": 1, "actual": torch_info["device_count"]},
                required=strict,
            )
            if expect_arch is not None:
                check(
                    "expected_arch",
                    isinstance(arch, str) and arch.startswith(expect_arch),
                    {"expected": expect_arch, "actual": arch},
                    required=strict,
                )
            check(
                "expected_wavefront",
                selected_device["wavefront"] == expect_wavefront,
                {"expected": expect_wavefront, "actual": selected_device["wavefront"]},
                required=strict,
            )
        else:
            check(
                "accelerator_available",
                False,
                "torch.cuda.is_available() is false",
                required=strict,
            )
    except Exception as exc:
        torch_info = {"imported": False, "error": f"{type(exc).__name__}: {exc}"}
        check("torch_import", False, torch_info["error"], required=strict)

    command_info = {
        "hipcc": _command("hipcc", "--version"),
        "rocminfo": _command("rocminfo"),
        "amd_smi": _command("amd-smi", "version"),
        "cmake": _command("cmake", "--version"),
        "ninja": _command("ninja", "--version"),
    }
    for name in ("hipcc", "rocminfo", "amd_smi"):
        command = command_info[name]
        check(
            f"{name}_available",
            command["path"] is not None and command["returncode"] == 0,
            command,
            required=strict,
        )

    git_info = _git_identity(repo)
    check(
        "git_head",
        bool(git_info.get("head")),
        {"head": git_info.get("head"), "error": git_info.get("error")},
        required=strict,
    )
    check(
        "git_clean",
        git_info.get("dirty") is False,
        {"dirty": git_info.get("dirty"), "worktree_sha256": git_info.get("worktree_sha256")},
        required=require_clean_git,
    )

    bound_files: list[dict[str, Any]] = []
    for requested in identity_files or []:
        path = requested.resolve()
        exists = path.is_file()
        entry: dict[str, Any] = {
            "requested": str(requested),
            "path": str(path),
            "exists": exists,
            "sha256": sha256_file(path) if exists else None,
        }
        bound_files.append(entry)
        check(
            f"identity_file:{requested}",
            exists,
            entry,
            required=strict,
        )

    capabilities_data = importlib.resources.files("p2g").joinpath("capabilities.json").read_bytes()
    capabilities = json.loads(capabilities_data)
    capabilities_identity = {
        "schema_version": capabilities.get("schema_version"),
        "sha256": sha256_bytes(capabilities_data),
        "target": capabilities.get("target"),
    }

    status = (
        "FAIL" if any(item["required"] and item["status"] == "FAIL" for item in checks) else "PASS"
    )
    python_path = Path(sys.executable)
    relevant_environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "HSA_OVERRIDE_GFX_VERSION",
            "PYTORCH_HIP_ALLOC_CONF",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
        )
    }
    return {
        "schema_version": "p2g.runtime_identity.v1",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "strict": strict,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": str(python_path),
            "python_executable_sha256": sha256_file(python_path) if python_path.is_file() else None,
        },
        "environment": relevant_environment,
        "distributions": _distribution_versions(),
        "torch": torch_info,
        "selected_device": selected_device,
        "commands": command_info,
        "git": git_info,
        "capabilities": capabilities_identity,
        "identity_files": bound_files,
        "checks": checks,
        "claim_boundary": (
            "Runtime identity only. It does not prove algorithm correctness, scientific validity, "
            "resource exclusivity, or performance."
        ),
    }
