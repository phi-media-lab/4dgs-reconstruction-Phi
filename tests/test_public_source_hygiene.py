from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
VERIFIER = ROOT / "tools" / "release" / "check_tracked_source.py"


def _commit(root: Path, message: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Source verifier test",
            "-c",
            "user.email=verifier@example.invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=root,
        check=True,
    )


def _repository(tmp_path: Path, payload: bytes = b"source\n") -> Path:
    (tmp_path / "README.md").write_bytes(payload)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    _commit(tmp_path, "fixture")
    return tmp_path


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        HIP_VISIBLE_DEVICES="-1",
        ROCR_VISIBLE_DEVICES="-1",
        CUDA_VISIBLE_DEVICES="-1",
    )
    return subprocess.run(
        [sys.executable, str(VERIFIER), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_source_verifier_accepts_a_clean_source_commit(tmp_path: Path) -> None:
    result = _run(_repository(tmp_path))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "credential_pattern_matches": 0,
        "forbidden_artifact_count": 0,
        "maximum_tracked_file_bytes": 1048576,
        "result": "PASS",
        "scanned_bytes": 7,
        "schema_version": "p2g.source_hygiene.v1",
        "symlink_or_submodule_count": 0,
        "tracked_file_count": 1,
    }


def test_source_verifier_reports_a_label_without_echoing_secret(tmp_path: Path) -> None:
    shaped_secret = b"hf" + b"_" + b"ABCDEFGHIJKLMNOPQRSTUVWX"
    result = _run(_repository(tmp_path, shaped_secret + b"\n"))

    assert result.returncode == 2
    assert "README.md:huggingface_token" in result.stderr
    assert shaped_secret.decode("ascii") not in result.stderr


def test_source_verifier_rejects_uncommitted_changes(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "README.md").write_text("changed\n", encoding="utf-8")

    result = _run(root)

    assert result.returncode == 2
    assert "clean committed HEAD" in result.stderr


def test_source_verifier_rejects_staged_changes(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "README.md").write_text("changed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)

    result = _run(root)

    assert result.returncode == 2
    assert "clean committed HEAD" in result.stderr


def test_source_verifier_rejects_non_ignored_untracked_files(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "unexpected.txt").write_text("not committed\n", encoding="utf-8")

    result = _run(root)

    assert result.returncode == 2
    assert "clean committed HEAD" in result.stderr


def test_source_verifier_allows_ignored_local_files(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / ".git/info/exclude").write_text("local-cache/\n", encoding="utf-8")
    cache = root / "local-cache"
    cache.mkdir()
    (cache / "state.json").write_text("{}\n", encoding="utf-8")

    result = _run(root)

    assert result.returncode == 0, result.stderr


def test_source_verifier_rejects_training_artifacts(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "weights.safetensors").write_bytes(b"not really a model")
    subprocess.run(["git", "add", "weights.safetensors"], cwd=root, check=True)
    _commit(root, "add forbidden artifact")

    result = _run(root)

    assert result.returncode == 2
    assert "training/data artifact is forbidden" in result.stderr


def test_source_verifier_rejects_symlinks(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "link").symlink_to("README.md")
    subprocess.run(["git", "add", "link"], cwd=root, check=True)
    _commit(root, "add symlink")

    result = _run(root)

    assert result.returncode == 2
    assert "unsupported tracked Git mode" in result.stderr
