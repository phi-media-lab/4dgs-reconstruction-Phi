from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from p2g.canonical import canonical_json_bytes, sha256_file, write_new_json
from p2g.errors import ContractError
from p2g.orchestrator import (
    GPU_STAGES,
    PIPELINE_PLAN_SCHEMA,
    STAGE_ORDER,
    DefaultPipelineBackend,
    PipelinePaths,
    PipelinePlan,
    PreflightOptions,
    StageName,
    pipeline_status,
    run_pipeline,
)
from p2g.synthetic_fixture import create_synthetic_multiview_fixture
from p2g.training.config import PortableProfile


def _quoted(path: Path) -> str:
    return json.dumps(path.as_posix())


def _write_plan(
    root: Path,
    *,
    observation_manifest: Path | None = None,
) -> tuple[Path, dict[str, Path]]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    profile = inputs / "profile.toml"
    profile.write_bytes(PortableProfile().to_toml_bytes())
    manifest = (
        inputs / "observations.json" if observation_manifest is None else observation_manifest
    )
    if observation_manifest is None:
        manifest.write_text("{}\n", encoding="utf-8")
    roma = inputs / "roma-indoor.pth"
    dino = inputs / "dinov2.pth"
    environment = inputs / "roma-environment.lock"
    roma.write_bytes(b"test roma weight\n")
    dino.write_bytes(b"test dino weight\n")
    environment.write_bytes(b"test environment lock\n")
    plan = root / "pipeline.toml"
    plan.write_text(
        f'''schema_version = "p2g.pipeline_plan.v3"
source_git_revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
profile = {_quoted(profile)}
observation_manifest = {_quoted(manifest)}
roma_indoor_weight = {_quoted(roma)}
dinov2_weight = {_quoted(dino)}
environment_lock = {_quoted(environment)}

[preflight]
gpu_index = 0
maximum_gpu_use_percent = 5.0
maximum_vram_percent = 1.0
admission_mode = "shared_quality"
allowed_stopped_processes = []
command_timeout_seconds = 20

[proposal]
frame_start = 0
frame_stop_exclusive = 2
points_per_frame = 16
nearest_cameras = 2
seed = 7
world_bound = 10.0

[initialization]
num_gaussians = 8
seed = 11
velocity_neighbors = 2
scale_multiplier = 0.1
sampling_mode = "paired_multiview_consensus_rank_mixture"
sampling_voxel_size = 0.02
sampling_evidence_fraction = 0.5
opacity = 0.5
duration_seconds = 0.1
duration_min_seconds = 0.0016666666666666668
duration_max_seconds = 1.0
time_offset_seconds = 0.0

[asset]
producer_git_revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
asset_license = "LicenseRef-test-only"
redistribution = "review_required"
provenance_summary = "Synthetic orchestrator test; no publication grant."
world_unit = "calibration_unit"
calibration_scale = 1.0
''',
        encoding="utf-8",
    )
    return plan, {
        "dinov2_weight": dino,
        "environment_lock": environment,
        "manifest": manifest,
        "profile": profile,
        "roma_indoor_weight": roma,
    }


class FakeBackend:
    def __init__(self, *, preflight_statuses: tuple[str, ...] = ()) -> None:
        self.calls: list[StageName] = []
        self.preflight_calls = 0
        self._preflight_statuses = list(preflight_statuses)

    def preflight(self, options: PreflightOptions) -> dict[str, Any]:
        self.preflight_calls += 1
        status = self._preflight_statuses.pop(0) if self._preflight_statuses else "PASS"
        return {
            "schema_version": "test.mi300x_preflight.v1",
            "status": status,
            "gpu_index": options.gpu_index,
            "attempt": self.preflight_calls,
        }

    def execute(
        self,
        stage: StageName,
        *,
        plan: PipelinePlan,
        paths: PipelinePaths,
        progress: Callable[[str], None],
    ) -> Path:
        del plan
        self.calls.append(stage)
        progress(f"fake {stage}")
        receipt = paths.receipt(stage)
        if receipt.is_file() and not receipt.is_symlink():
            return receipt
        write_new_json(
            receipt,
            {
                "schema_version": f"test.{stage}.v1",
                "status": "COMPLETE",
                "stage": stage,
            },
        )
        return receipt

    def execute_guarded(
        self,
        stage: StageName,
        *,
        plan: PipelinePlan,
        paths: PipelinePaths,
        progress: Callable[[str], None],
        resource_window: Path,
    ) -> Path:
        receipt = self.execute(
            stage,
            plan=plan,
            paths=paths,
            progress=progress,
        )
        write_new_json(
            resource_window,
            {
                "schema_version": "test.mi300x_resource_window.v1",
                "operation_status": "RETURNED",
                "status": "PASS",
                "stage": stage,
            },
        )
        return receipt


def _json(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_full_pipeline_is_hash_bound_path_free_and_idempotent(tmp_path: Path) -> None:
    plan, _ = _write_plan(tmp_path)
    workspace = tmp_path / "workspace"
    backend = FakeBackend()
    progress: list[str] = []

    status = run_pipeline(plan, workspace=workspace, backend=backend, progress=progress.append)

    assert status["status"] == "COMPLETE"
    assert status["completed_stage_count"] == len(STAGE_ORDER)
    assert status["next_stage"] is None
    assert status["preflight_attempt_count"] == len(GPU_STAGES)
    assert status["resource_window_attempt_count"] == len(GPU_STAGES)
    assert backend.calls == list(STAGE_ORDER)
    assert backend.preflight_calls == len(GPU_STAGES)
    assert (workspace / "pipeline.json").is_file()
    assert _json(workspace / "pipeline.json")["revision_consistency"] == (
        "SINGLE_REVISION"
    )
    assert set(_json(workspace / "pipeline.json")["final_outputs"]) == {
        "asset_manifest_sha256"
    }
    for index, stage in enumerate(STAGE_ORDER):
        record = _json(workspace / "stages" / f"{index:02d}-{stage}.json")
        request = record["request"]
        assert request["upstream"] == {
            prior: _json(workspace / "stages" / f"{prior_index:02d}-{prior}.json")[
                "logical_sha256"
            ]
            for prior_index, prior in enumerate(STAGE_ORDER[:index])
        }
        assert ("preflight" in request) is (stage in GPU_STAGES)
        assert ("resource_window" in record) is (stage in GPU_STAGES)
    for state_file in workspace.rglob("*.json"):
        assert tmp_path.as_posix() not in state_file.read_text(encoding="utf-8")

    second = FakeBackend()
    repeated = run_pipeline(plan, workspace=workspace, backend=second)
    assert repeated == status
    assert second.calls == []
    assert second.preflight_calls == 0


def test_stop_after_then_resume_skips_verified_work(tmp_path: Path) -> None:
    plan, _ = _write_plan(tmp_path)
    workspace = tmp_path / "workspace"
    first = FakeBackend()

    partial = run_pipeline(plan, workspace=workspace, stop_after="prepare", backend=first)

    assert partial["status"] == "PARTIAL"
    assert partial["next_stage"] == "propose"
    assert first.calls == ["prepare"]
    assert first.preflight_calls == 0

    second = FakeBackend()
    progress: list[str] = []
    complete = run_pipeline(
        plan,
        workspace=workspace,
        backend=second,
        progress=progress.append,
    )

    assert complete["status"] == "COMPLETE"
    assert second.calls == list(STAGE_ORDER[1:])
    assert progress[0] == "SKIP prepare: verified completion record"


def test_workspace_reuses_unaffected_stage_after_a_scoped_plan_change(tmp_path: Path) -> None:
    plan, _ = _write_plan(tmp_path)
    workspace = tmp_path / "workspace"
    first = FakeBackend()
    run_pipeline(plan, workspace=workspace, stop_after="prepare", backend=first)
    plan.write_text(
        plan.read_text(encoding="utf-8").replace("points_per_frame = 16", "points_per_frame = 17"),
        encoding="utf-8",
    )

    second = FakeBackend()
    status = run_pipeline(plan, workspace=workspace, backend=second)

    assert status["status"] == "COMPLETE"
    assert status["plan_snapshot_count"] == 2
    assert first.calls == ["prepare"]
    assert second.calls == list(STAGE_ORDER[1:])


def test_workspace_rejects_a_plan_change_that_affects_a_completed_stage(
    tmp_path: Path,
) -> None:
    plan, inputs = _write_plan(tmp_path)
    workspace = tmp_path / "workspace"
    run_pipeline(plan, workspace=workspace, stop_after="prepare", backend=FakeBackend())
    inputs["manifest"].write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(ContractError, match="completed prepare stage is incompatible"):
        run_pipeline(plan, workspace=workspace, backend=FakeBackend())


def test_stage_scoped_reuse_is_explicitly_marked_mixed_revision(tmp_path: Path) -> None:
    plan, _ = _write_plan(tmp_path)
    workspace = tmp_path / "workspace"
    run_pipeline(plan, workspace=workspace, stop_after="propose", backend=FakeBackend())
    plan.write_text(
        plan.read_text(encoding="utf-8").replace("a" * 40, "b" * 40),
        encoding="utf-8",
    )

    status = run_pipeline(plan, workspace=workspace, backend=FakeBackend())
    completion = _json(workspace / "pipeline.json")

    assert status["status"] == "COMPLETE"
    assert status["plan_snapshot_count"] == 2
    assert completion["revision_consistency"] == "MIXED_REVISION"
    assert set(completion["stage_source_revisions"].values()) == {"a" * 40, "b" * 40}


@pytest.mark.parametrize("target", ["receipt", "record"])
def test_status_rejects_completed_stage_tampering(tmp_path: Path, target: str) -> None:
    plan, _ = _write_plan(tmp_path)
    workspace = tmp_path / "workspace"
    run_pipeline(plan, workspace=workspace, stop_after="prepare", backend=FakeBackend())
    if target == "receipt":
        path = workspace / "artifacts" / "tensor-cache" / "tensor_cache.json"
        payload = _json(path)
        payload["tampered"] = True
    else:
        path = workspace / "stages" / "00-prepare.json"
        payload = _json(path)
        payload["request"]["parameters"]["tampered"] = True
    path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ContractError, match=r"changed|identity"):
        pipeline_status(workspace)


def test_busy_preflight_is_retained_and_retry_gets_a_new_attempt(tmp_path: Path) -> None:
    plan, _ = _write_plan(tmp_path)
    workspace = tmp_path / "workspace"
    busy = FakeBackend(preflight_statuses=("BUSY",))

    with pytest.raises(ContractError, match="did not pass"):
        run_pipeline(plan, workspace=workspace, stop_after="propose", backend=busy)

    partial = pipeline_status(workspace)
    assert partial["completed_stage_count"] == 1
    assert partial["next_stage"] == "propose"
    assert partial["preflight_attempt_count"] == 1
    assert not (workspace / "stages" / "01-propose.json").exists()
    assert _json(workspace / "preflight" / "01-propose-000001.json")["status"] == "BUSY"

    passing = FakeBackend()
    resumed = run_pipeline(
        plan,
        workspace=workspace,
        stop_after="propose",
        backend=passing,
    )
    assert resumed["completed_stage_count"] == 2
    assert resumed["preflight_attempt_count"] == 2
    assert _json(workspace / "preflight" / "01-propose-000002.json")["status"] == "PASS"


def test_real_default_backend_connects_fixture_to_prepare_without_gpu(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    create_synthetic_multiview_fixture(fixture)
    plan, _ = _write_plan(
        tmp_path / "configuration",
        observation_manifest=fixture / "observation_manifest.json",
    )
    workspace = tmp_path / "workspace"

    status = run_pipeline(
        plan,
        workspace=workspace,
        stop_after="prepare",
        backend=DefaultPipelineBackend(),
    )

    cache = _json(workspace / "artifacts" / "tensor-cache" / "tensor_cache.json")
    assert status["completed_stage_count"] == 1
    assert status["preflight_attempt_count"] == 0
    assert cache["schema_version"] == "p2g.tensor_cache.v1"
    assert cache["observation_manifest_sha256"] == sha256_file(
        fixture / "observation_manifest.json"
    )
    assert cache["camera_ids"] == ["cam000", "cam001", "cam002"]
    assert cache["frame_ids"] == [0, 1, 2]
    assert set(cache["arrays"]) == {
        "intrinsic",
        "rgb",
        "timestamp_seconds",
        "world_to_camera",
    }


def test_incomplete_non_resumable_stage_is_quarantined_before_retry(tmp_path: Path) -> None:
    plan, _ = _write_plan(tmp_path)
    workspace = tmp_path / "workspace"
    backend = FakeBackend()

    def fail_before_write(
        stage: StageName,
        *,
        plan: PipelinePlan,
        paths: PipelinePaths,
        progress: Callable[[str], None],
    ) -> Path:
        del stage, plan, paths, progress
        raise ContractError("injected failure before output")

    backend.execute = fail_before_write  # type: ignore[method-assign]
    with pytest.raises(ContractError, match="injected failure"):
        run_pipeline(plan, workspace=workspace, stop_after="prepare", backend=backend)
    paths = PipelinePaths(workspace.resolve())
    paths.tensor_cache.mkdir()
    marker = paths.tensor_cache / "user-data"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(ContractError):
        run_pipeline(
            plan,
            workspace=workspace,
            stop_after="prepare",
            backend=DefaultPipelineBackend(),
        )
    assert not marker.exists()
    preserved = (
        workspace
        / "artifacts"
        / "quarantine"
        / "00-prepare-000001"
        / "payload"
        / "user-data"
    )
    assert preserved.read_text(encoding="utf-8") == "keep\n"
    quarantine = _json(preserved.parents[1] / "quarantine.json")
    assert quarantine["status"] == "QUARANTINED"
    assert quarantine["reason"] == "INCOMPLETE_NON_RESUMABLE_OUTPUT"


def test_resource_window_violation_quarantines_the_stage_attempt(tmp_path: Path) -> None:
    from p2g.gpu_resource_window import ResourceWindowViolation

    plan, _ = _write_plan(tmp_path)
    workspace = tmp_path / "workspace"
    backend = FakeBackend()
    run_pipeline(plan, workspace=workspace, stop_after="prepare", backend=backend)

    def contaminated(
        stage: StageName,
        *,
        plan: PipelinePlan,
        paths: PipelinePaths,
        progress: Callable[[str], None],
        resource_window: Path,
    ) -> Path:
        del plan, progress
        paths.proposals.mkdir()
        (paths.proposals / "partial").write_text("preserve", encoding="utf-8")
        write_new_json(
            resource_window,
            {
                "schema_version": "test.mi300x_resource_window.v1",
                "operation_status": "RETURNED",
                "status": "BUSY",
                "stage": stage,
            },
        )
        raise ResourceWindowViolation("injected foreign client")

    backend.execute_guarded = contaminated  # type: ignore[method-assign]
    with pytest.raises(ResourceWindowViolation, match="foreign client"):
        run_pipeline(plan, workspace=workspace, stop_after="propose", backend=backend)

    preserved = (
        workspace
        / "artifacts"
        / "quarantine"
        / "01-propose-000001"
        / "payload"
        / "partial"
    )
    assert preserved.read_text(encoding="utf-8") == "preserve"
    assert not (workspace / "artifacts" / "proposals").exists()


def test_plan_is_fail_closed_for_unknown_fields_and_input_symlinks(tmp_path: Path) -> None:
    plan, inputs = _write_plan(tmp_path)
    plan.write_text(plan.read_text(encoding="utf-8") + "\nunknown = true\n", encoding="utf-8")
    with pytest.raises(ContractError, match="unknown fields"):
        PipelinePlan.load(plan)

    plan, inputs = _write_plan(tmp_path / "symlink-case")
    roma = inputs["roma_indoor_weight"]
    target = roma.with_name("real-roma.pth")
    roma.rename(target)
    roma.symlink_to(target)
    with pytest.raises(ContractError, match="roma_indoor_weight must not be a symlink"):
        PipelinePlan.load(plan)


def test_plan_v3_ends_at_asset_and_rejects_the_old_render_table(tmp_path: Path) -> None:
    plan, _ = _write_plan(tmp_path)
    assert PIPELINE_PLAN_SCHEMA == "p2g.pipeline_plan.v3"
    PipelinePlan.load(plan)

    plan.write_text(
        plan.read_text(encoding="utf-8") + '\n[render]\ndevice = "cuda"\n',
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="pipeline plan contains unknown fields"):
        PipelinePlan.load(plan)


def test_plan_reports_malformed_asset_identity_as_a_contract_error(tmp_path: Path) -> None:
    plan, _ = _write_plan(tmp_path)
    plan.write_text(
        plan.read_text(encoding="utf-8").replace(
            'producer_git_revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
            "producer_git_revision = 17",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="producer_git_revision"):
        PipelinePlan.load(plan)
