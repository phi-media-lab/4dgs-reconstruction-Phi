"""Narrow, lazy command surface for the public Pixel4DGS pipeline."""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from p2g.canonical import canonical_json_bytes, sha256_file, sha256_json, write_new_json
from p2g.errors import ContractError, P2GError

app = typer.Typer(
    name="p2g",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Trainable, explicit pixel-to-4D-Gaussian pipeline for one MI300X.",
)
asset_app = typer.Typer(
    name="asset",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Inspect and verify portable, pickle-free 4D Gaussian assets.",
)
app.add_typer(asset_app, name="asset")
fixture_app = typer.Typer(
    name="fixture",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Generate bounded, deterministic multiview contract inputs.",
)
app.add_typer(fixture_app, name="fixture")
data_app = typer.Typer(
    name="data",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Convert supported local datasets into the canonical observation manifest.",
)
app.add_typer(data_app, name="data")
camera_path_app = typer.Typer(
    name="camera-path",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Validate asset-independent trajectories and bind them to exported assets.",
)
app.add_typer(camera_path_app, name="camera-path")


def _expected[T](operation: Callable[[], T]) -> T:
    try:
        return operation()
    except (P2GError, ImportError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _emit(payload: object, output: Path | None = None) -> None:
    if output is not None:
        _expected(lambda: write_new_json(output, payload))
    typer.echo(canonical_json_bytes(payload).decode("utf-8"), nl=False)


def _progress(message: str) -> None:
    typer.echo(message, err=True)


def _stopped_process_specs(values: list[str] | None) -> tuple[tuple[int, int], ...]:
    parsed: list[tuple[int, int]] = []
    for value in values or []:
        pid_text, separator, ticks_text = value.partition(":")
        if (
            not separator
            or not pid_text.isdecimal()
            or not ticks_text.isdecimal()
            or int(pid_text) <= 0
            or int(ticks_text) <= 0
        ):
            raise ContractError(
                "--allow-stopped-process must use PID:STARTTIME_TICKS with positive integers"
            )
        parsed.append((int(pid_text), int(ticks_text)))
    parsed.sort()
    if len({pid for pid, _ in parsed}) != len(parsed):
        raise ContractError("--allow-stopped-process contains a duplicate PID")
    return tuple(parsed)


@app.command("run")
def run_pipeline_command(
    plan: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Complete p2g.pipeline_plan.v3 TOML."),
    ],
    workspace: Annotated[
        Path,
        typer.Option(help="New pipeline workspace, or the same workspace when resuming."),
    ],
    stop_after: Annotated[
        str | None,
        typer.Option(help="Stop after one named stage; intended for bounded validation."),
    ] = None,
) -> None:
    """Run or resume the complete fixed-order pixel-to-4DGS pipeline."""

    def run() -> dict[str, Any]:
        from p2g.orchestrator import STAGE_ORDER, run_pipeline

        if stop_after is not None and stop_after not in STAGE_ORDER:
            choices = ", ".join(STAGE_ORDER)
            raise ContractError(f"--stop-after must be one of: {choices}")
        return run_pipeline(
            plan,
            workspace=workspace,
            stop_after=stop_after,
            progress=_progress,
        )

    _emit(_expected(run))


@app.command("status")
def pipeline_workspace_status(
    workspace: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Pipeline workspace to verify."),
    ],
) -> None:
    """Verify a pipeline workspace and report completed and pending stages."""

    def inspect() -> dict[str, Any]:
        from p2g.orchestrator import pipeline_status

        return pipeline_status(workspace)

    _emit(_expected(inspect))


@fixture_app.command("create")
def create_fixture(
    output: Annotated[Path, typer.Option(help="New synthetic fixture directory.")],
    camera_count: Annotated[int, typer.Option(min=2, max=8)] = 3,
    frame_count: Annotated[int, typer.Option(min=2, max=16)] = 3,
    width: Annotated[int, typer.Option(min=8, max=256)] = 32,
    height: Annotated[int, typer.Option(min=8, max=256)] = 24,
) -> None:
    """Create a path-free fixture for installation and contract smoke tests."""

    def build() -> dict[str, Any]:
        from p2g.synthetic_fixture import create_synthetic_multiview_fixture

        return create_synthetic_multiview_fixture(
            output,
            camera_count=camera_count,
            frame_count=frame_count,
            width=width,
            height=height,
        )

    _emit(_expected(build))


@data_app.command("import-charge")
def import_charge(
    task_root: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Local Charge task directory containing camera JSON and RGB folders.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New observation_manifest.v2 JSON; source pixels are not copied."),
    ],
    dataset_id: Annotated[str, typer.Option(help="Stable identifier for this local selection.")],
    source_repository: Annotated[
        str,
        typer.Option(help="HTTPS identity of the exact upstream Charge scene repository."),
    ],
    source_revision: Annotated[
        str,
        typer.Option(help="Lowercase 40-character upstream dataset revision."),
    ],
    sealed_camera_count: Annotated[
        int,
        typer.Option(
            min=0,
            help="Lexicographically last test cameras reserved for the sealed gate.",
        ),
    ],
    train_transforms: Annotated[
        Path,
        typer.Option(help="Train camera JSON relative to the task root."),
    ] = Path("transforms_train.json"),
    test_transforms: Annotated[
        Path,
        typer.Option(help="Test camera JSON relative to the task root."),
    ] = Path("transforms_test.json"),
    fps: Annotated[float, typer.Option(min=0.001, help="Published source frame rate.")] = 96.0,
) -> None:
    """Hash and convert one fixed-rig Charge v1.0 RGB task."""

    def build() -> dict[str, Any]:
        from p2g.charge import import_charge_manifest

        return import_charge_manifest(
            task_root,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            output=output,
            dataset_id=dataset_id,
            source_repository=source_repository,
            source_revision=source_revision,
            sealed_camera_count=sealed_camera_count,
            fps=fps,
            progress=_progress,
        )

    _emit(_expected(build))


@data_app.command("import-selfcap")
def import_selfcap(
    dataset_root: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help=(
                "Local SelfCap capture containing videos/*.mp4 and "
                "optimized/{intri.yml,extri.yml,sync.json}."
            ),
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New or exactly resumable RGB-materialization directory."),
    ],
    dataset_id: Annotated[str, typer.Option(help="Stable identifier for this selection.")],
    source_start_frame: Annotated[
        int,
        typer.Option(min=0, help="First common-time source frame."),
    ] = 200,
    frame_count: Annotated[
        int,
        typer.Option(min=2, help="Number of synchronized output frames."),
    ] = 60,
    fps: Annotated[
        float,
        typer.Option(min=0.001, help="Expected source video frame rate."),
    ] = 60.0,
    scale: Annotated[
        float,
        typer.Option(min=0.001, max=1.0, help="Scale after common-ROI cropping."),
    ] = 0.5,
    diagnostic_camera: Annotated[
        str,
        typer.Option(help="Development-view camera excluded from training."),
    ] = "0007",
    sealed_camera: Annotated[
        str,
        typer.Option(help="Final-gate camera excluded from training and routine evaluation."),
    ] = "0015",
    workers: Annotated[
        int,
        typer.Option(min=1, help="Maximum independent camera conversion processes."),
    ] = 1,
) -> None:
    """Materialize synchronized SelfCap videos as audited RGB8 PNG observations."""

    def build() -> dict[str, Any]:
        from p2g.selfcap import import_selfcap as import_implementation

        return import_implementation(
            dataset_root,
            output=output,
            dataset_id=dataset_id,
            source_start_frame=source_start_frame,
            frame_count=frame_count,
            fps=fps,
            scale=scale,
            diagnostic_camera=diagnostic_camera,
            sealed_camera=sealed_camera,
            workers=workers,
            progress=_progress,
        )

    _emit(_expected(build))


@app.command("doctor")
def doctor(
    output: Annotated[Path | None, typer.Option(help="New preflight receipt JSON.")] = None,
    gpu_index: Annotated[int, typer.Option(min=0, help="ROCm device index.")] = 0,
    maximum_gpu_use_percent: Annotated[
        float,
        typer.Option(min=0.0, max=100.0, help="Largest admitted observed GPU use."),
    ] = 100.0,
    maximum_vram_percent: Annotated[
        float,
        typer.Option(min=0.0, max=100.0, help="Largest admitted observed VRAM use."),
    ] = 80.0,
    admission_mode: Annotated[
        str,
        typer.Option(help="shared_quality or exclusive_performance."),
    ] = "shared_quality",
    allow_stopped_process: Annotated[
        list[str] | None,
        typer.Option(
            help="Exact stopped client PID:STARTTIME_TICKS; repeat as needed."
        ),
    ] = None,
    command_timeout_seconds: Annotated[int, typer.Option(min=1)] = 20,
) -> None:
    """Capture a non-reserving, three-plane MI300X occupancy observation."""

    def capture() -> dict[str, Any]:
        from p2g.gpu_preflight import (
            Mi300xPreflightConfig,
            StoppedProcessIdentity,
            capture_mi300x_preflight,
        )

        identities = tuple(
            StoppedProcessIdentity(pid=pid, starttime_ticks=ticks)
            for pid, ticks in _stopped_process_specs(allow_stopped_process)
        )
        config = Mi300xPreflightConfig(
            gpu_index=gpu_index,
            maximum_gpu_use_percent=maximum_gpu_use_percent,
            maximum_vram_percent=maximum_vram_percent,
            admission_mode=cast(Any, admission_mode),
            allowed_stopped_processes=identities,
            command_timeout_seconds=command_timeout_seconds,
        )
        return capture_mi300x_preflight(config=config)

    receipt = _expected(capture)
    _emit(receipt, output)
    if receipt["status"] != "PASS":
        raise typer.Exit(1)


@app.command("prepare")
def prepare(
    observation_manifest: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="observation_manifest.v2 JSON."),
    ],
    output: Annotated[Path, typer.Option(help="New p2g.tensor_cache.v1 directory.")],
    image_root: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Defaults to the manifest directory."),
    ] = None,
) -> None:
    """Convert admitted RGB images into the hash-bound public tensor cache."""

    def build() -> dict[str, Any]:
        from p2g.training.prepare import build_tensor_cache

        return build_tensor_cache(
            output,
            observation_manifest=observation_manifest,
            image_root=image_root,
            progress=_progress,
        )

    _emit(_expected(build))


@app.command("propose")
def propose(
    tensor_cache: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="p2g.tensor_cache.v1 root."),
    ],
    observation_manifest: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Exact observation_manifest.v2 file bound by the tensor cache.",
        ),
    ],
    roma_indoor_weight: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Registry-matching external weight."),
    ],
    dinov2_weight: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Registry-matching external weight."),
    ],
    environment_lock: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Hash-bound RoMa environment lock."),
    ],
    output: Annotated[Path, typer.Option(help="Append-only proposal-sequence root.")],
    frame_start: Annotated[int, typer.Option(min=0)] = 0,
    frame_stop_exclusive: Annotated[int, typer.Option(min=1)] = 60,
    points_per_frame: Annotated[int, typer.Option(min=1)] = 700_000,
    nearest_cameras: Annotated[int, typer.Option(min=1)] = 2,
    seed: Annotated[int, typer.Option(min=0)] = 0,
    world_bound: Annotated[float, typer.Option(min=0.0)] = 1_000.0,
) -> None:
    """Build or resume a hash-bound RoMa proposal sequence."""

    if frame_stop_exclusive <= frame_start:
        raise typer.BadParameter(
            "must be greater than --frame-start",
            param_hint="--frame-stop-exclusive",
        )
    if world_bound <= 0.0:
        raise typer.BadParameter("must be greater than zero", param_hint="--world-bound")

    def build() -> dict[str, Any]:
        from p2g.training.roma_point_sequence import build_roma_point_sequence

        with contextlib.redirect_stdout(sys.stderr):
            return build_roma_point_sequence(
                output,
                memmap_root=tensor_cache,
                observation_manifest=observation_manifest,
                roma_weight=roma_indoor_weight,
                dino_weight=dinov2_weight,
                environment_lock=environment_lock,
                frame_ids=tuple(range(frame_start, frame_stop_exclusive)),
                num_points_per_frame=points_per_frame,
                nearest_cameras=nearest_cameras,
                seed=seed,
                world_bound=world_bound,
            )

    _emit(_expected(build))


@app.command("initialize")
def initialize(
    proposal_sequence: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Complete proposal-sequence root."),
    ],
    tensor_cache: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Matching tensor-cache root."),
    ],
    output: Annotated[Path, typer.Option(help="New Gaussian initialization directory.")],
    num_gaussians: Annotated[int, typer.Option(min=1)] = 500_000,
    seed: Annotated[int, typer.Option(min=0)] = 0,
    velocity_neighbors: Annotated[int, typer.Option(min=1)] = 3,
    scale_multiplier: Annotated[float, typer.Option(min=0.0)] = 0.1,
    sampling_mode: Annotated[str, typer.Option()] = ("paired_multiview_consensus_rank_mixture"),
    sampling_voxel_size: Annotated[float, typer.Option(min=0.0)] = 0.02,
    sampling_evidence_fraction: Annotated[
        float,
        typer.Option(min=0.0, max=1.0),
    ] = 0.5,
    opacity: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    duration_seconds: Annotated[float, typer.Option(min=0.0)] = 0.1,
    duration_min_seconds: Annotated[float, typer.Option(min=0.0)] = 1.0 / 600.0,
    duration_max_seconds: Annotated[float, typer.Option(min=0.0)] = 1.0,
    time_offset_seconds: Annotated[float, typer.Option()] = 0.0,
) -> None:
    """Build the strict, fixed-capacity public Gaussian initialization."""

    if not 0.0 < opacity < 1.0:
        raise typer.BadParameter("must lie in (0, 1)", param_hint="--opacity")
    if not 0.0 < sampling_evidence_fraction <= 1.0:
        raise typer.BadParameter(
            "must lie in (0, 1]",
            param_hint="--sampling-evidence-fraction",
        )
    if not duration_min_seconds < duration_seconds < duration_max_seconds:
        raise typer.BadParameter(
            "must satisfy min < duration < max",
            param_hint="--duration-seconds",
        )
    if scale_multiplier <= 0.0 or sampling_voxel_size <= 0.0:
        raise typer.BadParameter("scale multiplier and sampling voxel size must be positive")

    def build() -> dict[str, Any]:
        from p2g.training.build_initialization import build_initialization

        return build_initialization(
            output,
            proposal_sequence=proposal_sequence,
            tensor_cache=tensor_cache,
            num_gaussians=num_gaussians,
            seed=seed,
            velocity_neighbors=velocity_neighbors,
            scale_multiplier=scale_multiplier,
            sampling_mode=cast(Any, sampling_mode),
            sampling_voxel_size=sampling_voxel_size,
            sampling_evidence_fraction=sampling_evidence_fraction,
            opacity=opacity,
            duration_seconds=duration_seconds,
            duration_min_seconds=duration_min_seconds,
            duration_max_seconds=duration_max_seconds,
            time_offset_seconds=time_offset_seconds,
            progress=_progress,
        )

    _emit(_expected(build))


@app.command("train")
def train(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Resolved p2g.resolved_run.v1 TOML."),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New run directory, or its existing directory when resuming."),
    ],
    resume_checkpoint: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Latest hash-closed checkpoint from --output.",
        ),
    ] = None,
) -> None:
    """Train the explicit 4D Gaussian system from a resolved run contract."""

    def run() -> dict[str, Any]:
        from p2g.training.config import RunConfig

        try:
            training_module = importlib.import_module("p2g.training.train")
        except ModuleNotFoundError as exc:
            if exc.name != "p2g.training.train":
                raise
            raise ContractError(
                "training implementation is unavailable in this installation"
            ) from exc
        run_training = cast(Any, training_module).run_training
        result = run_training(
            RunConfig.load(config),
            run_dir=output,
            resume_checkpoint=resume_checkpoint,
        )
        return {
            "completed_steps": result.completed_steps,
            "final_checkpoint": result.final_checkpoint,
            "model": result.model_path,
            "receipt": result.receipt_path,
            "run_dir": result.run_dir,
        }

    _emit(_expected(run))


@asset_app.command("export")
def export_asset(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Completed training run."),
    ],
    output: Annotated[Path, typer.Option(help="New portable AssetBundle directory.")],
    producer_git_revision: Annotated[
        str,
        typer.Option(help="Exact 40-character source revision that produced the run."),
    ],
    asset_license: Annotated[
        str,
        typer.Option(help="License assertion for the derived AssetBundle."),
    ],
    redistribution: Annotated[
        str,
        typer.Option(help="One of: allowed, restricted, review_required."),
    ],
    provenance_summary: Annotated[
        str,
        typer.Option(help="Non-empty human-readable provenance assertion."),
    ],
    world_unit: Annotated[str, typer.Option(help="Unit of calibrated world coordinates.")] = (
        "calibration_unit"
    ),
    calibration_scale: Annotated[
        float,
        typer.Option(min=0.0, help="Positive scale of one world unit."),
    ] = 1.0,
    default_sh_degree: Annotated[
        int | None,
        typer.Option(min=0, max=3, help="Defaults to the trained model maximum."),
    ] = None,
) -> None:
    """Convert one complete, hash-closed training run into an AssetBundle."""

    if redistribution not in {"allowed", "restricted", "review_required"}:
        raise typer.BadParameter(
            "must be allowed, restricted, or review_required",
            param_hint="--redistribution",
        )
    if calibration_scale <= 0.0:
        raise typer.BadParameter("must be greater than zero", param_hint="--calibration-scale")

    def build() -> dict[str, Any]:
        training_module = importlib.import_module("p2g.training.train")
        publication = cast(Any, training_module).AssetPublication(
            output=output,
            producer_git_revision=producer_git_revision,
            asset_license=asset_license,
            redistribution=redistribution,
            provenance_summary=provenance_summary,
            world_unit=world_unit,
            calibration_scale=calibration_scale,
            default_sh_degree=default_sh_degree,
        )
        destination = cast(Any, training_module).export_asset(run_dir, publication)
        from p2g.training.asset import asset_summary, load_asset_bundle

        return {
            "asset": destination,
            "summary": asset_summary(load_asset_bundle(destination)),
        }

    _emit(_expected(build))


@app.command("evaluate")
def evaluate(
    source: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Checkpoint directory or exported training run.",
        ),
    ],
    output: Annotated[Path | None, typer.Option(help="New evaluation directory.")] = None,
) -> None:
    """Evaluate a checkpoint or exported run on diagnostic observations."""

    def run() -> dict[str, Any]:
        from p2g.training.evaluate import evaluate_checkpoint, evaluate_exported_run

        if (source / "state.pt").is_file():
            return evaluate_checkpoint(source, output_dir=output)
        if (source / "model.safetensors").is_file():
            return evaluate_exported_run(source, output_dir=output)
        raise ContractError("source is neither a checkpoint nor an exported training run")

    _emit(_expected(run))


@app.command("evaluate-sealed")
def evaluate_sealed(
    run_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Complete final training run."),
    ],
    gate: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Preregistered sealed gate JSON."),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New write-once sealed result directory."),
    ],
) -> None:
    """Evaluate the sealed role once and publish a tamper-evident PASS/FAIL receipt."""

    def run() -> dict[str, Any]:
        from p2g.training.sealed_evaluate import evaluate_sealed_run

        return evaluate_sealed_run(run_dir, gate_file=gate, output_dir=output)

    receipt = _expected(run)
    _emit(receipt)
    if receipt["status"] != "PASS":
        raise typer.Exit(1)


@app.command("verify-sealed")
def verify_sealed(
    result_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Published sealed result directory."),
    ],
    run_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Bound final training run."),
    ],
    gate: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Bound preregistered gate JSON."),
    ],
    expected_receipt_id: Annotated[
        str,
        typer.Option(help="Receipt SHA-256 retained when evaluation was published."),
    ],
) -> None:
    """Verify a sealed receipt and every bound byte without rerendering."""

    def run() -> dict[str, Any]:
        from p2g.training.sealed_evaluate import verify_sealed_receipt

        return verify_sealed_receipt(
            result_dir,
            run_dir=run_dir,
            gate_file=gate,
            expected_receipt_id=expected_receipt_id,
        )

    _emit(_expected(run))


@app.command("render-video")
def render_video(
    asset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    camera_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Explicit camera_path.v1 JSON."),
    ],
    output: Annotated[Path, typer.Option(help="New H.264 MP4 path.")],
    receipt: Annotated[Path | None, typer.Option(help="New render receipt JSON.")] = None,
    device: Annotated[str, typer.Option(help="Torch render device.")] = "cuda",
    crf: Annotated[int, typer.Option(min=0, max=51)] = 18,
) -> None:
    """Render an AssetBundle using only an explicit, hash-bound camera path."""

    def render() -> dict[str, Any]:
        from p2g.training.asset_render import render_asset_video

        return render_asset_video(
            asset,
            camera_path_file=camera_path,
            output=output,
            receipt=receipt,
            device=device,
            crf=crf,
        )

    _emit(_expected(render))


@camera_path_app.command("bind")
def bind_camera_path(
    asset: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Verified AssetBundle directory."),
    ],
    trajectory: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Asset-independent p2g.camera_trajectory.v1 JSON.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New asset-bound p2g.camera_path.v1 JSON."),
    ],
) -> None:
    """Bind a reviewed trajectory to the identity and time interval of one asset."""

    def bind() -> dict[str, Any]:
        from p2g.training.asset_render import bind_camera_trajectory

        return bind_camera_trajectory(
            asset,
            trajectory_file=trajectory,
            output=output,
        )

    _emit(_expected(bind))


def _asset_summary(asset: Path) -> tuple[Any, dict[str, Any]]:
    from p2g.training.asset import asset_summary, load_asset_bundle

    bundle = load_asset_bundle(asset)
    return bundle, asset_summary(bundle)


@asset_app.command("inspect")
def inspect_asset(
    asset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Verify every asset byte and print a path-free semantic summary."""

    _, summary = _expected(lambda: _asset_summary(asset))
    _emit(summary)


def _inside(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _asset_verification_receipt(bundle: Any, summary: dict[str, Any]) -> dict[str, Any]:
    root = Path(bundle.root).resolve()
    receipt: dict[str, Any] = {
        "schema_version": "p2g.asset_verification.v1",
        "status": "PASS",
        "asset": {
            "schema_version": summary["schema_version"],
            "bundle_id": summary["bundle_id"],
            "gaussian_count": summary["gaussian_count"],
            "tensor_count": summary["tensor_count"],
            "model_sha256": summary["model_sha256"],
            "equation_version": summary["equation_version"],
            "redistribution": summary["rights"]["redistribution"],
        },
        "files": {
            "manifest.json": sha256_file(root / "manifest.json"),
            "asset.json": sha256_file(root / "asset.json"),
            "model.safetensors": sha256_file(root / "model.safetensors"),
        },
        "claim_boundary": (
            "Every declared AssetBundle byte and semantic field was accepted; no render, "
            "source-data entitlement, visual-quality, or performance claim was made."
        ),
    }
    if receipt["asset"]["model_sha256"] != receipt["files"]["model.safetensors"]:
        raise ContractError("asset summary and model file digest disagree")
    receipt["logical_sha256"] = sha256_json(receipt)
    from p2g.schema import validate_payload

    validate_payload("asset_verification", receipt)
    return receipt


@asset_app.command("verify")
def verify_asset(
    asset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(help="New verification receipt JSON.")],
) -> None:
    """Publish a path-free receipt for a structurally valid AssetBundle."""

    def verify() -> dict[str, Any]:
        bundle, summary = _asset_summary(asset)
        destination = output.expanduser().resolve()
        root = Path(bundle.root).resolve()
        if destination.suffix.casefold() != ".json":
            raise ContractError("asset verification output must use a .json filename")
        if _inside(destination, root):
            raise ContractError("asset verification receipt must be outside the AssetBundle")
        receipt = _asset_verification_receipt(bundle, summary)
        write_new_json(destination, receipt)
        return receipt

    _emit(_expected(verify))


def main() -> None:
    """Console-script entry point with stable handling for expected failures."""

    try:
        app()
    except (P2GError, ImportError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc


__all__ = ["app", "main"]
