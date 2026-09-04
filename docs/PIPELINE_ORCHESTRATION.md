# Runnable pipeline orchestration

`p2g run` is the thin, resumable entry point for the public Pixel4DGS main
path. It does not implement a second trainer or hide stage policy. It calls the
same six independently invocable reconstruction stages in one fixed order:

```text
prepare -> propose -> initialize -> train -> evaluate -> asset
```

The runner exists to make one real scene reproducible from reviewed plans, to
stop before an occupied MI300X is touched, and to resume without deleting prior
work. Stage algorithms remain in their named modules and retain their
standalone CLI commands.

## 1. Inputs that must already exist

The runner never downloads data, code, or weights. Before starting, the
operator supplies:

- a calibrated `p2g.observation_manifest.v2` and its RGB files;
- a `p2g.portable_profile.v1` training profile;
- the exact external RoMa indoor and DINOv2 weight files accepted by the
  registered provider;
- the environment lock used to establish the RoMa runtime identity;
- explicit AssetBundle provenance, license, and redistribution assertions.

Paths may be absolute or relative to the plan file. Home expansion, file URIs,
missing files, and leaf symbolic links are rejected. Exact plan bytes are
stored in append-only history. A completed stage is reusable only when its own
semantic request remains identical; changing a later stage does not invalidate
an unrelated expensive upstream result.

## 2. Plan format

The complete plan is TOML. This example shows every supported field and the
MI300X-oriented defaults; it is a template, not a quality claim:

```toml
schema_version = "p2g.pipeline_plan.v3"
source_git_revision = "0123456789abcdef0123456789abcdef01234567"
profile = "profiles/quality.toml"
observation_manifest = "scene/observation_manifest.json"
# Omit image_root when image paths are relative to the manifest directory.
image_root = "scene"
roma_indoor_weight = "weights/roma_indoor.pth"
dinov2_weight = "weights/dinov2_vitl14_pretrain.pth"
environment_lock = "environments/roma.lock"

[preflight]
gpu_index = 0
maximum_gpu_use_percent = 100.0
maximum_vram_percent = 60.0
admission_mode = "shared_quality"
allowed_stopped_processes = [
  # { pid = 12345, starttime_ticks = 987654321 },
]
command_timeout_seconds = 20

[proposal]
frame_start = 0
frame_stop_exclusive = 60
points_per_frame = 700000
nearest_cameras = 2
seed = 0
world_bound = 1000.0

[initialization]
num_gaussians = 500000
seed = 0
velocity_neighbors = 3
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
producer_git_revision = "0123456789abcdef0123456789abcdef01234567"
asset_license = "LicenseRef-project-approved-asset-license"
redistribution = "review_required"
provenance_summary = "Describe the admitted source capture and derived-output review."
world_unit = "calibration_unit"
calibration_scale = 1.0
# default_sh_degree = 3
```

`source_git_revision` must equal `asset.producer_git_revision`. The profile controls the model, losses, optimizer, relocation schedule,
renderer ABI, iteration count, checkpoint cadence, and evaluation cadence. The
plan controls the cross-stage sources and builder parameters. A value is not
implicitly taken from an unrelated experiment directory.

The exact filenames, byte extents, and SHA-256 values are listed in the RoMa
provider registry. In particular, the RoMa indoor checkpoint remains
`NOASSERTION` and review-required; the runner does not turn possession of that
external file into redistribution permission.

## 3. Commands

Start a new run or resume one compatible stage-scoped plan in the workspace:

```bash
p2g run pipeline.toml --workspace runs/scene-a
```

Inspect the workspace without starting a stage:

```bash
p2g status runs/scene-a
```

For a bounded CPU connectivity check, stop after preparation:

```bash
p2g run pipeline.toml --workspace runs/scene-a-smoke --stop-after prepare
```

`--stop-after` accepts exactly one of `prepare`, `propose`, `initialize`,
`train`, `evaluate`, or `asset`. Continuing later uses the same plan and
workspace without that option. Only one `p2g run` process may write a workspace
at a time.

## 4. What each stage does

| Stage | Main operation | Canonical terminal receipt | GPU preflight |
|---|---|---|---|
| `prepare` | Audit calibrated observations and create the NumPy tensor cache | `artifacts/tensor-cache/tensor_cache.json` | No |
| `propose` | Build train-role-only RoMa point proposals | `artifacts/proposals/collection.json` | Yes |
| `initialize` | Assemble the fixed-capacity Safetensors Gaussian state | `artifacts/initialization/initialization.json` | No |
| `train` | Resolve the profile and optimize the explicit 4DGS model | `artifacts/run/training.json` | Yes |
| `evaluate` | Render and score diagnostic-role observations | `artifacts/evaluation/evaluation.json` | Yes |
| `asset` | Export the completed run as a portable AssetBundle | `artifacts/asset/manifest.json` | No |

The runner captures a fresh occupancy observation immediately before every GPU
stage. A hard `BUSY` result is retained and the stage is not called. In
`shared_quality`, external clients are recorded as contention and the stage may
continue when the capacity limit is satisfied; no timing from that stage is
admissible. In `exclusive_performance`, an external client is hard `BUSY` and a
scheduler allocation is additionally required.

## 5. Workspace and resume model

The workspace is append-oriented:

```text
runs/scene-a/
  workspace.json
  plans/
    000001-<plan-sha256>.toml
  stages/
    00-prepare.json
    ...
    05-asset.json
  preflight/
    01-propose-000001.json
    03-train-000001.json
    04-evaluate-000001.json
  resource-window/
    01-propose-000001.json
    03-train-000001.json
    04-evaluate-000001.json
  artifacts/
    quarantine/
    tensor-cache/
    proposals/
    initialization/
    resolved-run.toml
    run/
    evaluation/
    asset/
  pipeline.json
```

`workspace.json` fixes the layout and stage-scoped policy; `plans/` preserves
every distinct plan used with the workspace. Each stage record binds:

- the stage parameters;
- SHA-256 identities of direct external inputs used by that stage;
- the recursive in-package Python source closure used by that stage;
- all preceding stage-record identities;
- the exact terminal receipt;
- the passing preflight and full-stage resource-window receipts for GPU stages.

The final `pipeline.json` closes all six stage records, terminal plan, per-stage
source revisions, and AssetBundle manifest identity. `SINGLE_REVISION` is
required for release qualification. Development reuse across revisions is
explicitly marked `MIXED_REVISION` and cannot be promoted as a clean RC run.
Records contain workspace-relative paths only; host paths from the plan are
represented by hashes.

On restart, completed records are skipped only after their semantic request is
recomputed and matched. Proposal generation can resume verified frame shards;
training can resume only from the latest hash-closed checkpoint. An incomplete
non-resumable output, a training run without a safe checkpoint, or output from
an invalid resource window is atomically moved to
`artifacts/quarantine/<stage-attempt>/payload`. Its canonical receipt inventories
every regular file by size, mode, and SHA-256. Nothing is silently deleted.

`p2g status` is intentionally fast. It verifies canonical orchestration JSON,
receipt-file hashes, the stage chain, and recorded admission files. It does not
reread every byte of large tensor, PLY, or checkpoint payloads. The next
stage's loader verifies the payload hashes and semantic constraints before
consuming those artifacts. Asset inspection and render adoption perform their
own complete declared-file checks.

## 6. Bind a trajectory and render after asset publication

A camera trajectory is geometry and timestamps, not a reconstruction input.
It therefore stays outside the six-stage plan. Before training, an operator may
prepare and review an asset-independent `p2g.camera_trajectory.v1` file. Only
after the AssetBundle exists can the exact asset identity and valid-time
interval be bound into a `p2g.camera_path.v1`:

```bash
p2g camera-path bind runs/scene-a/artifacts/asset \
  --trajectory paths/reviewed-trajectory.json \
  --output paths/scene-a.camera-path.json

p2g render-video runs/scene-a/artifacts/asset \
  --camera-path paths/scene-a.camera-path.json \
  --output previews/scene-a.mp4 \
  --receipt evidence/scene-a.render.json
```

This ordering removes a circular dependency: a camera path cannot truthfully
name the final `bundle_id` before training and asset publication have produced
it. Binding validates every camera matrix and timestamp against the finished
asset before writing a new path. Rendering remains independently invocable,
GPU-preflighted by the operator or scheduler, and does not alter or reopen the
completed reconstruction workspace.

## 7. Failure and evidence semantics

- A failed preflight writes an immutable attempt receipt but no stage record.
- A stage exception leaves native resumable state visible; a non-resumable
  retry first creates a closed quarantine transaction.
- A plan change is admitted only where every completed stage has the same
  scoped parameters, inputs, implementation closure, and upstream identities.
- A changed receipt, stage record, admission record, or stage order fails closed.
- A completed pipeline means the configured operations produced their declared
  receipts. It does not by itself establish perceptual quality, convergence,
  throughput, or legal permission to distribute the source data or output.

The evaluation directory is diagnostic evidence. Release-quality claims still
need a preregistered real scene, fresh MI300X execution, declared metrics and
thresholds, one write-once sealed evaluation, and review of a separately bound
moving-camera video.

## 8. Rights boundary

The Apache-2.0 synthetic fixture can exercise `fixture create`, `run
--stop-after prepare`, and status/resume behavior without third-party media.
It is still not a real-scene quality fixture.

Likewise, setting `asset_license` in a plan is an assertion, not an automatic
grant. The source-data license in the observation manifest, the project source
license, third-party notices, model-weight terms, and derived-output review are
separate release gates. Apache-2.0 covers repository-owned source and the
synthetic fixture; it does not supply any missing grant at the other layers.
