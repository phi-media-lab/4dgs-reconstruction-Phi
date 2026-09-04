# Pixel4DGS

Pixel4DGS is a trainable, explicit pixel-to-4D-Gaussian pipeline designed for
one AMD Instinct MI300X (`gfx942`). Its target input is synchronized,
calibrated multi-camera RGB video; its target output is a portable
`AssetBundle` that can be reloaded, queried at continuous times, and rendered
from an explicit moving-camera path.

> **Open-source alpha:** repository-owned source, documentation, and the
> generated synthetic fixture are licensed under Apache-2.0; see
> [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). External dependencies, model
> weights, input data, trained assets, and rendered media retain separate
> terms and are not bundled. See
> [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Start here

- [Quickstart](docs/QUICKSTART.md): install the CPU review environment and run
  the bounded fixture-to-prepare smoke path.
- [Architecture](docs/ARCHITECTURE.md): understand the pixel-to-4DGS data flow,
  continuous-time representation, optimization, and runtime boundaries.
- [Documentation map](docs/README.md): find the detailed mechanism, MI300X,
  reproducibility, troubleshooting, and provenance guides.
- [Release process](docs/RELEASE_PROCESS.md): reproduce the no-publish archive
  check and distinguish source-release checks from demo and quality claims.
- [Changelog](CHANGELOG.md): see the initial alpha feature set.

## Scope

The v0 contract is deliberately narrow:

| Supported | Not claimed |
|---|---|
| Linux x86-64 | Windows or macOS |
| One MI300X / `gfx942` | CUDA, other AMD GPUs, or multi-GPU |
| CPython 3.12 | Other Python ABIs for GPU execution |
| PyTorch 2.10.0 + ROCm 7.0 | Arbitrary Torch/ROCm combinations |
| Synchronized, calibrated multi-view RGB | Monocular or uncalibrated video |
| Offline-undistorted pinhole observations | Rolling shutter or train-time distortion |
| Explicit linear motion and temporal activation | General video generation |

FreeTimeGS++ and 3DGS-MCMC are research references, not runtime dependencies.
Their source code is not included. Pixel4DGS implements population control
independently and tests it against explicit project-owned contracts.

## Intended pipeline

```text
calibrated multi-view videos
        -> observation manifest
        -> ProposalCollection
        -> GaussianInitialization
        -> MI300X 4DGS training
        -> AssetBundle
        -> evaluate / inspect / render camera path
```

The `p2g` command surface includes a `run`/`status` orchestrator for the six
reconstruction stages `prepare`, `propose`, `initialize`, `train`, `evaluate`,
and asset publication. It also exposes `doctor`, `camera-path bind`,
`render-video`, `evaluate-sealed`, `verify-sealed`, `asset export`, `asset
inspect`, `asset verify`, `fixture create`, and offline Charge and
SelfCap-style data adapters.
Command imports are lazy, so help, status, schema-level workflows, fixture
generation, and asset inspection do not initialize a GPU runtime. The
role-bound trainer and independently specified fixed-budget relocation
mechanism are present and CPU contract tested. The alpha source license is not
evidence of MI300X scene quality or performance.

Generate and prepare a tiny, third-party-payload-free contract fixture without
accessing a GPU:

```bash
p2g fixture create --output fixture
p2g prepare fixture/observation_manifest.json --output runs/smoke/scene
```

This proves input generation, schema/audit, and preparation connectivity only.
It is deliberately too small to establish matching, training, visual-quality,
or performance claims.

The Charge adapter hashes rather than copies the selected RGB files, converts
the empirically verified Blender camera-to-world convention into the public
OpenCV world-to-camera convention, and keeps official train/test cameras in
disjoint roles. The quickstart gives a fixed-revision `010_0050` Dense import
example without bundling it. See the
[quickstart](docs/QUICKSTART.md#2-import-a-local-charge-task)
for the exact offline command and the
[data contract](docs/DATA_CONTRACT.md#charge-v10-adapter) for the coordinate
proof and claim boundary.

The SelfCap adapter materializes synchronized videos into ordinary RGB8 PNGs
and a per-frame, per-camera observation manifest. It records source hashes,
fractional synchronization, undistortion, the common valid crop, resize and
quantization identities, and disjoint train/diagnostic/sealed roles. Source
media and generated pixels stay outside the source distribution; see the
[SelfCap adapter contract](docs/DATA_CONTRACT.md#selfcap-video-adapter).

For a real admitted scene, put all six reconstruction stages in one reviewed TOML plan and
use a dedicated workspace:

```bash
p2g run pipeline.toml --workspace runs/scene-a
p2g status runs/scene-a
```

The stage-scoped plan history, fixed stage order, MI300X admission windows,
workspace layout, quarantine, and exact resume semantics are documented in
[the runnable pipeline guide](docs/PIPELINE_ORCHESTRATION.md).

The staged [CPU workflow](.github/workflows/cpu-ci.yml) runs the complete CPU
suite with an explicit CPU-only Torch wheel and verifies the committed source
boundary. The separate
[release archive workflow](.github/workflows/release-check.yml) reproducibly
builds, compares, and smoke-installs the wheel and sdist without uploading or
publishing them. Development expectations and the alpha reporting boundary are
recorded in [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Implemented components

The repository contains:

- canonical JSON hashing and atomic artifact publication primitives;
- a deterministic calibrated multiview fixture generator with no downloaded
  media, weights, or learned output;
- an offline Charge v1.0 adapter that hash-binds local RGB inputs, reindexes
  source frames and preserves disjoint train, diagnostic, and sealed cameras;
- an offline SelfCap video adapter that performs hash-bound fractional-time
  sampling, undistortion, common-ROI cropping, RGB8 PNG materialization, and
  per-observation manifest generation;
- the v2 calibrated observation schema and semantic audit;
- an append-only, dataset-neutral `prepare` stage that converts admitted RGB8
  observations into the hash-bound public NumPy tensor cache;
- path-containment, image hash/header, camera geometry, synchronization, and
  train/diagnostic/sealed role-isolation checks;
- separate portable profiles, scene path inputs, and resolved run records with
  deterministic TOML serialization;
- an audited RGB8 loader plus a manifest-bound NumPy mmap cache whose camera,
  timestamp, dtype, shape, and file hashes are checked before training;
- an independently owned manifest-and-tensor-cache-to-point-proposal path that
  excludes diagnostic/sealed/free-view observations before camera pairing,
  records canonical role admission, exposes two-view geometry and rejection
  planes, and uses an optional hash-pinned RoMa provider that never downloads
  or bundles weights;
- a hash-bound proposal-to-Gaussian initializer with paired multi-view
  evidence sampling, explicit KNN motion/scale rules, physical duration
  calibration, and a strict alias-free Safetensors output;
- a train-only, hash-bound optimization orchestrator with sparse GPU-scalar
  diagnostics, checkpoint-aligned evaluation, checkpoint-exact resume, a
  completion receipt, and a separately retryable AssetBundle export;
- a digest-bound top-level runner that connects all six reconstruction stages,
  records preflight and continuously sampled KFD windows, reuses only
  stage-compatible work, and atomically quarantines invalid partial outputs;
- an independent fixed-budget relocation controller with explicit source
  utility, capacity, peak/far-time alpha conservation, projected-alpha-mass
  correction, stable slot lineage, and precise optimizer-state invalidation;
- a strict Safetensors-only Gaussian initialization boundary and an explicit,
  differentiable continuous-time Gaussian model;
- explicit L1, Gaussian-window SSIM, LPIPS, PSNR, and Gaussian regularizer
  equations with fail-closed pinned fused-SSIM and TorchMetrics AlexNet
  adapters;
- a fail-closed single-MI300X renderer adapter that pins the AMD gsplat wheel,
  native provider, call ABI, tensor layouts, and differentiable packed metadata;
- replayable MI300X admission with exact stopped-process identities, separate
  contention-tolerant shared-quality/strict exclusive-performance modes, and
  dynamic KFD-arrival detection;
- hash-closed local resume checkpoints using restricted tensor-state loading,
  kept explicitly separate from distributable AssetBundles;
- portable asset, asset-independent camera-trajectory, and bound camera-path schemas;
- project-owned analytic geometry tests;
- a hash-pinned MI300X renderer and fused-SSIM source-build recipe.

The exact raw-image restrictions, tensor-cache layout, photometric conversion,
role capabilities, and resumable sampler state are specified in
[the public data contract](docs/DATA_CONTRACT.md).
Portable asset inspection, hash-closed verification, and source-independent
moving-camera rendering are specified in
[the AssetBundle consumption guide](docs/ASSET_CONSUMPTION.md).
The initialization tensor catalog, temporal equations, physical invariants,
checkpoint boundary, and MI300X execution layout are specified in
[the public model contract](docs/MODEL_CONTRACT.md).
Loss equations, provider selection, and regularizer gradients are specified in
[the public loss contract](docs/LOSS_CONTRACT.md).
The exact raster inputs, fixed provider switches, output metadata, and MI300X
runtime identity are specified in
[the public renderer contract](docs/RENDERER_CONTRACT.md).
The correspondence-provider identity, external-weight policy, geometry,
provenance planes, and append-only point artifacts are specified in
[the public RoMa point-provider contract](docs/ROMA_POINT_PROVIDER_CONTRACT.md).
The fixed-capacity allocation, sampling equations, parameter construction,
duration calibration, and receipt boundary are specified in
[the public initialization-stage contract](docs/INITIALIZATION_STAGE.md).
Training input closure, optimization ordering, MI300X synchronization policy,
checkpoint/evaluation transactions, and AssetBundle handoff are specified in
[the public training contract](docs/TRAINING_CONTRACT.md).
Preregistered camera-role isolation, write-once PASS/FAIL evidence, and
externally anchored receipt verification are specified in
[the sealed quality evaluation guide](docs/SEALED_EVALUATION.md).
Population-control scheduling, source capacity, 4D split equations, optimizer
state mutation, and residual claim boundaries are specified in
[the fixed-budget relocation contract](docs/RELOCATION_CONTRACT.md).
The exact occupancy inputs, process-union rule, PID/start-time identity,
full-stage KFD guard, privacy boundary, verdict, and replay semantics are specified in
[the MI300X preflight contract](docs/MI300X_PREFLIGHT_CONTRACT.md).
The generated smoke input and its rights/claim limits are specified in
[the synthetic fixture guide](docs/SYNTHETIC_FIXTURE.md).

For CPU-side contract development, use CPython 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest \
  tests/test_canonical.py \
  tests/test_geometry.py \
  tests/test_public_schema_registry.py \
  tests/test_public_observation_audit.py \
  tests/test_public_import_boundary.py
```

PyTorch is intentionally not a generic PyPI dependency. The verified MI300X
build requires the ROCm-specific Torch wheel and native extensions described in
[the MI300X runtime guide](docs/MI300X_RUNTIME_BUILD.md). Installing an
arbitrary package named `torch` does not establish a supported GPU runtime.

## Correctness boundary

An observation is admitted only after both JSON-schema and semantic checks.
In particular:

- every image path must remain below the declared scene root after symlink
  resolution;
- optional byte verification checks SHA-256 and the encoded image header;
- intrinsics must use the declared pixel domain and each extrinsic must be a
  finite rigid world-to-camera transform;
- per-camera timestamps must increase and synchronized frame spreads must stay
  within the declared tolerance;
- identical image bytes cannot occur across optimization and evaluation roles.

`diagnostic` observations may be used for debugging. `sealed` observations may
only be used for the preregistered final evaluation; they must not influence
optimization, early stopping, hyperparameter choice, or checkpoint selection.

## Source and claim boundary

`tools/release/check_tracked_source.py` checks a clean committed tree for
credentials, symlinks, submodules, oversized files, and training/data artifact
types. Passing that check and the CPU suite establishes only the named source
and contract behavior; it does not establish fresh installation, full
training, MI300X performance, or scene-level quality.

The source package contains no real-scene pixels, model weights, checkpoints,
trained Gaussian assets, metrics, or videos. Those artifacts are not required
to use or redistribute the Apache-2.0 source. Any separate quality,
performance, or demo claim must identify the exact code, runtime, inputs, and
rights evidence that supports it.
