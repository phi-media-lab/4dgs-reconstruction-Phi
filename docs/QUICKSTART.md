# Quickstart

Repository-owned source and the generated fixture are available under
Apache-2.0. External weights, datasets, and generated real-scene artifacts are
not covered by that source license; see
[`LICENSE_AND_PROVENANCE.md`](LICENSE_AND_PROVENANCE.md).

## 1. CPU installation and smoke check

Use Linux x86-64 and CPython 3.12. Install an explicit CPU-only Torch wheel so
the development checks cannot accidentally select a CUDA or ROCm runtime:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev]'
```

Verify the reviewed source boundary and run the complete CPU suite with every
GPU visibility variable disabled:

```bash
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/python tools/release/check_tracked_source.py --root .
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/ruff check .
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/pyright
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/python -m pytest -q
```

Create and prepare the bounded synthetic fixture:

```bash
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/p2g fixture create --output fixture
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/p2g prepare fixture/observation_manifest.json \
  --output runs/smoke/tensor-cache
```

Expected terminal files are `fixture/fixture.json` and
`runs/smoke/tensor-cache/tensor_cache.json`. This path proves installation,
input audit, deterministic generation, and preparation. It does not run
matching or training and cannot support a visual-quality claim.

## 2. Import a local Charge task

The Charge umbrella release provides synchronized, calibrated train and test
cameras and labels the dataset CC BY 4.0. A scene shard has no standalone license file or
dataset card, so retain the umbrella and scene identities rather than inferring
a new per-shard grant. The adapter never downloads or copies pixels. Obtain the
example input from the
[official downloader](https://huggingface.co/charge-benchmark/Charge/commit/6c0255d5a4c3e87d334f79d737c846295187fbdd)
and keep it outside this source tree.

The example uses the complete full-length `010_0050` Dense RGB selection: 25
train cameras, 16 test cameras, and 381 common frames.
Download only its camera JSON and RGB payload from the fixed scene revision.
`huggingface_hub` is a data-acquisition tool, not a Pixel4DGS runtime
dependency:

```bash
python3.12 -m venv /data/hf-acquire-1.5.0
/data/hf-acquire-1.5.0/bin/python -m pip install 'huggingface_hub==1.5.0'
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  /data/hf-acquire-1.5.0/bin/hf download \
  charge-benchmark/Charge-010_0050-Dense \
  --repo-type dataset \
  --revision 322af4681d1bb5a196157caf36b5d1442cdf7317 \
  --include 'Charge_v1_0/010_0050/Dense/transforms_*.json' \
  --include 'Charge_v1_0/010_0050/Dense/*/frame_????.png' \
  --local-dir /data/charge-010-0050 \
  --max-workers 1
```

Keep an adjacent provenance record with the fixed umbrella and scene URLs and
revisions, the [project](https://charge-benchmark.github.io/), current
[paper](https://arxiv.org/abs/2512.13639), full
[Blender Charge credits](https://studio.blender.org/projects/charge/pages/credits/),
and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) URLs, plus the
acquisition date, selected task/modalities, and any changes. The importer
records the scene identity and input hashes, but it cannot reconstruct missing
attribution context later.

Then import the fixed selection without network access:

```bash
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/p2g data import-charge \
  /data/charge-010-0050/Charge_v1_0/010_0050/Dense \
  --train-transforms transforms_train.json \
  --test-transforms transforms_test.json \
  --dataset-id charge_010_0050_dense \
  --source-repository \
  https://huggingface.co/datasets/charge-benchmark/Charge-010_0050-Dense \
  --source-revision 322af4681d1bb5a196157caf36b5d1442cdf7317 \
  --sealed-camera-count 8 \
  --output charge-010-0050-dense.json
```

The command verifies every local RGB header and SHA-256, requires one static
fixed-rig calibration and common contiguous frame set, maps source frame IDs to
zero-based canonical IDs, assigns the lexicographically last requested test
cameras to `sealed`, and writes only the canonical manifest.

Stop here if the goal is only input identity, provenance, or importer
validation. Preparing this complete selection is a separate large CPU and I/O
operation: the RGB array payload alone is 82,346,913,792 bytes and the four
array payloads total 82,348,600,860 bytes (about 76.7 GiB). The actual `.npy`
files are slightly larger because they include headers, and the workspace has
additional metadata and filesystem overhead. Reserve at least 100 GB of free
space. An interrupted standalone prepare leaves an incomplete output that is
not a valid cache. Under `p2g run`, retry first moves it into a hash-inventoried
workspace quarantine.

When that cost is intentional, prepare the manifest with the original task
directory as the explicit image root:

```bash
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/p2g prepare charge-010-0050-dense.json \
  --image-root /data/charge-010-0050/Charge_v1_0/010_0050/Dense \
  --output runs/charge-010-0050/tensor-cache
```

Do not commit the downloaded data or treat a successful import as permission
to publish a trained derivative. Any published demo still needs content and
rights review, attribution to the Charge dataset and Blender movie creators, a
change notice, and an explicit license for the resulting asset and video.

## 3. Import an authorized SelfCap-style capture

The source distribution does not contain SelfCap media. Obtain and use a
capture only under its upstream terms, and keep both source and generated RGB
outside this repository. Install the exact CPU conversion dependencies in a
separate environment or add the `selfcap` extra to the review environment:

```bash
.venv/bin/python -m pip install -e '.[selfcap,dev]'
```

For a capture with `videos/*.mp4` and
`optimized/{intri.yml,extri.yml,sync.json}`, materialize a synchronized 60-frame
selection as follows:

```bash
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/p2g data import-selfcap /data/selfcap-sequence \
  --output /data/materialized/selfcap-sequence-f200-259 \
  --dataset-id selfcap_sequence_f200_259 \
  --source-start-frame 200 \
  --frame-count 60 \
  --fps 60 \
  --scale 0.5 \
  --diagnostic-camera 0007 \
  --sealed-camera 0015
```

The terminal `import.json` receipt binds the request, source inventory,
conversion runtime, camera receipts, and final `observation_manifest.json`.
Each observation names one ordinary RGB8 PNG and its SHA-256. A `PASS` proves
conversion and manifest integrity, not reconstruction quality or permission to
redistribute source or derived content.

## 4. Before a real MI300X run

The admitted GPU path is narrower than the CPU development path. Confirm all
of the following before starting it:

- one visible AMD Instinct MI300X (`gfx942`);
- CPython 3.12, PyTorch `2.10.0+rocm7.0`, and HIP `7.0.51831`;
- the pinned AMD gsplat/GLM runtime built as documented in the
  [MI300X runtime guide](MI300X_RUNTIME_BUILD.md);
- synchronized, calibrated, offline-undistorted multi-camera RGB observations;
- separately obtained RoMa and DINOv2 weights whose hashes match the registry;
- an explicit camera path and reviewed source/derived-asset rights assertions;
- enough MI300X HBM headroom for a quality run; a dedicated scheduler
  allocation is required only for performance claims.

`p2g doctor` observes occupancy but does not reserve the accelerator. Select
the claim class explicitly; a shared-quality observation cannot support timing:

```bash
p2g doctor --admission-mode shared_quality --output preflight.json
```

Shared-quality mode records other GPU clients as contention and continues while
the configured VRAM-capacity limit passes. Use a plan-specific limit that leaves
enough headroom for the stage. High utilization can slow the run but is not, by
itself, a quality failure.

When a pre-existing client is intentionally signal-stopped, bind both values
from `/proc/<pid>/stat`, never the PID alone:

```bash
p2g doctor --admission-mode shared_quality \
  --allow-stopped-process 12345:987654321 \
  --output preflight.json
```

Then create the complete TOML described in
[pipeline orchestration](PIPELINE_ORCHESTRATION.md) and run:

```bash
p2g run pipeline.toml --workspace runs/scene-a
p2g status runs/scene-a
```

For an RC, invoke the exact clean checkout through `tools/run_from_source.py`
with its full expected revision; this removes a stale Pixel4DGS editable finder
and fails if the imported package is outside that checkout. The runner resumes
only stage-compatible completed work and continuously monitors KFD during GPU
stages. Do not launch a second writer for the same workspace, delete quarantine
evidence to force progress, or use sealed observations for tuning.

```bash
python tools/run_from_source.py \
  --expected-revision 0123456789abcdef0123456789abcdef01234567 \
  run pipeline.toml --workspace runs/scene-a
```

## 5. Read the result correctly

A completed pipeline proves that declared artifacts were produced and verified.
It does not by itself prove convergence, throughput, unseen-view quality, or
permission to redistribute the input, weights, trained asset, or preview. See
[reproducibility](REPRODUCIBILITY.md) for the evidence levels and
[troubleshooting](TROUBLESHOOTING.md) for common failures.
