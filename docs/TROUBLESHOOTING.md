# Troubleshooting

Start with the first failing stage. Preserve its output and receipt: deleting a
partial workspace usually removes the evidence needed to diagnose it.

## Installation or import failure

Run:

```bash
python --version
python -m pip check
p2g --help
python tools/release/check_tracked_source.py --root .
```

The supported ABI is CPython 3.12. A generic `pip install torch` is not the
MI300X installation path; use the exact ROCm package and native build described
in [MI300X runtime build](MI300X_RUNTIME_BUILD.md). If `p2g --help` imports Torch
or initializes a GPU, treat that as a lazy-import regression.

A source-verifier failure means the checkout is not a clean `HEAD`, contains a
non-ignored untracked file, or includes a forbidden artifact, unsafe Git mode,
oversized file, or credential-shaped value. Review the reported path before
building a release.

## `doctor` reports busy or unsupported hardware

`p2g doctor` observes three occupancy planes but does not reserve a GPU. Check:

- exactly one intended device is visible;
- the device reports AMD Instinct MI300X and `gfx942`;
- no unapproved process owns compute or substantial memory;
- scheduler state agrees with the process/device observations;
- Torch reports the pinned ROCm/HIP runtime.

For `shared_quality`, another job is reported as contention rather than a hard
failure. Confirm that the configured VRAM limit leaves enough capacity, then
expect slower and timing-inadmissible execution. For `exclusive_performance`,
wait for or obtain a dedicated allocation; do not raise thresholds to disguise
contention. A passing observation is still not a scheduler lease.

## Native renderer rejected

The adapter rejects an import-name shadow, editable substitute, wrong package
version, changed call signature, missing native module, or non-`gfx942` binary.
Rebuild from the registered public sources and patches. Confirm the new overlay
precedes any older environment package, then rerun the public numerical checks.

Import success alone is insufficient. Do not enable an upstream JIT fallback
or loosen the ABI check to bypass a missing provider.

## RoMa provider or weight failure

The proposal stage never downloads weights. Verify that both supplied files
match the exact filenames, sizes, and SHA-256 values in the registry and that
the environment lock identifies the admitted RoMa code. The RoMa indoor weight
has unresolved redistribution metadata; local possession does not authorize
bundling it.

Provider stdout must remain machine-readable where specified. Capture verbose
provider diagnostics on stderr or in its stage directory.

## Observation audit failed

The most common causes are:

- image paths escape the declared root or pass through a symlink;
- recorded SHA-256 or encoded dimensions no longer match the file;
- intrinsics use a different pixel domain;
- world-to-camera matrices are not finite rigid transforms;
- timestamps are non-monotonic or exceed synchronization tolerance;
- images are not RGB8 PNG/JPEG in the declared photometric space;
- the same image bytes appear across optimization and evaluation roles.

Fix the source manifest or preparation process and create a new output
directory. The optimizer does not infer calibration, silently undistort frames,
or repair role leakage.

## Resume refused

Resume is intentionally scoped and strict. Check that:

- the active plan's parameters and inputs for every completed stage are unchanged;
- the relevant stage implementation closure has the same file hashes;
- the selected checkpoint is the latest valid checkpoint below the run;
- its manifest and tensor state hashes still match;
- terminal receipts have not been edited;
- only one process writes the workspace.

An incomplete non-resumable directory is moved into the workspace quarantine
before retry. Inspect its `quarantine.json`; do not delete or consume the
payload as a completed stage. A plan change affecting completed work still
needs a new workspace. Training checkpoints are trusted local artifacts; never
open an untrusted checkpoint as an AssetBundle.

## Rendering or video encoding failed

First run `p2g asset verify` on the AssetBundle. Then verify that the camera-path
JSON is hash-bound, uses the admitted pinhole convention, and queries times
inside the declared interval. The renderer needs the asset and camera path, not
the training scene.

Video output additionally requires a working ImageIO/ffmpeg backend. Separate
an encoding failure from a raster failure by retaining the render receipt and
any produced frames. Do not treat a playable video as proof that the asset,
camera path, or source rights were verified.

## Out of memory or poor throughput

Record Gaussian count, resolution, runtime identity, peak allocated/reserved
HBM, warm-step timing, and whether another process was present. Avoid timing
first-step compilation and checkpoint/evaluation steps as ordinary warm
iterations. The public profile is fixed-capacity and single-camera; lowering
capacity or resolution changes the experiment and must be recorded as a new
profile rather than reported as the same result.

## Quality regressed

Compare the earliest divergence, not only the final video:

1. source, runtime, plan, seed, and input hashes;
2. step-zero materialization and raster outputs;
3. sampled observation sequence and loss terms;
4. first relocation or screen-guard event;
5. checkpoint-aligned diagnostic metrics;
6. export/reload render equality;
7. sealed evaluation only after the source and protocol are frozen.

Keep a failing run. Do not tune on sealed observations or repeatedly rerun and
report only the best trajectory.
