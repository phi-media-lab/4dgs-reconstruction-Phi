# Public data contract

This document specifies the data boundary between a calibrated capture and the
Pixel4DGS optimizer. The implementation is intentionally narrow: unsupported
inputs fail before pixels can enter a training step.

## Configuration layers

Configuration has three distinct records:

1. `p2g.portable_profile.v1` contains algorithm and runtime policy but no data
   paths.
2. `p2g.scene_inputs.v1` contains the observation-manifest and initialization
   paths for one scene. Relative paths are resolved against this file.
3. `p2g.resolved_run.v1` contains absolute, resolved paths and is saved with a
   run. Its TOML serialization is deterministic and never overwrites an
   existing evidence file.

Routine evaluation is fixed to observations whose role is `diagnostic`.
`sealed` observations cannot be placed in `data.eval_roles`.

## Admission sequence

`PreparedScene.load` performs the following checks in order:

1. validate the v2 observation JSON schema;
2. run the semantic audit, including path containment, image SHA-256 and
   encoded header, rigid calibration, timestamp/synchronization, rectangular
   camera-frame coverage, and cross-role image isolation;
3. enforce the training decode subset described below; and
4. if configured, validate and open the tensor cache.

The public training subset accepts only:

- PNG or JPEG files that decode as three-channel RGB;
- 8-bit, full-range samples;
- `pinhole` cameras in the `undistorted` pixel domain with no distortion
  coefficients; and
- `srgb_encoded` inputs with a declared sRGB decode profile, or `linear_rgb`
  inputs with the linear pass-through profile.

Lens distortion must be removed as a recorded preparation transform. It is not
silently corrected inside the optimizer.

An image is hashed again on its first raw decode. This closes the interval
between scene admission and lazy loading. Decoded batches use `[H,W,3]`
float32 RGB in the scene's declared photometric space. Resizing happens in that
space, and the first two rows of the intrinsic matrix are scaled by the exact
output/input width and height ratios.

## Charge v1.0 adapter

`p2g data import-charge` converts an already downloaded fixed-rig Charge task
into `p2g.observation_manifest.v2`; it performs no network access and copies no
pixels. Both camera JSON paths must be inside the task root. Every train and
test camera must contain the same contiguous source frame IDs, static `PERSP`
calibration, square pixels, and full-range RGB8 PNG files whose headers and
SHA-256 digests are recorded. Source frame IDs are sorted and reindexed to
`0..F-1`; time is `(source_frame_id - first_source_frame_id) / fps`.

The source `transformation_matrix` is a Blender camera-to-world transform. With

```text
S = diag(1, -1, -1, 1)
```

the canonical camera matrix is computed explicitly as

```text
world_to_camera_opencv = inverse(camera_to_world_blender * S)
```

This is not a field-name guess. It was selected from four c2w/w2c and
Blender/OpenCV hypotheses by a cross-camera RGB+metric-depth projection test on
the fixed `Charge-050_0130` revision
`3a2b0a91af66c02bf7444a8a2d6cef48b91bbf0c`. From train camera
`Dense_00_02` into disjoint test camera `DenseTest_00_02`, 17,630 pixels were
sampled at each of source frames 416, 462, and 508. The selected conversion,
with source depth interpreted as camera-space Z, produced median relative depth
errors from `1.73e-4` to `1.78e-4`, 72.9–74.3% of all samples within 2%, and
median RGB L1 error of 1.0–1.33 on the depth-consistent samples. The other
hypotheses had median relative depth error of at least 10.3% or at most 11.3%
of all samples within 2%. Occlusions account for most samples outside the
selected depth threshold.

Official train cameras are always `train`. Test cameras sort by identifier;
the last `--sealed-camera-count` cameras become `sealed` and the remainder are
`diagnostic`. At least one diagnostic camera is required. Charge PNGs carry no
embedded color profile in the audited sample, so the adapter records the
explicit `srgb_reference_assumption_v1` decode profile when no sRGB chunk is
present. That assumption, the frame mapping, input revision, JSON hashes, and
all image hashes contribute to the source identity.

## SelfCap video adapter

`p2g data import-selfcap` converts an authorized local SelfCap-style capture
with `videos/<camera>.mp4` and EasyMocap
`optimized/{intri.yml,extri.yml,sync.json}` inputs. It performs no network
access and writes no source media into the package. Install the exact conversion
runtime with `pixel4dgs[selfcap]`; the adapter records the NumPy, OpenCV, Pillow,
OpenCV-build, and implementation identities in its request.

For target frame `f` and camera synchronization offset `s`, the decoded source
position is

```text
source_start_frame + f + source_fps * s
```

An integral position uses one decoded frame. A fractional position linearly
interpolates the adjacent decoded BGR frames in float32. Each camera is then
undistorted with OpenCV's calibrated pinhole map. The adapter intersects the
`alpha=0` valid ROIs across every camera, applies that one common crop, scales
with `INTER_AREA`, clips to `[0,255]`, rounds half up once, converts BGR to RGB,
and writes a lossless RGB8 PNG. These operations and their parameters are
included in the manifest transform identity; the optimizer never repeats them.

Outputs are organized as `rgb/<camera>/<frame>.png`. Every PNG has its own byte
length and SHA-256 in the camera receipt and in the final
`p2g.observation_manifest.v2`. The importer publishes one camera directory at a
time through an atomic rename, verifies a completed camera before resuming it,
and publishes the final manifest and import receipt only after all camera
directories pass image-header, hash, schema, and semantic audit. Re-running an
identical request verifies and returns the existing receipt; changed source,
runtime, arguments, or output bytes fail closed.

The designated diagnostic and sealed cameras must be distinct calibrated
camera IDs. They are excluded from training for every output frame; all other
cameras receive the `train` role. Frame IDs and timestamps start at zero in the
generated observation grid, while the original inclusive source-frame range is
preserved in the request and receipt.

The adapter records the upstream SelfCap research/noncommercial source terms
as a restricted license reference. It does not grant permission to publish the
input videos, materialized frames, learned assets, or rendered media. Keeping
those payloads external is independent from the project's Apache-2.0 source
license.

## Role capabilities

The scene exposes disjoint index sets:

- `train_indices` for optimization;
- `diagnostic_indices` (and the compatibility name `eval_indices`) for routine
  evaluation;
- `sealed_indices` for the preregistered final quality gate; and
- `free_view_indices` for explicitly requested view access.

`load_batch(index)` accepts only train or diagnostic indices. A sealed record
requires `load_batch(index, access="sealed")`; a free-view record similarly
requires `access="free_view"`. The role check happens before consulting the
decoded-image cache, so previously caching a sealed batch cannot bypass the
capability check.

## Tensor cache v1

The optional cache is an I/O optimization, not a second scene definition. Its
root contains `tensor_cache.json` plus four ordinary C-order `.npy` arrays:

| Array | dtype | shape |
|---|---|---|
| `rgb` | `uint8` | `[F,C,H,W,3]` |
| `intrinsic` | `float32` | `[F,C,3,3]` |
| `world_to_camera` | `float32` | `[F,C,4,4]` |
| `timestamp_seconds` | `float64` | `[F,C]` |

`tensor_cache.json` follows `p2g.tensor_cache.v1`. It records the ordered
camera and frame axes, the SHA-256 of the source observation manifest, and the
relative path, SHA-256, dtype, shape, and C-order declaration for every array.
Absolute paths, parent traversal, symlinked array files, missing axes, and
undeclared arrays are rejected.

At open time, array headers are compared with the declarations. Intrinsics,
world-to-camera matrices, and timestamps are then compared observation by
observation with the source manifest. The mmap and raw-image paths share the
same photometric conversion, resizing, tensor layout, and role checks.

The cache currently requires one RGB resolution across the complete
camera-frame grid. Variable-resolution scenes remain supported through the raw
image path.

The public preparation command builds this cache directly from an already
admitted observation manifest:

```text
p2g prepare scene/observations.json \
  --image-root scene \
  --output artifacts/tensor-cache
```

Preparation does not infer calibration, synchronization, distortion, color
space, or roles. It first performs the full manifest and file audit, then
claims a new directory, writes ordinary `.npy` arrays, flushes and hashes every
array, and publishes `tensor_cache.json` last. An interrupted directory has no
manifest and is never resumed or overwritten implicitly.

The cache deliberately retains all manifest roles so one artifact can serve
training, diagnostic, and sealed evaluation. A proposal builder therefore must
not treat cache membership as optimization permission. `p2g propose` requires
the exact source observation manifest in addition to the cache, verifies its
byte hash and semantic audit, checks cache axes/calibration/timestamps against
it, and admits only observations whose role is `train`. Diagnostic, sealed,
and free-view camera IDs are recorded as excluded in each frame receipt. The
proposal timestamp is computed only from the admitted train observations.

## Sampling and resume

Two public sampling policies are defined:

- `shuffled_epoch` visits every selected training observation once per epoch;
- `frame_camera_with_replacement` chooses a frame uniformly, then chooses one
  available training camera uniformly within that frame.

Sampler state uses `p2g.scene_sampler_state.v1` and contains only JSON-safe
values. The policy, seed, exact observation inventory, frame partition,
permutation/cursor, epoch, and Python PRNG words are validated before state is
replaced. A failed restore leaves the live sampler unchanged.

## Current proof boundary

Project-owned synthetic tests exercise raw decoding, sRGB conversion,
intrinsic scaling, sealed access, selection accounting, post-audit tampering,
tensor-cache hash and coordinate substitution, and JSON-roundtripped sampler
resume. These tests establish the data contract; they do not by themselves
establish scene-level 4DGS quality or MI300X throughput.
