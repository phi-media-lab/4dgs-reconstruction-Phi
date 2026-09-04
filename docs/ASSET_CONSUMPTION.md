# Inspecting, verifying, and rendering an AssetBundle

An `AssetBundle` is the portable output of the Pixel4DGS training pipeline. It
contains exactly three files: a Safetensors model, semantic metadata, and a
manifest that binds both by byte length and SHA-256. It deliberately contains
no training configuration, optimizer state, source images, observation paths,
or checkpoint pickle.

## Export from a completed training run

Asset publication is deliberately separate from optimization. The exporter
first verifies the path-free `training.json` completion receipt and all bound
run artifacts, then requires explicit producer and rights assertions:

```text
p2g asset export RUN \
  --output ASSET \
  --producer-git-revision FULL_40_CHARACTER_REVISION \
  --asset-license SPDX_OR_LICENSE_ASSERTION \
  --redistribution review_required \
  --provenance-summary "How this derived asset was produced"
```

The source-data license and train-role time interval come from the audited
observation manifest; they are not silently replaced by command-line values.
The output path is append-only and must be outside the immutable training run. See
[the public training contract](TRAINING_CONTRACT.md) for the upstream binding.

## Inspect and verify

Inspection loads the bundle on CPU, verifies its complete file inventory and
hashes, validates the metadata schema and tensor catalog, reconstructs the
continuous-time Gaussian model, and prints a path-free summary:

```text
p2g asset inspect artifacts/scene.asset
```

Verification performs the same checks and atomically publishes a separate
`p2g.asset_verification.v1` receipt:

```text
p2g asset verify artifacts/scene.asset \
  --output evidence/scene-asset-verification.json
```

The receipt records the bundle ID, Gaussian and tensor counts, equation
version, redistribution state, and hashes of all three files. Its
`logical_sha256` covers every other receipt field. It contains no local asset
path. The output must be a new `.json` path outside the AssetBundle; the tool
will neither alter a bundle nor overwrite prior evidence.

A successful verification means that the bytes and declared semantics are
self-consistent and accepted by this implementation. It does not grant source
data or asset redistribution rights and does not make a visual-quality,
runtime-compatibility, or performance claim.

## Explicit moving-camera render

A reusable `p2g.camera_trajectory.v1` contains only reviewed camera geometry,
timestamps, resolution, and frame rate. It deliberately has no AssetBundle ID.
After export, bind that trajectory to the exact bundle and validate every
timestamp against its declared valid interval:

```text
p2g camera-path bind artifacts/scene.asset \
  --trajectory paths/reviewed-trajectory.json \
  --output paths/scene.camera-path.json
```

This publishes a new `p2g.camera_path.v1`; it never modifies the trajectory or
the asset. A path cannot be bound before export because the final bundle ID is
derived from the published asset bytes.

Video rendering then consumes only that AssetBundle and bound camera path:

```text
p2g render-video artifacts/scene.asset \
  --camera-path paths/orbit.json \
  --output previews/orbit.mp4 \
  --receipt evidence/orbit.render.json
```

The bound camera path names the exact AssetBundle ID and provides, for every frame,
the timestamp, 3x3 pinhole intrinsic, and 4x4 world-to-camera transform. The
loader checks finite values, positive focal lengths, zero skew, homogeneous
matrix rows, a right-handed orthonormal rotation, nondecreasing time, and the
asset's valid-time interval before rendering.

The v0 command fixes the device to the ROCm Torch `cuda` compatibility device.
The renderer independently admits the exact MI300X runtime and checks the
runtime dependency identities against those recorded by the asset. It does not
consult a run directory, checkpoint, observation manifest, or source dataset.

Frames are rendered in the asset's declared photometric space, encoded with
H.264/yuv420p, then checked with ffprobe for codec, pixel format, padded even
resolution, frame count, and rate. The video and receipt are new outputs. A
temporary video is fully encoded and probed before it is hard-linked into
place; receipt publication is also append-only. The receipt binds the camera
path bytes, video bytes, asset/model identity, renderer implementation, runtime,
timings, encoder identity, and redistribution state.

On success each standalone command writes exactly one canonical JSON object to
stdout. Progress is sent to stderr. Expected contract, filesystem, or
existing-output failures return status 2 without replacing any artifact.

The source distribution also retains the standalone tools for environments
that intentionally do not install console scripts; they implement the same
artifact boundary and never consult training-run state.

## Separation from training verification

`tools/verify_asset_bundle.py` inspects only the portable export. Comparing a
local checkpoint with an export would require run state and is therefore not a
portable consumer test. Continuous-time behavior is explicit in the asset
equation metadata and is exercised by rendering caller-supplied timestamps in
the camera path.
