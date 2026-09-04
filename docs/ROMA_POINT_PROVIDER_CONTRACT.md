# Public RoMa point-provider contract

This document specifies the optional correspondence stage that turns one
synchronized multi-camera frame into an auditable bank of two-view 3D point
hypotheses. The stage proposes initialization capacity; it does not claim
temporal identity, final Gaussian quality, or scene reconstruction by itself.

## Ownership and dependency boundary

Pixel4DGS owns the tensor-cache reader, camera graph, seed derivation,
coordinate conversion, triangulation, admission rules, provenance format, PLY
writer, and atomic publication logic in p2g.training.roma_point_provider.

RoMa supplies dense image correspondences through its public Python API. It is
not vendored:

- distribution: romatch==0.1.2;
- repository: https://github.com/Parskatt/RoMa.git;
- revision: 77f8d68803526dcddfd9b7a46bc76125bdc25f15;
- factory: romatch.roma_indoor; and
- source license: MIT, copyright Johan Edstedt.

The package-owned registry is p2g/registries/roma_provider_v1.json. It records
the source identity, license notice hash, runtime matrix, weight URLs, byte
lengths, hashes, and redistribution decisions. Provider construction fails
unless the installed distribution has the exact version, owns the imported
module, and has a direct_url.json record for the registered Git commit.

## Weight policy

No model weight is included in the source tree, wheel, source distribution, or
container recipe, and the provider contains no downloader.

| Weight | Bytes | SHA-256 | Registry status |
|---|---:|---|---|
| RoMa indoor | 445,646,911 | 4d3dca889ae1ef245123dc62aab914475c7bbf41f2c8002606450fb6cf2d91e6 | NOASSERTION; license review required |
| DINOv2 ViT-L/14 | 1,217,586,395 | d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428 | Apache-2.0 declared upstream |

Users obtain these files from the URLs shown in the registry and pass local
paths explicitly. Before loading a checkpoint, the provider rejects symlinks,
the wrong byte extent, or a hash mismatch. A source-code license does not by
itself grant rights to redistribute a model checkpoint; the unresolved RoMa
weight status is deliberately visible rather than inferred.

## Runtime admission

The currently admitted inference environment is intentionally narrow:

- Linux x86-64 and CPython 3.12;
- PyTorch 2.10.0+rocm7.0;
- torchvision 0.25.0+rocm7.0;
- HIP 7.0.51831;
- exactly one visible accelerator; and
- AMD gfx942.

The caller supplies the uv.lock used for inference. Its hash is embedded in
the receipt, and its romatch, PyTorch, torchvision, Python, Git-source, and
ROCm-index records must match the public registry. Runtime inspection repeats
the version and device checks rather than trusting the lock alone.

The model is constructed from the two verified local state dictionaries with
coarse resolution 560, upsample resolution 864, float16 inference, directed
matching, prediction upsampling, no padding, no compilation, and the
pure-PyTorch correlation path. Both state dictionaries are passed to the
factory, preventing its optional network-download branches. Sampling uses
threshold mode with threshold 0.05. Deterministic Torch algorithms and highest
float32 matmul precision are enabled before model construction.

## Public inputs and role authority

The provider requires both the exact `p2g.observation_manifest.v2` and the
matching `p2g.tensor_cache.v1` used by training. The manifest remains the role
authority; the cache is only a pixel/calibration transport. Its layout is:

    tensor_cache.json
    rgb.npy                 uint8    [F,C,H,W,3]
    intrinsic.npy           float32  [F,C,3,3]
    world_to_camera.npy     float32  [F,C,4,4]
    timestamp_seconds.npy   float64  [F,C]

Every array must be a regular, non-symlinked, C-order NumPy file whose path,
shape, dtype, and SHA-256 agree with the manifest. Frame and camera axes must
agree across all four arrays. The selected frame must have finite timestamps
that satisfy the source manifest's synchronization audit, positive focal
lengths, homogeneous camera matrices, and proper orthonormal rotations. The
receipt binds both transport hashes and canonical hashes of the selected frame
payload.

Before any correspondence call, the provider schema-validates and semantically
audits the observation manifest, requires its exact file hash to equal the
cache binding, compares the complete camera/frame axes and the selected
frame's dimensions, intrinsics, extrinsics, and timestamps, then selects only
`train` observations. Cameras marked `diagnostic`, `sealed`, or `free_view`
never enter the camera graph or provider images. A canonical role-admission
record names admitted observation/camera IDs, groups excluded camera IDs by
role, and is carried through the frame receipt and sequence collection. The
proposal time is the explicitly recorded arithmetic mean of the admitted
train-observation timestamps; excluded roles cannot shift the temporal axis.

## Directed pair plan

Camera centers are computed from the world-to-camera matrices. Each source
camera is connected to the requested number of nearest distinct cameras.
Squared center distance is the primary order and camera index is the explicit
tie-break, so the graph is deterministic.

The requested frame budget is divided equally over directed pairs:

    samples_per_pair = requested_points // (camera_count * neighbor_count)
    actual_samples   = samples_per_pair * camera_count * neighbor_count

The receipt records both requested and actual counts. A pair seed is the first
64 bits of SHA-256 over canonical JSON containing the global seed, frame ID,
source camera, target camera, and neighbor rank. Results inside each pair are
stably sorted by dense source-grid linear index.

## Coordinate and geometry equations

For an image of width W, height H, and a RoMa normalized coordinate
(u_n, v_n), pixel coordinates are

\[
u = \frac{W}{2}(u_n + 1), \qquad
v = \frac{H}{2}(v_n + 1).
\]

For a dense output grid of width W_d and height H_d, the inverse pixel-center
mapping is

\[
x_d = \operatorname{round}\left(\frac{(u_n+1)W_d-1}{2}\right), \qquad
y_d = \operatorname{round}\left(\frac{(v_n+1)H_d-1}{2}\right).
\]

Each pair is triangulated in float64 by the project-owned pinhole geometry
implementation. Every sampled row retains:

- normalized and pixel-space matches;
- raw RoMa certainty and its sampling score;
- world position and both ray parameters;
- camera-space depth in each view;
- reprojection errors in both views;
- world-space ray gap and triangulation angle;
- normalized Sampson residual;
- every individual validity predicate; and
- its admitted PLY row, or -1 if rejected.

Only representation-safety conditions determine admission: finite geometry and
scores, a recoverable dense source cell, both pixels inside their images,
positive camera-space depth in both views, and position inside the declared
world bound. Confidence, reprojection error, ray gap, angle, and epipolar
residual remain continuous diagnostics; they are not hidden quality filters.

Color is sampled bilinearly at pixel centers from the lower indexed camera in
the pair. This canonical rule gives the two directions of a camera pair the
same color source without depending on a reference implementation.

## Published artifact

One frame is published as an append-only directory:

    fNNNNNN.ply
    provenance.safetensors
    receipt.json

The PLY contains admitted float32 XYZ and uint8 RGB. Safetensors contains every
sampled row, including rejected rows and their reason planes. A canonical
semantic digest hashes sorted tensor names, little-endian dtypes, shapes, and
payload bytes independently of Safetensors metadata ordering. The receipt
records the camera graph, pair seeds, policy, per-pair and aggregate
diagnostics, provider/runtime/weight identities, artifact hashes, and
limitations. It also records `frame.role = train` and the hash-bound role
admission. Resume rejects a frame shard if that admission differs from the
currently supplied source manifest.

A temporary sibling directory is completely written and synchronized before a
single rename publishes the result. Existing destinations are never
overwritten. The sequence layer verifies completed frame shards and then
publishes a hard-linked point root atomically, so interrupted work can resume
without silently accepting changed inputs or policy.

The downstream initializer accepts only this complete collection together with
its matching tensor cache; it does not accept an unbound PLY directory. Its
fixed-capacity allocation and evidence sampling are specified in
[the public initialization-stage contract](INITIALIZATION_STAGE.md).

## Reference command entry points

The standalone tools expose only names from the public artifact vocabulary.
They resolve relative paths against the invocation directory, reject implicit
home and file-URI expansion, and never download weights. For one frame:

```text
python tools/build_roma_point_proposals.py \
  --tensor-cache scene/tensor-cache \
  --observation-manifest scene/observations.json \
  --frame-id 0 \
  --roma-indoor-weight external-models/roma_indoor.pth \
  --dinov2-weight external-models/dinov2_vitl14_pretrain.pth \
  --environment-lock environments/roma/uv.lock \
  --output artifacts/proposals/f000000
```

For a resumable sequence, the frame interval is explicitly half-open:

```text
p2g propose \
  --tensor-cache scene/tensor-cache \
  --observation-manifest scene/observations.json \
  --frame-start 0 \
  --frame-stop-exclusive 60 \
  --roma-indoor-weight external-models/roma_indoor.pth \
  --dinov2-weight external-models/dinov2_vitl14_pretrain.pth \
  --environment-lock environments/roma/uv.lock \
  --output artifacts/proposal-sequence
```

On success, stdout is exactly one canonical JSON receipt followed by a newline.
Provider and per-frame progress is redirected to stderr, so stdout can be piped
directly into a JSON consumer. Expected contract, filesystem, or existing-output
failures return status 2 with a concise stderr message. Argument syntax errors
also return status 2 through argparse. A Python traceback is reserved for an
unexpected implementation failure.

## Proof boundary

Project-owned CPU fixtures prove tensor-cache validation, exact manifest
binding, exclusion of sealed/diagnostic cameras before provider dispatch,
deterministic graph and seeds, coordinate conversion, two-view geometry,
diagnostic-only confidence, canonical provenance, PLY/Safetensors publication,
tamper rejection, and the no-eager-Torch import boundary. The pinned source
and distribution metadata have also been checked in the MI300X provider
environment.

These checks do not authorize redistribution of the RoMa indoor checkpoint and
do not establish full-scene initialization quality or throughput. A fresh
GPU-enabled provider replay remains a separate MI300X release gate.
