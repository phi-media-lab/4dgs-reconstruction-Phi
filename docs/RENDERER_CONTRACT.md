# Public MI300X renderer contract

This document fixes the only renderer profile admitted by Pixel4DGS v0.  The
project-owned Python adapter validates and materializes the inputs; the
projection, spherical-harmonic evaluation, tile sorting, and alpha compositing
are executed by the separately licensed, pinned AMD gsplat provider.

## Runtime identity

The release profile is deliberately one binary ABI:

| Layer | Required identity |
|---|---|
| Host | Linux x86-64, CPython 3.12 |
| Accelerator visibility | exactly one device, addressed as `cuda:0` by Torch |
| Accelerator | AMD Instinct MI300X, `gfx942` |
| PyTorch | `2.10.0+rocm7.0` |
| HIP runtime | `7.0.51831` |
| AMD gsplat distribution | `amd-gsplat==1.5.3+b01acd43e3c7fa942f95fda0974e9125e4de7395` |
| AMD gsplat source | commit `b01acd43e3c7fa942f95fda0974e9125e4de7395` |

The adapter checks the normalized distribution name, full version, module
version, Python call signature, and ownership of both `gsplat/__init__.py` and
the prebuilt `gsplat/csrc.so`.  It refuses an import-name shadow, an editable
substitute, and a missing native provider.  The last check happens before the
backend import so upstream cannot silently fall back to a local JIT build.

These runtime checks complement rather than replace the source-build receipt.
The public build recipe additionally binds the source archives, packaging-only
patches, license files, and `gfx942` code objects.  The numerical release gate
checks RGB and SH3 forward results plus gradients for means, quaternions,
scales, opacities, and appearance.

## Input representation

One call rasterizes one materialized Gaussian population into one camera.  It
does not accept Gaussian batch dimensions or camera batches.

| Input | Shape | Meaning |
|---|---:|---|
| means | `N x 3` | world-space centers at the queried time |
| quaternions | `N x 4` | normalized `wxyz` rotations |
| scales | `N x 3` | positive world-space axis scales |
| opacities | `N` | base opacity multiplied by temporal activation |
| colors | `N x K x 3` | real spherical-harmonic coefficients, `K=(degree+1)^2` |
| world-to-camera | `1 x 4 x 4` | rigid pinhole view transform |
| intrinsic | `1 x 3 x 3` | pixel-domain pinhole calibration |

Every tensor above is contiguous float32 on the same execution device.  The
active SH degree is an integer from zero through three.  The temporal model is
evaluated before this boundary, so the renderer never owns motion, duration,
or persistence equations; the `MaterializedGaussians` returned with the result
records the exact physical state it consumed.

Targets are admitted as offline-undistorted images.  Radial or tangential
coefficients are rejected because the provider's unscented-transform path does
not supply the complete scale/quaternion gradient contract used by training.
Camera parameters are constants: this ABI does not claim gradients for
intrinsics or extrinsics.

## Fixed provider call

The adapter supplies every behavioral switch explicitly:

- packed projection and rasterization, with no Gaussian batch dimensions;
- `tile_size=8`, selected for the registered AMD implementation;
- classic EWA rasterization with `eps2d=0.3` by default;
- RGB output, pinhole camera, global shutter, and one camera;
- no covariance shortcut, distortion, UT, Eval3D, segmentation, distribution,
  sparse gradients, or absolute-gradient mode;
- a three-element linear-RGB background for the registered packed ABI; and
- a 32-channel chunk, although the admitted output has only three channels.

AMD gsplat evaluates real SH coefficients along the camera-to-Gaussian
direction and applies its documented `max(SH + 0.5, 0)` conversion before
front-to-back alpha compositing.  Pixel4DGS optionally clamps the final RGB
image to `[0,1]`; alpha and projection metadata are not clamped or detached.

## Output and gradient surface

The exact outputs are one `H x W x 3` RGB tensor, one `H x W x 1` alpha tensor,
and packed projection metadata.  The adapter validates the following metadata
before returning it:

- aligned `camera_ids`, `gaussian_ids`, `means2d`, two-axis `radii`, `depths`,
  projected `opacities`, and `tiles_per_gauss` rows;
- one-dimensional `flatten_ids` for pixel-intersection accounting; and
- the exact `ceil(width/8) x ceil(height/8)` tile grid.

When `means2d` participates in autograd, its gradient is retained.  Population
control can therefore accumulate screen-space gradients by explicit packed
Gaussian ID; screen-influence checks can use the same depth, radius, opacity,
and tile-coverage rows.  The adapter also adds canonical width, height,
single-camera count, and renderer ABI fields without deleting provider
metadata.

Shape, dtype, device, package identity, and call-signature checks require no
GPU tensor-value reads.  Per-pixel finiteness and gradient finiteness are
checked at the training loop's existing synchronization boundary rather than
forcing an extra device synchronization inside every render.

## Correctness and support boundary

A successful CPU fake-provider test proves argument mapping, rejection rules,
output validation, and metadata/gradient preservation.  It does not prove a
native kernel. A successful MI300X synthetic numerical gate proves only the
pinned inputs and tolerances. Held-out full-scene evaluation is required for a
training-quality claim; datasets, checkpoints, assets, and videos are not part
of the renderer source distribution.

Other AMD GPUs, CUDA, multiple visible GPUs, alternate Torch/ROCm versions,
distorted cameras, multi-camera batches, and other gsplat modes are outside v0;
the adapter fails closed instead of presenting them as best-effort support.
