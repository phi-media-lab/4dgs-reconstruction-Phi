# 4DGS Viewer Phi interoperability

Pixel4DGS owns calibrated-pixel ingestion, MI300X training, and publication of
the portable `p2g.asset_bundle.v1` plus a separately bound
`p2g.camera_path.v1`. The sister
[4DGS Viewer Phi](https://github.com/phi-media-lab/4dgs-viewer-Phi) repository
owns the offline bridge, its `phi.4dgs.explicit.v1` format, the AMD Linux
renderer, media transport, and browser receiver.

This split is intentional. A training workspace, checkpoint, optimizer,
dataset, PyTorch installation, or ROCm runtime never becomes a Viewer input.
The bridge consumes only the two hash-closed inference artifacts.

## Compatibility authority

The consumer-side
[`tools/convert_p2g_asset.py`](https://github.com/phi-media-lab/4dgs-viewer-Phi/blob/main/tools/convert_p2g_asset.py)
is the executable authority for acceptance. This document describes how to
produce its current supported input profile; it does not duplicate that parser
inside Pixel4DGS. The corresponding mapping and validation procedure are in the
Viewer's
[`P2G_ASSET_BRIDGE.md`](https://github.com/phi-media-lab/4dgs-viewer-Phi/blob/main/docs/P2G_ASSET_BRIDGE.md).

The integration described here was reviewed against Viewer revision
`ad2fba774bcb11411c32921a3ffbe6a3019ee5c1`. Later consumers must still be
checked with the converter shipped by the revision being deployed.

## Accepted producer profile

The Viewer bridge deliberately accepts a strict subset of valid Pixel4DGS
outputs:

| Boundary | Required value |
|---|---|
| Bundle | `p2g.asset_bundle.v1`, format major 1, exactly `asset.json`, `manifest.json`, and `model.safetensors` |
| Tensor model | `p2g.asset_model.v1`, exact 14-plane catalog, finite values, unique runtime IDs |
| Equation | `p2g.linear_motion_gaussian_gate.v1` |
| Temporal model | learned persistence and a finite positive gate-logit scale |
| Appearance | real SH3, `gsplat_real_sh_v1`, linear-RGB coefficients, default degree 3 |
| Photometric output | `linear_rgb` or `srgb_reference_profile` |
| Camera | pre-undistorted pinhole, OpenCV axes, pixel-center intrinsics, right-handed world-to-camera extrinsics |
| Raster ABI | `p2g.gsplat_rocm.v1`, `radius_clip = 0`, clamped RGB, finite near/far/`eps2d` |
| Camera path | exact bundle ID, at least two ordered frames, every timestamp inside the asset interval |

All bundle bytes, Safetensors metadata and tensor catalog entries must agree.
The bridge validates every camera-path frame even though it selects only one as
the Player's initial camera.

The normal Pixel4DGS configuration surface is broader. In particular,
`model.persistence` defaults to `off`, SH degree can be lower than 3, and a
caller may request a non-zero `radius_clip` or unclamped output. Those remain
valid Pixel4DGS choices but are not accepted by the current Viewer bridge.

## Reference training policy

[`examples/viewer-interop/profile.toml`](../examples/viewer-interop/profile.toml)
is a complete, load-tested MI300X profile with the supported representation and
raster settings made explicit. It is an interoperability profile, not a claim
that one set of losses, population-control settings, or cache sizes is optimal
for every scene.

The fields that determine bridge eligibility are:

```toml
[initialization]
sh_degree = 3

[model]
persistence = "learned"
gate_logit_scale = 20.0

[renderer]
radius_clip = 0.0
clamp_rgb = true
```

The bridge accepts any finite positive gate-logit scale and converts it to the
Player's fixed scale of 20. The reference profile uses 20 so that this
particular mapping is numerically direct.

Asset publication must also retain degree 3. In the pipeline plan, state it
instead of relying on the model maximum:

```toml
[asset]
default_sh_degree = 3
```

The scene's admitted observation manifest determines whether exported output
is `linear_rgb` or `srgb_reference_profile`; both paths are supported. The
bridge applies an sRGB transfer only for linear output and does not apply a
second transfer to the reference-sRGB path.

## Export and bind a camera path

First export, inspect, and verify the completed run using the normal Pixel4DGS
commands:

```bash
p2g asset export RUN \
  --output ASSET \
  --producer-git-revision FULL_40_CHARACTER_REVISION \
  --asset-license SPDX_OR_LICENSE_ASSERTION \
  --redistribution review_required \
  --provenance-summary "Describe the admitted inputs and derived-output review."

p2g asset inspect ASSET
p2g asset verify ASSET --output ASSET_VERIFICATION.json
```

Bind a reviewed trajectory only after the final bundle ID exists:

```bash
p2g camera-path bind ASSET \
  --trajectory TRAJECTORY.json \
  --output CAMERA_PATH.json
```

The camera path must remain separate from the three-file AssetBundle. Neither
command modifies an existing artifact or silently replaces an output.

## Convert with the Viewer

Run these commands from a checkout of 4DGS Viewer Phi. The output directory
must not already exist:

```bash
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  python3 tools/convert_p2g_asset.py \
  ASSET CAMERA_PATH.json PHI_EXPLICIT_ASSET \
  --camera-frame 0 \
  --name scene-name

HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  python3 tools/validate_asset.py PHI_EXPLICIT_ASSET/manifest.json
```

Conversion is CPU-only. It reads neither source images nor training state and
produces:

```text
PHI_EXPLICIT_ASSET/
  manifest.json
  gaussians.bin
  sh3.f16
```

The converted payload remains on the renderer host. In Remote Frame Mode the
browser receives H.264 video and sends camera/time controls; it never downloads
the Gaussian asset or creates the renderer's GPU device. Follow the
[Player guide](https://github.com/phi-media-lab/4dgs-viewer-Phi/blob/main/player/README.md)
for the supported AMD Linux, Vulkan, DMA-BUF, VA-API, GStreamer, and WebRTC
profile.

## Semantic mapping

For source time interval `[t0,t1]`, let `D = t1 - t0` and normalized Player
time be `u = (t - t0) / D`. The bridge applies:

```text
mean_phi       = mean_p2g
center_phi     = (center_seconds - t0) / D
velocity_phi   = velocity_per_second * D
sigma_phi      = sigma_seconds / D
raw_gate_phi   = persistence_logit * p2g_gate_scale / 20
quaternion_phi = xyzw reordered from p2g wxyz
```

Pixel4DGS reconstructs temporal width as

```text
sigma = sigma_min + (sigma_max - sigma_min) * sigmoid(duration_logit).
```

The Player uses a different bounded raw-duration parameterization. When the
source duration bounds are not the common `[0,sigma_max]` form, the bridge
reparameterizes each Gaussian so the normalized physical `sigma` is preserved.
It rejects a normalized width below the Player's `1e-6` floor instead of
silently widening it. Pixel4DGS therefore must not weaken its positive
duration-bound invariant merely to preserve the source raw logit.

Means, log scales, opacity logits and SH0 remain `f32`. The 15 non-constant SH3
coefficients are converted from `f32` to coefficient-major IEEE binary16; this
is the only intended lossy parameter conversion, and the conversion receipt
reports its maximum and mean absolute error.

## Camera-path behavior

Pixel4DGS `render-video` consumes every frame in a bound camera path and emits
an offline moving-camera video. The current Viewer bridge has a different use:
it validates the complete path, then `--camera-frame` selects one frame's
calibration and timestamp as the interactive Player's initial state. Subsequent
orbit, zoom and time changes come from the browser control channel. Automatic
playback of the full imported camera trajectory is not currently claimed by the
Viewer.

## Evidence and rights boundary

The hand-off has completed on a real 499,980-Gaussian SH3 scene through
conversion, AMD Vulkan rendering, VA-API H.264 encoding, WebRTC presentation,
and browser orbit/zoom controls. The authorized rendered preview and its media
terms are maintained in the Viewer repository. Real-scene source media,
AssetBundles, converted payloads, raw frames, private receipts, machine paths,
and service configuration are not part of either source release.

That integration proves one artifact and serving path. It does not establish:

- training reproduction from the current public checkout;
- Pixel4DGS/Phi same-camera, same-time pixel parity;
- scene-quality metrics, long-duration stability, multi-user service, or TURN;
- redistribution rights for a source dataset or a derived model.

Keep those claims independent. In particular, a passing bridge or structural
validator is not a substitute for an independently rendered cross-implementation
image comparison.
