# Pixel4DGS architecture

Pixel4DGS turns synchronized, calibrated pixels into an explicit, trainable 4D
Gaussian scene representation. The design keeps geometry, time, appearance,
optimization, and rendering visible as separate mechanisms while specializing
the execution path for one MI300X.

## Data flow

```text
RGB files + calibration + timestamps + roles
                    |
                    v
        observation manifest -- prepare --> tensor cache
                    |                            |
                    +---------- propose --------+
                                   |
                                   v
                         proposal collection
                                   |
                              initialize
                                   |
                                   v
                      Gaussian initialization
                                   |
                                train
                                   |
                    +--------------+--------------+
                    v                             v
             training run                   evaluation
                    |
                asset export
                    |
                    v
             Safetensors AssetBundle
                    +
        reviewed trajectory -- bind --> bound camera path
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
          Pixel4DGS render-video             Viewer CPU bridge
                     |                                 |
                     v                                 v
        offline moving-camera video       phi.4dgs.explicit.v1
                                                       |
                                                       v
                                        AMD Linux Player -- WebRTC --> browser
```

Every Pixel4DGS arrow through asset export and offline rendering is implemented
by the same library function used by its standalone CLI command. `p2g run` only
sequences the six reconstruction stages, records their receipts, and resumes
verified outputs; it does not contain an alternate hidden pipeline. The Viewer
branch begins at the published artifact boundary and is implemented entirely
by the sister repository.

| Stage | Main input | Main output | MI300X work |
|---|---|---|---|
| `prepare` | admitted observation manifest and RGB files | `p2g.tensor_cache.v1` | no |
| `propose` | manifest, tensor cache, registered matcher weights | proposal collection | yes |
| `initialize` | proposals and tensor cache | `p2g.gaussian_initialization.v1` | no |
| `train` | resolved run and initialization | hash-closed training run | yes |
| `evaluate` | run/checkpoint and diagnostic observations | evaluation receipt | yes |
| `asset export` | completed training run and rights assertions | `p2g.asset_bundle.v1` | no |
| `render-video` | AssetBundle and camera path | video and render receipt | yes |

The optional Viewer branch is implemented by the separate
[4DGS Viewer Phi](https://github.com/phi-media-lab/4dgs-viewer-Phi)
repository, not by another hidden Pixel4DGS stage. Its CPU-only bridge accepts
the strict producer profile documented in
[Viewer interoperability](VIEWER_INTEROP.md), converts it to
`phi.4dgs.explicit.v1`, and hands that asset to an AMD Linux Vulkan renderer.
The browser is a thin H.264/WebRTC receiver and never owns the Gaussian
payload.

## Representation

Each Gaussian owns a reference mean, velocity, log scale, quaternion, opacity
logit, spherical-harmonic appearance, center time, bounded duration, optional
persistence, and stable runtime identity. At query time `t`:

```text
mean_i(t) = mean_i + velocity_i * (t - center_time_i)

transient_i(t) = exp(-0.5 * ((t - center_time_i) / sigma_i)^2)

activation_i(t) = persistent_i
                  + (1 - persistent_i) * transient_i(t)

opacity_i(t) = sigmoid(opacity_logit_i) * activation_i(t)
```

Duration is reconstructed from a bounded sigmoid parameter. Quaternions are
normalized and log scales are exponentiated immediately before rasterization.
Temporal activation never hides inside color, scale, or rotation. These choices
make a materialized Gaussian state directly inspectable at any continuous time.

The representation uses contiguous struct-of-arrays tensors. One vectorized
materialization evaluates all Gaussians for one scalar time, followed by one
single-camera packed raster call. There is no Python loop over Gaussians and no
host transfer in the hot materialization path.

## Geometry and role isolation

The observation manifest is the authority for camera calibration, timestamp,
photometric interpretation, and role. Only `train` observations can influence
pair planning, proposal construction, initialization, optimization, or the
screen-influence guard. `diagnostic` observations can be rendered for routine
debugging. `sealed` observations require explicit access and remain outside
training, early stopping, checkpoint selection, and routine evaluation.

The proposal stage restores matcher coordinates to the admitted pixel domain,
forms camera rays from explicit intrinsics/extrinsics, triangulates them, and
records rejection reasons and aggregate geometry statistics. The initializer
then selects a deterministic multi-view evidence mixture and constructs the
fixed-capacity Gaussian planes with explicit KNN motion, scale, and duration
rules.

## Optimization and population control

One training step has a fixed order: sample a train observation, materialize,
rasterize, compute declared losses, backpropagate, reject invalid gradients,
update parameters, then run scheduled population-control and screen-guard
events. The image objective exposes L1 and Gaussian-window SSIM separately;
regularizers remain individually named in the metric stream.

Gaussian capacity is fixed. Relocation reuses dead slots and records stable
lineage rather than growing an unbounded population. Its source utility,
capacity allocation, parameter construction, alpha conservation, approximation
residual, and optimizer-row invalidation are implemented explicitly and tested
independently of the referenced research repositories.

## Runtime boundary

The Python code owns admission, temporal materialization, loss construction,
optimizer behavior, receipts, and runtime identity checks. The pinned AMD
gsplat provider owns projection, spherical-harmonic evaluation, tile sorting,
and alpha compositing. The adapter admits only the tested `gfx942` ABI and
supplies every renderer switch explicitly.

The supported runtime is one MI300X with CPython 3.12, PyTorch 2.10 ROCm 7.0,
and the registered public-source native build. CPU fake-provider tests prove
argument mapping and failure behavior, not native-kernel quality. Native
numerical checks and a preregistered full-scene quality gate remain separate.

The sister Viewer has a different runtime boundary. Its bridge is CPU-only;
its reference Player uses Rust, wgpu/WGSL, Vulkan/RADV, linear DMA-BUF,
VA-API, GStreamer, and WebRTC on an AMD Linux graphics node. ROCm and the
training workspace are not dependencies of that serving process. Conversely,
the Viewer is not a Pixel4DGS training or offline-evaluation dependency.

## Artifact and trust boundaries

Stage outputs are append-oriented and bind inputs by SHA-256. Terminal manifests
are published last, and a changed input or partial output cannot be silently
adopted. Training checkpoints contain optimizer and RNG state and are trusted
local resume artifacts. External exchange uses only JSON plus Safetensors in an
AssetBundle; the renderer does not need the training workspace or source images.

A Viewer conversion creates a new, hash-bound serving asset and copies the
source rights declaration into provenance; it cannot grant new redistribution
rights. Real-scene bundles, converted payloads, frames, and private receipts
remain outside both source trees unless separately authorized.

Source permission, dependency licenses, model-weight terms, input-data rights,
and derived-asset rights are independent. A mechanically valid artifact never
creates a license grant.

## Detailed mechanism references

- [Data admission and caching](DATA_CONTRACT.md)
- [Proposal provider and geometry](ROMA_POINT_PROVIDER_CONTRACT.md)
- [Initialization](INITIALIZATION_STAGE.md)
- [Continuous-time model](MODEL_CONTRACT.md)
- [Losses](LOSS_CONTRACT.md)
- [Training](TRAINING_CONTRACT.md)
- [Relocation](RELOCATION_CONTRACT.md)
- [Renderer](RENDERER_CONTRACT.md)
- [Asset consumption](ASSET_CONSUMPTION.md)
- [Viewer interoperability](VIEWER_INTEROP.md)
