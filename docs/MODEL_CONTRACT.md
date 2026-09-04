# Public initialization and continuous-time model contract

This document defines the trainable representation independently of any
reference repository. It is the contract shared by initialization, training,
checkpointing, AssetBundle export, and rendering.
The equation-family identifier is `p2g.linear_motion_gaussian_gate.v1`.

## Safe initialization boundary

`p2g.gaussian_initialization.v1` is a Safetensors file. It must contain metadata
key `schema_version` with exactly that value. The loader accepts no pickle
fallback, symbolic aliases, arbitrary Python objects, symlinked input, or
unknown tensor planes.

For `N > 0` Gaussians, the required tensor catalog is:

| Tensor | dtype | shape | meaning |
|---|---:|---:|---|
| `means` | float32 | `[N,3]` | reference position in world coordinates |
| `log_scales` | float32 | `[N,3]` | logarithm of positive principal scales |
| `quaternions` | float32 | `[N,4]` | nonzero `wxyz` rotation quaternion |
| `opacity_logits` | float32 | `[N,1]` | unconstrained base-opacity logit |
| `sh0` | float32 | `[N,1,3]` | degree-zero real-SH appearance |
| `center_times` | float32 | `[N,1]` | temporal center in seconds |
| `duration_logits` | float32 | `[N,1]` | unconstrained duration-fraction logit |
| `velocities` | float32 | `[N,3]` | world displacement per second |
| `runtime_ids` | int64 | `[N]` | unique stable identity within the run |

`sh_rest` is the only optional plane. When present it must be float32 with
shape `[N,(D+1)^2-1,3]`, where `D` is the configured SH degree. When absent it
is deterministically initialized to zero. No other plane is admitted.

All floating tensors must be finite, CPU-resident, and contiguous at this
boundary. The run profile supplies the global duration bounds, initial
persistence logit, and time offset. The resolved in-memory state expands scalar
policy into per-Gaussian planes so later fixed-budget population operations can
preserve each Gaussian's complete state explicitly.

The file is hashed before and after parsing. The in-memory provenance record
contains its SHA-256 and schema identity, but never its machine path.

Files produced by the public proposal-to-Gaussian builder also declare their
proposal-sequence hash, tensor-cache hash, sampling policy, physical duration
bounds, target duration, and time offset in Safetensors metadata. The loader
requires those duration bounds and time offset to equal the resolved run
configuration and verifies that every duration logit reconstructs the declared
target. Generic independently produced files remain governed by the base
tensor contract above. The construction equations and receipt are specified in
[the initialization-stage contract](INITIALIZATION_STAGE.md).

## Physical parameterization

For Gaussian `i`, define the scalar time displacement

```text
delta_i(t) = t - center_time_i
```

and linear world-space motion

```text
mean_i(t) = mean_i + velocity_i * delta_i(t).
```

Principal scales are strictly positive:

```text
scale_i = exp(log_scale_i).
```

The stored quaternion is normalized immediately before rasterization. A zero
quaternion is rejected at initialization, rather than silently mapped to a
rotation.

Temporal width is bounded in seconds:

```text
fraction_i = sigmoid(duration_logit_i)
sigma_i = sigma_min_i + (sigma_max_i - sigma_min_i) * fraction_i
```

with the invariant `0 < sigma_min_i < sigma_max_i`. Transient activation is a
unit-height Gaussian in time:

```text
transient_i(t) = exp(-0.5 * (delta_i(t) / sigma_i)^2).
```

When learned persistence is disabled, activation equals `transient_i(t)`.
When enabled, the persistent mixture fraction and activation are

```text
persistent_i = sigmoid(gate_logit_scale * persistence_logit_i)
activation_i(t) = persistent_i + (1 - persistent_i) * transient_i(t).
```

This mixture has useful limiting behavior: a zero persistent fraction is purely
transient; a unit persistent fraction is always active; and activation remains
inside `(0,1]`. The physical opacity sent to the rasterizer is

```text
opacity_i(t) = sigmoid(opacity_logit_i) * activation_i(t).
```

No temporal factor is folded into color, scale, or rotation. Consequently each
part of the representation can be inspected and tested independently.

## Appearance and active SH degree

The model stores degree zero separately from the remaining real spherical
harmonic coefficients so the optimizer may use different learning rates.
Materialization concatenates exactly the coefficients requested by active
degree `d`: `1 + ((d+1)^2-1)` coefficients. Degrees outside the initialized
catalog are rejected.

## MI300X execution shape

The representation is struct-of-arrays: every parameter plane is contiguous
and all Gaussians are evaluated with vectorized PyTorch operations at one scalar
time. There is no Python loop over Gaussians and no host transfer in the hot
materialization path. After model construction, one explicit `.to("cuda")`
moves parameters and registered duration/identity buffers into the ROCm device
address space. The renderer consumes these physical planes through the
separately pinned `gfx942` runtime contract.

This layout is a correctness and ABI decision, not a performance claim by
itself. Release performance still requires a measured MI300X training run and
the registered scene-quality gate.

## Checkpoint boundary

A model state is tensor-only and must contain exactly the ten parameter planes
plus `duration_min_seconds`, `duration_max_seconds`, `runtime_ids`, and
`gate_logit_scale`. Missing and unknown planes are rejected. The persistence
mode remains run policy and must be supplied when reconstructing the module;
an externally supplied gate scale must agree with the stored scalar.

This state is appropriate only inside the separately defined trusted-local
checkpoint envelope. A distributable inference artifact uses AssetBundle and
Safetensors, never a Python object archive.

The local resume checkpoint is an atomically published directory containing
`config.toml`, `metadata.json`, `state.pt`, and `manifest.json`. The manifest
binds the byte length and SHA-256 of every payload and undeclared files are
rejected. `state.pt` is permitted only for local resume state and is loaded with
Torch's restricted `weights_only=True` loader. Python and NumPy RNG states are
converted to primitive containers before saving, so resume does not need an
arbitrary-object unpickler. The checkpoint metadata explicitly declares
`local_resume_only` and `redistributable=false`.

The smaller `model.safetensors` plus `model.json` export is also tensor-only and
hash-bound, but remains a local training output because it carries no rights or
scene provenance. AssetBundle is the only intended publication and inference
exchange format.
