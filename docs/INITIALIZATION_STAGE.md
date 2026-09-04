# Public proposal-to-Gaussian initialization stage

This document specifies the first-party bridge from auditable multi-view point
proposals to the strict `p2g.gaussian_initialization.v1` tensor asset consumed
by training. It is an initialization policy, not a substitute for pixel-loss
optimization, population control, or a scene-quality claim.

## Inputs and closure

The builder accepts exactly two artifact roots:

1. one complete `p2g.roma_point_proposal_sequence.v1`; and
2. the matching `p2g.tensor_cache.v1` from which those proposals were made.

A loose PLY directory is deliberately not an input. For every frame the
builder verifies the collection row, frame receipt, binary PLY, provenance
Safetensors, source RGB digest, tensor-cache manifest digest, observation
manifest digest, train-only role-admission digest, admitted camera inventory,
point count, timestamp, and canonical relative filename. It rejects legacy or
tampered proposals without a canonical `train` admission, symlinks, paths
escaping an artifact root, unknown PLY layouts, changed hashes, non-increasing
timestamps, and any source disagreement.

The accepted PLY layout is only the provider's exact binary little-endian
`float32 XYZ + uint8 RGB` payload. The richer correspondence and geometry
planes remain in the hash-bound provenance file and drive the sampling policy.

## Fixed-capacity allocation

For `F` proposal frames and a requested budget `B`, the initial allocation is

```text
slots_per_frame = floor(B / F)
assembled_slots = F * slots_per_frame
discarded_remainder = B - assembled_slots
```

The default is `B = 500,000`. With the intended 60-frame input this produces
`8,333` slots per frame and exactly `499,980` initial Gaussians. The small
remainder is reported rather than hidden or assigned to a privileged frame.

Sampling is with replacement because the Gaussian capacity budget is distinct
from the number of admitted two-view hypotheses. A fixed NumPy PCG64 stream
and content-derived evidence streams make the result replayable.

## Default multi-view sampling policy

The default mode is
`paired_multiview_consensus_rank_mixture`, with a 0.02 world-unit support voxel
and a 0.5 evidence-replacement fraction. It is a soft prior: no confidence,
angle, residual, or support threshold silently removes capacity.

Within each directed camera pair, tied values receive the same normalized
midrank. For proposal `i`, the candidate-quality term is

\[
q_i = \operatorname{gmean}\left(
  r^{\text{matcher}}_i,
  \sin^2(\operatorname{clip}(\theta_i,0,90^\circ)),
  r^{\text{inverse-gap}}_i,
  r^{\text{inverse-reprojection}}_i
\right).
\]

The support voxel records the number of distinct directed-pair ordinals and
distinct cameras corroborating its location. Their global midranks form

\[
s_i = \operatorname{gmean}\left(
  r^{\text{pair-support}}_i,
  r^{\text{camera-support}}_i
\right),
\qquad
w_i = \operatorname{gmean}(q_i,s_i).
\]

The main random stream first draws the exact raw-uniform baseline indices. A
separate SHA-256-derived stream selects half of the output slots and redraws
only those slots with probability proportional to `w_i`. This pairing has two
purposes:

- unchanged slots and subsequent time jitter are identical to the raw arm;
- experiments can attribute differences to the evidence population rather
  than to an unrelated random trajectory.

The receipt reports source-versus-selected quantiles for matcher certainty,
triangulation angle, ray gap, maximum bidirectional reprojection error,
pair/camera support, and final score. These are diagnostics, not declarations
of geometric truth.

Four alternate modes remain explicit for ablation:

| Mode | Rule |
|---|---|
| `raw_candidate_uniform` | uniform proposal rows |
| `occupied_voxel_uniform` | uniform occupied voxels, then uniform rows within each voxel |
| `triangulation_information_mixture` | uniform plus `sin²(angle)` probability mixture |
| `paired_matcher_support_rank_mixture` | paired replacement by within-pair matcher midrank |

## White-box parameter construction

For each selected point `x_i` in frame `f`, the next frame is the motion
reference. The last frame uses the preceding frame with a negative time delta,
which preserves the same forward velocity convention. With `K = 3` by
default,

\[
v_i = \frac{\operatorname{mean}(KNN_K(x_i,P_{f'}))-x_i}{t_{f'}-t_f}.
\]

This is an explicit adjacent-frame nearest-neighbor prior. It does not assert
persistent point identity or learned scene flow.

Spatial scale uses the root-mean-square distance to the nearest three selected
points in the same frame:

\[
\ell_i = 0.1\sqrt{\max\left(
  \operatorname{mean}_{j\in KNN_3(i)} d_{ij}^2,
  10^{-7}
\right)}.
\]

The isotropic initial scale is stored as three copies of `log(ell_i)`. The
rotation is identity in `wxyz` order, opacity defaults to `logit(0.5)`, and
RGB8 is converted to degree-zero real spherical harmonics by

\[
SH_0 = \frac{RGB/255 - 0.5}{0.28209479177387814}.
\]

Each temporal center is jittered uniformly by half the signed adjacent-frame
interval around its source timestamp. For the public default bounds

```text
sigma_min = 1 / 600 seconds
sigma_max = 1 second
sigma_target = 0.1 second
```

the stored duration logit is recomputed as

\[
\operatorname{logit}\left(
  \frac{\sigma_{target}-\sigma_{min}}
       {\sigma_{max}-\sigma_{min}}
\right).
\]

This recalibration matters: carrying over a logit from different physical
bounds would silently change the initialized temporal width.

## Output contract

The append-only output directory contains:

```text
initialization.safetensors
initialization.json
```

Safetensors contains exactly these planes:

```text
means, log_scales, quaternions, opacity_logits, sh0,
center_times, duration_logits, velocities, runtime_ids
```

Legacy aliases and scalar policy planes are rejected. Higher-order SH is not
stored; the public loader creates the configured coefficient count at zero.
Runtime IDs are unique, contiguous, and zero-based.

Safetensors metadata binds the proposal collection, tensor-cache manifest,
sampling mode, duration policy, and time-offset policy. The loader verifies
that physical duration bounds and time offset match the run configuration and
that the stored duration logits reconstruct the declared target.

The JSON receipt contains no local filesystem paths or wall-clock durations.
It binds source hashes, per-frame reference choices, sampling diagnostics,
population arithmetic, tensor catalog, container hash, an ordering-independent
canonical tensor digest, and a logical receipt digest. Publication uses a
fully written sibling directory followed by one rename and never overwrites an
existing destination.

## Command

```text
p2g initialize \
  --proposal-sequence artifacts/proposal-sequence \
  --tensor-cache scene/tensor-cache \
  --output artifacts/initialization
```

Defaults reproduce the intended 60-frame, 499,980-Gaussian initialization.
Progress is written to stderr; stdout is exactly one canonical JSON receipt.
Expected contract, filesystem, or existing-output failures return status 2.
Importing the tool or requesting help does not import Torch.

## Claim boundary

The stage can prove that a trainer-readable Gaussian initialization was built
deterministically from the declared proposal and pixel-cache artifacts. It
cannot prove correspondence correctness, temporal identity, convergence,
final reconstruction quality, throughput, or rights to redistribute the
source dataset and model weights. Those remain separate training, evaluation,
MI300X, and release gates.
