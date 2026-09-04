# Fixed-budget relocation contract

This document specifies Pixel4DGS population control independently of any
reference implementation. The objective is narrow: recycle parameter slots
whose base opacity is no longer useful while keeping the Gaussian tensor
capacity, runtime identities, optimizer topology, and continuous-time model
equations fixed.

The public mode name is `fixed_budget_relocation_v1`. It is an inspectable
training mechanism, not a claim of numerical equivalence with another
codebase. Release still requires a fresh MI300X quality and throughput gate.

## State observed during optimization

The registered packed renderer retains the gradient of each projected 2D mean.
For every rendered packed row `j`, the controller converts the normalized
screen gradient to pixel units:

```text
gx_pixel = gx * width  * camera_count / 2
gy_pixel = gy * height * camera_count / 2
g_pixel  = sqrt(gx_pixel^2 + gy_pixel^2).
```

Using the renderer-provided `gaussian_ids`, it accumulates for each fixed slot
`i`:

```text
gradient_sum[i]    += g_pixel
visibility_count[i] += 1.
```

Duplicate packed IDs are reduced with `index_add_`; no dense `[camera,
Gaussian]` table is created. These two vectors are the only per-step relocation
state. They are structure-of-arrays tensors resident on the training device.

At a scheduled event, the mean observed position-gradient magnitude is

```text
mean_gradient[i] = gradient_sum[i] / max(visibility_count[i], 1).
```

The window is reset only after the event has completed. It is included in every
local resume checkpoint, so an interrupted window resumes exactly.

## Event schedule

For completed-step count `s`, an event occurs exactly when

```text
start <= s < stop
and
(s - start) mod every == 0.
```

Anchoring cadence to `start` makes non-round start values unambiguous. A
scheduled event is still recorded when no slot can be moved.

## Dead slots, source utility, and capacity

Base opacity is `a[i] = sigmoid(opacity_logits[i])`. A destination candidate
satisfies

```text
a[i] <= opacity_threshold.
```

A source must satisfy all of the following:

1. it is not a destination candidate;
2. it was visible at least once in the current window;
3. its mean pixel-position gradient is positive; and
4. at least one split piece can remain above the opacity threshold.

Its source utility is

```text
a[i] * mean_gradient[i] / max_eligible(mean_gradient).
```

The normalization changes no relative utility; it prevents unnecessary dynamic
range. Clone allocation is breadth-first over capacity strata. Every eligible
source receives at most one clone in a stratum, and a second stratum begins only
after every source that has second-piece capacity has been considered in the
first. If only part of the last stratum is needed, sources in that frontier are
drawn without replacement using the utility above. The final source-to-slot
pairing is shuffled. Both operations use the checkpointed Torch RNG.

This rule keeps multiplicities as low and as broadly distributed as capacity
allows before utility chooses a partial frontier. It also makes control flow
finite: at most 63 clone strata exist under the multiplicity bound below. One
high-utility source therefore cannot monopolize an event through unbounded
replacement sampling.

For source alpha `a`, threshold `tau`, and total coincident multiplicity `r`,
the split alpha is

```text
a_piece = 1 - (1 - a)^(1/r).
```

The largest permitted `r` is therefore

```text
floor(log(1 - a) / log(1 - tau)),
```

clamped to `[1, 64]`. Clone capacity is `r - 1`. If aggregate source capacity
is smaller than the number of dead candidates, the lowest-opacity destination
slots are recycled first and the rest are reported as deferred. Capacity never
changes: deferred slots retain their prior parameter rows. Capacity calculation
adds an eight-float32-epsilon guard to `tau`, so conversion back to stored
float32 logits cannot place a newly split piece just below the declared
threshold through rounding alone.

## Three explicit conservation laws and one visible approximation

Suppose one source is represented by `r` coincident pieces after relocation.
Every piece receives the alpha above. Their center composite is

```text
1 - (1 - a_piece)^r = a,
```

so center alpha is exact at peak temporal activation before floating-point
rounding.

Keeping source scale unchanged would not preserve the integrated projected
footprint. For an ideal 2D Gaussian profile, expansion of front-to-back alpha
composition gives the dimensionless integral

```text
D(a_piece, r)
  = sum(k=1..r) (-1)^(k+1) * C(r,k) * a_piece^k / k
  = sum(j=1..r) (1 - (1 - a_piece)^j) / j.
```

The second form is evaluated in float64 because it is positive and avoids an
unstable alternating binomial sum. Each scale axis is multiplied by

```text
scale_factor = sqrt(a / D).
```

A projected covariance area scales by `scale_factor^2`; therefore this preserves
the integrated projected alpha mass for any locally linear camera projection.
When learned persistence is enabled, let `p` be the source persistent fraction
and `p_piece` the fraction assigned to every split piece. The controller solves

```text
1 - (1 - a_piece * p_piece)^r = a * p
```

analytically. Thus the composite alpha is also exact at the far-time persistent
limit. The corresponding persistence logit is obtained through the model's
declared `gate_logit_scale`.

No constant `a_piece`, Gaussian duration, and persistence fraction can make
the split composite equal the original for every transient activation in
`[0,1]` when `r > 1`. The controller therefore does not claim full temporal
equivalence. It evaluates 17 evenly spaced transient activations and reports
the maximum interior alpha residual, alongside measured peak-alpha,
far-persistent-alpha, projected-mass, multiplicity, and scale diagnostics.

These conservation statements concern coincident ideal Gaussian profiles at
the peak and far-time endpoints. They do not claim that intermediate temporal
activation, subsequent nonlinear training, or finite raster tiles leave
rendered pixels unchanged.

## Parameter and identity mutation

For each admitted destination, the controller copies the source rows of:

- mean and velocity;
- quaternion;
- degree-0 and higher-order spherical harmonics;
- temporal center and duration logit;
- persistence logit; and
- per-slot minimum and maximum duration buffers.

Source and destination opacity/scale rows are then replaced by the conserved
split values. When persistence is learned, their persistence rows receive the
far-time-conserving value as well. `runtime_ids` are slot identities and are
never copied or renumbered. The total leading dimension of every parameter
plane remains unchanged.

Adam state invalidation follows actual mutation:

- every destination row is cleared for every trainable Gaussian plane;
- source rows are additionally cleared for opacity, log-scale, and learned
  persistence, the source planes changed by the split; and
- unrelated optimizer entries, including a same-length camera-correction
  optimizer, are not touched.

This is stricter and less destructive than identifying optimizer tensors only
by their leading dimension. All parameter, buffer, optimizer-ownership, and
row-state shapes are validated before the first model write, so a malformed
topology cannot leave a partially relocated model. Adam's scalar per-parameter
step counter is deliberately retained; only row-shaped gradient and moment
state made stale by relocation is cleared.

## Lineage and checkpoint state

The controller stores, per stable slot, the most recent relocation step and one
role:

```text
0 = never involved
1 = destination
2 = source.
```

The fixed-size lineage vectors support the formation screen guard without an
unbounded event log. Checkpoint state also contains the exact relocation policy,
gradient window, event counters, and last event step. Restore rejects unknown
fields, a changed policy or population, wrong tensor dtypes/shapes, non-finite
accumulators, inconsistent counters, and impossible lineage chronology.

## MI300X execution boundary

The raster hot path gains only two fixed-size accumulator updates per step.
Source classification, capacity calculation, weighted frontier sampling, float64
split correction, diagnostic scalar materialization, and optimizer-row clearing
run only at relocation events. Capacity allocation uses one 64-bin histogram,
at most one weighted draw without replacement, and one final permutation; it
has no unbounded rejection loop. The implementation calls no custom relocation
kernel and imports no reference package.

CPU analytic tests establish the equations, capacity bound, slot identity,
precise optimizer invalidation, scheduling, lineage, and checkpoint rejection
rules. They do not establish scene convergence or device throughput. Those are
separate fresh-source MI300X gates and must use train-role observations for
optimization, diagnostic observations for routine evaluation, and sealed
observations only for the preregistered final decision.
