# Public loss and image-metric contract

This document fixes the reconstruction equations, regularizer meanings, and
provider boundary used by the public training path. The goal is to make a loss
configuration interpretable without reading a reference implementation.

## Image domain

Every metric consumes one `H x W x 3` RGB tensor for the prediction and one for
the target. Shapes must match exactly; both tensors must be float32 on the same
device. The constants below assume a unit photometric range. Scene admission
provides targets in `[0, 1]`; predictions are deliberately not clamped before
L1, SSIM, or PSNR, because clamping would hide overshoot and suppress its
gradient.

## SSIM

For images at least 11 pixels along each axis, the reference implementation
uses an `11 x 11` Gaussian window with sigma `1.5`. For a smaller image it uses
the largest odd width that fits, with sigma scaled by `width / 11`. The same
normalized two-dimensional kernel is applied independently to R, G, and B.

For local means `mu_x`, `mu_y`, population variances `var_x`, `var_y`, and
covariance `cov_xy`, the per-pixel score is

```text
((2 mu_x mu_y + C1) (2 cov_xy + C2))
-------------------------------------------------
((mu_x^2 + mu_y^2 + C1) (var_x + var_y + C2))
```

where `C1 = 0.01^2` and `C2 = 0.03^2`. The reported score is the mean over
pixels and channels. `same` uses zero padding and preserves image dimensions;
`valid` excludes the window radius at every border. The training term is
`weight_ssim * (1 - score)`.

The pure-Torch equation is the executable reference. The optional fused path
has no silent fallback: selecting it requires `fused-ssim==1.0.0`, built from
the registered public revision
`a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38`. The runtime release check compares
both score and prediction gradient for both padding policies. Merely finding a
module with the expected import name is insufficient; the module must belong
to the registered distribution, and the MI300X runtime receipt remains the
source-revision and binary-ABI proof.

## PSNR

PSNR assumes the same unit range:

```text
MSE  = mean((prediction - target)^2)
PSNR = -10 log10(max(MSE, 1e-12))
```

The floor makes an exact match finite and equal to 120 dB. It is a reporting
metric and is not included in the optimized objective.

## Training objective

`LossFunction` always returns the same ordered scalar catalog. Disabled terms
are explicit zero tensors, so metrics never change shape across profiles.

| Term | Unweighted value | Intended pressure |
|---|---|---|
| `l1` | mean absolute RGB error | reconstruct observed color |
| `ssim` | `1 - SSIM` | reconstruct local structure |
| `lpips` | pinned TorchMetrics AlexNet LPIPS score | preserve perceptual structure through the frozen quality-reproduction path |
| `opacity` | mean base opacity weighted by detached temporal activation | discourage unnecessary opacity without letting this regularizer collapse time gates |
| `scale` | mean physical Gaussian scale | discourage unnecessarily broad splats |
| `persistence` | mean persistent mixture fraction | prefer transient support when enabled |
| `gate` | mean `1 - persistent_fraction` | prefer persistent support when enabled |
| `color_correction` | caller-supplied weighted affine-camera penalty | keep nuisance correction near identity |

The two persistence terms represent opposite priors and are separately visible
in metrics. A nonzero persistence prior is rejected when persistence is not a
learned model plane. The color-correction value is already weighted by its own
configuration before it enters this catalog; the loss layer does not weight it
a second time.

## LPIPS compatibility path

The optional perceptual term uses the registered provider profile:
`torchmetrics==1.9.0`,
`LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="mean",
normalize=False)`, PyTorch `2.10.0+rocm7.0`, and torchvision
`0.25.0+rocm7.0`. Prediction and target are each clamped to `[0,1]`, exposed as
a strided NCHW view, passed to the provider, and multiplied by the configured
loss weight. Provider state is reset after every observation.

This versioned input convention is intentionally explicit: TorchMetrics
documents `normalize=False` for `[-1,1]` tensors, whereas this profile supplies
clamped `[0,1]` tensors. The implementation does not relabel the domain or
silently change it to `normalize=True`.

`registries/lpips_alex_v1.json` fixes the TorchMetrics wheel/sdist identities,
the installed metric and functional source files, its packaged Alex linear
weights, the torchvision AlexNet source, and the external AlexNet checkpoint.
The loader verifies installed ownership, sizes, SHA-256 values, runtime
versions, and the pre-existing Torch Hub checkpoint before constructing the
metric, then verifies the files again. Automatic download is actively blocked.
The AlexNet checkpoint remains external and unbundled with a `NOASSERTION`
license status; enabling LPIPS without the exact local file fails closed.

## MI300X execution shape

All image work is vectorized. The Torch SSIM kernel is cached per window,
device, and dtype; fused SSIM receives contiguous `1 x 3 x H x W` tensors,
while LPIPS preserves the registered strided NCHW compatibility view. A
disabled term performs no provider import or image computation.
The hot loss path does not perform Python truth-value reads of GPU tensor
contents. Finiteness is checked after backward by the training-loop loss and
gradient gate, avoiding extra device synchronization inside each loss term.
