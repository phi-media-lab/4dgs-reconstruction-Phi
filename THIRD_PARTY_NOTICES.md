# Third-party notices

Pixel4DGS itself is licensed under Apache-2.0. This source distribution does
not vendor third-party runtime source trees, Python wheels, model weights,
datasets, checkpoints, trained Gaussian assets, or preview videos. Those
items retain their own terms. The records below identify the external
components supported by the repository and the way they are handled.

## Native MI300X renderer build

- [AMD Ecosystem gsplat](https://github.com/AMD-Ecosystem/gsplat), revision
  `b01acd43e3c7fa942f95fda0974e9125e4de7395`, is Apache-2.0. Pixel4DGS ships
  an independently maintained build recipe and two patch files, not the
  upstream source archive. The build verifies and preserves upstream
  `LICENSE` and `NOTICE.txt`.
- [OpenGL Mathematics (GLM)](https://github.com/g-truc/glm), revision
  `33b4a621a697a305bc3a7610d290677b96beb181`, is fetched only while building
  the renderer. The recipe selects the MIT option from upstream's
  dual-license text and preserves `copying.txt`.
- [fused-ssim](https://github.com/rahul-goel/fused-ssim), revision
  `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38`, is an MIT-licensed, optional
  experiment recorded by the runtime manifest. It is not part of the default
  admitted quality path and is not included in this distribution.

The exact archive, license, notice, and patch hashes are in
[`third_party/manifests/mi300x_runtime_v1.json`](third_party/manifests/mi300x_runtime_v1.json).

## External Python and model providers

- PyTorch and torchvision are installed as external ROCm wheels. Their
  licenses and notices must accompany any redistribution of those binaries.
- [RoMa](https://github.com/Parskatt/RoMa) code is loaded from the pinned MIT
  release `romatch==0.1.2`; it is not vendored here.
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) `1.9.0` is an
  external Apache-2.0 dependency used by the admitted LPIPS provider.
- The DINOv2 ViT-L/14 checkpoint, RoMa indoor checkpoint, and torchvision
  AlexNet feature checkpoint are external-only, hash-verified files. They are
  never downloaded automatically or bundled. The registry records DINOv2 as
  Apache-2.0; terms for the RoMa indoor and AlexNet feature weights remain
  `NOASSERTION`, so users must establish their right to obtain and use them.

Exact provider revisions, artifact hashes, source license links, and
redistribution policy are in
[`src/p2g/registries/roma_provider_v1.json`](src/p2g/registries/roma_provider_v1.json)
and
[`src/p2g/registries/lpips_alex_v1.json`](src/p2g/registries/lpips_alex_v1.json).

## Research references and data

FreeTimeGS++ and 3DGS-MCMC are reference-only research codebases. Their code is
not copied into, linked by, or required at runtime by Pixel4DGS.

Charge and SelfCap-style inputs remain external. No dataset license, likeness
permission, or right in a derived checkpoint, Gaussian asset, image, or video
is granted by Pixel4DGS's Apache-2.0 source license. The repository contains
only offline adapters and source-only recipes. See
[`docs/LICENSE_AND_PROVENANCE.md`](docs/LICENSE_AND_PROVENANCE.md) for the
full rights-layer boundary.
