# MI300X runtime build

This guide builds the two native components used by the Pixel4DGS v0 training
profile from immutable public sources. The supported boundary is intentionally
one platform and one ABI; passing these checks does not imply support for other
ROCm releases, GPU architectures, or renderer modes.

## Verified platform

| Component | Required identity |
|---|---|
| Host | Linux x86-64 |
| Python | CPython 3.12 (`cp312`) |
| PyTorch | `2.10.0+rocm7.0` |
| HIP runtime | `7.0.51831` |
| GPU code object | `gfx942` only |
| AMD gsplat | `b01acd43e3c7fa942f95fda0974e9125e4de7395` |
| GLM | `33b4a621a697a305bc3a7610d290677b96beb181` |
| fused-SSIM | `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38` |
| LPIPS provider | `torchmetrics==1.9.0`, AlexNet, registry-bound external weights |

The host must provide `git`, `patch`, `sha256sum`, `roc-obj-ls`, and `uv`, plus
an existing CPython 3.12 environment containing the exact ROCm PyTorch wheel,
`setuptools`, and `wheel`. The scripts fail before compilation if Torch, HIP,
the visible device, or the target architecture differs.

PyTorch is not declared as a generic project dependency because the supported
artifact is the ROCm-specific wheel. Accidentally resolving a CPU or CUDA wheel
with the same distribution name would produce a misleading installation.

The frozen quality profile also requires `torchvision==0.25.0+rocm7.0` and
`torchmetrics==1.9.0`. Install them in the explicit training environment, not
as generic project dependencies. Place the external
`alexnet-owt-7be5be79.pth` checkpoint in the active Torch Hub `checkpoints`
directory before training. The LPIPS registry fixes its size and SHA-256; the
loss constructor refuses a missing or changed file and blocks network download.
The project package contains neither that checkpoint nor the TorchMetrics
linear-weight payload.

## 1. Fetch immutable sources

Choose a new source-cache directory. The fetcher refuses to overwrite an
existing path:

```bash
tools/release/fetch_mi300x_runtime_sources.sh /work/p2g-runtime-sources
```

It fetches exact Git commits and creates deterministic archives:

| Archive | SHA-256 |
|---|---|
| `amd-gsplat-b01acd43.tar.gz` | `04050fbfc4a329ed760baf58362498290a735ed5cd74cb86f2c9e53c0b3f78f3` |
| `glm-33b4a621.tar.gz` | `4755eb000b1400cddd6f94255e4f70886ed4a3dee07231811bd1b04c2ed75b0a` |
| `fused-ssim-a7c48d6d.tar.gz` | `95d68b3ac3e7c29e76a9a7384454ca7b946e6a793c989941fb6929c6ffa99927` |

The command fails if a remote resolves to a different commit or the resulting
archive has different bytes.

## 2. Build isolated wheels

Provide a new build root and the Python executable from the verified training
environment. Keep the virtual-environment path itself; resolving the executable
symlink can lose its package context.

```bash
tools/release/build_mi300x_runtime.sh \
  /work/p2g-runtime-sources \
  /work/p2g-runtime-build \
  /work/p2g-train-venv/bin/python \
  8
```

The builder:

1. verifies the source archives and repository-owned patches;
2. verifies Torch, HIP, the visible GPU, and `gfx942`;
3. extracts only into the new build root;
4. applies two packaging-only AMD gsplat patches;
5. builds with `PYTORCH_ROCM_ARCH=gfx942`;
6. installs both wheels into `BUILD_ROOT/overlay` without dependencies;
7. imports the public APIs from that overlay; and
8. verifies that both native providers contain `gfx942` code objects.

The patches do not change renderer equations or kernels:

- `amd-gsplat-b01acd43-build-identity.patch` gives a verified source archive
  the exact upstream revision in its distribution version;
- `amd-gsplat-b01acd43-glm-include.patch` supplies the exact GLM tree as an
  external include, preventing HIPify from rewriting GLM's own include paths.

Wheel ZIP bytes are not the reproducibility boundary: compiler paths and ZIP
metadata can vary. The source commits and archives, patches, distribution
identity, ABI, code-object architecture, and registered numerical behavior are.

## 3. Run public numerical checks

Use the newly built overlay before any packages already installed in the train
environment:

```bash
runtime_build=/work/p2g-runtime-build
train_python=/work/p2g-train-venv/bin/python

PYTHONPATH="$runtime_build/overlay" "$train_python" \
  tools/release/validate_mi300x_runtime.py capture-gsplat \
  --module-root "$runtime_build/overlay" \
  --output "$runtime_build/gsplat-capture-a"

PYTHONPATH="$runtime_build/overlay" "$train_python" \
  tools/release/validate_mi300x_runtime.py capture-gsplat \
  --module-root "$runtime_build/overlay" \
  --output "$runtime_build/gsplat-capture-b"

PYTHONPATH= "$train_python" \
  tools/release/validate_mi300x_runtime.py compare-gsplat \
  --baseline "$runtime_build/gsplat-capture-a" \
  --candidate "$runtime_build/gsplat-capture-b" \
  --output "$runtime_build/gsplat-repeatability.json"

PYTHONPATH="$runtime_build/overlay" "$train_python" \
  tools/release/validate_mi300x_runtime.py validate-fused-ssim \
  --module-root "$runtime_build/overlay" \
  --p2g-source-root "$PWD/src" \
  --output "$runtime_build/fused-ssim-reference.json"
```

The gsplat capture covers RGB and spherical harmonics degree 3 forward
rasterization, alpha, and gradients for means, quaternions, scales, opacity,
and color in the exact admitted profile. Comparing two captures checks bounded
repeatability of the build. It does **not** establish compatibility with an
arbitrary training run or final scene quality.

The fused-SSIM command compares the extension against the explicit Pixel4DGS
PyTorch equation over three image shapes and both supported padding modes. It
checks the scalar, prediction gradient, finiteness, and repeat execution.

Runtime qualification additionally compares gsplat against a project-owned
synthetic reference fixture. Full-scene quality validation is a separate
end-to-end test; its input and output payloads are not required by the source
commands or distribution.

## 4. Admitted renderer profile

Pixel4DGS v0 admits only:

- one visible GPU and no Gaussian batch dimensions;
- one offline-undistorted pinhole camera per training batch;
- float32 tensors;
- packed rasterization;
- `tile_size=8`;
- materialized RGB or SH3 appearance;
- classic rasterization with no sparse or absolute gradients.

The adapter must reject unsupported background shapes, camera models, batch
dimensions, devices, dtypes, and ABI identities before launching a kernel.
Passing this build does not broaden that profile.

## 5. License boundary

- AMD Ecosystem gsplat is Apache-2.0; preserve its upstream `LICENSE` and
  `NOTICE.txt`.
- fused-SSIM is MIT; preserve its upstream `LICENSE`.
- GLM offers multiple terms; this recipe selects its MIT option and preserves
  the complete upstream `copying.txt`.
- TorchMetrics is Apache-2.0 and torchvision carries its upstream BSD notice;
  the external AlexNet checkpoint remains unbundled with a `NOASSERTION`
  license status.

The fetch/build process obtains those license files with the source trees. The
resulting terms do not cover Pixel4DGS datasets, provider weights, trained
assets, or preview media. Those remain separate release gates.

## 6. Interpreting failures

- A source hash mismatch means the fetch is not the registered source; do not
  patch around it.
- A Torch/HIP/architecture mismatch means the host is outside the supported
  matrix; do not use the result as release evidence.
- A missing `gfx942` code object means the native wheel is not admitted even if
  Python import succeeds.
- A numerical comparison failure is a correctness failure, not a performance
  fluctuation.
- A passing synthetic check does not replace the full-chain quality gate.
