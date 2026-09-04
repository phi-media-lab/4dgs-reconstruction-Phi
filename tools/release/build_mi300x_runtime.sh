#!/usr/bin/env bash
# Build the pinned public Pixel4DGS runtime from hash-verified source archives.

set -euo pipefail

usage() {
    echo "usage: $0 SOURCE_CACHE NEW_BUILD_ROOT TRAIN_PYTHON [MAX_JOBS]" >&2
    exit 64
}

[[ $# -ge 3 && $# -le 4 ]] || usage

source_cache=$(realpath "$1")
build_root=$(realpath -m "$2")
train_python_dir=$(cd "$(dirname "$3")" && pwd -P)
train_python="$train_python_dir/$(basename "$3")"
max_jobs=${4:-8}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)

[[ -d "$source_cache" ]] || { echo "source cache is not a directory" >&2; exit 66; }
[[ -x "$train_python" ]] || { echo "training Python is not executable" >&2; exit 66; }
[[ "$max_jobs" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_JOBS must be positive" >&2; exit 64; }
[[ ! -e "$build_root" ]] || { echo "build root already exists: $build_root" >&2; exit 73; }
[[ "$build_root" != / && "$build_root" != "$HOME" ]] || {
    echo "unsafe build root: $build_root" >&2
    exit 64
}

gsplat_revision=b01acd43e3c7fa942f95fda0974e9125e4de7395
gsplat_archive="$source_cache/amd-gsplat-b01acd43.tar.gz"
gsplat_archive_sha=04050fbfc4a329ed760baf58362498290a735ed5cd74cb86f2c9e53c0b3f78f3
glm_revision=33b4a621a697a305bc3a7610d290677b96beb181
glm_archive="$source_cache/glm-33b4a621.tar.gz"
glm_archive_sha=4755eb000b1400cddd6f94255e4f70886ed4a3dee07231811bd1b04c2ed75b0a
fused_ssim_revision=a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38
fused_ssim_archive="$source_cache/fused-ssim-a7c48d6d.tar.gz"
fused_ssim_archive_sha=95d68b3ac3e7c29e76a9a7384454ca7b946e6a793c989941fb6929c6ffa99927
identity_patch="$repo_root/third_party/patches/amd-gsplat-b01acd43-build-identity.patch"
identity_patch_sha=c6ac18feb5ccf3a76b61e0f7e65fec2b7154de53b1cc3d1535beaad8524252bd
glm_patch="$repo_root/third_party/patches/amd-gsplat-b01acd43-glm-include.patch"
glm_patch_sha=3aea0f0e87854d2134b4b3d2adf5f16155995e219f81388f9b33ceea73ff12ea

verify_sha256() {
    local path=$1
    local expected=$2
    [[ -f "$path" && ! -L "$path" ]] || {
        echo "required regular file is missing: $path" >&2
        exit 66
    }
    local observed
    observed=$(sha256sum "$path" | cut -d' ' -f1)
    [[ "$observed" == "$expected" ]] || {
        echo "SHA-256 mismatch for $(basename "$path"): $observed" >&2
        exit 65
    }
}

verify_sha256 "$gsplat_archive" "$gsplat_archive_sha"
verify_sha256 "$glm_archive" "$glm_archive_sha"
verify_sha256 "$fused_ssim_archive" "$fused_ssim_archive_sha"
verify_sha256 "$identity_patch" "$identity_patch_sha"
verify_sha256 "$glm_patch" "$glm_patch_sha"

command -v uv >/dev/null
command -v patch >/dev/null
command -v roc-obj-ls >/dev/null

env PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= \
    "$train_python" - <<'PY'
import torch

assert str(torch.__version__) == "2.10.0+rocm7.0", torch.__version__
assert str(torch.version.hip) == "7.0.51831", torch.version.hip
assert torch.cuda.is_available()
architecture = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
assert architecture == "gfx942", architecture
PY

mkdir -p \
    "$build_root/sources/amd-gsplat" \
    "$build_root/sources/glm" \
    "$build_root/sources/fused-ssim" \
    "$build_root/wheels" \
    "$build_root/logs" \
    "$build_root/tmp/gsplat" \
    "$build_root/tmp/fused-ssim" \
    "$build_root/torch-extensions/gsplat" \
    "$build_root/torch-extensions/fused-ssim"

tar -xzf "$gsplat_archive" --strip-components=1 -C "$build_root/sources/amd-gsplat"
tar -xzf "$glm_archive" --strip-components=1 -C "$build_root/sources/glm"
tar -xzf "$fused_ssim_archive" --strip-components=1 -C "$build_root/sources/fused-ssim"

patch --batch --fuzz=0 -p1 -d "$build_root/sources/amd-gsplat" < "$identity_patch"
patch --batch --fuzz=0 -p1 -d "$build_root/sources/amd-gsplat" < "$glm_patch"

(
    cd "$build_root/sources/amd-gsplat"
    env \
        PYTHONNOUSERSITE=1 \
        PYTHONSAFEPATH=1 \
        PYTHONPATH= \
        TMPDIR="$build_root/tmp/gsplat" \
        TORCH_EXTENSIONS_DIR="$build_root/torch-extensions/gsplat" \
        PYTORCH_ROCM_ARCH=gfx942 \
        AMD_GSPLAT_BUILD_REVISION="$gsplat_revision" \
        AMD_GSPLAT_GLM_INCLUDE="$build_root/sources/glm" \
        MAX_JOBS="$max_jobs" \
        uv build --wheel --no-build-isolation --no-python-downloads \
            --no-create-gitignore --python "$train_python" \
            --out-dir "$build_root/wheels" \
        2>&1 | tee "$build_root/logs/amd-gsplat-build.log"
)

(
    cd "$build_root/sources/fused-ssim"
    env \
        PYTHONNOUSERSITE=1 \
        PYTHONSAFEPATH=1 \
        PYTHONPATH= \
        TMPDIR="$build_root/tmp/fused-ssim" \
        TORCH_EXTENSIONS_DIR="$build_root/torch-extensions/fused-ssim" \
        PYTORCH_ROCM_ARCH=gfx942 \
        MAX_JOBS="$max_jobs" \
        uv build --wheel --no-build-isolation --no-python-downloads \
            --no-create-gitignore --python "$train_python" \
            --out-dir "$build_root/wheels" \
        2>&1 | tee "$build_root/logs/fused-ssim-build.log"
)

gsplat_wheel="$build_root/wheels/amd_gsplat-1.5.3+${gsplat_revision}-cp312-cp312-linux_x86_64.whl"
fused_ssim_wheel="$build_root/wheels/fused_ssim-1.0.0-cp312-cp312-linux_x86_64.whl"
[[ -f "$gsplat_wheel" && ! -L "$gsplat_wheel" ]] || {
    echo "AMD gsplat wheel was not produced" >&2
    exit 70
}
[[ -f "$fused_ssim_wheel" && ! -L "$fused_ssim_wheel" ]] || {
    echo "fused-SSIM wheel was not produced" >&2
    exit 70
}

mkdir "$build_root/overlay"
env PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= \
    uv pip install --no-deps --target "$build_root/overlay" \
        --python "$train_python" "$gsplat_wheel" "$fused_ssim_wheel"

env PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="$build_root/overlay" \
    "$train_python" - <<'PY'
import importlib.metadata

import fused_ssim
import gsplat

assert importlib.metadata.version("amd-gsplat") == (
    "1.5.3+b01acd43e3c7fa942f95fda0974e9125e4de7395"
)
assert importlib.metadata.version("fused-ssim") == "1.0.0"
assert callable(gsplat.rasterization)
assert callable(fused_ssim.fused_ssim)
PY

roc-obj-ls "$build_root/overlay/gsplat/csrc.so" | grep -q -- "--gfx942"
roc-obj-ls "$build_root/overlay"/fused_ssim_cuda*.so | grep -q -- "--gfx942"

printf '%s  %s\n' \
    "$(sha256sum "$gsplat_wheel" | cut -d' ' -f1)" "$(basename "$gsplat_wheel")" \
    "$(sha256sum "$fused_ssim_wheel" | cut -d' ' -f1)" "$(basename "$fused_ssim_wheel")"
printf 'sources: amd-gsplat=%s glm=%s fused-ssim=%s\n' \
    "$gsplat_revision" "$glm_revision" "$fused_ssim_revision"
