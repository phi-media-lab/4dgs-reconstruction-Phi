#!/usr/bin/env bash
set -euo pipefail

unset PIP_NO_BUILD_ISOLATION PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export PYTHONSAFEPATH=1

for variable in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
    if [[ ${!variable:-} != "-1" ]]; then
        echo "$variable must be -1 for the CPU-only release check" >&2
        exit 2
    fi
done

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

python - <<'PY'
import importlib.metadata
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"release check requires CPython 3.12, found {sys.version.split()[0]}")
expected = {"build": "1.6.0", "setuptools": "78.1.1", "wheel": "0.45.1"}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"release build tool mismatch: expected {expected!r}, found {actual!r}")
PY

repository=$(git rev-parse --show-toplevel)
output=$(realpath -m "$1")
if [[ -e $output ]]; then
    echo "refusing to overwrite release-check output: $output" >&2
    exit 2
fi
if [[ -n $(git -C "$repository" status --porcelain) ]]; then
    echo "release check requires a clean Git worktree" >&2
    exit 2
fi

mkdir -p "$output"
epoch=$(git -C "$repository" show -s --format=%ct HEAD)
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH=$epoch

archive_source() {
    local destination=$1
    mkdir -p "$destination"
    git -C "$repository" archive --format=tar HEAD | tar -xf - -C "$destination"
}

single_artifact() {
    local directory=$1
    local suffix=$2
    local matches=()
    mapfile -t matches < <(find "$directory" -maxdepth 1 -type f -name "*$suffix" -print)
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "expected one $suffix artifact in $directory, found ${#matches[@]}" >&2
        exit 2
    fi
    printf '%s\n' "${matches[0]}"
}

for route in direct-wheel isolated-wheel direct-sdist isolated-sdist; do
    archive_source "$output/source-$route"
done

(
    cd "$output/source-direct-wheel"
    python -m build --wheel --no-isolation --outdir "$output/direct-wheel"
)
(
    cd "$output/source-isolated-wheel"
    python -m build --wheel --outdir "$output/isolated-wheel"
)
(
    cd "$output/source-direct-sdist"
    python -m build --sdist --no-isolation --outdir "$output/direct-sdist"
)
(
    cd "$output/source-isolated-sdist"
    python -m build --sdist --outdir "$output/isolated-sdist"
)

direct_wheel=$(single_artifact "$output/direct-wheel" .whl)
isolated_wheel=$(single_artifact "$output/isolated-wheel" .whl)
direct_sdist=$(single_artifact "$output/direct-sdist" .tar.gz)
isolated_sdist=$(single_artifact "$output/isolated-sdist" .tar.gz)

python "$repository/tools/release/check_python_archives.py" \
    --wheel "$direct_wheel" \
    --wheel "$isolated_wheel" \
    --sdist "$direct_sdist" \
    --sdist "$isolated_sdist"

for label in direct isolated; do
    mkdir -p "$output/unpack-$label"
done
python -m wheel unpack --dest "$output/unpack-direct" "$direct_wheel"
python -m wheel unpack --dest "$output/unpack-isolated" "$isolated_wheel"
diff -ru "$output/unpack-direct" "$output/unpack-isolated"

mkdir -p "$output/extract-direct-sdist" "$output/extract-isolated-sdist"
tar -xzf "$direct_sdist" -C "$output/extract-direct-sdist"
tar -xzf "$isolated_sdist" -C "$output/extract-isolated-sdist"
diff -ru "$output/extract-direct-sdist" "$output/extract-isolated-sdist"

mapfile -t sdist_roots < <(find "$output/extract-direct-sdist" -mindepth 1 -maxdepth 1 -type d -print)
if [[ ${#sdist_roots[@]} -ne 1 ]]; then
    echo "source distribution must contain one root directory" >&2
    exit 2
fi
(
    cd "${sdist_roots[0]}"
    python -m build --wheel --no-isolation --outdir "$output/sdist-wheel"
)
sdist_wheel=$(single_artifact "$output/sdist-wheel" .whl)
cmp "$direct_wheel" "$isolated_wheel"
cmp "$direct_wheel" "$sdist_wheel"
mkdir -p "$output/unpack-sdist-wheel"
python -m wheel unpack --dest "$output/unpack-sdist-wheel" "$sdist_wheel"
diff -ru "$output/unpack-direct" "$output/unpack-sdist-wheel"

for unpacked in "$output/unpack-direct" "$output/unpack-isolated" "$output/unpack-sdist-wheel"; do
    mapfile -t sboms < <(find "$unpacked" -type f -path '*/share/pixel4dgs/sbom.cdx.json' -print)
    if [[ ${#sboms[@]} -ne 1 ]]; then
        echo "wheel must contain exactly one Pixel4DGS SBOM" >&2
        exit 2
    fi
    cmp "$repository/sbom.cdx.json" "${sboms[0]}"
    for notice in LICENSE NOTICE THIRD_PARTY_NOTICES.md; do
        mapfile -t notice_files < <(find "$unpacked" -type f -path "*/*.dist-info/licenses/$notice" -print)
        if [[ ${#notice_files[@]} -ne 1 ]]; then
            echo "wheel must contain exactly one $notice" >&2
            exit 2
        fi
        cmp "$repository/$notice" "${notice_files[0]}"
    done
done
for required in CITATION.cff LICENSE NOTICE THIRD_PARTY_NOTICES.md sbom.cdx.json; do
    if [[ ! -f ${sdist_roots[0]}/$required ]]; then
        echo "sdist is missing $required" >&2
        exit 2
    fi
    cmp "$repository/$required" "${sdist_roots[0]}/$required"
done

index=0
for artifact in "$isolated_wheel" "$direct_sdist"; do
    index=$((index + 1))
    environment="$output/smoke-env-$index"
    fixture="$output/smoke-fixture-$index"
    cache="$output/smoke-cache-$index"
    python -m venv "$environment"
    "$environment/bin/python" -m pip install "$artifact"
    "$environment/bin/python" -m pip check
    "$environment/bin/p2g" --help >/dev/null
    "$environment/bin/p2g" fixture create --output "$fixture"
    "$environment/bin/p2g" prepare "$fixture/observation_manifest.json" --output "$cache"
    "$environment/bin/python" - <<'PY'
import importlib.util
import pathlib
import p2g
import sys

origin = pathlib.Path(p2g.__file__).resolve()
environment = pathlib.Path(sys.prefix).resolve()
if sys.prefix == sys.base_prefix or not origin.is_relative_to(environment):
    raise SystemExit(f"p2g did not load from the fresh virtual environment: {origin}")
if importlib.util.find_spec("torch") is not None or "torch" in sys.modules:
    raise SystemExit("generic release smoke unexpectedly exposed Torch")
PY
done

sha256sum "$direct_wheel" "$isolated_wheel" "$sdist_wheel" "$direct_sdist" "$isolated_sdist"
