#!/usr/bin/env bash
# Fetch immutable public runtime sources and emit bit-reproducible archives.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 NEW_SOURCE_CACHE" >&2
    exit 64
fi

source_cache=$(realpath -m "$1")
[[ ! -e "$source_cache" ]] || {
    echo "source cache already exists: $source_cache" >&2
    exit 73
}
[[ "$source_cache" != / && "$source_cache" != "$HOME" ]] || {
    echo "unsafe source cache: $source_cache" >&2
    exit 64
}

command -v git >/dev/null
command -v sha256sum >/dev/null

parent=$(dirname "$source_cache")
mkdir -p "$parent"
temporary=$(mktemp -d "$parent/.p2g-runtime-sources.XXXXXXXX")
work="$temporary/work"
payload="$temporary/payload"
mkdir "$work" "$payload"
cleanup() {
    rm -rf -- "$temporary"
}
trap cleanup EXIT

fetch_archive() {
    local name=$1
    local repository=$2
    local revision=$3
    local prefix=$4
    local archive_name=$5
    local expected_sha256=$6
    local checkout="$work/$name"
    local archive="$payload/$archive_name"

    git init --quiet "$checkout"
    git -C "$checkout" remote add origin "$repository"
    git -C "$checkout" fetch --quiet --depth=1 origin "$revision"
    local observed_revision
    observed_revision=$(git -C "$checkout" rev-parse FETCH_HEAD)
    [[ "$observed_revision" == "$revision" ]] || {
        echo "revision mismatch for $name: $observed_revision" >&2
        exit 65
    }
    git -C "$checkout" archive --format=tar.gz --prefix="$prefix/" \
        -o "$archive" "$revision"
    local observed_sha256
    observed_sha256=$(sha256sum "$archive" | cut -d' ' -f1)
    [[ "$observed_sha256" == "$expected_sha256" ]] || {
        echo "archive mismatch for $name: $observed_sha256" >&2
        exit 65
    }
}

fetch_archive \
    amd-gsplat \
    https://github.com/AMD-Ecosystem/gsplat.git \
    b01acd43e3c7fa942f95fda0974e9125e4de7395 \
    amd-gsplat \
    amd-gsplat-b01acd43.tar.gz \
    04050fbfc4a329ed760baf58362498290a735ed5cd74cb86f2c9e53c0b3f78f3

fetch_archive \
    glm \
    https://github.com/g-truc/glm.git \
    33b4a621a697a305bc3a7610d290677b96beb181 \
    glm \
    glm-33b4a621.tar.gz \
    4755eb000b1400cddd6f94255e4f70886ed4a3dee07231811bd1b04c2ed75b0a

fetch_archive \
    fused-ssim \
    https://github.com/rahul-goel/fused-ssim.git \
    a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38 \
    fused-ssim \
    fused-ssim-a7c48d6d.tar.gz \
    95d68b3ac3e7c29e76a9a7384454ca7b946e6a793c989941fb6929c6ffa99927

(
    cd "$payload"
    sha256sum \
        amd-gsplat-b01acd43.tar.gz \
        glm-33b4a621.tar.gz \
        fused-ssim-a7c48d6d.tar.gz \
        > SHA256SUMS
)

mv "$payload" "$source_cache"
printf 'public runtime source cache: %s\n' "$source_cache"
