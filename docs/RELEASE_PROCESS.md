# Release process

Pixel4DGS source is distributed under Apache-2.0. This document defines the
mechanical checks for a release commit. The workflow intentionally contains no
upload step; publishing remains an explicit maintainer action.

## Release scope

A release commit must contain coherent `LICENSE`, `NOTICE`, third-party
notices, SBOM, and citation metadata and pass the checks below. A real-scene
demo, hosted-CI badge, sealed-quality claim, or performance claim requires its
own evidence, but none is required to distribute the clearly labeled alpha
source package.

## CPU archive check

The [release archive workflow](../.github/workflows/release-check.yml) calls
the same repository-owned checker available to a local reviewer:

```bash
python3.12 -m venv /tmp/pixel4dgs-release-tools
source /tmp/pixel4dgs-release-tools/bin/activate
python -m pip install \
  build==1.6.0 \
  setuptools==78.1.1 \
  wheel==0.45.1

HIP_VISIBLE_DEVICES=-1 \
ROCR_VISIBLE_DEVICES=-1 \
CUDA_VISIBLE_DEVICES=-1 \
tools/release/check_python_distributions.sh /tmp/pixel4dgs-release-check
```

Use a new output directory, a clean checkout, and a fresh CPython 3.12 virtual
environment created without `--system-site-packages`. The checker enforces the
documented Python and build-tool versions, removes Python path and user-site
inheritance, and:

1. verifies all three GPU visibility variables are disabled;
2. builds direct, PEP 517-isolated, and sdist-derived wheels from clean Git
   archives with the release commit time as `SOURCE_DATE_EPOCH`;
3. rejects unsafe or non-canonical archive paths, normalized extraction-path
   aliases, duplicate members, and non-regular sdist payloads;
4. validates every wheel `RECORD` member, SHA-256 digest, and declared size
   before unpacking, then compares all three wheel trees and archive bytes;
5. compares direct and isolated sdist contents;
6. verifies that wheel and sdist copies of the license, notices, and SBOM
   exactly match the source snapshot;
7. installs one wheel and one sdist in fresh environments and runs the bounded
   fixture-to-prepare smoke path without Torch.

The workflow has only `contents: read`, disables persisted checkout
credentials, and neither uploads nor publishes its outputs. Its result covers
Python source archives only. It does not validate a ROCm runtime, run training,
approve dependency or dataset rights, or establish reconstruction quality.

## Authorized release gate

From the exact release commit, regenerate any environment-specific dependency
lock and rerun the source, archive, and fresh-install checks. Clean-checkout
MI300X and preregistered quality runs are required
for claims tied to those results, not for the Apache-2.0 source grant itself.
Only an authorized maintainer may create a release tag or publish archives;
there is intentionally no publishing credential or command in the workflow.
