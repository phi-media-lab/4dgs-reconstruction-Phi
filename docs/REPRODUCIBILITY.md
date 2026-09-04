# Reproducibility and evidence

Pixel4DGS separates reproducibility into source, package, runtime, run, and
quality layers. Passing a lower layer never stands in for a higher one.

## 1. Source identity

A release is identified by an immutable Git commit. Verify that the checked-out
index and working-tree bytes match that commit:

```bash
python tools/release/check_tracked_source.py --root .
git fsck --strict
```

The verifier requires `HEAD`, the index, and working tree to agree and rejects
non-ignored untracked files, symlinks, submodules, trained/data artifact
suffixes, oversized files, and common credential shapes. The Apache-2.0 grant
is in the top-level `LICENSE`.

## 2. Python distributions

Build from a clean release commit with a fixed epoch derived from that commit:

```bash
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" \
PYTHONHASHSEED=0 \
python -m build --wheel --sdist --outdir dist
```

The release gate compares wheels built directly, in PEP 517 isolation, and
from the sdist. Before extraction it verifies archive member bytes, wheel
`RECORD`, canonical relative paths, and the absence of normalized path aliases.
It then verifies the embedded license, notice, and SBOM files and fresh
installation of both archive types. The source distribution carries user
documentation, tests, build recipes, and source-verification tooling; the
wheel carries runtime code, schemas, registries, project license/notice files,
and the pre-build SBOM.

Run the repository-owned no-publish implementation locally or through the
hosted workflow as described in the [release process](RELEASE_PROCESS.md).

PyTorch is intentionally outside the generic package dependencies. CPU checks
install the explicit PyTorch CPU wheel. The admitted MI300X environment instead
uses the hash-pinned ROCm wheel and native-provider recipe in
[MI300X runtime build](MI300X_RUNTIME_BUILD.md).

## 3. Runtime identity

Before a GPU stage, record:

- host/ABI and CPython version;
- Torch distribution and ROCm/HIP version;
- the one visible GPU and `gfx942` architecture;
- AMD gsplat distribution, source revision, native module ownership, and call
  signature;
- provider registry and external-weight SHA-256 values;
- resolved profile and environment-lock identities.

`p2g doctor` captures occupancy from device and process planes. Shared-quality
runs may proceed under recorded contention when HBM capacity is sufficient, but
their timing is inadmissible. A passing observation is not a reservation; use a
scheduler or another exclusive-allocation mechanism for performance evidence.

## 4. Run identity and resume

Use append-only plan history and one workspace for a compatible development
line. Each stage binds only its parameters, direct inputs, recursive Python
source closure, preceding receipts, admission evidence, and terminal artifact.
Completed work is skipped only when that scoped request is identical. A change
that affects an already completed stage requires a new workspace; a downstream
change can retain unaffected upstream work. `pipeline.json` marks any
cross-revision reuse as `MIXED_REVISION`; a release qualification must be
`SINGLE_REVISION`.

Training records the deterministic sampler, model, optimizer, relocation,
color-correction, and Python/NumPy/Torch RNG state at a checkpoint. Metric rows
newer than the selected checkpoint are removed before resume; older canonical
rows are retained. Evaluation-aligned checkpoints make each reported periodic
metric recoverable from an exact state.

Retain at least:

```text
pipeline.toml
workspace.json
pipeline.json
artifacts/resolved-run.toml
artifacts/run/runtime.json
artifacts/run/metrics.jsonl
artifacts/run/training.json
artifacts/evaluation/evaluation.json
artifacts/asset/manifest.json
paths/scene.camera-path.json
evidence/scene.render.json
```

The camera path and render receipt are post-export evidence, not members of the
six-stage reconstruction workspace. Retain them beside the immutable plan and
completion seal when moving-camera review is part of the experiment record.

Receipts avoid machine paths where a digest or workspace-relative path is
sufficient. Large inputs and outputs remain external to source control.

## 5. Quality and performance claims

Evidence levels are deliberately distinct:

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| CPU unit/contract suite | equations, boundaries, deterministic state transitions | native kernel execution or scene quality |
| MI300X synthetic parity | pinned runtime and bounded operator behavior | training convergence or generalization |
| completed training transaction | mechanical optimization/export closure | unseen-view quality or redistribution rights |
| diagnostic evaluation | declared development-view behavior | sealed quality |
| preregistered sealed evaluation | one unchanged run against fixed thresholds | support for arbitrary scenes or platforms |
| qualified timing window | throughput/HBM for the exact runtime and workload | performance on a shared or different GPU |

Register the source commit, runtime identities, dataset/role split, seed,
iteration count, Gaussian count, thresholds, and evaluation code before a
sealed run. Do not tune from sealed observations, weaken a threshold after the
result, or select only a favorable rerun. Record failures as evidence.

## 6. Release reproducibility checklist

- tracked-source verification and `git fsck --strict` pass;
- Ruff, strict Pyright, and the complete CPU suite pass with GPU visibility
  disabled;
- direct, isolated, and sdist-derived wheels agree;
- fresh wheel and sdist installations pass the documented smoke path;
- the no-publish release archive workflow passes from the exact commit;
- dependency lock, third-party notices, and CycloneDX SBOM bind the release;
- clean public-source native build and numerical checks pass on the admitted
  MI300X runtime;
- held-out full-scene, sealed-quality, moving-camera, resume, throughput, and HBM
  gates pass without changing the preregistration;
- project source, dependency, weight, data, asset, and media rights are each
  approved for the intended distribution.
