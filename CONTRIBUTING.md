# Contributing to Pixel4DGS

Pixel4DGS welcomes source contributions under the Apache License 2.0. Unless a
submission is conspicuously marked otherwise, an intentional contribution is
licensed as described by section 5 of [`LICENSE`](LICENSE). The project does
not currently require or imply a separate CLA or DCO.

## Engineering scope

Changes should preserve the two project priorities:

1. numerical and semantic correctness with inspectable mechanisms; and
2. an execution path deliberately optimized and tested for one MI300X.

Do not add private datasets, trained assets, checkpoints, videos, credentials,
machine paths, or code copied from a research reference. A reference may
motivate an independently implemented mechanism, but the change must record
its provenance and pass the project-owned tests and quality gate.

## CPU development checks

Use CPython 3.12. The CPU suite deliberately installs an explicit CPU PyTorch
wheel because the project does not declare a generic Torch runtime dependency:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev]'

HIP_VISIBLE_DEVICES=-1 \
ROCR_VISIBLE_DEVICES=-1 \
CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/python tools/release/check_tracked_source.py --root .

HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/ruff check .
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/pyright
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/python -m pytest -q
```

The synthetic fixture and CPU tests establish contracts and connectivity only.
They cannot support a claim about MI300X throughput, convergence, or visual
quality.

## Change evidence

A reviewable change should state:

- the user-visible behavior or invariant being changed;
- the source/provenance of the design;
- tests that fail before and pass after the change;
- whether artifact formats or resume compatibility change;
- whether an MI300X run is required, and the preregistered metrics when it is;
- any license, weight, dataset, or derived-output implications.

GPU evidence must come from the documented, hash-bound MI300X environment.
Never weaken a quality threshold, use sealed observations for tuning, or select
only a favorable rerun without recording the change in evaluation policy.
