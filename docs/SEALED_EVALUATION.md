# Sealed quality evaluation

The sealed evaluator answers one narrow question: did one exact final training
run meet a quality threshold that was fixed before the sealed observations were
opened? It is separate from routine `p2g evaluate`, which remains restricted to
the diagnostic role.

## Freeze before training

A `p2g.sealed_quality_gate.v1` file binds:

- the exact recipe and portable-profile bytes;
- the observation-manifest SHA-256 and dataset ID;
- the diagnostic and sealed camera IDs and observation counts;
- the required final step, Gaussian count, and SH degree;
- the PSNR and SSIM equations, SSIM padding, and aggregation rule; and
- diagnostic floors plus a sealed anchor, allowed regression, and resulting
  absolute floors.

The gate has a canonical `gate_id`. Review and commit that ID before the run.
Changing a threshold, camera, dependency-bound recipe, or profile changes the
ID. The policy reserves sealed observations for a single post-freeze result;
the software cannot prevent someone from copying held-out data and violating
that experimental rule, so process review is still required.

## Evaluate the final run

The input must be a complete 30k-style run, not an intermediate checkpoint.
Its training receipt, final tensor-only export, final diagnostic evaluation,
resolved configuration, input binding, manifest, profile, and recipe are all
verified before sealed access is granted.

```bash
p2g evaluate-sealed runs/scene-final \
  --gate protocols/scene-v1.gate.json \
  --output evidence/scene-v1-sealed
```

The command evaluates only `scene.sealed_indices` and calls the dataset loader
with `access="sealed"`. It publishes a new directory containing canonical
`receipt.json` and one PNG render per sealed observation. The destination must
not already exist and must be outside the training run.

Both quality outcomes are evidence:

- `PASS` means all four preregistered diagnostic/sealed PSNR/SSIM checks pass;
- `FAIL` means at least one check fails, returns exit status 1, and still
  preserves the complete receipt and renders.

Input, provider, rendering, or structural errors return exit status 2 and do
not publish a completed result. A quality failure is never converted into an
exception before its evidence is committed.

The receipt contains no machine paths. It binds the gate, recipe, profile,
manifest, final run files, relevant implementation files, renderer runtime,
every target identity, every rendered PNG, per-observation metrics, aggregates,
threshold checks, and final status. Candidate and implementation files are
hashed before and after rendering so a mid-run change aborts publication.

## Verify without rerendering

Retain the printed `receipt_id` outside the result directory, for example in a
review record tied to the source commit. It is the external anchor needed to
detect replacement of both a receipt and its self-hash.

```bash
p2g verify-sealed evidence/scene-v1-sealed \
  --run-dir runs/scene-final \
  --gate protocols/scene-v1.gate.json \
  --expected-receipt-id RECEIPT_SHA256
```

Verification is CPU-only: it rechecks the externally retained ID, canonical
receipt logic, gate inputs, complete-run hashes, manifest and scene roles,
implementation hashes, diagnostic evidence, and the exact render inventory.
It does not rerender, so it does not independently reproduce floating-point
metrics.

“Write-once” here is an application guarantee: the evaluator refuses to
overwrite an existing destination and publishes by directory rename.
“Tamper-evident” means changes are detected relative to the externally retained
receipt ID and bound bytes. This is not a cryptographic signature, trusted
timestamp, append-only storage service, or operating-system immutability flag.

## Claim and rights boundary

A PASS applies only to the exact dataset selection, candidate bytes, software
identity, runtime, and thresholds named by the receipt. It is not a general
benchmark, performance result, or guarantee on another capture. Source-data,
likeness, render, trained-model, and redistribution rights remain independent
of the project source license; evaluation outputs are not added to the source
distribution.
