# Public MI300X training contract

This document defines the optimization loop between a completed
`p2g.gaussian_initialization.v1` artifact and a portable AssetBundle. It is a
runtime and artifact contract, not a scene-quality claim.

## Input closure before GPU setup

`p2g train` accepts one resolved `p2g.resolved_run.v1` configuration. Before
constructing a model or loading the renderer, the trainer requires all of the
following:

1. an audited observation manifest;
2. its `p2g.tensor_cache.v1` directory with transport verification enabled;
3. `initialization.safetensors`; and
4. the sibling `initialization.json` receipt produced by the first-party
   initializer.

The trainer recomputes the byte hashes of the observation manifest, tensor-cache
manifest, initialization tensor, and initialization receipt. It validates the
two generated JSON files, recomputes the receipt's logical hash, and compares
the receipt with the Safetensors metadata. The resulting
`p2g.training_input_binding.v1` record includes the upstream proposal-sequence
identity without storing a machine path.

Training stops before GPU setup if any edge disagrees:

```text
observation manifest hash
          │
          ├── tensor_cache.json
          │         │
          │         └── initialization.json
          │                    │
          └────────────────────┴── initialization.safetensors metadata
```

Raw-image training and disabled tensor transport verification are outside the
first public MI300X execution contract. The ordinary dataset loader then
performs the deeper image, array, calibration, timestamp, and role audit before
the first optimization step.

## Observation capabilities

Only indices in `PreparedScene.train_indices` enter `SceneSampler`. The loop
also checks `batch.role == "train"` after every sample, before rendering.

- `train`: optimization and formation screen-guard observations;
- `diagnostic`: periodic evaluation only;
- `sealed`: never loaded by the trainer or routine evaluator;
- `free_view`: never loaded by the trainer or routine evaluator.

Per-camera affine nuisance parameters, when enabled, are allocated only for
cameras present in the admitted training partition. Diagnostic results use the
uncorrected model output so they measure the representation that can actually
be placed in an AssetBundle.

## One optimization step

For completed-step index `s + 1`, the order is fixed:

1. sample one train observation from the checkpointed deterministic sampler;
2. materialize the 4D Gaussian state at that observation time;
3. render through the registered `gfx942` gsplat provider;
4. form the declared reconstruction and regularization terms;
5. backpropagate once;
6. reject a missing or non-finite trainable gradient before mutation;
7. update every per-plane optimizer and learning-rate schedule;
8. apply the configured fixed-budget relocation event, if scheduled; and
9. apply the formation screen-influence guard, if scheduled.

The Gaussian population is fixed. A population-control implementation may move
and reinitialize slots, optimizer rows, and lineage state, but may not silently
change tensor capacity or runtime IDs.

The public `fixed_budget_relocation_v1` source-selection rule, capacity bound,
peak and far-time alpha conservation, projected-alpha-mass equation, temporal
residual, and optimizer-row mutation are specified in
[the relocation contract](RELOCATION_CONTRACT.md). It is an independent
mechanism, not a compatibility path.

Active SH degree is

```text
min(floor(s / sh_degree_interval), model.max_sh_degree).
```

The public sampling names describe behavior rather than a reference codebase:

- `shuffled_epoch` visits a shuffled permutation of train observations; and
- `frame_camera_with_replacement` samples a train frame and then one train
  camera from that frame, both with replacement.

## MI300X hot-loop policy

The trainer contains no unconditional `torch.cuda.synchronize()` in the
per-step loop. It makes one host decision per step for the finite-loss/gradient
guard. Loss values, PSNR, gradient norms, visibility counts, and memory counters
are copied to the host only for:

- the first step;
- `log_every` steps;
- checkpoint or diagnostic-evaluation steps;
- relocation or screen-guard events; and
- the final step.

This produces a sparse, ordered metric journal instead of synchronizing the
MI300X repeatedly for every diagnostic scalar. The journal is canonical JSONL;
every row carries both `step` and `next_step` and identifies the sampled role.
This policy reduces avoidable host serialization but is not itself a throughput
claim. Release performance must be measured on the registered MI300X runtime.

## Checkpoint and resume transaction

The run directory is append-only and is created through a staged directory
rename. A new run never reuses an existing path. `config.toml` is the resolved
configuration saved inside that directory.

At a checkpoint, the trainer flushes and fsyncs the metric prefix before
publishing the hash-closed local checkpoint. Diagnostic evaluation also forces
a checkpoint at the same completed step. Therefore every published periodic
evaluation has an exact resume state.

Resume accepts only the numerically latest checkpoint below the selected run,
and the checkpoint configuration must equal the requested configuration. RNG,
sampler, optimizer, model, color-correction, and relocation state are restored.
Metric rows newer than the checkpoint are removed atomically; older rows must be
canonical and strictly ordered, but need not contain every step. If a process
stopped after publishing an evaluation checkpoint but before publishing its
evaluation, resume completes that evaluation before further optimization.

If `training.json` is already present, resume verifies the completion receipt
and every artifact it binds, then returns before runtime or GPU initialization.
Conversely, final model files that exist without a valid completion receipt are
an unbound publication conflict: the trainer neither adopts nor overwrites
them. This fail-closed boundary makes a hard crash during final model publication
visible instead of silently blessing ambiguous bytes.

Local `state.pt` remains a trusted local resume envelope loaded with
`weights_only=True`. It is not an inference or redistribution format.

## Completed run layout

The important outputs are:

```text
RUN/
├── config.toml
├── runtime.json
├── metrics.jsonl
├── checkpoints/step_NNNNNNNN/
├── renders/step_NNNNNNNN/evaluation.json
├── model.safetensors
├── model.json
└── training.json
```

`runtime.json` records the path-free upstream binding, role counts, renderer
identity, screen-guard configuration, and synchronization policy. A final
diagnostic evaluation is produced even when the configured periodic cadence
does not land on the final step.

`training.json` is published last. Its logical hash covers the input binding,
completed step, claim boundary, and hashes of the runtime record, metric
journal, final checkpoint manifest, local model export, and final diagnostic
evaluation. Its presence means the mechanical training transaction completed;
it does not mean a sealed quality or redistribution gate passed.

## Sealed quality is a separate terminal action

Routine checkpoint and exported-run evaluation remains diagnostic-only. A
sealed role can be read only through the preregistered terminal evaluator after
the complete run and tensor-only export have passed their existing hash
closure. The gate fixes candidate shape, roles, metric equations, and quality
floors before access; PASS and quality FAIL are both published as write-once,
externally anchorable receipts. See
[Sealed quality evaluation](SEALED_EVALUATION.md) for the exact command,
receipt inventory, verification procedure, and claim limits.

## Asset publication is separate and retryable

Training first emits a local Safetensors model. Converting that result into an
AssetBundle is a separate command so a rights assertion or filesystem failure
cannot make a completed optimization run unusable:

```text
p2g asset export RUN \
  --output ASSET \
  --producer-git-revision FULL_40_CHARACTER_REVISION \
  --asset-license SPDX_OR_LICENSE_ASSERTION \
  --redistribution review_required \
  --provenance-summary "How this derived asset was produced"
```

The exporter verifies `training.json` and every file hash again, loads only the
safe local model export, derives the train-role time interval and source-data
license from the observation manifest, and binds the upstream, checkpoint, and
training receipts into the AssetBundle metadata. The producer revision and
derived-asset rights are explicit user assertions because they cannot be
inferred safely from a Python environment.

## Commands

Start a new run:

```text
p2g train resolved-config.toml --output RUN
```

Resume only from the selected run's latest checkpoint:

```text
p2g train resolved-config.toml \
  --output RUN \
  --resume-checkpoint RUN/checkpoints/step_NNNNNNNN
```

Expected contract, hash, runtime, and existing-output failures return status 2.
Help remains lazy and does not import Torch.

## Claim boundary

CPU tests can prove orchestration order, source binding, role exclusion,
checkpoint publication, sparse-journal recovery, and AssetBundle handoff. They
cannot prove ROCm kernel availability, numerical parity, GPU occupancy,
convergence, visual quality, or relocation quality. Those require separate
MI300X execution and preregistered held-out evaluation.
