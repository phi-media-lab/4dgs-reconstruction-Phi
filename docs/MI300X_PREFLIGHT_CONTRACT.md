# MI300X occupancy-preflight contract

This contract defines the evidence collected before a GPU stage is allowed to
start. It answers one narrow question: did the three host observation planes
show the selected device below configured capacity limits, and which other
client processes were present?

It does not reserve the GPU, identify the accelerator architecture, validate a
Torch/ROCm ABI, or prove that the device remains idle after the observation.
Those are separate scheduler and runtime-admission responsibilities.

The pipeline follows this snapshot with a continuous `/dev/kfd` resource-window
guard. The two receipts are complementary: preflight sees AMD process/memory
and utilization planes; the guard detects process arrival and state changes
during the stage.

## Inputs and admission modes

`Mi300xPreflightConfig` has seven explicit fields:

| Field | Default | Meaning |
|---|---:|---|
| `gpu_index` | 0 | AMD SMI and `cardN` index to inspect |
| `maximum_gpu_use_percent` | 100.0 | Inclusive utilization limit; quality runs normally do not gate on occupancy |
| `maximum_vram_percent` | 80.0 | Inclusive allocated-VRAM capacity limit |
| `admission_mode` | `shared_quality` | `shared_quality` or `exclusive_performance` |
| `owner_process` | null | Runtime-injected runner PID plus `/proc` start-time identity |
| `allowed_stopped_processes` | empty | Sorted PID/start-time identities that must remain in state `T` |
| `command_timeout_seconds` | 20 | Per-command timeout |

Unknown configuration fields, booleans used as numbers, non-finite values,
unsorted identities, and percentages outside `[0, 100]` fail closed. An
`exclusive_performance` observation cannot contain stopped-process allowances.
The canonical configuration receives a `gpupreflight_...` content ID.

`shared_quality` permits other clients. Their identities and any stopped-client
state change are retained as warnings and set `contention_observed`; they do not
make the stage busy. GPU utilization therefore need not be low, while the VRAM
limit remains an explicit capacity/headroom decision. The runner identity is
start-time bound and a runner mismatch still fails closed. This mode supports
correctness/quality execution, never timing claims.

`exclusive_performance` permits the exact runner but no external KFD/AMD
client. A passing snapshot is only a performance precondition; a scheduler
lease, full-window guard, and postflight remain mandatory.

## Three observation planes

Capture resolves and hashes the exact executable behind each command, then runs
the following commands sequentially:

```text
amd-smi process --gpu N --json
fuser /dev/kfd
rocm-smi --showuse --showmemuse --csv
```

The AMD SMI plane supplies per-device process records and reported VRAM bytes.
The KFD plane catches clients that are absent from the AMD process list. The
ROCm SMI plane supplies GPU-use and allocated-VRAM percentages. The process
inventory is the union of the first two planes; source membership is preserved
per PID.

`fuser` status 0 must accompany at least one parsed PID, and status 1 must
accompany an empty PID inventory. AMD SMI and ROCm SMI must return 0. Duplicate
PIDs, duplicate selected-device rows, malformed nested records, non-finite
metrics, unexpected return codes, and ambiguous device rows are errors rather
than an idle verdict. On stderr, `fuser` may emit only its ordinary `/dev/kfd:`
device label; permission or execution diagnostics are rejected so an access
failure cannot be mistaken for an idle device.

Hard verdict reasons are independent and ordered:

1. `GPU_USE_ABOVE_LIMIT`;
2. `VRAM_ABOVE_LIMIT`; and
3. `OWNER_PROCESS_IDENTITY_MISMATCH`.

In `shared_quality`, `FOREIGN_GPU_PROCESSES_PRESENT` and
`STOPPED_PROCESS_ALLOWANCE_MISMATCH` are warnings. In
`exclusive_performance`, foreign-process findings are hard reasons. Stopped
allowances are not legal in exclusive mode.

The status is `PASS` only when the reason list is empty; otherwise it is
`BUSY`. A stopped identity affects only process classification. It never waives
the utilization or VRAM limits.

## Process metadata and privacy

For the union of observed PIDs, capture takes a best-effort procfs snapshot.
PID, UID, parent PID, state, start time, `comm`, argument count, and a SHA-256 of
the command-line bytes are retained. Only the basenames of the executable and
`argv[0]` are stored. The working directory is deliberately recorded as null,
and command arguments are never persisted. A process may disappear between the
device query and procfs read; this is represented explicitly by nullable fields
instead of dropping the PID.

Raw AMD SMI, ROCm SMI, and `fuser` outputs remain in the local receipt so the
normalization and verdict can be replayed. Users should still treat receipts as
operational telemetry and review them before sharing.

## Receipt and replay

The `p2g.mi300x_preflight.v2` receipt contains:

- tool paths and hashes;
- exact command vectors, return codes, UTC boundaries, and elapsed times;
- raw command outputs and normalized procfs metadata;
- the deterministic evaluation and policy content ID;
- the top-level `PASS` or `BUSY` observation; and
- a canonical logical SHA-256 over every field except the digest itself.

`audit_mi300x_preflight` validates the JSON schema, recomputes the logical hash,
checks command identity and return-code semantics, checks that raw AMD JSON
matches its decoded copy, and independently derives the evaluation again.
Current executable hashes are required for an on-host audit and may be made an
explicitly optional check for offline replay. A `BUSY` receipt can therefore be
authentic and replayable: the audit passes, while the non-required
`admission_passed` check remains visibly failed. A contended shared-quality
receipt is `PASS` with explicit warnings, not a clean-device assertion.

## Full-stage resource window

For `propose`, `train`, and `evaluate`, the orchestrator immediately follows a
passing preflight with `p2g.mi300x_resource_window.v1`. It samples `/dev/kfd`
before the operation, every two seconds, and after the operation. The guard:

- admits the exact runner and live descendants of that start-time-bound runner;
- identifies registered stopped identities and any state/identity change;
- records unrelated KFD processes or resumed/reused stopped identities as
  contention in `shared_quality`, while continuing the stage;
- treats unrelated KFD processes as violations in `exclusive_performance`;
- hashes every normalized observation into a chain while expanding only state
  transitions; and
- rejects an observation gap above 15 seconds or any capture failure.

A hard violation prevents publication of the stage record and atomically moves
any stage output into the hash-inventoried quarantine. A contended shared-quality
receipt remains valid quality evidence but makes all elapsed-time, throughput,
occupancy, and bandwidth observations inadmissible. Polling narrows the race but
cannot prove exclusivity between samples, so performance evidence still needs
an external allocation.

## Integration boundary

`p2g doctor` captures only the replayable preflight. `p2g run` additionally
requires and binds the passing resource-window receipt for every GPU stage.
Neither mechanism is interpreted as a lock.

Exact MI300X `gfx942`, Python, Torch, HIP, native-extension, and code-object
identity is enforced by the renderer/runtime admission described in
`MI300X_RUNTIME_BUILD.md`. Keeping that validation separate prevents an idle
but unsupported GPU from being mistaken for a valid execution platform.

All parser, union, identity, process-tree, dynamic-arrival, privacy, capture,
schema, tamper, and replay tests use synthetic command output and procfs
fixtures. They neither import Torch nor open a GPU runtime.
