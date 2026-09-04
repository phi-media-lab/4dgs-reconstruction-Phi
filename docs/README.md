# Documentation map

Start with the smallest document that answers the current question:

| Need | Read |
|---|---|
| Install and run a CPU smoke check | [Quickstart](QUICKSTART.md) |
| Understand the end-to-end system | [Architecture](ARCHITECTURE.md) |
| Prepare calibrated observations | [Data contract](DATA_CONTRACT.md) |
| Run or resume the six reconstruction stages | [Pipeline orchestration](PIPELINE_ORCHESTRATION.md) |
| Build the supported ROCm runtime | [MI300X runtime build](MI300X_RUNTIME_BUILD.md) |
| Diagnose GPU occupancy | [MI300X preflight](MI300X_PREFLIGHT_CONTRACT.md) |
| Understand proposals and initialization | [RoMa provider](ROMA_POINT_PROVIDER_CONTRACT.md) and [initialization](INITIALIZATION_STAGE.md) |
| Inspect model, losses, and rasterization | [Model](MODEL_CONTRACT.md), [loss](LOSS_CONTRACT.md), and [renderer](RENDERER_CONTRACT.md) |
| Understand training and population control | [Training](TRAINING_CONTRACT.md) and [relocation](RELOCATION_CONTRACT.md) |
| Run or verify a preregistered sealed gate | [Sealed quality evaluation](SEALED_EVALUATION.md) |
| Verify or render a portable asset | [Asset consumption](ASSET_CONSUMPTION.md) |
| Serve a compatible asset with the sister Viewer | [Viewer interoperability](VIEWER_INTEROP.md) |
| Reproduce and interpret evidence | [Reproducibility](REPRODUCIBILITY.md) |
| Build and verify release archives without publishing | [Release process](RELEASE_PROCESS.md) |
| Diagnose a failed command | [Troubleshooting](TROUBLESHOOTING.md) |
| Understand source, weight, data, and asset rights | [License and provenance](LICENSE_AND_PROVENANCE.md) |
| Exercise the generated smoke input | [Synthetic fixture](SYNTHETIC_FIXTURE.md) |

The detailed mechanism documents intentionally state exact equations and
failure boundaries. The quickstart and architecture guides provide the shorter
user path through the same implementation; they do not define a second API.
