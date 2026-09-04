# Changelog

## 0.1.0.dev0 - 2026-09-04

- Added the six-stage `prepare → propose → initialize → train → evaluate →
  asset` reconstruction pipeline for one AMD Instinct MI300X.
- Added a continuous-time dynamic Gaussian model, fixed-budget relocation,
  temporal gating, color correction, explicit losses, and the admitted
  ROCm/gsplat renderer.
- Added calibrated observation, tensor-cache, proposal, initialization,
  checkpoint, evaluation, AssetBundle, camera-path, and render contracts.
- Added offline Charge and SelfCap-style input adapters without bundled media.
- Added deterministic synthetic fixtures and a complete CPU contract suite.
- Added reproducible wheel/sdist checks, source hygiene checks, Apache-2.0
  licensing, third-party notices, citation metadata, and a CycloneDX SBOM.
