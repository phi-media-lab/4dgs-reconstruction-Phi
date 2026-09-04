# License and provenance

Repository-owned source, documentation, tests, build files, and the generated
synthetic fixture are licensed under Apache-2.0. The operative terms are in
[`LICENSE`](../LICENSE); project attribution is in [`NOTICE`](../NOTICE).

That grant applies only to material owned by Pixel4DGS contributors. It does
not relicense a dependency, model weight, dataset, captured person, trained
asset, rendered image, or video.

## Independent rights layers

| Layer | Examples | Required evidence |
|---|---|---|
| Project source | Python, tests, docs, build scripts | project `LICENSE`, `NOTICE`, source identity |
| Third-party code | PyTorch, AMD gsplat, GLM, RoMa | exact revision and upstream license/notice |
| Model weights | RoMa, DINOv2, AlexNet checkpoints | exact hash and weight-specific terms |
| Input data | camera frames and calibration | dataset, privacy, likeness, and redistribution rights |
| Derived output | Gaussian assets, images, videos | input-derived rights review and an explicit output license |

Permission at one layer does not imply permission at another. In particular,
the Apache-2.0 project license does not automatically license external weights
or an output derived from user-supplied footage.

## Included and excluded material

The source distribution includes first-party Python, schemas, tests,
documentation, build recipes, and small text patches. It contains no Python
wheels, native source archives, model weights, datasets, checkpoints, trained
Gaussian payloads, metric results, images, or preview videos.

FreeTimeGS++ and 3DGS-MCMC are research references, not runtime dependencies.
Their source code is not included. Pixel4DGS population control is an
independent implementation with its own equations and tests.

## Runtime and provider handling

| Component | Handling |
|---|---|
| AMD Ecosystem gsplat | build pinned Apache-2.0 source; preserve upstream `LICENSE` and `NOTICE.txt` |
| GLM | fetch the pinned revision, select its MIT option, preserve `copying.txt` |
| PyTorch/torchvision ROCm | install external wheels; preserve their terms if redistributing binaries |
| RoMa code | install pinned MIT source and preserve attribution |
| DINOv2 weight | external, hash-verified, Apache-2.0 declared by its provider; not bundled |
| RoMa indoor weight | external, hash-verified, `NOASSERTION`; not bundled |
| TorchMetrics LPIPS | external Apache-2.0 distribution with hash-checked provider files |
| AlexNet feature weight | external, hash-verified, `NOASSERTION`; not bundled |
| fused-SSIM | pinned MIT optional experiment; not in the default admitted path |

Exact identities and hashes are recorded in the runtime manifest and provider
registries. Provider code refuses automatic weight download and requires the
user to supply matching local files. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and
[`sbom.cdx.json`](../sbom.cdx.json).

The CycloneDX file is a pre-build SBOM and declares its dependency composition
incomplete. Resolved transitive packages, system ROCm components, external
weights, datasets, and generated artifacts belong to environment- or
run-specific inventories.

## Synthetic fixture

`p2g fixture create` generates deterministic pixels without downloading or
embedding third-party payload. Its manifest declares Apache-2.0, matching the
project source. It checks installation and data contracts, not reconstruction
quality.

## External data adapters

The SelfCap adapter is source code, not a redistribution of a capture. It
records the upstream restricted license reference in generated manifests and
does not authorize redistribution of input frames or derived output. Users
must obtain and review their own input.

The Charge adapter likewise operates on a user-provided local copy and bundles
no pixels. For the documented example, preserve the exact scene revision,
acquisition date, selected modalities, change history, the
[Charge paper](https://arxiv.org/abs/2512.13639),
[project page](https://charge-benchmark.github.io/),
[Blender Charge credits](https://studio.blender.org/projects/charge/pages/credits/),
and [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/). The
fixed scene shard is mapped through the
[official Charge release](https://huggingface.co/charge-benchmark/Charge/commit/6c0255d5a4c3e87d334f79d737c846295187fbdd),
whose metadata declares CC BY 4.0.

If Charge material or a derivative is shared, retain the attribution,
copyright, license, disclaimer, source, and modification notices required by
its terms. Do not imply endorsement or use Charge/Blender branding as project
branding.

## Derived assets

`p2g asset export` requires an explicit asset-license assertion and records the
source-data license. This is provenance metadata, not an automatic grant. The
publisher remains responsible for checking the input, likeness, trademark,
weight, and output layers before distributing a trained AssetBundle or render.
