# Synthetic multiview contract fixture

`p2g fixture create` generates a tiny, calibrated RGB scene without downloading
media, weights, or model outputs. Its purpose is to test installation, public
artifact contracts, role isolation, and the `prepare` boundary.

```bash
p2g fixture create --output fixture
p2g prepare fixture/observation_manifest.json --output runs/smoke/scene
```

The default fixture contains three pinhole cameras and three synchronized
frames at 32×24 pixels. The first two frames have the `train` role; the last
frame is `diagnostic`. Pixel bytes are produced by a bounded integer algorithm,
PNG containers use fixed uncompressed DEFLATE blocks, and two runs with the
same arguments produce identical files.

The output is append-only:

```text
fixture/
├── fixture.json
├── observation_manifest.json
└── images/
    ├── cam000/
    ├── cam001/
    └── cam002/
```

`fixture.json` records every image byte count and SHA-256 digest, the generation
parameters, the observation-manifest digest, and a path-free claim boundary.
`observation_manifest.json` is a normal `p2g.observation_manifest.v2` input and
is verified against the generated image bytes before preparation.

## Claim and rights boundary

This is a contract fixture, not a reconstruction-quality benchmark. Its tiny
images do not prove useful RoMa matches, convergence, visual quality, or MI300X
performance.

The generator contains no third-party payload. Generated manifests and
receipts now use `license: Apache-2.0`, `license_status: declared`, and
`publication_status: source_fixture_redistributable`. This grant covers the
synthetic fixture only; it does not attach to user-provided inputs or to a
real-scene trained asset.
