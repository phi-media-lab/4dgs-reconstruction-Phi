# Generated synthetic multiview example

Create the example locally rather than storing opaque image payloads in Git:

```bash
p2g fixture create --output fixture
p2g prepare fixture/observation_manifest.json --output runs/smoke/scene
```

See `docs/SYNTHETIC_FIXTURE.md` for the deterministic format, rights status,
and the strict limit of the resulting smoke-test claim.
