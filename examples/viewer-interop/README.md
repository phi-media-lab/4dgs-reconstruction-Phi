# Viewer-interoperable MI300X profile

`profile.toml` is a complete Pixel4DGS portable profile whose representation
and raster settings satisfy the currently documented 4DGS Viewer Phi bridge
profile. It differs from the ordinary defaults by enabling learned persistence
and using the reference gate scale of 20; SH3, a zero radius clip, clamped RGB,
and the `gfx942` renderer remain explicit rather than implicit.

This is an interoperability example, not a universal quality preset. Dataset
selection, initialization size, loss weights, color correction, relocation,
cache size, and evaluation policy still require scene-specific review. The
example deliberately leaves relocation and screen-guard correction disabled.

Reference it from a pipeline plan:

```toml
profile = "examples/viewer-interop/profile.toml"

[asset]
default_sh_degree = 3
```

Relative plan paths resolve from the plan file, so adjust `profile` for its
actual location. Follow
[`docs/VIEWER_INTEROP.md`](../../docs/VIEWER_INTEROP.md) for export, camera-path
binding, conversion, and the exact claim boundary.
