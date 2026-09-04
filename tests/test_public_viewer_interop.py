from __future__ import annotations

from pathlib import Path

from p2g.training.config import PortableProfile

ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "examples" / "viewer-interop" / "profile.toml"
VIEWER_URL = "https://github.com/phi-media-lab/4dgs-viewer-Phi"


def test_viewer_interop_profile_is_complete_canonical_and_loadable() -> None:
    profile = PortableProfile.load(PROFILE)

    assert profile.initialization.sh_degree == 3
    assert profile.model.persistence == "learned"
    assert profile.model.gate_logit_scale == 20.0
    assert profile.renderer.backend == "gsplat_rocm"
    assert profile.renderer.radius_clip == 0.0
    assert profile.renderer.clamp_rgb is True
    assert profile.renderer.require_gfx942 is True
    assert PROFILE.read_bytes() == profile.to_toml_bytes()


def test_public_docs_expose_the_viewer_handoff_without_private_evidence() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "docs" / "ARCHITECTURE.md",
        ROOT / "docs" / "ASSET_CONSUMPTION.md",
        ROOT / "docs" / "QUICKSTART.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "VIEWER_INTEROP.md",
        ROOT / "examples" / "viewer-interop" / "README.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert all(path.is_file() and not path.is_symlink() for path in paths)
    assert VIEWER_URL in combined
    assert "p2g.asset_bundle.v1" in combined
    assert "phi.4dgs.explicit.v1" in combined
    assert "learned persistence" in combined
    assert "cross-renderer pixel parity" in combined
    for forbidden in ("/home/", "/mnt/", "/private/", "phi" + "-amd"):
        assert forbidden not in combined
