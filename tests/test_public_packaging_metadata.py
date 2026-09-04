from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_project_metadata_is_narrow_and_does_not_resolve_generic_torch() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["build-system"] == {
        "requires": ["setuptools>=78.1,<79", "wheel>=0.45,<0.46"],
        "build-backend": "setuptools.build_meta",
    }
    project = metadata["project"]
    assert project["name"] == "pixel4dgs"
    assert project["version"] == "0.1.0.dev0"
    assert project["requires-python"] == ">=3.12,<3.13"
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]
    assert project["urls"] == {
        "Repository": "https://github.com/phi-media-lab/4dgs-reconstruction-phi",
        "Issues": "https://github.com/phi-media-lab/4dgs-reconstruction-phi/issues",
    }
    assert project["scripts"] == {"p2g": "p2g.cli:main"}
    dependencies = project["dependencies"]
    assert dependencies == sorted(dependencies, key=str.casefold)
    assert "imageio>=2.37,<3" in dependencies
    assert "Pillow>=12,<13" in dependencies
    assert "scipy>=1.18,<2" in dependencies
    assert not any(
        requirement.casefold().startswith(("torch", "msgspec")) for requirement in dependencies
    )
    assert project["optional-dependencies"]["selfcap"] == [
        "numpy==2.5.2",
        "opencv-python-headless==5.0.0.93",
        "Pillow==12.3.0",
    ]
    setuptools = metadata["tool"]["setuptools"]
    assert setuptools["package-dir"] == {"": "src"}
    assert setuptools["packages"]["find"]["where"] == ["src"]
    assert setuptools["package-data"]["p2g"] == ["registries/*.json", "schemas/*.json"]
    assert setuptools["data-files"] == {"share/pixel4dgs": ["sbom.cdx.json"]}
    assert (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines() == [
        (
            "include CHANGELOG.md CITATION.cff CONTRIBUTING.md LICENSE NOTICE README.md "
            "SECURITY.md THIRD_PARTY_NOTICES.md"
        ),
        "include sbom.cdx.json",
        "graft docs",
        "graft examples",
        "graft src",
        "graft tests",
        "graft third_party",
        "graft tools",
        "global-exclude __pycache__ *.py[cod]",
    ]


def test_release_license_notice_citation_and_sbom_are_coherent() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    third_party = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    sbom: dict[str, Any] = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))

    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "Pixel4DGS contributors" in notice
    assert "AMD Ecosystem gsplat" in third_party
    assert "external-only" in third_party
    assert "license: Apache-2.0" in citation
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.6"
    assert sbom["metadata"]["component"] == {
        "type": "application",
        "bom-ref": "pkg:pypi/pixel4dgs@0.1.0.dev0",
        "name": "pixel4dgs",
        "version": "0.1.0.dev0",
        "licenses": [{"license": {"id": "Apache-2.0"}}],
    }
    assert sbom["compositions"] == [
        {
            "aggregate": "incomplete",
            "dependencies": ["pkg:pypi/pixel4dgs@0.1.0.dev0"],
        }
    ]


def test_runtime_guide_matches_hash_closed_public_source_manifest() -> None:
    manifest: dict[str, Any] = json.loads(
        (ROOT / "third_party/manifests/mi300x_runtime_v1.json").read_text(encoding="utf-8")
    )
    guide = (ROOT / "docs/MI300X_RUNTIME_BUILD.md").read_text(encoding="utf-8")

    assert manifest["status"] == "PUBLIC_SOURCE_BUILD_VERIFIED"
    support = manifest["support_matrix"]
    for expected in (
        support["python_abi"],
        support["torch"],
        support["hip"],
        support["gpu_architecture"],
    ):
        assert expected in guide
    for component in manifest["components"]:
        assert component["revision"] in guide
        assert component["archive"] in guide
        assert component["archive_sha256"] in guide
        assert component["license"] in guide
    recipe = manifest["recipe"]
    for record in (recipe["fetch_script"], recipe["build_script"], *recipe["patches"]):
        path = ROOT / record["path"]
        assert path.is_file() and not path.is_symlink()
        assert _sha256(path) == record["sha256"]


def test_user_facing_packaging_docs_have_no_internal_machine_dependency() -> None:
    paths = (
        ROOT / "CHANGELOG.md",
        ROOT / "CITATION.cff",
        ROOT / "CONTRIBUTING.md",
        ROOT / "NOTICE",
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / ".github/workflows/release-check.yml",
        ROOT / "pyproject.toml",
        ROOT / "docs/ARCHITECTURE.md",
        ROOT / "docs/ASSET_CONSUMPTION.md",
        ROOT / "docs/DATA_CONTRACT.md",
        ROOT / "docs/INITIALIZATION_STAGE.md",
        ROOT / "docs/LICENSE_AND_PROVENANCE.md",
        ROOT / "docs/MI300X_PREFLIGHT_CONTRACT.md",
        ROOT / "docs/MI300X_RUNTIME_BUILD.md",
        ROOT / "docs/MODEL_CONTRACT.md",
        ROOT / "docs/PIPELINE_ORCHESTRATION.md",
        ROOT / "docs/QUICKSTART.md",
        ROOT / "docs/README.md",
        ROOT / "docs/RELOCATION_CONTRACT.md",
        ROOT / "docs/RELEASE_PROCESS.md",
        ROOT / "docs/REPRODUCIBILITY.md",
        ROOT / "docs/ROMA_POINT_PROVIDER_CONTRACT.md",
        ROOT / "docs/SEALED_EVALUATION.md",
        ROOT / "docs/TRAINING_CONTRACT.md",
        ROOT / "docs/TROUBLESHOOTING.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for forbidden in (
        "/home/",
        "/mnt/",
        "/private/",
        "phi" + "-amd",
    ):
        assert forbidden not in combined
    assert "clean committed" in combined
    assert "Apache-2.0" in combined
    assert "external weights" in combined
    assert "held-out full-scene" in combined
    assert "p2g.tensor_cache.v1" in combined
    assert "p2g.gaussian_initialization.v1" in combined
    assert "p2g.linear_motion_gaussian_gate.v1" in combined
    assert "sealed" in combined


def test_user_facing_local_markdown_links_resolve_inside_the_tree() -> None:
    markdown_files = [
        ROOT / "CHANGELOG.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "examples").rglob("*.md")),
    ]
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    root = ROOT.resolve()
    for markdown in markdown_files:
        for raw_target in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            if raw_target.startswith(("#", "https://", "http://", "mailto:")):
                continue
            relative = raw_target.split("#", 1)[0]
            if not relative:
                continue
            target = (markdown.parent / relative).resolve()
            assert target.is_relative_to(root), (markdown, raw_target)
            assert target.exists(), (markdown, raw_target)


def test_charge_docs_pin_current_attribution_and_disclose_prepare_size() -> None:
    quickstart = (ROOT / "docs/QUICKSTART.md").read_text(encoding="utf-8")
    provenance = (ROOT / "docs/LICENSE_AND_PROVENANCE.md").read_text(encoding="utf-8")
    combined = f"{quickstart}\n{provenance}"
    normalized_quickstart = " ".join(quickstart.split())

    assert "https://arxiv.org/abs/2512.13639" in combined
    assert "https://studio.blender.org/projects/charge/pages/credits/" in combined
    assert "paper_v2_lr.pdf" not in combined

    observations = 381 * 41
    rgb_payload_bytes = observations * 858 * 2048 * 3
    tensor_payload_bytes = rgb_payload_bytes + observations * (3 * 3 * 4 + 4 * 4 * 4 + 8)
    assert rgb_payload_bytes == 82_346_913_792
    assert tensor_payload_bytes == 82_348_600_860
    assert f"{rgb_payload_bytes:,}" in quickstart
    assert f"{tensor_payload_bytes:,}" in quickstart
    assert "Reserve at least 100 GB of free space" in normalized_quickstart
    assert "Stop here if the goal is only input identity" in normalized_quickstart


def test_selfcap_docs_expose_exact_offline_conversion_and_rights_boundary() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    quickstart = (ROOT / "docs/QUICKSTART.md").read_text(encoding="utf-8")
    data_contract = (ROOT / "docs/DATA_CONTRACT.md").read_text(encoding="utf-8")
    provenance = (ROOT / "docs/LICENSE_AND_PROVENANCE.md").read_text(encoding="utf-8")

    assert metadata["project"]["optional-dependencies"]["selfcap"] == [
        "numpy==2.5.2",
        "opencv-python-headless==5.0.0.93",
        "Pillow==12.3.0",
    ]
    assert "p2g data import-selfcap" in quickstart
    assert "--source-start-frame 200" in quickstart
    assert "--frame-count 60" in quickstart
    assert "--diagnostic-camera 0007" in quickstart
    assert "--sealed-camera 0015" in quickstart
    assert "source_start_frame + f + source_fps * s" in data_contract
    assert "rounds half up once" in data_contract
    assert "per-frame, per-camera observation manifest" in (
        ROOT / "README.md"
    ).read_text(encoding="utf-8")
    assert "source code, not a redistribution" in provenance


def test_release_archive_check_is_cpu_only_read_only_and_never_publishes() -> None:
    workflow = (ROOT / ".github/workflows/release-check.yml").read_text(encoding="utf-8")
    checker = ROOT / "tools/release/check_python_distributions.sh"
    script = checker.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert workflow.count('VISIBLE_DEVICES: "-1"') == 3
    assert "build==1.6.0" in workflow
    assert "setuptools==78.1.1" in workflow
    assert "wheel==0.45.1" in workflow
    assert "tools/release/check_python_distributions.sh" in workflow
    assert [
        line.strip().removeprefix("uses: ")
        for line in workflow.splitlines()
        if line.strip().startswith("uses: ")
    ] == [
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1",
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0",
    ]
    for forbidden in (
        "id-token: write",
        "actions/upload-artifact",
        "gh release",
        "twine upload",
        "pypa/gh-action-pypi-publish",
    ):
        assert forbidden not in workflow.casefold()
        assert forbidden not in script.casefold()
    assert checker.stat().st_mode & 0o111 == 0o111
    assert 'git -C "$repository" archive' in script
    assert "SOURCE_DATE_EPOCH" in script
    assert "PYTHONNOUSERSITE=1" in script
    assert "PYTHONSAFEPATH=1" in script
    assert "unset PIP_NO_BUILD_ISOLATION PYTHONOPTIMIZE" in script
    assert "python -m wheel unpack" in script
    assert 'cmp "$direct_wheel" "$isolated_wheel"' in script
    assert 'cmp "$direct_wheel" "$sdist_wheel"' in script
    assert script.index('cmp "$direct_wheel" "$sdist_wheel"') < script.index(
        'python -m wheel unpack --dest "$output/unpack-sdist-wheel"'
    )
    assert 'python "$repository/tools/release/check_python_archives.py"' in script
    assert script.count('--wheel "$') == 2
    assert script.count('--sdist "$') == 2
    assert 'importlib.util.find_spec("torch") is not None' in script
    assert "assert " not in script
