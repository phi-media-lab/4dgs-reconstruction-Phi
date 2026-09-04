from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

from p2g.errors import ContractError, OutputExistsError
from p2g.training.config import (
    PROFILE_SCHEMA,
    RESOLVED_RUN_SCHEMA,
    SCENE_INPUTS_SCHEMA,
    DataPolicyConfig,
    InitializationPolicyConfig,
    OptimizerConfig,
    PortableProfile,
    RelocationConfig,
    RendererConfig,
    RunConfig,
    SceneInputs,
    TrainingConfig,
)

ROOT = Path(__file__).parents[1]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_default_portable_profile_is_deterministic_and_standard_toml() -> None:
    profile = PortableProfile()

    first = profile.to_toml_bytes()
    second = profile.to_toml_bytes()
    decoded = tomllib.loads(first.decode("utf-8"))

    assert first == second
    assert first.endswith(b"\n")
    assert decoded["schema_version"] == PROFILE_SCHEMA
    assert decoded["data"]["train_roles"] == ["train"]
    assert decoded["training"]["device"] == "cuda"
    assert decoded["renderer"]["backend"] == "gsplat_rocm"


def test_equivalent_mapping_order_has_identical_serialization() -> None:
    default = PortableProfile()
    reversed_optimizer = OptimizerConfig(
        lrs=dict(reversed(tuple(default.optimizer.lrs.items()))),
        lr_final_factors=dict(reversed(tuple(default.optimizer.lr_final_factors.items()))),
    )

    assert PortableProfile(optimizer=reversed_optimizer).to_toml_bytes() == (
        default.to_toml_bytes()
    )


def test_profile_and_scene_inputs_resolve_to_a_roundtrippable_run(tmp_path: Path) -> None:
    profile_path = _write(
        tmp_path / "profiles" / "quality.toml",
        f'''schema_version = "{PROFILE_SCHEMA}"

[data]
train_roles = ["train"]
eval_roles = ["diagnostic"]

[training]
iterations = 1000
sampling = "frame_camera_with_replacement"

[training.relocation]
mode = "fixed_budget_relocation_v1"
start = 100
stop = 900
every = 100
''',
    )
    scene_path = _write(
        tmp_path / "scenes" / "capture.toml",
        f'''schema_version = "{SCENE_INPUTS_SCHEMA}"
manifest = "metadata/observations.json"
initialization = "initialization/gaussians.safetensors"
image_root = "images"
''',
    )

    config = RunConfig.from_files(profile=profile_path, scene=scene_path)

    assert config.data.manifest == (scene_path.parent / "metadata/observations.json").resolve()
    assert config.data.image_root == (scene_path.parent / "images").resolve()
    assert (
        config.initialization.path
        == (scene_path.parent / "initialization/gaussians.safetensors").resolve()
    )
    assert config.data.eval_roles == ("diagnostic",)
    assert config.training.relocation.mode == "fixed_budget_relocation_v1"

    resolved_path = tmp_path / "run" / "config.toml"
    config.save(resolved_path)
    reloaded = RunConfig.load(resolved_path)

    assert reloaded == config
    assert reloaded.to_toml_bytes() == resolved_path.read_bytes()
    assert tomllib.loads(resolved_path.read_text(encoding="utf-8"))["schema_version"] == (
        RESOLVED_RUN_SCHEMA
    )
    with pytest.raises(OutputExistsError, match="refusing to overwrite"):
        config.save(resolved_path)


def test_scene_tensor_inventory_roundtrips_with_absolute_resolution(tmp_path: Path) -> None:
    scene_path = _write(
        tmp_path / "capture" / "scene.toml",
        f'''schema_version = "{SCENE_INPUTS_SCHEMA}"
manifest = "observations.json"
initialization = "init.safetensors"

[tensor_memmap]
root = "tensors"
camera_ids = ["cam-a", "cam-b"]
frame_ids = [0, 2, 4]
verify_transport_sha256 = true
''',
    )

    scene = SceneInputs.load(scene_path)

    assert scene.tensor_memmap is not None
    assert scene.tensor_memmap.root == (scene_path.parent / "tensors").resolve()
    assert scene.tensor_memmap.camera_ids == ("cam-a", "cam-b")
    assert scene.tensor_memmap.frame_ids == (0, 2, 4)
    saved = tmp_path / "resolved-scene.toml"
    scene.save(saved)
    assert SceneInputs.load(saved) == scene


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[data]\ntrain_roles = ["sealed"]\n', "only the train role"),
        ('[data]\neval_roles = ["sealed"]\n', "fixed to the diagnostic role"),
        ('[data]\neval_roles = ["free_view"]\n', "fixed to the diagnostic role"),
        ('[data]\ntrain_roles = [["train"]]\n', "only the train role"),
        ('[data]\nmanifest = "private.json"\n', "unknown fields"),
        ("[renderer]\ntile_size = 16\n", "tile_size=8"),
        ("[renderer]\npacked = false\n", "packed=true"),
        ('[training]\nsampling = "uniform"\n', "public sampling policy"),
        ('[training.relocation]\nmode = "mcmc"\n', "not a public mode"),
        ('[initialization]\nformat = "external_bridge"\n', "must be p2g_safetensors"),
        ("[loss]\nl1 = inf\n", "finite number"),
    ],
)
def test_profile_rejects_non_public_or_malformed_policy(
    tmp_path: Path, body: str, message: str
) -> None:
    profile_path = _write(
        tmp_path / "profile.toml",
        f'schema_version = "{PROFILE_SCHEMA}"\n\n{body}',
    )

    with pytest.raises(ContractError, match=message):
        PortableProfile.load(profile_path)


@pytest.mark.parametrize(
    "manifest",
    ["~/observations.json", "file:///datasets/observations.json"],
)
def test_scene_paths_reject_implicit_or_uri_resolution(tmp_path: Path, manifest: str) -> None:
    scene_path = _write(
        tmp_path / "scene.toml",
        f'''schema_version = "{SCENE_INPUTS_SCHEMA}"
manifest = "{manifest}"
initialization = "init.safetensors"
''',
    )

    with pytest.raises(ContractError, match="home expansion or a file URI"):
        SceneInputs.load(scene_path)


def test_resolved_objects_reject_ambiguous_programmatic_relative_paths() -> None:
    scene = SceneInputs(
        manifest=Path("observations.json"),
        initialization=Path("initialization.safetensors"),
    )

    with pytest.raises(ContractError, match="resolved absolute path"):
        scene.validate()


def test_schema_and_top_level_fields_are_fail_closed(tmp_path: Path) -> None:
    missing_schema = _write(tmp_path / "missing.toml", "[data]\ndownscale = 1\n")
    unknown_field = _write(
        tmp_path / "unknown.toml",
        f'schema_version = "{PROFILE_SCHEMA}"\nprivate_adapter = true\n',
    )

    with pytest.raises(ContractError, match="must declare"):
        PortableProfile.load(missing_schema)
    with pytest.raises(ContractError, match="unknown fields"):
        PortableProfile.load(unknown_field)


def test_direct_configuration_validation_has_exact_public_enums_and_optimizer_keys() -> None:
    with pytest.raises(ContractError, match="tile_size=8"):
        RendererConfig(tile_size=16).validate()
    with pytest.raises(ContractError, match="packed=true"):
        RendererConfig(packed=False).validate()
    with pytest.raises(ContractError, match="every public model parameter exactly"):
        OptimizerConfig(lrs={"means": 1.0}).validate()
    with pytest.raises(ContractError, match="public sampling policy"):
        TrainingConfig(sampling=cast(Any, "uniform")).validate()
    with pytest.raises(ContractError, match="not a public mode"):
        RelocationConfig(mode=cast(Any, "mcmc")).validate(iterations=30_000)
    with pytest.raises(ContractError, match="must be p2g_safetensors"):
        InitializationPolicyConfig(format=cast(Any, "external_bridge")).validate()


def test_role_contract_never_raises_an_untyped_collection_error() -> None:
    malformed = DataPolicyConfig(train_roles=cast(Any, (["train"],)))

    with pytest.raises(ContractError, match="only the train role"):
        malformed.validate()


def test_public_config_source_has_no_bridge_specific_runtime_dependency() -> None:
    source = (ROOT / "src/p2g/training/config.py").read_text(encoding="utf-8").casefold()
    forbidden = ("msg" + "spec", "free" + "time", "ft" + "gs")

    assert not any(token in source for token in forbidden)
