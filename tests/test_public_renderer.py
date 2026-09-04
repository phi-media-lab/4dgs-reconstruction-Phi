# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import importlib
import importlib.metadata
import math
from dataclasses import replace
from enum import Enum
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from p2g.errors import ContractError
from p2g.training import renderer as renderer_module
from p2g.training.config import RendererConfig
from p2g.training.dataset import TrainingBatch
from p2g.training.model import MaterializedGaussians
from p2g.training.renderer import GsplatRenderer

ROOT = Path(__file__).parents[1]


def _materialized(*, count: int = 3, degree: int = 1) -> MaterializedGaussians:
    return MaterializedGaussians(
        means=torch.zeros((count, 3), dtype=torch.float32),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        scales=torch.full((count, 3), 0.1),
        opacities=torch.full((count,), 0.5),
        colors=torch.zeros((count, (degree + 1) ** 2, 3), dtype=torch.float32),
        temporal_activation=torch.ones((count, 1), dtype=torch.float32),
        temporal_sigma=torch.ones((count, 1), dtype=torch.float32),
        time_delta=torch.zeros((count, 1), dtype=torch.float32),
    )


def _batch(*, height: int = 5, width: int = 7) -> TrainingBatch:
    return TrainingBatch(
        observation_id="camera-0/frame-0",
        camera_id="camera-0",
        frame_id=0,
        role="train",
        timestamp=torch.tensor(0.25, dtype=torch.float32),
        rgb=torch.zeros((height, width, 3), dtype=torch.float32),
        intrinsic=torch.tensor(
            [[[20.0, 0.0, width / 2], [0.0, 20.0, height / 2], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        world_to_camera=torch.eye(4, dtype=torch.float32)[None].contiguous(),
    )


def _metadata(*, count: int, height: int, width: int) -> dict[str, Any]:
    projected_leaf = torch.zeros((count, 2), dtype=torch.float32, requires_grad=True)
    return {
        "camera_ids": torch.zeros(count, dtype=torch.int64),
        "gaussian_ids": torch.arange(count, dtype=torch.int64),
        "radii": torch.ones((count, 2), dtype=torch.int32),
        "means2d": projected_leaf + 0.0,
        "depths": torch.ones(count, dtype=torch.float32),
        "opacities": torch.full((count,), 0.5, dtype=torch.float32),
        "tiles_per_gauss": torch.ones(count, dtype=torch.int32),
        "flatten_ids": torch.arange(count * 2, dtype=torch.int32),
        "tile_width": math.ceil(width / 8),
        "tile_height": math.ceil(height / 8),
        "width": width,
        "height": height,
    }


def _bind_cpu(renderer: GsplatRenderer, rasterization: Any) -> object:
    shutter = object()
    provider = SimpleNamespace(
        rasterization=rasterization,
        global_shutter=shutter,
        identity={},
    )
    binding = SimpleNamespace(provider=provider, device=torch.device("cpu"), identity={})
    cast(Any, renderer)._binding = binding
    return shutter


def test_renderer_maps_the_entire_fixed_abi_and_preserves_grad_metadata() -> None:
    materialized = _materialized()
    batch = _batch()
    calls: list[dict[str, Any]] = []

    def rasterization(**arguments: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        calls.append(arguments)
        rendered = torch.zeros((1, batch.height, batch.width, 3), dtype=torch.float32)
        rendered[0, 0, 0, 0] = -0.25
        rendered[0, 0, 0, 1] = 1.25
        alpha = torch.full((1, batch.height, batch.width, 1), 0.4, dtype=torch.float32)
        return (
            rendered,
            alpha,
            _metadata(
                count=materialized.means.shape[0],
                height=batch.height,
                width=batch.width,
            ),
        )

    renderer = GsplatRenderer(RendererConfig())
    shutter = _bind_cpu(renderer, rasterization)
    result = renderer.render_materialized(materialized, batch, sh_degree=1)

    assert len(calls) == 1
    arguments = calls[0]
    expected_keywords = set(
        cast(tuple[str, ...], cast(Any, renderer_module)._RASTERIZATION_KEYWORDS)
    )
    assert set(arguments) == expected_keywords
    assert arguments["means"] is materialized.means
    assert arguments["quats"] is materialized.quaternions
    assert arguments["colors"] is materialized.colors
    assert arguments["viewmats"] is batch.world_to_camera
    assert arguments["Ks"] is batch.intrinsic
    assert arguments["backgrounds"].shape == (3,)
    assert arguments["backgrounds"].dtype == torch.float32
    assert arguments["rolling_shutter"] is shutter
    assert arguments["packed"] is True
    assert arguments["tile_size"] == 8
    assert arguments["render_mode"] == "RGB"
    assert arguments["camera_model"] == "pinhole"
    assert arguments["distributed"] is False
    assert arguments["with_ut"] is False
    assert arguments["with_eval3d"] is False
    assert result.image.shape == (batch.height, batch.width, 3)
    assert result.alpha.shape == (batch.height, batch.width, 1)
    assert float(result.image.min()) == 0.0
    assert float(result.image.max()) == 1.0
    assert result.materialized is materialized
    assert result.aux["width"] == batch.width
    assert result.aux["height"] == batch.height
    assert result.aux["n_cameras"] == 1
    assert result.aux["renderer_abi"] == "p2g.gsplat_rocm.v1"
    assert cast(torch.Tensor, result.aux["means2d"]).retains_grad


def test_renderer_can_leave_final_rgb_unclamped_without_changing_alpha() -> None:
    materialized = _materialized(degree=0)
    batch = _batch()

    def rasterization(**_: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return (
            torch.full((1, batch.height, batch.width, 3), 1.5),
            torch.full((1, batch.height, batch.width, 1), 1.25),
            _metadata(count=materialized.means.shape[0], height=batch.height, width=batch.width),
        )

    renderer = GsplatRenderer(RendererConfig(clamp_rgb=False))
    _bind_cpu(renderer, rasterization)
    result = renderer.render_materialized(materialized, batch, sh_degree=0)

    assert torch.equal(result.image, torch.full_like(result.image, 1.5))
    assert torch.equal(result.alpha, torch.full_like(result.alpha, 1.25))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda state, batch: (replace(state, colors=state.colors[:, :3]), batch),
            "materialized.colors must have shape",
        ),
        (
            lambda state, batch: (replace(state, means=state.means.to(torch.float64)), batch),
            "materialized.means must use torch.float32",
        ),
        (
            lambda state, batch: (
                state,
                replace(batch, rgb=torch.zeros((7, 5, 3)).transpose(0, 1)),
            ),
            "batch.rgb must be contiguous",
        ),
        (
            lambda state, batch: (
                state,
                replace(batch, intrinsic=batch.intrinsic.requires_grad_(True)),
            ),
            "does not expose camera gradients",
        ),
        (
            lambda state, batch: (
                state,
                replace(batch, radial_coeffs=torch.zeros((1, 6), dtype=torch.float32)),
            ),
            "offline-undistorted pinhole",
        ),
    ],
)
def test_renderer_rejects_out_of_profile_inputs_before_provider_call(
    mutate: Any,
    message: str,
) -> None:
    calls = 0

    def rasterization(**_: Any) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("must not be called")

    renderer = GsplatRenderer(RendererConfig())
    _bind_cpu(renderer, rasterization)
    materialized, batch = mutate(_materialized(), _batch())
    with pytest.raises(ContractError, match=message):
        renderer.render_materialized(materialized, batch, sh_degree=1)
    assert calls == 0


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda rgb, alpha, metadata: (rgb[:, :-1], alpha, metadata), "unexpected RGB shape"),
        (
            lambda rgb, alpha, metadata: (rgb, alpha[..., 0], metadata),
            "unexpected alpha shape",
        ),
        (
            lambda rgb, alpha, metadata: (
                rgb,
                alpha,
                {name: value for name, value in metadata.items() if name != "means2d"},
            ),
            "metadata means2d is missing",
        ),
        (
            lambda rgb, alpha, metadata: (rgb, alpha, {**metadata, "tile_width": 99}),
            "tile_width metadata differs",
        ),
    ],
)
def test_renderer_rejects_provider_output_outside_the_abi(corrupt: Any, message: str) -> None:
    materialized = _materialized()
    batch = _batch()

    def rasterization(**_: Any) -> object:
        rgb = torch.zeros((1, batch.height, batch.width, 3), dtype=torch.float32)
        alpha = torch.zeros((1, batch.height, batch.width, 1), dtype=torch.float32)
        metadata = _metadata(
            count=materialized.means.shape[0], height=batch.height, width=batch.width
        )
        return corrupt(rgb, alpha, metadata)

    renderer = GsplatRenderer(RendererConfig())
    _bind_cpu(renderer, rasterization)
    with pytest.raises(ContractError, match=message):
        renderer.render_materialized(materialized, batch, sh_degree=1)


class _Shutter(Enum):
    GLOBAL = "global"


def _signature_fixture(
    means: object,
    quats: object,
    scales: object,
    opacities: object,
    colors: object,
    viewmats: object,
    Ks: object,
    width: object,
    height: object,
    near_plane: object = 0.01,
    far_plane: object = 1.0e10,
    radius_clip: object = 0.0,
    eps2d: object = 0.3,
    sh_degree: object = None,
    packed: object = True,
    tile_size: object = 8,
    backgrounds: object = None,
    render_mode: object = "RGB",
    sparse_grad: object = False,
    absgrad: object = False,
    rasterize_mode: object = "classic",
    channel_chunk: object = 32,
    distributed: object = False,
    camera_model: object = "pinhole",
    segmented: object = False,
    covars: object = None,
    with_ut: object = False,
    with_eval3d: object = False,
    radial_coeffs: object = None,
    tangential_coeffs: object = None,
    thin_prism_coeffs: object = None,
    ftheta_coeffs: object = None,
    rolling_shutter: object = _Shutter.GLOBAL,
    viewmats_rs: object = None,
) -> object:
    del (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        near_plane,
        far_plane,
        radius_clip,
        eps2d,
        sh_degree,
        packed,
        tile_size,
        backgrounds,
        render_mode,
        sparse_grad,
        absgrad,
        rasterize_mode,
        channel_chunk,
        distributed,
        camera_model,
        segmented,
        covars,
        with_ut,
        with_eval3d,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        rolling_shutter,
        viewmats_rs,
    )
    return None


class _FakeDistribution:
    def __init__(self, root: Path, *, version: str | None = None) -> None:
        self.root = root
        self.version = version or renderer_module.AMD_GSPLAT_VERSION
        self.metadata = {"Name": "amd_gsplat"}
        self.files = [PurePosixPath("gsplat/__init__.py"), PurePosixPath("gsplat/csrc.so")]

    def locate_file(self, item: object) -> Path:
        return self.root / str(item)


def _provider_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str | None = None,
    shadow_module: bool = False,
) -> None:
    package = tmp_path / "site" / "gsplat"
    package.mkdir(parents=True)
    module_file = package / "__init__.py"
    native_file = package / "csrc.so"
    module_file.write_text("# fixture\n", encoding="utf-8")
    native_file.write_bytes(b"fixture")
    if shadow_module:
        module_file = tmp_path / "shadow" / "gsplat.py"
        module_file.parent.mkdir()
        module_file.write_text("# shadow\n", encoding="utf-8")
    module = SimpleNamespace(
        __file__=str(module_file),
        __version__=renderer_module.AMD_GSPLAT_MODULE_VERSION,
        rasterization=_signature_fixture,
    )
    backend = SimpleNamespace(_C=SimpleNamespace(__file__=str(native_file)))
    distribution = _FakeDistribution(tmp_path / "site", version=version)
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _: cast(Any, distribution),
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: {"gsplat": module, "gsplat.cuda._backend": backend}[name],
    )


def test_provider_loader_binds_exact_distribution_module_native_file_and_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provider_fixture(tmp_path, monkeypatch)

    provider = cast(Any, renderer_module)._load_provider()

    assert provider.rasterization is _signature_fixture
    assert provider.global_shutter is _Shutter.GLOBAL
    assert provider.identity == {
        "distribution": "amd-gsplat",
        "version": renderer_module.AMD_GSPLAT_VERSION,
        "module_version": "1.5.3",
        "source_revision": renderer_module.AMD_GSPLAT_REVISION,
    }


@pytest.mark.parametrize(
    ("version", "shadow", "message"),
    [
        ("1.5.3", False, "renderer requires amd-gsplat"),
        (None, True, "not owned by amd-gsplat"),
    ],
)
def test_provider_loader_rejects_wrong_version_or_import_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str | None,
    shadow: bool,
    message: str,
) -> None:
    _provider_fixture(tmp_path, monkeypatch, version=version, shadow_module=shadow)

    with pytest.raises(ContractError, match=message):
        cast(Any, renderer_module)._load_provider()


def test_environment_gate_requires_one_exact_mi300x_and_caches_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    provider = SimpleNamespace(
        rasterization=_signature_fixture,
        global_shutter=_Shutter.GLOBAL,
        identity={
            "distribution": "amd-gsplat",
            "version": renderer_module.AMD_GSPLAT_VERSION,
            "module_version": "1.5.3",
            "source_revision": renderer_module.AMD_GSPLAT_REVISION,
        },
    )

    def load_provider() -> Any:
        nonlocal calls
        calls += 1
        return provider

    monkeypatch.setattr(renderer_module, "_load_provider", load_provider)
    monkeypatch.setattr(renderer_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(renderer_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        renderer_module.torch,
        "__version__",
        renderer_module.EXPECTED_TORCH_VERSION,
    )
    monkeypatch.setattr(renderer_module.torch.version, "hip", renderer_module.EXPECTED_HIP_VERSION)
    monkeypatch.setattr(renderer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(renderer_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        renderer_module.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="AMD Instinct MI300X VF",
            gcnArchName="gfx942:sramecc+:xnack-",
        ),
    )
    renderer = GsplatRenderer(RendererConfig())

    first = renderer.validate_environment("cuda")
    second = renderer.validate_environment("cuda:0")

    assert first == second
    assert calls == 1
    assert first["architecture"] == "gfx942"
    assert first["visible_device_count"] == 1
    assert first["gsplat_distribution"] == renderer_module.AMD_GSPLAT_VERSION
    assert first["renderer_abi"] == "p2g.gsplat_rocm.v1"
    assert not any("/" in str(value) for value in first.values())


def test_environment_and_configuration_gates_fail_before_provider_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ContractError, match="require_gfx942=true"):
        GsplatRenderer(RendererConfig(require_gfx942=False))

    renderer = GsplatRenderer(RendererConfig())
    with pytest.raises(ContractError, match="sole visible device cuda:0"):
        renderer.validate_environment("cpu")
    with pytest.raises(ContractError, match="sole visible device cuda:0"):
        renderer.validate_environment("cuda:1")

    monkeypatch.setattr(renderer_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(renderer_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        renderer_module.torch,
        "__version__",
        renderer_module.EXPECTED_TORCH_VERSION,
    )
    monkeypatch.setattr(renderer_module.torch.version, "hip", renderer_module.EXPECTED_HIP_VERSION)
    monkeypatch.setattr(renderer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(renderer_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        renderer_module,
        "_load_provider",
        lambda: pytest.fail("provider must not load for an invalid device topology"),
    )
    with pytest.raises(ContractError, match="exactly one visible"):
        renderer.validate_environment("cuda")


def test_renderer_source_has_no_reference_bridge_or_hot_value_reads() -> None:
    source = (ROOT / "src/p2g/training/renderer.py").read_text(encoding="utf-8").casefold()
    forbidden = ("free" + "time", "ft" + "gs", "torch.isfinite", ".item()")

    assert not any(token in source for token in forbidden)
