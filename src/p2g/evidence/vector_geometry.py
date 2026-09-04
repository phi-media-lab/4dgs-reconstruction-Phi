from __future__ import annotations

import importlib
import math
from typing import Any, cast

from p2g.errors import ContractError

VECTOR_GEOMETRY_BACKEND = "numpy_float64_vectorized_v1"


def _camera_arrays(camera: dict[str, Any]) -> tuple[Any, Any, Any, list[float]]:
    np: Any = importlib.import_module("numpy")
    try:
        intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
        world_to_camera = np.asarray(camera["world_to_camera"], dtype=np.float64)
        distortion = cast(list[float], camera["distortion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"vector geometry received a malformed camera: {exc}") from exc
    if intrinsic.shape != (3, 3) or world_to_camera.shape != (4, 4):
        raise ContractError("vector geometry camera matrices have invalid shape")
    if not bool(np.isfinite(intrinsic).all()) or not bool(np.isfinite(world_to_camera).all()):
        raise ContractError("vector geometry camera matrices are non-finite")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ContractError("vector geometry camera focal length must be positive")
    if camera["model"] == "opencv_radtan":
        if len(distortion) not in {4, 5}:
            raise ContractError("OpenCV radial-tangential distortion requires 4 or 5 values")
    elif camera["model"] == "pinhole":
        if distortion:
            raise ContractError("pinhole camera must not declare distortion")
    else:
        raise ContractError(f"unsupported vector geometry camera model: {camera['model']}")
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    center = -(rotation.T @ translation)
    return intrinsic, rotation, center, distortion


def _distort(x: Any, y: Any, distortion: list[float]) -> tuple[Any, Any]:
    k1, k2, p1, p2 = distortion[:4]
    k3 = distortion[4] if len(distortion) == 5 else 0.0
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    xy2 = 2.0 * x * y
    return (
        x * radial + p1 * xy2 + p2 * (radius2 + 2.0 * x * x),
        y * radial + p1 * (radius2 + 2.0 * y * y) + p2 * xy2,
    )


def _normalized_camera_xy(camera: dict[str, Any], pixels: Any) -> tuple[Any, Any]:
    np: Any = importlib.import_module("numpy")
    intrinsic, _, _, distortion = _camera_arrays(camera)
    values = np.asarray(pixels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ContractError("vector geometry pixels must have shape (N,2)")
    distorted_y = (values[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1]
    distorted_x = (values[:, 0] - intrinsic[0, 2] - intrinsic[0, 1] * distorted_y) / intrinsic[0, 0]
    valid = np.isfinite(distorted_x) & np.isfinite(distorted_y)
    if camera["model"] == "pinhole":
        return np.stack((distorted_x, distorted_y), axis=1), valid

    x = distorted_x.copy()
    y = distorted_y.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(40):
            projected_x, projected_y = _distort(x, y, distortion)
            residual_x = distorted_x - projected_x
            residual_y = distorted_y - projected_y
            x += residual_x
            y += residual_y
            if bool(
                np.max(np.maximum(np.abs(residual_x), np.abs(residual_y)), initial=0.0) <= 1e-13
            ):
                break
        check_x, check_y = _distort(x, y, distortion)
    residual = np.maximum(np.abs(check_x - distorted_x), np.abs(check_y - distorted_y))
    valid &= np.isfinite(x) & np.isfinite(y) & (residual <= 1e-10)
    return np.stack((x, y), axis=1), valid


def _rays(camera: dict[str, Any], pixels: Any) -> tuple[Any, Any, Any]:
    np: Any = importlib.import_module("numpy")
    normalized, valid = _normalized_camera_xy(camera, pixels)
    _, rotation, center, _ = _camera_arrays(camera)
    camera_directions = np.column_stack(
        (normalized[:, 0], normalized[:, 1], np.ones(len(normalized), dtype=np.float64))
    )
    camera_norm = np.linalg.norm(camera_directions, axis=1)
    valid &= np.isfinite(camera_norm) & (camera_norm > 1e-15)
    camera_directions = camera_directions / camera_norm[:, None]
    world_directions = camera_directions @ rotation
    world_norm = np.linalg.norm(world_directions, axis=1)
    valid &= np.isfinite(world_norm) & (world_norm > 1e-15)
    world_directions = world_directions / world_norm[:, None]
    return center, world_directions, valid


def _triangulate_positions(
    source_camera: dict[str, Any],
    target_camera: dict[str, Any],
    source_pixels: Any,
    target_pixels: Any,
) -> dict[str, Any]:
    np: Any = importlib.import_module("numpy")
    source_origin, source_direction, source_valid = _rays(source_camera, source_pixels)
    target_origin, target_direction, target_valid = _rays(target_camera, target_pixels)
    baseline = source_origin - target_origin
    direction_dot = np.sum(source_direction * target_direction, axis=1)
    source_offset_dot = source_direction @ baseline
    target_offset_dot = target_direction @ baseline
    denominator = 1.0 - direction_dot * direction_dot
    valid = source_valid & target_valid & np.isfinite(denominator) & (denominator > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        source_depth = (direction_dot * target_offset_dot - source_offset_dot) / denominator
        target_depth = (target_offset_dot - direction_dot * source_offset_dot) / denominator
        source_point = source_origin + source_depth[:, None] * source_direction
        target_point = target_origin + target_depth[:, None] * target_direction
        position = 0.5 * (source_point + target_point)
        ray_gap = np.linalg.norm(source_point - target_point, axis=1)
        angle = np.degrees(np.arccos(np.clip(direction_dot, -1.0, 1.0)))
    finite = np.isfinite(position).all(axis=1)
    finite &= np.isfinite(source_depth) & np.isfinite(target_depth)
    finite &= np.isfinite(ray_gap) & np.isfinite(angle)
    valid &= finite
    return {
        "position": position,
        "source_depth": source_depth,
        "target_depth": target_depth,
        "ray_gap_world": ray_gap,
        "triangulation_angle_degrees": angle,
        "valid": valid,
    }


def _project(camera: dict[str, Any], positions: Any) -> tuple[Any, Any, Any]:
    np: Any = importlib.import_module("numpy")
    intrinsic, rotation, center, distortion = _camera_arrays(camera)
    translation = -(rotation @ center)
    camera_positions = np.asarray(positions, dtype=np.float64) @ rotation.T + translation
    depth = camera_positions[:, 2]
    valid = np.isfinite(camera_positions).all(axis=1) & np.isfinite(depth) & (np.abs(depth) > 1e-15)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = camera_positions[:, 0] / depth
        y = camera_positions[:, 1] / depth
        if camera["model"] == "opencv_radtan":
            x, y = _distort(x, y, distortion)
        pixel_x = intrinsic[0, 0] * x + intrinsic[0, 1] * y + intrinsic[0, 2]
        pixel_y = intrinsic[1, 1] * y + intrinsic[1, 2]
    pixels = np.stack((pixel_x, pixel_y), axis=1)
    valid &= np.isfinite(pixels).all(axis=1)
    return pixels, depth, valid


def _sampson(
    source_camera: dict[str, Any],
    target_camera: dict[str, Any],
    source_pixels: Any,
    target_pixels: Any,
) -> tuple[Any, Any]:
    np: Any = importlib.import_module("numpy")
    source_xy, source_valid = _normalized_camera_xy(source_camera, source_pixels)
    target_xy, target_valid = _normalized_camera_xy(target_camera, target_pixels)
    _, source_rotation, source_center, _ = _camera_arrays(source_camera)
    _, target_rotation, target_center, _ = _camera_arrays(target_camera)
    source_translation = -(source_rotation @ source_center)
    target_translation = -(target_rotation @ target_center)
    relative_rotation = target_rotation @ source_rotation.T
    relative_translation = target_translation - relative_rotation @ source_translation
    skew = np.asarray(
        [
            [0.0, -relative_translation[2], relative_translation[1]],
            [relative_translation[2], 0.0, -relative_translation[0]],
            [-relative_translation[1], relative_translation[0], 0.0],
        ],
        dtype=np.float64,
    )
    essential = skew @ relative_rotation
    source_h = np.column_stack((source_xy, np.ones(len(source_xy), dtype=np.float64)))
    target_h = np.column_stack((target_xy, np.ones(len(target_xy), dtype=np.float64)))
    target_line = source_h @ essential.T
    source_line = target_h @ essential
    numerator = np.sum(target_h * target_line, axis=1)
    denominator = (
        target_line[:, 0] ** 2
        + target_line[:, 1] ** 2
        + source_line[:, 0] ** 2
        + source_line[:, 1] ** 2
    )
    valid = source_valid & target_valid & np.isfinite(denominator) & (denominator > 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.abs(numerator) / np.sqrt(denominator)
    valid &= np.isfinite(distance)
    return distance, valid


def evaluate_two_view_batch(
    source_camera: dict[str, Any],
    target_camera: dict[str, Any],
    source_pixels: Any,
    target_pixels: Any,
    *,
    pixel_sigma: float,
) -> dict[str, Any]:
    """Vectorized float64 implementation of the first-party scalar oracle."""
    if not math.isfinite(pixel_sigma) or pixel_sigma <= 0:
        raise ContractError("vector geometry pixel sigma must be finite and positive")
    np: Any = importlib.import_module("numpy")
    source = np.asarray(source_pixels, dtype=np.float64)
    target = np.asarray(target_pixels, dtype=np.float64)
    diagnostics = evaluate_two_view_diagnostics(
        source_camera,
        target_camera,
        source,
        target,
    )

    jacobian_columns: list[Any] = []
    covariance_valid = np.ones(len(source), dtype=np.bool_)
    coordinates = np.column_stack((source, target))
    for axis in range(4):
        positive = coordinates.copy()
        negative = coordinates.copy()
        positive[:, axis] += pixel_sigma
        negative[:, axis] -= pixel_sigma
        positive_result = _triangulate_positions(
            source_camera,
            target_camera,
            positive[:, :2],
            positive[:, 2:],
        )
        negative_result = _triangulate_positions(
            source_camera,
            target_camera,
            negative[:, :2],
            negative[:, 2:],
        )
        covariance_valid &= positive_result["valid"] & negative_result["valid"]
        jacobian_columns.append(
            (positive_result["position"] - negative_result["position"]) * (0.5 / pixel_sigma)
        )
    jacobian = np.stack(jacobian_columns, axis=2)
    covariance = pixel_sigma**2 * np.einsum("nik,njk->nij", jacobian, jacobian)
    covariance_valid &= np.isfinite(covariance).all(axis=(1, 2))

    valid = diagnostics["valid"] & covariance_valid
    return {
        **diagnostics,
        "covariance_xx": covariance[:, 0, 0],
        "covariance_xy": covariance[:, 0, 1],
        "covariance_xz": covariance[:, 0, 2],
        "covariance_yy": covariance[:, 1, 1],
        "covariance_yz": covariance[:, 1, 2],
        "covariance_zz": covariance[:, 2, 2],
        "valid": valid,
    }


def evaluate_two_view_diagnostics(
    source_camera: dict[str, Any],
    target_camera: dict[str, Any],
    source_pixels: Any,
    target_pixels: Any,
) -> dict[str, Any]:
    """Triangulate and expose correctness diagnostics without covariance work.

    ``source_depth`` and ``target_depth`` are the closest-ray parameters retained
    for compatibility with the scalar oracle.  The explicitly named
    ``*_camera_z`` planes are the correct cheirality quantities: the z coordinate
    after applying each camera's world-to-camera transform.
    """

    np: Any = importlib.import_module("numpy")
    source = np.asarray(source_pixels, dtype=np.float64)
    target = np.asarray(target_pixels, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ContractError("vector geometry source/target pixels must share shape (N,2)")

    base = _triangulate_positions(source_camera, target_camera, source, target)
    position = base["position"]
    source_projection, source_camera_z, source_project_valid = _project(source_camera, position)
    target_projection, target_camera_z, target_project_valid = _project(target_camera, position)
    sampson, sampson_valid = _sampson(
        source_camera,
        target_camera,
        source,
        target,
    )
    source_reprojection = np.linalg.norm(source_projection - source, axis=1)
    target_reprojection = np.linalg.norm(target_projection - target, axis=1)
    valid = base["valid"].copy()
    valid &= source_project_valid & target_project_valid & sampson_valid
    valid &= np.isfinite(source_reprojection) & np.isfinite(target_reprojection)
    return {
        **base,
        "source_camera_z": source_camera_z,
        "target_camera_z": target_camera_z,
        "epipolar_sampson_normalized": sampson,
        "source_reprojection_error_pixels": source_reprojection,
        "target_reprojection_error_pixels": target_reprojection,
        "valid": valid,
    }
