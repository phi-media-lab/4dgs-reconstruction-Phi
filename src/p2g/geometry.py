from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from p2g.errors import ContractError

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class Ray:
    origin: Vector3
    direction: Vector3


@dataclass(frozen=True)
class Triangulation:
    position: Vector3
    source_depth: float
    target_depth: float
    ray_gap: float
    angle_degrees: float


@dataclass(frozen=True)
class PositionCovariance:
    xx: float
    xy: float
    xz: float
    yy: float
    yz: float
    zz: float
    assumed_pixel_sigma: float


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _add(left: Vector3, right: Vector3) -> Vector3:
    return cast(Vector3, tuple(a + b for a, b in zip(left, right, strict=True)))


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return cast(Vector3, tuple(a - b for a, b in zip(left, right, strict=True)))


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return cast(Vector3, tuple(value * scalar for value in vector))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    norm = _norm(vector)
    if not math.isfinite(norm) or norm <= 1e-15:
        raise ContractError("cannot normalize a degenerate 3D vector")
    return _scale(vector, 1.0 / norm)


def distort_radtan(x: float, y: float, distortion: list[float]) -> tuple[float, float]:
    if len(distortion) not in {4, 5}:
        raise ContractError("OpenCV radial-tangential distortion requires 4 or 5 values")
    k1, k2, p1, p2 = distortion[:4]
    k3 = distortion[4] if len(distortion) == 5 else 0.0
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    xy2 = 2.0 * x * y
    return (
        x * radial + p1 * xy2 + p2 * (radius2 + 2.0 * x * x),
        y * radial + p1 * (radius2 + 2.0 * y * y) + p2 * xy2,
    )


def undistort_radtan(
    distorted_x: float,
    distorted_y: float,
    distortion: list[float],
) -> tuple[float, float]:
    x, y = distorted_x, distorted_y
    for _ in range(40):
        projected_x, projected_y = distort_radtan(x, y, distortion)
        residual_x = distorted_x - projected_x
        residual_y = distorted_y - projected_y
        x += residual_x
        y += residual_y
        if max(abs(residual_x), abs(residual_y)) <= 1e-13:
            break
        if not math.isfinite(x) or not math.isfinite(y):
            raise ContractError("OpenCV distortion inversion diverged")
    check_x, check_y = distort_radtan(x, y, distortion)
    if max(abs(check_x - distorted_x), abs(check_y - distorted_y)) > 1e-10:
        raise ContractError("OpenCV distortion inversion did not converge")
    return x, y


def _camera_values(
    camera: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    intrinsic = cast(list[list[float]], camera["intrinsic"])
    world_to_camera = cast(list[list[float]], camera["world_to_camera"])
    distortion = cast(list[float], camera["distortion"])
    return intrinsic, world_to_camera, distortion


def camera_center(camera: dict[str, Any]) -> Vector3:
    _, world_to_camera, _ = _camera_values(camera)
    rotation = [row[:3] for row in world_to_camera[:3]]
    translation = cast(Vector3, tuple(row[3] for row in world_to_camera[:3]))
    return cast(
        Vector3,
        tuple(
            -sum(rotation[axis][row] * translation[axis] for axis in range(3)) for row in range(3)
        ),
    )


def camera_forward_world(camera: dict[str, Any]) -> Vector3:
    _, world_to_camera, _ = _camera_values(camera)
    rotation = [row[:3] for row in world_to_camera[:3]]
    return _normalize((rotation[2][0], rotation[2][1], rotation[2][2]))


def normalized_camera_xy(
    camera: dict[str, Any], pixel_x: float, pixel_y: float
) -> tuple[float, float]:
    intrinsic, _, distortion = _camera_values(camera)
    fx, skew, cx = intrinsic[0]
    fy, cy = intrinsic[1][1], intrinsic[1][2]
    if fx <= 0 or fy <= 0:
        raise ContractError("camera focal length must be positive")
    distorted_y = (pixel_y - cy) / fy
    distorted_x = (pixel_x - cx - skew * distorted_y) / fx
    if camera["model"] == "opencv_radtan":
        return undistort_radtan(distorted_x, distorted_y, distortion)
    if distortion:
        raise ContractError("pinhole camera must not declare distortion")
    return distorted_x, distorted_y


def unproject_pixel(camera: dict[str, Any], pixel_x: float, pixel_y: float) -> Ray:
    x, y = normalized_camera_xy(camera, pixel_x, pixel_y)
    direction_camera = _normalize((x, y, 1.0))
    _, world_to_camera, _ = _camera_values(camera)
    rotation = [row[:3] for row in world_to_camera[:3]]
    direction_world = cast(
        Vector3,
        tuple(
            sum(rotation[axis][row] * direction_camera[axis] for axis in range(3))
            for row in range(3)
        ),
    )
    return Ray(origin=camera_center(camera), direction=_normalize(direction_world))


def project_world(camera: dict[str, Any], position: Vector3) -> tuple[float, float, float]:
    intrinsic, world_to_camera, distortion = _camera_values(camera)
    camera_position = cast(
        Vector3,
        tuple(
            sum(world_to_camera[row][axis] * position[axis] for axis in range(3))
            + world_to_camera[row][3]
            for row in range(3)
        ),
    )
    depth = camera_position[2]
    if not math.isfinite(depth) or abs(depth) <= 1e-15:
        raise ContractError("point projects at zero or non-finite depth")
    x, y = camera_position[0] / depth, camera_position[1] / depth
    if camera["model"] == "opencv_radtan":
        x, y = distort_radtan(x, y, distortion)
    elif distortion:
        raise ContractError("pinhole camera must not declare distortion")
    fx, skew, cx = intrinsic[0]
    fy, cy = intrinsic[1][1], intrinsic[1][2]
    return fx * x + skew * y + cx, fy * y + cy, depth


def triangulate_rays(source: Ray, target: Ray) -> Triangulation:
    baseline = _subtract(source.origin, target.origin)
    source_target_dot = _dot(source.direction, target.direction)
    source_offset_dot = _dot(source.direction, baseline)
    target_offset_dot = _dot(target.direction, baseline)
    denominator = 1.0 - source_target_dot * source_target_dot
    if denominator <= 1e-12:
        raise ContractError("triangulation rays are parallel or numerically degenerate")
    source_depth = (source_target_dot * target_offset_dot - source_offset_dot) / denominator
    target_depth = (target_offset_dot - source_target_dot * source_offset_dot) / denominator
    source_point = _add(source.origin, _scale(source.direction, source_depth))
    target_point = _add(target.origin, _scale(target.direction, target_depth))
    position = _scale(_add(source_point, target_point), 0.5)
    ray_gap = _norm(_subtract(source_point, target_point))
    cosine = min(1.0, max(-1.0, source_target_dot))
    angle = math.degrees(math.acos(cosine))
    values = (*position, source_depth, target_depth, ray_gap, angle)
    if not all(math.isfinite(value) for value in values):
        raise ContractError("triangulation produced non-finite geometry")
    return Triangulation(
        position=position,
        source_depth=source_depth,
        target_depth=target_depth,
        ray_gap=ray_gap,
        angle_degrees=angle,
    )


def triangulate_pixels(
    source_camera: dict[str, Any],
    target_camera: dict[str, Any],
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> Triangulation:
    return triangulate_rays(
        unproject_pixel(source_camera, *source_xy),
        unproject_pixel(target_camera, *target_xy),
    )


def triangulation_covariance(
    source_camera: dict[str, Any],
    target_camera: dict[str, Any],
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    pixel_sigma: float,
) -> PositionCovariance:
    if not math.isfinite(pixel_sigma) or pixel_sigma <= 0:
        raise ContractError("triangulation pixel sigma must be finite and positive")
    coordinates = (*source_xy, *target_xy)
    jacobian_columns: list[Vector3] = []
    for axis in range(4):
        positive = list(coordinates)
        negative = list(coordinates)
        positive[axis] += pixel_sigma
        negative[axis] -= pixel_sigma
        positive_position = triangulate_pixels(
            source_camera,
            target_camera,
            (positive[0], positive[1]),
            (positive[2], positive[3]),
        ).position
        negative_position = triangulate_pixels(
            source_camera,
            target_camera,
            (negative[0], negative[1]),
            (negative[2], negative[3]),
        ).position
        jacobian_columns.append(
            _scale(_subtract(positive_position, negative_position), 0.5 / pixel_sigma)
        )

    covariance = [
        [
            pixel_sigma**2 * sum(column[row] * column[column_index] for column in jacobian_columns)
            for column_index in range(3)
        ]
        for row in range(3)
    ]
    values = (
        covariance[0][0],
        covariance[0][1],
        covariance[0][2],
        covariance[1][1],
        covariance[1][2],
        covariance[2][2],
    )
    if not all(math.isfinite(value) for value in values):
        raise ContractError("triangulation covariance is non-finite")
    return PositionCovariance(*values, assumed_pixel_sigma=pixel_sigma)


def epipolar_sampson_distance_normalized(
    source_camera: dict[str, Any],
    target_camera: dict[str, Any],
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> float:
    source_normalized = (*normalized_camera_xy(source_camera, *source_xy), 1.0)
    target_normalized = (*normalized_camera_xy(target_camera, *target_xy), 1.0)
    _, source_w2c, _ = _camera_values(source_camera)
    _, target_w2c, _ = _camera_values(target_camera)
    source_rotation = [row[:3] for row in source_w2c[:3]]
    target_rotation = [row[:3] for row in target_w2c[:3]]
    source_translation = [row[3] for row in source_w2c[:3]]
    target_translation = [row[3] for row in target_w2c[:3]]

    relative_rotation = [
        [
            sum(target_rotation[row][axis] * source_rotation[column][axis] for axis in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]
    relative_translation = [
        target_translation[row]
        - sum(relative_rotation[row][axis] * source_translation[axis] for axis in range(3))
        for row in range(3)
    ]
    skew = [
        [0.0, -relative_translation[2], relative_translation[1]],
        [relative_translation[2], 0.0, -relative_translation[0]],
        [-relative_translation[1], relative_translation[0], 0.0],
    ]
    essential = [
        [
            sum(skew[row][axis] * relative_rotation[axis][column] for axis in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]
    target_line = [
        sum(essential[row][axis] * source_normalized[axis] for axis in range(3)) for row in range(3)
    ]
    source_line = [
        sum(essential[axis][row] * target_normalized[axis] for axis in range(3)) for row in range(3)
    ]
    numerator = sum(target_normalized[axis] * target_line[axis] for axis in range(3))
    denominator = (
        target_line[0] ** 2 + target_line[1] ** 2 + source_line[0] ** 2 + source_line[1] ** 2
    )
    if not math.isfinite(denominator) or denominator <= 1e-30:
        raise ContractError("epipolar Sampson distance is degenerate")
    distance = abs(numerator) / math.sqrt(denominator)
    if not math.isfinite(distance):
        raise ContractError("epipolar Sampson distance is non-finite")
    return distance


def reprojection_error_pixels(
    camera: dict[str, Any], position: Vector3, expected_xy: tuple[float, float]
) -> float:
    projected_x, projected_y, _ = project_world(camera, position)
    return math.hypot(projected_x - expected_xy[0], projected_y - expected_xy[1])


def baseline_distance(source_camera: dict[str, Any], target_camera: dict[str, Any]) -> float:
    return _norm(_subtract(camera_center(source_camera), camera_center(target_camera)))


def direction_angle_degrees(source: Vector3, target: Vector3) -> float:
    cosine = min(1.0, max(-1.0, _dot(_normalize(source), _normalize(target))))
    return math.degrees(math.acos(cosine))
