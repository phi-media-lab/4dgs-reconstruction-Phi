from __future__ import annotations

import math
import unittest
from typing import Any

from p2g.geometry import (
    baseline_distance,
    camera_center,
    epipolar_sampson_distance_normalized,
    project_world,
    reprojection_error_pixels,
    triangulate_pixels,
    triangulation_covariance,
    unproject_pixel,
)


def _camera(center_x: float, *, distorted: bool = False) -> dict[str, Any]:
    return {
        "model": "opencv_radtan" if distorted else "pinhole",
        "pixel_domain": "distorted" if distorted else "undistorted",
        "intrinsic": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "world_to_camera": [
            [1.0, 0.0, 0.0, -center_x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "distortion": [0.1, -0.05, 0.001, -0.002] if distorted else [],
    }


class GeometryTests(unittest.TestCase):
    def test_project_unproject_roundtrip_with_distortion(self) -> None:
        camera = _camera(0.25, distorted=True)
        point = (0.4, -0.2, 3.0)
        pixel_x, pixel_y, depth = project_world(camera, point)
        ray = unproject_pixel(camera, pixel_x, pixel_y)
        reconstructed = tuple(
            ray.origin[axis] + depth * ray.direction[axis] / ray.direction[2] for axis in range(3)
        )
        for actual, expected in zip(reconstructed, point, strict=True):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_two_view_triangulation_and_reprojection(self) -> None:
        source = _camera(-0.5)
        target = _camera(0.5)
        point = (0.1, 0.2, 4.0)
        source_projection = project_world(source, point)
        target_projection = project_world(target, point)
        result = triangulate_pixels(
            source,
            target,
            source_projection[:2],
            target_projection[:2],
        )
        for actual, expected in zip(result.position, point, strict=True):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertGreater(result.source_depth, 0.0)
        self.assertGreater(result.target_depth, 0.0)
        self.assertLess(result.ray_gap, 1e-10)
        self.assertGreater(result.angle_degrees, 0.0)
        self.assertLess(
            reprojection_error_pixels(source, result.position, source_projection[:2]),
            1e-9,
        )
        self.assertLess(
            epipolar_sampson_distance_normalized(
                source,
                target,
                source_projection[:2],
                target_projection[:2],
            ),
            1e-12,
        )
        covariance = triangulation_covariance(
            source,
            target,
            source_projection[:2],
            target_projection[:2],
            pixel_sigma=1.0,
        )
        self.assertGreater(covariance.xx, 0.0)
        self.assertGreater(covariance.yy, 0.0)
        self.assertGreater(covariance.zz, 0.0)

    def test_epipolar_residual_detects_bad_match(self) -> None:
        source = _camera(-0.5)
        target = _camera(0.5)
        point = (0.1, 0.2, 4.0)
        source_xy = project_world(source, point)[:2]
        target_xy = project_world(target, point)[:2]
        bad_target_xy = (target_xy[0], target_xy[1] + 20.0)
        self.assertGreater(
            epipolar_sampson_distance_normalized(source, target, source_xy, bad_target_xy),
            0.01,
        )

    def test_camera_centers_and_baseline(self) -> None:
        left = _camera(-0.5)
        right = _camera(0.5)
        self.assertEqual(camera_center(left), (-0.5, -0.0, -0.0))
        self.assertEqual(camera_center(right), (0.5, -0.0, -0.0))
        self.assertTrue(math.isclose(baseline_distance(left, right), 1.0))


if __name__ == "__main__":
    unittest.main()
