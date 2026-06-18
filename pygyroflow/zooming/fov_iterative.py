"""Iterative FOV computation algorithm.

Port of Gyroflow's fov_iterative.rs. Computes the minimum FOV scale factor
for each frame that avoids black borders, using a polygon contraction approach:

  1. Sample 31x31 points around the input image border
  2. Undistort these points (with rolling shutter correction)
  3. Find the nearest undistorted edge point to the image center
  4. Refine with interpolation around the nearest point (5 iterations)
  5. Calculate minimal FOV from the nearest edge distance
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.stabilization.frame_transform import _quat_at_timestamp


def _points_around_rect(
    w: float,
    h: float,
    w_div: int,
    h_div: int,
    margin: float = 2.0,
) -> list[tuple[float, float]]:
    """Generate evenly-spaced points along a rectangle border.

    Points are arranged clockwise starting from top-left.
    The rectangle is inset by `margin` pixels.

    Args:
        w: Rectangle width.
        h: Rectangle height.
        w_div: Number of divisions along width (min 2).
        h_div: Number of divisions along height (min 2).
        margin: Inset margin in pixels.

    Returns:
        List of (x, y) points in clockwise order.
    """
    w -= margin * 2.0
    h -= margin * 2.0

    wcnt = max(2, w_div) - 1
    hcnt = max(2, h_div) - 1
    wstep = w / wcnt
    hstep = h / hcnt

    points: list[tuple[float, float]] = []

    # Top edge (left to right)
    for i in range(wcnt):
        points.append((i * wstep, 0.0))
    # Right edge (top to bottom)
    for i in range(hcnt):
        points.append((w, i * hstep))
    # Bottom edge (right to left)
    for i in range(wcnt):
        points.append(((wcnt - i) * wstep, h))
    # Left edge (bottom to top)
    for i in range(hcnt):
        points.append((0.0, (hcnt - i) * hstep))

    # Add margin offset
    points = [(x + margin, y + margin) for x, y in points]

    return points


def _interpolate_points(
    pts: list[tuple[float, float]],
    steps: int,
) -> list[tuple[float, float]]:
    """Linearly interpolate between adjacent points to create a denser set.

    Used for refining the search around the nearest edge point.

    Args:
        pts: Input points.
        steps: Number of interpolation steps between each pair.

    Returns:
        Denser point list.
    """
    d = steps + 1
    new_len = d * len(pts) - steps
    result = []
    for i in range(new_len):
        idx1 = i // d
        idx2 = min(idx1 + 1, len(pts) - 1)
        f = (i % d) / d
        x = pts[idx1][0] + f * (pts[idx2][0] - pts[idx1][0])
        y = pts[idx1][1] + f * (pts[idx2][1] - pts[idx1][1])
        result.append((x, y))
    return result


def _undistort_points_simple(
    points: list[tuple[float, float]],
    timestamp_ms: float,
    frame: int,
    params: ComputeParams,
    lens_correction_amount: float,
) -> list[tuple[float, float]]:
    """Simplified point undistortion for FOV calculation.

    Applies the stabilization rotation to each point, without full
    lens distortion model. For the FOV algorithm we only need the
    approximate rotated positions, not pixel-perfect coordinates.

    Args:
        points: Input pixel coordinates.
        timestamp_ms: Frame timestamp.
        frame: Frame index.
        params: Compute parameters.
        lens_correction_amount: Lens correction strength (unused here but
            kept for API compatibility).

    Returns:
        Rotated/transformed point coordinates.
    """
    if not points:
        return []

    # Get rotation quaternion for this timestamp
    ts_us = timestamp_ms * 1000.0
    org_quat = _quat_at_timestamp(params.quaternions, ts_us).inverse()
    smoothed_quat = _quat_at_timestamp(params.smoothed_quaternions, ts_us)

    # Combined rotation
    combined_quat = smoothed_quat * org_quat
    rot_matrix = combined_quat.to_rotation_matrix()

    # Camera intrinsics
    fx = params.camera_matrix[0, 0]
    fy = params.camera_matrix[1, 1]
    cx = params.camera_matrix[0, 2]
    cy = params.camera_matrix[1, 2]

    # Scale intrinsics to video resolution if needed
    calib_w = params.calib_width if params.calib_width > 0 else params.width
    calib_h = params.calib_height if params.calib_height > 0 else params.height
    if calib_w > 0 and calib_h > 0:
        ratio_x = params.width / calib_w
        ratio_y = params.height / calib_h
        fx *= ratio_x
        fy *= ratio_y
        cx *= ratio_x
        cy *= ratio_y

    w = float(params.width)
    h = float(params.height)

    result = []
    for px, py in points:
        # Pixel to normalized camera coords
        xn = (px - cx) / fx
        yn = (py - cy) / fy

        # Apply rotation
        vec = rot_matrix @ np.array([xn, yn, 1.0])
        if vec[2] <= 0.0:
            # Behind camera
            result.append((-1e6, -1e6))
            continue

        xn_r = vec[0] / vec[2]
        yn_r = vec[1] / vec[2]

        # Back to pixel coords
        out_x = xn_r * fx + cx
        out_y = yn_r * fy + cy

        result.append((float(out_x), float(out_y)))

    return result


class FovIterative:
    """Iterative FOV calculator using polygon contraction.

    Finds the maximum inscribed rectangle (matching output aspect ratio)
    within the stabilized polygon, then derives the minimum FOV scale.
    """

    def __init__(
        self,
        compute_params: ComputeParams,
        org_output_size: tuple[int, int],
    ) -> None:
        """Create the FOV calculator.

        Args:
            compute_params: Stabilization parameters.
            org_output_size: Original output size (width, height) for aspect ratio.
        """
        ratio = compute_params.width / max(1, org_output_size[0])
        self.input_dim = (float(compute_params.width), float(compute_params.height))
        self.output_dim = (
            float(org_output_size[0]) * ratio,
            float(org_output_size[1]) * ratio,
        )
        self.output_inv_aspect = self.output_dim[1] / self.output_dim[0]
        self.compute_params = compute_params

    def compute(
        self,
        timestamps: list[tuple[int, float]],
        ranges: list[tuple[float, float]],
    ) -> list[float]:
        """Compute FOV values for all timestamps.

        Args:
            timestamps: List of (frame_index, timestamp_ms).
            ranges: Trim ranges as (start_frac, end_frac).

        Returns:
            List of FOV scale factors, one per timestamp.
        """
        if not timestamps:
            return []

        rect = _points_around_rect(
            self.input_dim[0], self.input_dim[1],
            31, 31,
            self.compute_params.fov_algorithm_margin,
        )

        center = (self.input_dim[0] / 2.0, self.input_dim[1] / 2.0)

        # Use constant keyframe values for now (no per-frame keyframe lookup)
        zoom_cx = self.compute_params.adaptive_zoom_center_offset[0]
        zoom_cy = self.compute_params.adaptive_zoom_center_offset[1]
        lens_corr = self.compute_params.lens_correction_amount
        kv = (zoom_cx, zoom_cy, lens_corr)

        fov_values = [
            self._find_fov(rect, ts, frame, center, kv)
            for frame, ts in timestamps
        ]

        # Apply trim ranges
        if ranges:
            l = (len(timestamps) - 1)
            max_fov = max(fov_values) if fov_values else 1.0
            for i, _ in enumerate(fov_values):
                within_range = any(
                    i >= math.floor(l * r[0]) and i <= math.ceil(l * r[1])
                    for r in ranges
                )
                if not within_range:
                    fov_values[i] = max_fov

        return fov_values

    def _find_fov(
        self,
        rect: list[tuple[float, float]],
        ts: float,
        frame: int,
        center: tuple[float, float],
        keyframe_values: tuple[float, float, float],
    ) -> float:
        """Find the minimum FOV for a single frame.

        Args:
            rect: Border sample points.
            ts: Timestamp in ms.
            frame: Frame index.
            center: Image center (cx, cy).
            keyframe_values: (zoom_cx, zoom_cy, lens_correction).

        Returns:
            FOV scale factor.
        """
        zoom_cx, zoom_cy, lens_corr = keyframe_values

        # Undistort border points
        polygon = _undistort_points_simple(
            rect, ts, frame, self.compute_params, lens_corr
        )

        # Apply zoom center offset
        for i, (x, y) in enumerate(polygon):
            polygon[i] = (
                x - zoom_cx * self.input_dim[0],
                y - zoom_cy * self.input_dim[1],
            )

        # Initial search rectangle: very large
        initial = (1e6, 1e6 * self.output_inv_aspect)
        nearest_idx: Optional[int] = None
        nearest_rect = initial

        for _ in range(5):
            nearest_idx, nearest_rect = self._nearest_edge(
                polygon, center, nearest_rect
            )
            if nearest_idx is not None and rect:
                n = len(rect)
                relevant = [
                    rect[(nearest_idx - 1) % n],
                    rect[nearest_idx],
                    rect[(nearest_idx + 1) % n],
                ]
                distorted = _interpolate_points(relevant, 30)
                polygon = _undistort_points_simple(
                    distorted, ts, frame, self.compute_params, lens_corr
                )
                for i, (x, y) in enumerate(polygon):
                    polygon[i] = (
                        x - zoom_cx * self.input_dim[0],
                        y - zoom_cy * self.input_dim[1],
                    )
                nearest_idx, nearest_rect = self._nearest_edge(
                    polygon, center, nearest_rect
                )
            else:
                break

        # Convert nearest edge distance to FOV scale
        fov = (nearest_rect[0] * 2.0 / self.output_dim[0])
        return fov

    @staticmethod
    def _nearest_edge(
        polygon: list[tuple[float, float]],
        center: tuple[float, float],
        initial: tuple[float, float],
    ) -> tuple[Optional[int], tuple[float, float]]:
        """Find the nearest polygon point that constrains the inscribed rectangle.

        The inscribed rectangle maintains the output aspect ratio. For each
        polygon point, compute the minimum rectangle (maintaining aspect ratio)
        that would contain that point, then track the overall minimum.

        Args:
            polygon: Transformed border points.
            center: Image center.
            initial: Current best rectangle (half_w, half_h).

        Returns:
            (nearest_point_index, contracted_rectangle).
        """
        best_idx = None
        best_rect = initial

        inv_aspect = initial[1] / initial[0] if initial[0] > 0 else 1.0

        for i, (x, y) in enumerate(polygon):
            ap = (abs(x - center[0]), abs(y - center[1]))
            if ap[0] < best_rect[0] and ap[1] < best_rect[1]:
                if ap[1] > ap[0] * inv_aspect:
                    best_idx = i
                    best_rect = (ap[1] / inv_aspect, ap[1])
                else:
                    best_idx = i
                    best_rect = (ap[0], ap[0] * inv_aspect)

        return best_idx, best_rect
