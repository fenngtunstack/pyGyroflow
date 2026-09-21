"""Pose estimation sub-package -- recover camera rotation from optical flow.

Upstream's ``EstimatePoseMethod`` has four strategies, dispatched by index:

* ``0`` ``PoseFindEssentialMat`` -- OpenCV LMEDS on undistorted normalized
  points (``find_essential_mat.py``, faithful port)
* ``1`` ``PoseAlmeida``          -- per-point ray deltas (``almeida.py``)
* ``2`` ``PoseEightPoint``       -- upstream runs ARRSAC over unit rays;
  **ARRSAC is not ported** (no OpenCV equivalent), so this index runs the
  eight-point RANSAC helper on undistorted *pixel* coordinates instead.
  Documented gap, not a faithful port.
* ``3`` ``PoseFindHomography``   -- homography + decomposition on undistorted
  normalized points (``homography.py``, faithful port)

An unknown index falls back to Almeida with an error log, as upstream's
``From<u32>`` does. Every strategy except Almeida undistorts both point sets
first, each at its own frame's timestamp (``undistort_points_for_optical_flow``);
Almeida undistorts internally inside its ``delta``.
"""

import numpy as np
import numpy.typing as npt
import logging

from pygyroflow.stabilization.cpu_undistort import (
    _optical_flow_lens_data,
    undistort_points,
    undistort_points_for_optical_flow,
)
from pygyroflow.synchronization.estimate_pose.eight_point import (
    estimate_pose_eight_point,
)
from pygyroflow.synchronization.estimate_pose.essential_matrix import (
    estimate_essential_matrix,
)
from pygyroflow.synchronization.estimate_pose.find_essential_mat import (
    estimate_pose_find_essential_mat,
)
from pygyroflow.synchronization.estimate_pose.homography import (
    estimate_homography,
    estimate_pose_find_homography,
)

logger = logging.getLogger(__name__)

# Upstream numbering (`estimate_pose/mod.rs:30-34`):
# 0 = PoseFindEssentialMat, 1 = PoseAlmeida, 2 = PoseEightPoint,
# 3 = PoseFindHomography; unknown -> Almeida.
_METHOD_MAP = {
    0: "find_essential_mat",
    1: "almeida",
    2: "eight_point",
    3: "homography",
}


def estimate_rotation(
    prev_pts: npt.NDArray[np.float32],
    curr_pts: npt.NDArray[np.float32],
    camera_matrix: npt.NDArray[np.float64],
    method: int = 0,
    size_wh: tuple[float, float] | None = None,
    params=None,
    timestamp_ms: float = 0.0,
    next_timestamp_ms: float | None = None,
) -> np.ndarray | None:
    """Estimate the 3x3 rotation matrix between two frames.

    Parameters
    ----------
    prev_pts, curr_pts:
        Matched point pairs, shape (N, 2), in the pixel space of ``size_wh``.
    camera_matrix:
        3x3 camera intrinsics, used directly only when ``params`` is None
        (pinhole fallback); with a lens the points are undistorted first and
        the estimators work on the result.
    method:
        Pose estimation method index -- upstream's ``pose_method``
        (0=findEssentialMat, 1=Almeida, 2=eight-point stand-in, 3=homography;
        unknown falls back to Almeida).
    size_wh:
        Frame size the points are expressed in (the optical-flow working
        size, which may be smaller than the video). Undistortion scales the
        calibration to it. ``None`` falls back to ``params``' dimensions.
    params:
        ``ComputeParams``, so the estimators can reach the lens. Upstream
        hands it to every estimator (``EstimatePoseTrait::init``). ``None``
        means "run on a pinhole camera", which is what every caller got
        before this argument existed.
    timestamp_ms:
        *prev_pts*' frame timestamp, for the per-timestamp lens lookup.
    next_timestamp_ms:
        *curr_pts*' frame timestamp. Upstream undistorts the two sets at
        their own timestamps (``estimate_pose(..., timestamp_us,
        next_timestamp_us)``) — on a zoom lens the intrinsics differ between
        the two frames. Defaults to ``timestamp_ms``.

    Returns
    -------
    3x3 rotation matrix (ndarray, float64) or None on failure.
    """
    if next_timestamp_ms is None:
        next_timestamp_ms = timestamp_ms

    method_name = _METHOD_MAP.get(method)
    if method_name is None:
        logger.error("Unknown pose method %d; falling back to Almeida", method)
        method_name = "almeida"

    if method_name == "almeida":
        from .almeida import estimate_pose_almeida

        if size_wh is None:
            # principal point centred on typical sensors: w ~ 2*cx, h ~ 2*cy
            size_wh = (2.0 * float(camera_matrix[0, 2]), 2.0 * float(camera_matrix[1, 2]))
        return estimate_pose_almeida(
            prev_pts,
            curr_pts,
            camera_matrix,
            size_wh,
            params=params,
            timestamp_ms=timestamp_ms,
        )

    if params is None:
        # Pinhole fallback: the raw points with the real intrinsics.
        # findEssentialMat normalizes by K internally, so method 0 computes
        # the same epipolar constraint it would from undistorted normalized
        # points under an identity K.
        if method_name == "find_essential_mat":
            return estimate_pose_find_essential_mat(prev_pts, curr_pts, camera_matrix)
        if method_name == "eight_point":
            result = estimate_pose_eight_point(prev_pts, curr_pts, camera_matrix)
            return result[0] if result is not None else None
        if method_name == "homography":
            return estimate_pose_find_homography(prev_pts, curr_pts)
        return None

    # Every upstream estimator except Almeida undistorts both point sets
    # first, each at its own frame's timestamp.
    dims = size_wh if size_wh is not None else (params.width, params.height)
    ts1_us = int(round(timestamp_ms * 1000.0))
    ts2_us = int(round(next_timestamp_ms * 1000.0))

    if method_name == "eight_point":
        # ARRSAC stand-in. Upstream feeds ARRSAC unit rays, which need no K;
        # findEssentialMat does need one, so the undistorted points are put
        # back into pixel units and the intrinsics at *prev_pts*' timestamp
        # stand in for both sets. On a zoom lens that is an approximation the
        # ARRSAC port will one day remove.
        scaled_k, coeffs = _optical_flow_lens_data(params, ts1_us, dims)
        prev = undistort_points(
            prev_pts, scaled_k, coeffs, scaled_k, p=None, rot_per_point=None,
            params=params, lens_correction_amount=1.0, timestamp_ms=ts1_us / 1000.0,
            shift_per_point=None, mesh=None,
        )
        curr = undistort_points(
            curr_pts, scaled_k, coeffs, scaled_k, p=None, rot_per_point=None,
            params=params, lens_correction_amount=1.0, timestamp_ms=ts2_us / 1000.0,
            shift_per_point=None, mesh=None,
        )
        result = estimate_pose_eight_point(
            np.asarray(prev, dtype=np.float64),
            np.asarray(curr, dtype=np.float64),
            scaled_k,
        )
        return result[0] if result is not None else None

    prev = np.asarray(
        undistort_points_for_optical_flow(prev_pts, ts1_us, params, dims),
        dtype=np.float64,
    )
    curr = np.asarray(
        undistort_points_for_optical_flow(curr_pts, ts2_us, params, dims),
        dtype=np.float64,
    )
    if method_name == "find_essential_mat":
        return estimate_pose_find_essential_mat(prev, curr, np.eye(3))
    if method_name == "homography":
        return estimate_pose_find_homography(prev, curr)
    return None


__all__ = [
    "estimate_rotation",
    "estimate_pose_eight_point",
    "estimate_pose_find_essential_mat",
    "estimate_essential_matrix",
    "estimate_homography",
    "estimate_pose_find_homography",
]
