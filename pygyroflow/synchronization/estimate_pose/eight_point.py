"""Eight-point algorithm for essential matrix estimation + pose recovery.

Uses OpenCV's ``findEssentialMat`` with RANSAC and ``recoverPose`` to
extract the rotation matrix and (unit-norm) translation vector from a set
of matched 2D-2D point correspondences and a known camera intrinsics
matrix.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# RANSAC parameters matching Gyroflow's defaults
_RANSAC_PROB = 0.999
_RANSAC_THRESHOLD = 1.0  # pixels
_MIN_INLIERS = 10


def estimate_pose_eight_point(
    prev_pts: npt.NDArray[np.float32],
    curr_pts: npt.NDArray[np.float32],
    camera_matrix: npt.NDArray[np.float64],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate rotation between two frames using the eight-point algorithm.

    Parameters
    ----------
    prev_pts:
        Matched points in the previous frame, shape (N, 2).
    curr_pts:
        Matched points in the current frame, shape (N, 2).
    camera_matrix:
        3x3 camera intrinsic matrix (fx, fy, cx, cy).

    Returns
    -------
    (R, t) on success where R is a 3x3 rotation matrix and t is a 3x1
    translation vector.  Returns ``None`` if estimation fails.
    """
    if len(prev_pts) < _MIN_INLIERS or len(curr_pts) < _MIN_INLIERS:
        return None

    prev = prev_pts.reshape(-1, 2).astype(np.float64)
    curr = curr_pts.reshape(-1, 2).astype(np.float64)
    K = camera_matrix.astype(np.float64)

    try:
        E, mask = cv2.findEssentialMat(
            prev, curr, K, cv2.RANSAC, _RANSAC_PROB, _RANSAC_THRESHOLD,
        )
    except cv2.error as exc:
        logger.warning("findEssentialMat failed: %s", exc)
        return None

    if E is None:
        return None

    # Validate essential matrix shape (should be 3x3; may be 3x3xN if
    # multiple hypotheses are returned).
    if E.ndim == 3:
        E = E[:, :, 0]

    inlier_count = int(np.count_nonzero(mask)) if mask is not None else 0
    if inlier_count < _MIN_INLIERS:
        logger.debug(
            "Too few inliers (%d < %d) for pose recovery",
            inlier_count, _MIN_INLIERS,
        )
        return None

    try:
        _, R, t, _ = cv2.recoverPose(E, prev, curr, K, mask=mask)
    except cv2.error as exc:
        logger.warning("recoverPose failed: %s", exc)
        return None

    return R, t
