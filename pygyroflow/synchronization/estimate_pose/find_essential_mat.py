"""Faithful port of upstream's ``PoseFindEssentialMat`` (pose method 0).

Upstream ``estimate_pose/find_essential_mat.rs``: undistort both point sets
(at each set's own timestamp — the caller does that part, see
``estimate_rotation``), run OpenCV ``findEssentialMat`` with **LMEDS** on the
normalized coordinates and an identity camera matrix, recover the pose with
the triangulated-points ``recoverPose`` overload (cheirality by
triangulation at ``distance = 1e5``), and fail below 10 inliers.

The dispatch used to route method 0 into the eight-point RANSAC helper, so
the *default* pose method never ran what upstream calls by that name; this
module is what method 0 dispatches to now. ``camera_matrix`` is the identity
for pre-normalized points; a real K is only passed by the ``params=None``
pinhole fallback, which feeds undistorted-in-name-only pixel coordinates —
``findEssentialMat`` normalizes by K internally, so both arrive at the same
epipolar constraint.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# `find_essential_mat.rs:37,42-45`: LMEDS, confidence 0.999, threshold 1e-5
# (ignored by LMEDS, kept for parity), maxIters 4000; triangulated recoverPose
# at distance 1e5; a model with fewer than 10 inliers is "not found".
_FIND_ARGS = (cv2.LMEDS, 0.999, 1e-5, 4000)
_RECOVER_DISTANCE = 100000.0
_MIN_INLIERS = 10


def estimate_pose_find_essential_mat(
    pts1: npt.NDArray[np.float32],
    pts2: npt.NDArray[np.float32],
    camera_matrix: npt.NDArray[np.float64],
) -> np.ndarray | None:
    """Recover the inter-frame rotation the way upstream's method 0 does.

    Parameters
    ----------
    pts1, pts2:
        Matched points, shape (N, 2) — undistorted normalized coordinates
        when the caller has a lens, pixel coordinates in the pinhole
        fallback. NaNs (a failed undistortion) drop the *pair*, which the
        C++ call would survive as garbage in the least-squares sense; the
        Python binding turns them into a hard error instead, so the filter
        is the difference between "one point fewer" and "no pose at all".
    camera_matrix:
        ``np.eye(3)`` for normalized points, the real intrinsics for the
        pinhole fallback.

    Returns
    -------
    3x3 rotation matrix, or ``None`` when OpenCV finds no model or the
    inlier count is under 10.
    """
    p1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
    p1, p2 = p1[finite], p2[finite]
    if len(p1) < 8:
        return None

    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    try:
        E, _mask = cv2.findEssentialMat(p1, p2, K, *_FIND_ARGS)
    except cv2.error as exc:
        logger.warning("findEssentialMat failed: %s", exc)
        return None
    if E is None:
        return None
    if E.ndim == 3:
        E = E[:, :, 0]

    try:
        # `recover_pose_triangulated(..., 100000.0, mask, ...)`: the overload
        # that filters correspondences by triangulated cheirality rather than
        # just the sign test.
        inliers, R, _t, _out_mask = cv2.recoverPose(E, p1, p2, K, _RECOVER_DISTANCE)
    except cv2.error as exc:
        logger.warning("recoverPose failed: %s", exc)
        return None

    if int(inliers) < _MIN_INLIERS:
        logger.debug("PoseFindEssentialMat: only %d inliers", int(inliers))
        return None
    return R
