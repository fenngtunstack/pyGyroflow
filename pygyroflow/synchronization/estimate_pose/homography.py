"""Homography estimation using OpenCV's findHomography.

Estimates a 3x3 homography matrix from matched 2D-2D point pairs using
RANSAC.  In the context of synchronization, homographies are useful when
the scene is approximately planar or the camera undergoes pure rotation,
because the inter-frame transform can be described by a single 2D projective
mapping.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_RANSAC_REPROJ_THRESHOLD = 5.0  # pixels
_MIN_POINTS = 4


def estimate_homography(
    pts1: npt.NDArray[np.float32],
    pts2: npt.NDArray[np.float32],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate a homography from matched point pairs.

    Parameters
    ----------
    pts1, pts2:
        Matched points, shape (N, 2), dtype float32/float64.

    Returns
    -------
    (H, mask) on success; (None, None) on failure.

    H : ndarray (3, 3)
        The estimated homography matrix.
    mask : ndarray (N, 1) uint8
        RANSAC inlier mask.
    """
    if len(pts1) < _MIN_POINTS or len(pts2) < _MIN_POINTS:
        return None, None

    p1 = pts1.reshape(-1, 2).astype(np.float64)
    p2 = pts2.reshape(-1, 2).astype(np.float64)

    try:
        H, mask = cv2.findHomography(
            p1, p2, cv2.RANSAC, _RANSAC_REPROJ_THRESHOLD,
        )
    except cv2.error as exc:
        logger.warning("findHomography failed: %s", exc)
        return None, None

    if H is None:
        return None, None

    return H, mask


# `find_homography.rs:38`: RANSAC, reprojection threshold 0.001 (the points
# are normalized, so this is ~0.1% of the frame half-width, not 5 px),
# maxIters 2000, confidence 0.999. Keywords, not positionals: the Python
# binding's fifth positional slot is the *output mask*, not maxIters.
_FIND_HOMOGRAPHY_KWARGS = {"method": cv2.RANSAC, "ransacReprojThreshold": 0.001,
                           "maxIters": 2000, "confidence": 0.999}


def estimate_pose_find_homography(
    pts1: npt.NDArray[np.float32],
    pts2: npt.NDArray[np.float32],
) -> np.ndarray | None:
    """Recover the inter-frame rotation the way upstream's method 3 does.

    Port of ``find_homography.rs``: ``findHomography`` on the undistorted
    normalized coordinates (done by the caller), ``decomposeHomographyMat``
    against an **identity** camera matrix — with normalized points the
    homography *is* in identity-K space — and then the solution pick. The
    upstream fold is easy to misread as "largest translation wins": its guard
    is ``m < dot => keep stored``, so a candidate only replaces the stored
    solution when its ``t·t`` is **not** greater — the fold keeps the
    *smallest* ``|t|²``. That reads inverted against the usual heuristic for
    pure rotation, but it is what the shipped code does, so it is what runs
    here. NaN pairs are dropped, as in the essential-matrix path.

    Returns the 3x3 rotation, or ``None`` when nothing decomposes.
    """
    p1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
    p1, p2 = p1[finite], p2[finite]
    if len(p1) < 4:
        return None

    try:
        H, _mask = cv2.findHomography(p1, p2, **_FIND_HOMOGRAPHY_KWARGS)
    except cv2.error as exc:
        logger.warning("findHomography failed: %s", exc)
        return None
    if H is None:
        return None

    try:
        num, Rs, Ts, _Ns = cv2.decomposeHomographyMat(H, np.eye(3))
    except cv2.error as exc:
        logger.warning("decomposeHomographyMat failed: %s", exc)
        return None

    # The fold: first solution seeds, then a candidate replaces the stored one
    # exactly when `not (stored_dot < candidate_dot)`.
    best = None
    for R, t in zip(Rs, Ts):
        dot = float(np.dot(t.ravel(), t.ravel()))
        if best is None or not (best[1] < dot):
            best = (R, dot)
    if best is None:
        return None
    return np.asarray(best[0], dtype=np.float64)
