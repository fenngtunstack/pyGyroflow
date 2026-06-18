"""Essential matrix estimation using OpenCV's findEssentialMat.

Thin wrapper that returns the essential matrix and the RANSAC inlier mask
without performing pose decomposition.  Useful when the caller wants to
inspect or post-process the essential matrix before recovering the pose.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_RANSAC_PROB = 0.999
_RANSAC_THRESHOLD = 1.0


def estimate_essential_matrix(
    pts1: npt.NDArray[np.float32],
    pts2: npt.NDArray[np.float32],
    camera_matrix: npt.NDArray[np.float64],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate the essential matrix from matched 2D point correspondences.

    Parameters
    ----------
    pts1, pts2:
        Matched points in frames 1 and 2, shape (N, 2).
    camera_matrix:
        3x3 camera intrinsic matrix.

    Returns
    -------
    (E, mask) on success; (None, None) on failure.

    E : ndarray (3, 3) or (3, 3, M)
        The estimated essential matrix.
    mask : ndarray (N, 1) uint8
        Inlier mask from RANSAC.
    """
    if len(pts1) < 8 or len(pts2) < 8:
        return None, None

    p1 = pts1.reshape(-1, 2).astype(np.float64)
    p2 = pts2.reshape(-1, 2).astype(np.float64)
    K = camera_matrix.astype(np.float64)

    try:
        E, mask = cv2.findEssentialMat(
            p1, p2, K, cv2.RANSAC, _RANSAC_PROB, _RANSAC_THRESHOLD,
        )
    except cv2.error as exc:
        logger.warning("findEssentialMat failed: %s", exc)
        return None, None

    return E, mask
