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
