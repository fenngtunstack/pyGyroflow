"""Pose estimation sub-package -- recover camera rotation from optical flow.

Three estimation strategies are provided:

* ``eight_point``     -- Essential matrix + recoverPose (default, robust)
* ``essential_matrix``-- Essential matrix without pose decomposition
* ``homography``      -- Homography for planar / pure-rotation scenes

The public entry point is ``estimate_rotation()``, which dispatches to the
appropriate method based on a method index.
"""

import cv2
import numpy as np
import numpy.typing as npt
import logging

from pygyroflow.synchronization.estimate_pose.eight_point import (
    estimate_pose_eight_point,
)
from pygyroflow.synchronization.estimate_pose.essential_matrix import (
    estimate_essential_matrix,
)
from pygyroflow.synchronization.estimate_pose.homography import (
    estimate_homography,
)

logger = logging.getLogger(__name__)

# Method index mapping -- mirrors Gyroflow's ``pose_method`` parameter.
# 0 = FindEssentialMat, 1 = Almeida (not ported), 2 = EightPoint, 3 = Homography
_METHOD_MAP = {
    0: "essential_matrix",
    2: "eight_point",
    3: "homography",
}


def estimate_rotation(
    prev_pts: npt.NDArray[np.float32],
    curr_pts: npt.NDArray[np.float32],
    camera_matrix: npt.NDArray[np.float64],
    method: int = 0,
) -> np.ndarray | None:
    """Estimate the 3x3 rotation matrix between two frames.

    Parameters
    ----------
    prev_pts, curr_pts:
        Matched point pairs, shape (N, 2).
    camera_matrix:
        3x3 camera intrinsics.
    method:
        Pose estimation method index (0=essential_matrix, 2=eight_point,
        3=homography).

    Returns
    -------
    3x3 rotation matrix (ndarray, float64) or None on failure.
    """
    method_name = _METHOD_MAP.get(method, "essential_matrix")

    if method_name == "eight_point":
        result = estimate_pose_eight_point(prev_pts, curr_pts, camera_matrix)
        if result is not None:
            return result[0]  # rotation matrix only
        return None

    if method_name == "homography":
        H, _mask = estimate_homography(prev_pts, curr_pts)
        if H is None:
            return None
        # Decompose homography into rotations.  OpenCV returns a list of
        # possible decompositions; pick the first valid one.
        try:
            K = camera_matrix.astype(np.float64)
            num, Rs, _norms, _Rts = cv2.decomposeHomographyMat(H, K)
            if num > 0 and len(Rs) > 0:
                return Rs[0]
        except cv2.error as exc:
            logger.warning("decomposeHomographyMat failed: %s", exc)
        return None

    # Default: essential_matrix  (method 0)
    result = estimate_pose_eight_point(prev_pts, curr_pts, camera_matrix)
    if result is not None:
        return result[0]
    return None


__all__ = [
    "estimate_rotation",
    "estimate_pose_eight_point",
    "estimate_essential_matrix",
    "estimate_homography",
]
