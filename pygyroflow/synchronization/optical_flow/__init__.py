"""Optical flow sub-package -- feature detection and frame-to-frame tracking.

Three backends are provided:

* ``AKazeDetector``  -- AKAZE keypoints + Hamming matching (robust, slower)
* ``DISDetector``    -- Dense Inverse Search flow (handles large displacements)
* ``PyrLKDetector``  -- Pyramidal Lucas-Kanade tracking (fastest, small motion)
"""

from pygyroflow.synchronization.optical_flow.base import OpticalFlowDetector
from pygyroflow.synchronization.optical_flow.akaze import AKazeDetector
from pygyroflow.synchronization.optical_flow.opencv_dis import DISDetector
from pygyroflow.synchronization.optical_flow.opencv_pyrlk import PyrLKDetector

# Method index mapping -- mirrors Gyroflow's ``of_method`` parameter.
METHOD_MAP: dict[int, type[OpticalFlowDetector]] = {
    0: AKazeDetector,
    1: PyrLKDetector,
    2: DISDetector,
}


def create_detector(method: int = 2) -> OpticalFlowDetector:
    """Factory: create a detector by method index (default 2 = DIS)."""
    cls = METHOD_MAP.get(method, DISDetector)
    return cls()


__all__ = [
    "OpticalFlowDetector",
    "AKazeDetector",
    "DISDetector",
    "PyrLKDetector",
    "METHOD_MAP",
    "create_detector",
]
