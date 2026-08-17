"""Synchronization module -- align gyroscope data with video frames.

This module implements the core synchronization pipeline that finds the
time offset between IMU (gyroscope) data and video frames.  It mirrors
Gyroflow's Rust synchronization architecture:

* **Optical flow** -- detect and track features between frames.
* **Pose estimation** -- recover inter-frame camera rotation from flow.
* **Offset search** -- cross-correlate visual and gyro rotations.
* **Autosync** -- one-call orchestration of the full pipeline.
* **OptimSync** -- frequency-domain analysis for sync-point selection.

Quick start::

    from pygyroflow.synchronization import AutosyncProcess

    autosync = AutosyncProcess(camera_matrix=K, fps=30.0)
    offset_ms = autosync.run(
        frames=[(ts, gray) for ts, gray in video],
        gyro_data=[(ts, omega) for ts, omega in imu],
    )
"""

from pygyroflow.synchronization.optical_flow import (
    OpticalFlowDetector,
    AKazeDetector,
    DISDetector,
    PyrLKDetector,
    create_detector,
    METHOD_MAP as OF_METHOD_MAP,
)
from pygyroflow.synchronization.estimate_pose import (
    estimate_rotation,
    estimate_pose_eight_point,
    estimate_essential_matrix,
    estimate_homography,
)
from pygyroflow.synchronization.find_offset import (
    find_time_offset,
    find_offset_visual_features,
    find_offset_rs_sync,
)
from pygyroflow.synchronization.pose_estimator import (
    FrameResult,
    PoseEstimator,
)
from pygyroflow.synchronization.autosync import AutosyncProcess
from pygyroflow.synchronization.optimsync import OptimSync

__all__ = [
    # Optical flow
    "OpticalFlowDetector",
    "AKazeDetector",
    "DISDetector",
    "PyrLKDetector",
    "create_detector",
    "OF_METHOD_MAP",
    # Pose estimation
    "estimate_rotation",
    "estimate_pose_eight_point",
    "estimate_essential_matrix",
    "estimate_homography",
    # Offset search
    "find_time_offset",
    "find_offset_visual_features",
    "find_offset_rs_sync",
    # Core classes
    "FrameResult",
    "PoseEstimator",
    "AutosyncProcess",
    "OptimSync",
]
