"""Abstract base class for optical flow detectors.

All optical flow methods share a common interface: given two consecutive
grayscale frames, they return matched point pairs (N,2) that describe
pixel-level correspondences.  The downstream pose estimator uses these
correspondences to recover inter-frame camera rotation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class OpticalFlowDetector(ABC):
    """Interface that every optical-flow backend must implement."""

    @abstractmethod
    def detect_and_track(
        self,
        prev_frame: npt.NDArray[np.uint8],
        curr_frame: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Detect features in *prev_frame* and find their locations in *curr_frame*.

        Parameters
        ----------
        prev_frame:
            Previous grayscale frame (H, W), dtype uint8.
        curr_frame:
            Current grayscale frame (H, W), dtype uint8.

        Returns
        -------
        prev_points : ndarray, shape (N, 2), dtype float32
            Pixel coordinates (x, y) of matched features in *prev_frame*.
        curr_points : ndarray, shape (N, 2), dtype float32
            Corresponding pixel coordinates in *curr_frame*.

        Notes
        -----
        Implementations must return empty (0,2) arrays on failure rather than
        raising, so callers never need to special-case errors.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable name of the detector (for logging / UI)."""
        ...
