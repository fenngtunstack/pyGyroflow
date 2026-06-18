"""Smoothing algorithms for PyGyroFlow.

Provides all smoothing algorithms ported from Gyroflow:
- NoSmoothing: passthrough (no smoothing)
- DefaultAlgo: adaptive velocity-sensing smoothing (primary algorithm)
- PlainSmoothing: fixed time constant exponential smoothing
- FixedSmoothing: locks camera to a fixed orientation
- HorizonLock: post-processor for horizon level correction
- Smoothing: registry/manager that ties everything together
"""

from pygyroflow.smoothing.base import SmoothingAlgorithm
from pygyroflow.smoothing.default_algo import DefaultAlgo
from pygyroflow.smoothing.fixed import FixedSmoothing
from pygyroflow.smoothing.horizon import HorizonLock
from pygyroflow.smoothing.none import NoSmoothing
from pygyroflow.smoothing.plain import PlainSmoothing
from pygyroflow.smoothing.registry import Smoothing
from pygyroflow.smoothing.trim import get_max_angles, get_trimmed_quats

__all__ = [
    "SmoothingAlgorithm",
    "NoSmoothing",
    "DefaultAlgo",
    "PlainSmoothing",
    "FixedSmoothing",
    "HorizonLock",
    "Smoothing",
    "get_trimmed_quats",
    "get_max_angles",
]
