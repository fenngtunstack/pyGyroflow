"""Distortion models package for PyGyroFlow.

Provides a factory ``from_name`` that returns the appropriate
``DistortionModelBase`` subclass for a given model identifier string.

Supported models:
  - opencv_fisheye   OpenCV Fisheye (Kannala-Brandt), 4 coeffs
  - opencv_standard  OpenCV Standard (Brown-Conrady), up to 12 coeffs
  - poly3            Single-coefficient cubic polynomial
  - poly5            Two-coefficient quintic polynomial
  - ptlens           PTLens three-coefficient polynomial
  - insta360         Insta360 (UCM + Brown-Conrady)
  - sony             Sony extended angle-based polynomial
  - gopro_superview  GoPro Superview digital stretch
  - gopro_hyperview  GoPro Hyperview digital stretch
  - digital_stretch  Simple anisotropic scaling
"""

from __future__ import annotations

from .base import DistortionModelBase
from .digital_stretch import DigitalStretchModel
from .gopro_hyperview import GoProHyperviewModel
from .gopro_superview import GoProSuperviewModel
from .insta360 import Insta360Model
from .opencv_fisheye import OpenCVFisheyeModel
from .opencv_standard import OpenCVStandardModel
from .poly3 import Poly3Model
from .poly5 import Poly5Model
from .ptlens import PTLensModel
from .sony import SonyModel

__all__ = [
    "DistortionModelBase",
    "from_name",
    "OpenCVFisheyeModel",
    "OpenCVStandardModel",
    "Poly3Model",
    "Poly5Model",
    "PTLensModel",
    "Insta360Model",
    "SonyModel",
    "GoProSuperviewModel",
    "GoProHyperviewModel",
    "DigitalStretchModel",
]

# Registry mapping model name -> class
_MODEL_REGISTRY: dict[str, type[DistortionModelBase]] = {
    "opencv_fisheye": OpenCVFisheyeModel,
    "opencv_standard": OpenCVStandardModel,
    "poly3": Poly3Model,
    "poly5": Poly5Model,
    "ptlens": PTLensModel,
    "insta360": Insta360Model,
    "sony": SonyModel,
    "gopro_superview": GoProSuperviewModel,
    "gopro_hyperview": GoProHyperviewModel,
    "digital_stretch": DigitalStretchModel,
}


def from_name(name: str) -> DistortionModelBase:
    """Create distortion model by name.

    Falls back to OpenCVFisheyeModel for unknown names, matching
    Gyroflow's default behaviour.
    """
    cls = _MODEL_REGISTRY.get(name, OpenCVFisheyeModel)
    return cls()
