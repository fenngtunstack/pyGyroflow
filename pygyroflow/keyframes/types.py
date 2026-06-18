"""Keyframe types, easing functions, and data classes.

Port of Gyroflow's src/core/keyframes.rs enum definitions.
The KeyframeType enum includes all types from the Rust macro definition,
plus metadata (display color, label, value format) that the Rust code
generates via the define_keyframes! macro.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


class Easing(Enum):
    """Easing function for keyframe interpolation.

    Uses sine-based easing from https://easings.net, matching the Rust
    simple_easing crate calls.
    """

    NoEasing = 0  # Linear
    EaseIn = 1  # sin(x*pi/2 - pi) + 1 -- slow start, fast end
    EaseOut = 2  # sin(x*pi/2) -- fast start, slow end
    EaseInOut = 3  # (sin(x*pi - pi/2) + 1) / 2 -- slow start and end

    @staticmethod
    def _sine_in(x: float) -> float:
        """EaseInSine: sin((x * pi / 2) - pi) + 1 = 1 - cos(x * pi / 2)."""
        return 1.0 - math.cos(x * math.pi / 2.0)

    @staticmethod
    def _sine_out(x: float) -> float:
        """EaseOutSine: sin(x * pi / 2)."""
        return math.sin(x * math.pi / 2.0)

    @staticmethod
    def _sine_in_out(x: float) -> float:
        """EaseInOutSine: (sin(x * pi - pi/2) + 1) / 2 = -(cos(pi*x) - 1) / 2."""
        return -(math.cos(math.pi * x) - 1.0) / 2.0

    def apply(self, alpha: float) -> float:
        """Apply this easing function to a linear alpha in [0, 1].

        Returns the eased alpha value.
        """
        if self == Easing.EaseIn:
            return self._sine_in(alpha)
        if self == Easing.EaseOut:
            return self._sine_out(alpha)
        if self == Easing.EaseInOut:
            return self._sine_in_out(alpha)
        return alpha  # NoEasing

    @staticmethod
    def resolve(easing_a: Easing, easing_b: Easing) -> Easing:
        """Determine the effective easing from two adjacent keyframes.

        Mirrors Rust's Easing::get(). The logic:
        - If previous keyframe is EaseOut or EaseInOut AND next is EaseIn or
          EaseInOut -> EaseInOut
        - If only next is EaseIn or EaseInOut -> EaseOut
        - If only previous is EaseOut or EaseInOut -> EaseIn
        - Otherwise -> NoEasing (linear)
        """
        a_out = easing_a in (Easing.EaseOut, Easing.EaseInOut)
        b_in = easing_b in (Easing.EaseIn, Easing.EaseInOut)

        if a_out and b_in:
            return Easing.EaseInOut
        if b_in:
            return Easing.EaseOut
        if a_out:
            return Easing.EaseIn
        return Easing.NoEasing

    @staticmethod
    def interpolate(
        easing_a: Easing, easing_b: Easing, value_a: float, value_b: float, alpha: float
    ) -> float:
        """Interpolate between two values using resolved easing.

        Args:
            easing_a: Easing of the left (earlier) keyframe.
            easing_b: Easing of the right (later) keyframe.
            value_a: Value at the left keyframe.
            value_b: Value at the right keyframe.
            alpha: Linear interpolation factor in [0, 1].

        Returns:
            Interpolated value.
        """
        resolved = Easing.resolve(easing_a, easing_b)
        x = resolved.apply(alpha)
        return value_a * (1.0 - x) + value_b * x


class KeyframeType(Enum):
    """All keyframe types from Gyroflow.

    Each member carries metadata: color (hex), display text, and a value
    formatter function -- matching the Rust define_keyframes! macro output.
    """

    # Camera
    Fov = ("#8ee6ea", "FOV", lambda v: f"{v:.2f}")
    VideoRotation = ("#eae38e", "Video rotation", lambda v: f"{v:.1f}°")
    ZoomingSpeed = ("#32e595", "Zooming speed", lambda v: f"{v:.2f}s")
    ZoomingCenterX = ("#6fefb6", "Zooming center offset X", lambda v: f"{v * 100:.0f}%")
    ZoomingCenterY = ("#5ddba2", "Zooming center offset Y", lambda v: f"{v * 100:.0f}%")
    MaxZoom = ("#184CC5", "Zoom limit", lambda v: f"{v:.0f}%")

    # Rotation
    AdditionalRotationX = ("#7817ef", "Additional 3D yaw", lambda v: f"{v:.2f}°")
    AdditionalRotationY = ("#9248ec", "Additional 3D pitch", lambda v: f"{v:.2f}°")
    AdditionalRotationZ = ("#ab7ce4", "Additional 3D roll", lambda v: f"{v:.2f}°")

    # Translation
    AdditionalTranslationX = ("#ea2487", "Additional 3D translation X", lambda v: f"{v:.0f}px")
    AdditionalTranslationY = ("#e0539a", "Additional 3D translation Y", lambda v: f"{v:.0f}px")
    AdditionalTranslationZ = ("#e98fbc", "Additional 3D translation Z", lambda v: f"{v:.0f}px")

    # Background
    BackgroundMargin = ("#6e5ddb", "Background margin", lambda v: f"{v:.0f}%")
    BackgroundFeather = ("#9d93e1", "Background feather", lambda v: f"{v:.0f}%")

    # Horizon
    LockHorizonAmount = ("#ed7789", "Horizon lock amount", lambda v: f"{v:.0f}%")
    LockHorizonRoll = ("#e86176", "Horizon lock roll correction", lambda v: f"{v:.1f}°")
    LockHorizonPitchEnabled = (
        "#e86176",
        "Horizon lock pitch enabled",
        lambda v: "On" if v != 0.0 else "Off",
    )
    LockHorizonPitch = ("#e86176", "Horizon lock pitch correction", lambda v: f"{v:.1f}°")

    # Lens
    LensCorrectionStrength = ("#e8ae61", "Lens correction strength", lambda v: f"{v * 100:.0f}%")
    LightRefractionCoeff = ("#CD7F19", "Light refraction coefficient", lambda v: f"{v:.3f}")

    # Smoothing
    SmoothingParamTimeConstant = ("#94ea8e", "Max smoothness", lambda v: f"{v:.2f}")
    SmoothingParamTimeConstant2 = ("#89df82", "Max smoothness at high velocity", lambda v: f"{v:.2f}")
    SmoothingParamSmoothness = ("#7ced74", "Smoothness", lambda v: f"{v:.2f}")
    SmoothingParamPitch = ("#59c451", "Pitch smoothness", lambda v: f"{v:.2f}")
    SmoothingParamRoll = ("#51c485", "Roll smoothness", lambda v: f"{v:.2f}")
    SmoothingParamYaw = ("#88c451", "Yaw smoothness", lambda v: f"{v:.2f}")

    # Speed
    VideoSpeed = ("#f6e926", "Video speed", lambda v: f"{v * 100:.1f}%")

    def __init__(self, color: str, text: str, format_fn: Callable[[float], str]):
        self.color = color
        self.text = text
        self.format_value = format_fn


@dataclass
class Keyframe:
    """A single keyframe with value and easing.

    Mirrors Rust's Keyframe struct. The id field is a random identifier;
    in Python we generate it via random.randint.
    """

    id: int
    value: float
    easing: Easing = Easing.EaseInOut  # Default matches Rust's set() method
