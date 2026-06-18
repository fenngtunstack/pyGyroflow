"""Enumerations matching Gyroflow's Rust enums used in GPU pipeline and lens models."""

from __future__ import annotations

from enum import IntEnum


class BackgroundMode(IntEnum):
    """How to fill pixels outside the stabilized frame."""

    SolidColor = 0
    RepeatPixels = 1
    MirrorPixels = 2
    MarginWithFeather = 3


class ReadoutDirection(IntEnum):
    """Rolling shutter readout direction."""

    TopToBottom = 0
    BottomToTop = 1
    LeftToRight = 2
    RightToLeft = 3

    def is_horizontal(self) -> bool:
        return self in (ReadoutDirection.LeftToRight, ReadoutDirection.RightToLeft)

    def is_inverted(self) -> bool:
        return self in (ReadoutDirection.BottomToTop, ReadoutDirection.RightToLeft)


class Interpolation(IntEnum):
    """Image interpolation method for GPU undistortion."""

    Nearest = 1
    Bilinear = 2
    Bicubic = 4
    Lanczos4 = 8
    EWA = 16


class DistortionModelType(IntEnum):
    """Lens distortion model identifier.

    Must match the values used in Gyroflow's Rust codebase and WGSL shader.
    """

    OpenCVFisheye = 0
    OpenCVStandard = 1
    Poly3 = 2
    Poly5 = 3
    PtLens = 4
    Insta360 = 5
    Sony = 6
    GoProSuperview = 7
    GoProHyperview = 8
    DigitalStretch = 9
