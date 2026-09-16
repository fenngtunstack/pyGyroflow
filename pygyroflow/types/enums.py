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
    """Resampling kernel, using upstream Gyroflow's values.

    These are the numbers that belong in ``KernelParams.interpolation`` and
    are uploaded to the WGSL shader, where the 2/4/8 entries double as the
    kernel's tap count.

    They are **not** the numbers ``stabilization.cpu_undistort`` takes — that
    path indexes 0=Bilinear / 1=Bicubic / 2=Lanczos4 / 3-6=EWA. The two
    conventions collide (both use "2"), so the conversion lives in
    ``stabilization.cpu_undistort.CPU_TO_UPSTREAM_INTERPOLATION``. The old
    values here (Nearest=1, EWA=16) existed in neither.
    """

    Bilinear = 2
    Bicubic = 4
    Lanczos4 = 8
    RobidouxSharp = 10
    Robidoux = 11
    Mitchell = 12
    CatmullRom = 13


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
