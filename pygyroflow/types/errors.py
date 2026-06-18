"""PyGyroFlow exception hierarchy."""


class GyroflowError(Exception):
    """Base exception for all PyGyroFlow errors."""


class TelemetryParseError(GyroflowError):
    """Failed to parse telemetry / motion data from video file."""


class LensProfileError(GyroflowError):
    """Invalid or missing lens profile."""


class StabilizationError(GyroflowError):
    """Error during stabilization computation."""


class GPUError(GyroflowError):
    """GPU / wgpu pipeline error."""


class VideoIOError(GyroflowError):
    """Video input/output error."""
