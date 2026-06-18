"""Native telemetry parser bridge via PyO3.

Interfaces with telemetry_parser_bridge (Rust + PyO3) for high-performance
telemetry extraction from video files. The bridge must be built separately:

    cd telemetry_parser_bridge && maturin develop --release
"""


def parse_telemetry(path: str, sample_index: int | None = None):
    """Parse telemetry using the native Rust bridge.

    Args:
        path: Path to the video file.
        sample_index: Optional sample index for multi-stream files.

    Returns:
        FileMetadata with parsed gyro data, or None if parsing fails.

    Raises:
        ImportError: If telemetry_parser_bridge is not installed.
    """
    try:
        import telemetry_parser_bridge as _tpb

        return _tpb.parse_telemetry_file(path, sample_index)
    except ImportError:
        raise ImportError(
            "telemetry_parser_bridge not available. "
            "Build it with: cd telemetry_parser_bridge && maturin develop --release"
        )
