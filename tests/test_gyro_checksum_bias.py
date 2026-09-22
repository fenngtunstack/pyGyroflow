"""A-11/A-12: checksum completeness and find_bias's offset shift."""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.gyro_source.file_metadata import FileMetadata
from pygyroflow.gyro_source.imu_transforms import IMUTransforms
from pygyroflow.gyro_source.source import GyroSource
from pygyroflow.types.time_types import TimeIMU


def _source_with_imu(samples, **transform_over):
    src = GyroSource()
    meta = FileMetadata(detected_source="Test")
    meta.raw_imu = [
        TimeIMU(timestamp_ms=t, gyro=np.array(g)) for t, g in samples
    ]
    src.file_metadata = meta
    src.raw_imu = list(meta.raw_imu)
    for name, value in transform_over.items():
        setattr(src.imu_transforms, name, value)
    return src


class TestGetChecksumCompleteness:
    def test_image_orientations_length_is_hashed(self):
        src = _source_with_imu([(0.0, (1.0, 0.0, 0.0))])
        base = src.get_checksum()
        src.file_metadata.image_orientations = {0: None, 1: None, 2: None}
        assert src.get_checksum() != base

    def test_transforms_and_method_move_the_checksum(self):
        src = _source_with_imu([(0.0, (1.0, 0.0, 0.0))])
        base = src.get_checksum()
        src.imu_transforms.imu_rotation_angles = (10.0, 0.0, 0.0)
        rotated = src.get_checksum()
        src.imu_transforms.gyro_bias = [0.1, 0.0, 0.0]
        biased = src.get_checksum()
        src.integration_method = 3
        method = src.get_checksum()
        assert len({base, rotated, biased, method}) == 4

    def test_offsets_move_the_checksum(self):
        src = _source_with_imu([(0.0, (1.0, 0.0, 0.0))])
        base = src.get_checksum()
        src.offsets[0] = 42.0
        assert src.get_checksum() != base

    def test_stream_endpoints_move_the_checksum(self):
        src = GyroSource()
        from pygyroflow.types.quaternion import Quat64
        from scipy.spatial.transform import Rotation

        src.quaternions = {
            0: Quat64(Rotation.identity()),
            1000: Quat64(Rotation.from_euler("y", 0.1)),
        }
        base = src.get_checksum()
        src.quaternions[1000] = Quat64(Rotation.from_euler("y", 0.2))
        assert src.get_checksum() != base


class TestFindBiasOffsetShift:
    def test_the_window_shifts_by_the_sync_offset(self):
        """Samples at gyro-time 100..200 ms; a +50 ms sync offset means the
        video window 100..200 covers gyro-time 50..150 — only the first
        sample falls inside (mod.rs:933-935)."""
        src = _source_with_imu([
            (100.0, (1.0, 2.0, 3.0)),
            (180.0, (10.0, 20.0, 30.0)),
        ])
        # set_offset (not a bare dict write): it also rebuilds
        # offsets_adjusted, which is what the video-timeline lookup reads.
        src.set_offset(0, 50.0)
        bias = src.find_bias(100.0, 200.0)
        assert bias == pytest.approx((-1.0, -2.0, -3.0))

    def test_zero_offset_is_the_plain_window(self):
        """The bounds are strict (``> ts_start && < ts_stop``): a sample
        exactly at the start is excluded, one at 180 included."""
        samples = [(100.0, (1.0, 2.0, 3.0)), (180.0, (5.0, 6.0, 7.0))]
        src = _source_with_imu(samples)
        bias = src.find_bias(100.0, 200.0)
        assert bias == pytest.approx((-5.0, -6.0, -7.0))
