"""Gyro data export.

The ``stab`` column is the stabilized camera motion, which is what an
NLE/Blender import expects. Upstream reaches it by reversing its own
composition — ``(quat_smooth / quat_org).inverse()`` (gyro_export.rs) — and
``gyro.smoothed_quaternions`` holds the *correction* (``smooth⁻¹ * org``), not
the smoothed orientation. Writing the correction through verbatim, as this
did, exports the inverse rotation: the imported track moves the wrong way.
"""

import csv
import math
import os

import numpy as np
import pytest

from pygyroflow.gyro_export import (
    export_gyro_csv,
    export_gyro_csv_full,
    export_gyro_json,
)
from pygyroflow.types.quaternion import Quat64


def _read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


class TestExportGyroCsvFull:
    @staticmethod
    def _export(tmp_path, org, smooth, **kwargs):
        # The pipeline stores the corpus as the correction smooth⁻¹ * org.
        correction = smooth.inverse() * org
        out = tmp_path / "gyro.csv"
        export_gyro_csv_full({0: org}, {0: correction}, str(out), **kwargs)
        return _read_rows(str(out))

    def test_header(self, tmp_path):
        rows = self._export(tmp_path, Quat64.identity(), Quat64.identity())
        assert rows[0] == [
            "timestamp_us", "timestamp_ms",
            "org_quat_w", "org_quat_x", "org_quat_y", "org_quat_z",
            "stab_quat_w", "stab_quat_x", "stab_quat_y", "stab_quat_z",
        ]

    def test_stab_column_is_the_smoothed_orientation(self, tmp_path):
        org = Quat64.from_euler_angles(np.deg2rad(1.0), 0.0, 0.0)
        smooth = Quat64.from_euler_angles(
            0.0, np.deg2rad(2.0), np.deg2rad(3.0)
        )
        row = self._export(tmp_path, org, smooth)[1]
        got = np.array([float(v) for v in row[6:10]])
        assert np.allclose(got, smooth.quaternion(), atol=1e-9)

    def test_org_column_is_the_raw_orientation(self, tmp_path):
        org = Quat64.from_euler_angles(0.0, 0.0, np.deg2rad(20.0))
        row = self._export(tmp_path, org, Quat64.identity())[1]
        got = np.array([float(v) for v in row[2:6]])
        assert np.allclose(got, org.quaternion(), atol=1e-9)

    def test_stab_differs_from_the_correction(self, tmp_path):
        """The regression: the correction is the inverse of the camera motion."""
        org = Quat64.from_euler_angles(np.deg2rad(1.0), 0.0, 0.0)
        smooth = Quat64.from_euler_angles(0.0, np.deg2rad(30.0), 0.0)
        correction = smooth.inverse() * org
        row = self._export(tmp_path, org, smooth)[1]
        got = np.array([float(v) for v in row[6:10]])
        assert not np.allclose(got, correction.quaternion(), atol=1e-6)

    def test_identity_smoothing_exports_identity(self, tmp_path):
        org = Quat64.from_euler_angles(np.deg2rad(7.0), 0.0, 0.0)
        row = self._export(tmp_path, org, Quat64.identity())[1]
        got = np.array([float(v) for v in row[6:10]])
        assert np.allclose(got, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_missing_timestamp_falls_back_to_original(self, tmp_path):
        org = Quat64.from_euler_angles(0.0, np.deg2rad(5.0), 0.0)
        out = tmp_path / "g.csv"
        export_gyro_csv_full({0: org}, {}, str(out))
        got = np.array([float(v) for v in _read_rows(str(out))[1][6:10]])
        assert np.allclose(got, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_fov_column(self, tmp_path):
        rows = self._export(
            tmp_path, Quat64.identity(), Quat64.identity(),
            fovs=[1.25] * 4, fps=30.0,
        )
        assert rows[0][-1] == "fov_scale"
        assert float(rows[1][-1]) == pytest.approx(1.25)


class TestSimpleExports:
    def test_csv_round_trip(self, tmp_path):
        quats = {
            0: Quat64.identity(),
            10_000: Quat64.from_euler_angles(0.1, 0.2, 0.3),
        }
        out = tmp_path / "q.csv"
        export_gyro_csv(quats, str(out))
        rows = _read_rows(str(out))
        assert rows[0][0] == "timestamp_us"
        assert [int(r[0]) for r in rows[1:]] == [0, 10_000]
        assert float(rows[2][5]) == pytest.approx(10.0)

    def test_json_round_trip(self, tmp_path):
        import json

        out = tmp_path / "q.json"
        export_gyro_json({5_000: Quat64.identity()}, str(out))
        with open(out) as handle:
            data = json.load(handle)
        assert list(data) == ["5000"]
        assert data["5000"]["w"] == pytest.approx(1.0)
        assert data["5000"]["timestamp_ms"] == pytest.approx(5.0)
