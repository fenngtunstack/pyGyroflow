"""The RS-aware sync's lens handling (gap item D-06, the rs_sync half).

Upstream's ``FindOffsetsRssync`` undistorts every track's points with
``undistort_points_for_optical_flow`` (``rs_sync.rs:120-121``) before turning
them into unit rays — its rotation-only cost model compares *directions*, so
feeding it distorted pixels biases every candidate delay the same wrong way.
The port normalized with the pinhole formula only (its own docstring carried
a ``distortion_coeffs: not yet used`` confession). Also ported here:
``FindOffsetsRssync::new``'s readout-time rules — zero falls back to half the
frame interval at *scaled* fps, and a global-shutter lens overrides even an
explicit value to 0.01 ms.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pygyroflow.synchronization.autosync import AutosyncProcess
from pygyroflow.synchronization.find_offset import rs_sync as rs_mod
from pygyroflow.synchronization.find_offset.rs_sync import RollingShutterSync
from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.stabilization.cpu_undistort import undistort_points_for_optical_flow

W, H = 1920, 1080
K = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
FISHEYE = [-0.05, 0.02, 0.0, 0.0]


def _params(**overrides) -> ComputeParams:
    values = dict(
        width=W,
        height=H,
        output_width=W,
        output_height=H,
        camera_matrix=K.copy(),
        distortion_coeffs=list(FISHEYE) + [0.0] * 8,
        distortion_model_name="opencv_fisheye",
    )
    values.update(overrides)
    return ComputeParams(**values)


def _rays(points, dims=(W, H), ts_us=0):
    """The undistorted unit rays the port must produce, built directly."""
    und = undistort_points_for_optical_flow(points, ts_us, _params(), dims)
    out = []
    for x, y in und:
        n = math.sqrt(x * x + y * y + 1.0)
        out.append((x / n, y / n, 1.0 / n))
    return out


class TestTheReadoutTimeRules:
    """``FindOffsetsRssync::new``:74-81, including the global-shutter
    override that wins even against an explicit value."""

    def test_explicit_value_is_kept(self):
        rs = RollingShutterSync({}, frame_readout_time_ms=12.0, fps=30.0)
        assert rs.readout_time_s == pytest.approx(0.012)

    def test_zero_falls_back_to_half_the_frame_interval(self):
        rs = RollingShutterSync({}, frame_readout_time_ms=0.0, fps=30.0)
        assert rs.readout_time_s == pytest.approx((1000.0 / 30.0 / 2.0) / 1000.0)

    def test_the_fallback_uses_scaled_fps_when_given(self):
        rs = RollingShutterSync(
            {}, frame_readout_time_ms=0.0, fps=30.0, scaled_fps=60.0
        )
        assert rs.readout_time_s == pytest.approx((1000.0 / 60.0 / 2.0) / 1000.0)

    def test_global_shutter_overrides_everything(self):
        params = _params(lens=SimpleNamespace(global_shutter=True))
        rs = RollingShutterSync(
            {}, frame_readout_time_ms=12.0, fps=30.0, compute_params=params
        )
        assert rs.readout_time_s == pytest.approx(0.01 / 1000.0)

    def test_no_params_never_trips_the_global_shutter_branch(self):
        rs = RollingShutterSync({}, frame_readout_time_ms=12.0, fps=30.0)
        assert rs.readout_time_s == pytest.approx(0.012)


class TestTracksGoThroughTheLens:
    def test_rays_are_the_undistorted_ones(self):
        pts = [(200.0, 150.0), (1700.0, 900.0), (960.0, 100.0)]
        rs = RollingShutterSync({}, fps=30.0, compute_params=_params())
        rs.add_track_from_frames(
            0, 33_333, pts, [(x + 4.0, y + 4.0) for x, y in pts],
            frame_height=float(H),
            compute_params=_params(),
        )
        track = rs.tracks[0]
        want_a = _rays(pts)
        for got, expected in zip(track.pts_a, want_a):
            assert got == pytest.approx(expected, abs=1e-12)

    def test_per_point_timestamps_use_the_original_rows(self):
        """``rs_sync.rs:132-133`` reads the raw flow point's ``y`` — the
        undistorted coordinate sits at a different height and would smear
        the per-point exposure times."""
        pts = [(200.0, 150.0), (1700.0, 900.0)]
        rs = RollingShutterSync(
            {}, frame_readout_time_ms=10.0, fps=30.0, compute_params=_params()
        )
        rs.add_track_from_frames(
            1_000_000, 1_033_333, pts, pts, frame_height=float(H),
            compute_params=_params(),
        )
        track = rs.tracks[0]
        for (x, y), ta, tb in zip(pts, track.ts_a, track.ts_b):
            assert ta == pytest.approx(1.0 + 0.010 * (y / H))
            assert tb == pytest.approx(1.033333 + 0.010 * (y / H))

    def test_flow_scale_is_honoured(self):
        """Downscaled flow must pass ``points_dims`` — the calibration is
        scaled to the points, and an unstated size silently gives ratio 1."""
        pts = [(100.0, 75.0), (850.0, 450.0)]
        rs = RollingShutterSync({}, fps=30.0, compute_params=_params())
        rs.add_track_from_frames(
            0, 33_333, pts, pts, frame_height=540.0,
            compute_params=_params(), points_dims=(960, 540),
        )
        track = rs.tracks[0]
        want = _rays(pts, dims=(960, 540))
        for got, expected in zip(track.pts_a, want):
            assert got == pytest.approx(expected, abs=1e-12)

    def test_nan_points_are_dropped_not_propagated(self, monkeypatch):
        """A failed undistortion comes back NaN; upstream propagates it into
        the cost (the whole sync dies). The pair is dropped instead."""
        pts = [(200.0, 150.0), (1700.0, 900.0), (960.0, 540.0)]

        def fake(points, ts_us, params, dims):
            out = undistort_points_for_optical_flow(points, ts_us, params, dims)
            out[1] = (float("nan"), float("nan"))
            return out

        # add_track_from_frames imports lazily inside the function body; the
        # package's __init__ shadows the module attribute with the
        # *function* `cpu_undistort`, so resolve the module explicitly.
        import importlib

        cu = importlib.import_module("pygyroflow.stabilization.cpu_undistort")

        monkeypatch.setattr(cu, "undistort_points_for_optical_flow", fake)
        rs = RollingShutterSync({}, fps=30.0, compute_params=_params())
        rs.add_track_from_frames(
            0, 33_333, pts, pts, frame_height=float(H),
            compute_params=_params(),
        )
        track = rs.tracks[0]
        assert len(track.pts_a) == 2
        for x, y, z in track.pts_a + track.pts_b:
            assert math.isfinite(x) and math.isfinite(y) and math.isfinite(z)

    def test_without_params_the_pinhole_path_is_untouched(self):
        pts = [(200.0, 150.0), (1700.0, 900.0)]
        rs = RollingShutterSync({}, fps=30.0)
        rs.add_track_from_frames(
            0, 33_333, pts, pts, frame_height=float(H),
            camera_matrix=K,
        )
        track = rs.tracks[0]
        nx, ny = (200.0 - 960.0) / 900.0, (150.0 - 540.0) / 900.0
        n = math.sqrt(nx * nx + ny * ny + 1.0)
        assert track.pts_a[0] == pytest.approx((nx / n, ny / n, 1.0 / n))


class TestTheAutosyncPlumbing:
    def test_compute_params_reach_the_rs_sync(self, monkeypatch):
        """``_rs_sync_offset`` must hand the lens (and the flow scale) to
        the track builder — otherwise none of the above runs in production."""
        seen = {}

        class StubRs:
            def __init__(self, quats, **kwargs):
                seen["kwargs"] = kwargs

            def add_track_from_frames(self, *args, **kwargs):
                seen["track_args"] = kwargs

            def full_sync(self, *a, **k):
                return None

        monkeypatch.setattr(rs_mod, "RollingShutterSync", StubRs)

        params = _params()
        proc = AutosyncProcess(
            camera_matrix=K, fps=30.0, compute_params=params
        )
        est = proc.pose_estimator
        pts = np.array([[200.0, 150.0], [1700.0, 900.0]])
        from pygyroflow.synchronization.pose_estimator import FrameResult

        est._frames[0] = FrameResult(
            timestamp_us=0, frame_no=0, prev_points=pts, curr_points=pts + 4.0
        )
        est._frames[33_333] = FrameResult(timestamp_us=33_333, frame_no=1)
        frames = [
            (0, np.zeros((540, 960), np.uint8)),
            (33_333, np.zeros((540, 960), np.uint8)),
        ]
        offset = proc._rs_sync_offset(
            frames, {0: None}, frame_readout_time_ms=0.0, search_range_ms=500.0
        )
        assert offset is None  # the stub finds nothing; the calls still ran
        assert seen["kwargs"]["compute_params"] is params
        assert seen["kwargs"]["scaled_fps"] == pytest.approx(30.0)
        assert seen["track_args"]["compute_params"] is params
        assert seen["track_args"]["points_dims"] == (960, 540)

    def test_the_process_keeps_a_reference_for_the_rs_path(self):
        proc = AutosyncProcess(camera_matrix=K, compute_params=_params())
        assert proc._compute_params is not None
