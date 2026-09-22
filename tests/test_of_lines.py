"""`filter_of_lines` and `get_of_lines_for_timestamp` (B-13)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygyroflow.synchronization.pose_estimator import (
    FrameResult,
    PoseEstimator,
)


def _est_with_pairs(frames, pts_fn):
    """frames: [(ts, frame_size)]; pts_fn(i) -> (prev, curr) arrays."""
    est = PoseEstimator()
    for i, (ts, size) in enumerate(frames):
        prev, curr = pts_fn(i)
        fr = FrameResult(
            timestamp_us=ts, frame_no=i,
            prev_points=prev, curr_points=curr,
        )
        fr.frame_size = size  # type: ignore[attr-defined]
        est._frames[ts] = fr
    return est


class TestFilterOfLines:
    def test_uniform_direction_keeps_everything(self):
        pts1 = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        pts2 = [(5.0, 0.0), (15.0, 0.0), (25.0, 0.0)]  # all +x
        out = PoseEstimator.filter_of_lines(((0, pts1), (33_333, pts2)), 1.0)
        assert out is not None
        assert len(out[0][1]) == 3 and len(out[1][1]) == 3

    def test_outliers_are_dropped(self):
        pts1 = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        pts2 = [(5.0, 0.0), (15.0, 0.0), (25.0, 0.0), (30.0, 20.0)]  # last is ~76°
        out = PoseEstimator.filter_of_lines(((0, pts1), (33_333, pts2)), 1.0)
        assert out is not None
        assert len(out[0][1]) == 3
        assert out[0][1][-1][0] == pytest.approx(20.0)  # the outlier gone

    def test_scale_multiplies_survivors(self):
        pts1 = [(0.0, 0.0)]
        pts2 = [(5.0, 0.0)]
        out = PoseEstimator.filter_of_lines(((0, pts1), (33_333, pts2)), 2.0)
        assert out is not None
        assert out[1][1][0] == (pytest.approx(10.0), pytest.approx(0.0))

    def test_none_passthrough(self):
        assert PoseEstimator.filter_of_lines(None, 1.0) is None

    def test_orthogonal_majority_kills_both_directions(self):
        """Two +x lines and one +90° line: mean ≈ 30°, every line deviates
        ~30° or more — with the strict ``<`` gate the whole set dies. That
        is upstream's behaviour with a naive circular mean; pinned."""
        pts1 = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        pts2 = [(5.0, 0.0), (15.0, 0.0), (20.0, 30.0)]  # 0°, 0°, 90°
        out = PoseEstimator.filter_of_lines(((0, pts1), (33_333, pts2)), 1.0)
        assert out is not None
        assert len(out[0][1]) == 0

    def test_consensus_direction_survives(self):
        """Three collinear + one outlier: the mean sits with the three and
        only the outlier is dropped."""
        pts1 = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        pts2 = [(5.0, 0.0), (15.0, 0.0), (25.0, 0.0), (30.0, 30.0)]  # 3x 0°, 1x 45°
        out = PoseEstimator.filter_of_lines(((0, pts1), (33_333, pts2)), 1.0)
        assert out is not None
        assert len(out[0][1]) == 3


class TestGetOfLinesForTimestamp:
    def _frames(self):
        return [(k * 33_333, (64, 48)) for k in range(5)]

    def _pts(self, i):
        prev = np.array([[10.0 + i, 20.0], [30.0, 40.0]], dtype=np.float32)
        return prev, prev + 3.0

    def test_returns_the_pairs_and_frame_size(self):
        est = _est_with_pairs(self._frames(), self._pts)
        pts, size = est.get_of_lines_for_timestamp(0)
        assert size == (64, 48)
        ts1, p1 = pts[0]
        ts2, p2 = pts[1]
        assert ts1 == 0 and ts2 == 0
        assert len(p1) == 2 and p2[0][0] == pytest.approx(13.0)

    def test_next_no_skips_frames(self):
        est = _est_with_pairs(self._frames(), self._pts)
        pts, _size = est.get_of_lines_for_timestamp(0, next_no=2)
        assert pts[0][1][0][0] == pytest.approx(12.0)  # frame 2's prev points

    def test_two_ms_tolerance(self):
        est = _est_with_pairs(self._frames(), self._pts)
        assert est.get_of_lines_for_timestamp(1500)[0] is not None
        assert est.get_of_lines_for_timestamp(9000)[0] is None

    def test_filter_is_applied_when_asked(self):
        # Two 0° lines + one 45° outlier: mean = 15°, the collinear pair
        # deviates 15° (< 30, kept), the outlier 30° (strictly-greater,
        # dropped). A 90° outlier would pull the mean to 30° and kill the
        # collinear pair too — the naive-circular-mean hazard one test up
        # pins separately.
        def pts(i):
            prev = np.array(
                [[10.0, 10.0], [40.0, 10.0], [70.0, 10.0]], dtype=np.float32)
            curr = np.array(
                [[13.0, 10.0], [43.0, 10.0], [85.0, 25.0]], dtype=np.float32)
            return prev, curr

        est = _est_with_pairs(self._frames(), pts)
        pts_out, _ = est.get_of_lines_for_timestamp(0, apply_filter=True)
        assert len(pts_out[0][1]) == 2
        pts_raw, _ = est.get_of_lines_for_timestamp(0, apply_filter=False)
        assert len(pts_raw[0][1]) == 3

    def test_scale_reaches_the_filter(self):
        est = _est_with_pairs(self._frames(), self._pts)
        pts, _ = est.get_of_lines_for_timestamp(0, scale=2.0, apply_filter=True)
        assert pts[1][1][0][0] == pytest.approx(26.0)

    def test_missing_points_is_none(self):
        est = PoseEstimator()
        est._frames[0] = FrameResult(timestamp_us=0, frame_no=0)
        assert est.get_of_lines_for_timestamp(0) == (None, None)

    def test_multi_frame_distance_is_explicitly_unimplemented(self):
        est = _est_with_pairs(self._frames(), self._pts)
        with pytest.raises(NotImplementedError, match="B-14"):
            est.get_of_lines_for_timestamp(0, num_frames=2)
