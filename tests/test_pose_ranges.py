"""`get_ranges` — contiguous-frame range splitting at >100 ms gaps."""

from __future__ import annotations

from pygyroflow.synchronization.pose_estimator import FrameResult, PoseEstimator


def _est_with(timestamps_us):
    est = PoseEstimator()
    for i, ts in enumerate(timestamps_us):
        est._frames[ts] = FrameResult(timestamp_us=ts, frame_no=i)
    return est


class TestGetRanges:
    def test_contiguous_frames_are_one_range(self):
        est = _est_with([k * 33_333 for k in range(10)])
        assert est.get_ranges() == [(0, 9 * 33_333)]

    def test_a_gap_splits_the_clip(self):
        ts = [k * 33_333 for k in range(5)] + \
             [k * 33_333 + 500_000 for k in range(5, 10)]
        est = _est_with(ts)
        ranges = est.get_ranges()
        assert len(ranges) == 2
        assert ranges[0] == (0, 4 * 33_333)
        assert ranges[1] == (5 * 33_333 + 500_000, 9 * 33_333 + 500_000)

    def test_exactly_100ms_is_not_a_split(self):
        """The guard is `> 100000`, strictly — a frame exactly 100 ms after
        the previous one stays in range."""
        est = _est_with([0, 100_000, 200_000])
        assert est.get_ranges() == [(0, 200_000)]

    def test_just_over_100ms_with_isolated_frames_yields_nothing(self):
        """Upstream's `!=` guards: two isolated frames produce NO ranges at
        all — the first fragment (start 0, end 0) is suppressed by
        ``curr_range_start != prev_ts`` and the second likewise. Pinned as
        upstream semantics, not as a desirable property."""
        est = _est_with([0, 100_001])
        assert est.get_ranges() == []

    def test_empty_and_single_frames(self):
        """Empty input: no ranges (the trailing push needs two distinct
        timestamps). A single frame: the range ``(0, ts)`` — the implicit
        range start is 0, upstream's initial value."""
        assert PoseEstimator().get_ranges() == []
        assert _est_with([42]).get_ranges() == [(0, 42)]

    def test_out_of_order_insertion_is_sorted_first(self):
        """The frame dict is keyed by timestamp; range detection walks them
        in order regardless of insertion order."""
        est = _est_with([3 * 33_333, 0, 33_333, 66_666])
        assert est.get_ranges() == [(0, 3 * 33_333)]
