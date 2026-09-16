"""The visual-rotation signal the offset search correlates against (gap G-10).

`get_visual_rotations` assembles the angular-velocity signal from the
per-frame pose estimates. Upstream does it in `recalculate_gyro_data`
(core/synchronization/mod.rs) and three details there were missing here:

* the sample belongs *between* two frames, not on one of them,
* a frame whose pose estimate failed can borrow an interpolated value,
* an optional low-pass filter can run over the result.

The first one matters most: the offset search measures a time shift, so a
half-frame bias applied to every sample is a constant error in the quantity
being measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.synchronization.pose_estimator import FrameResult, PoseEstimator

FPS = 30.0
STEP_US = int(round(1_000_000.0 / FPS))  # 33333


def estimator_with(eulers, step_us=STEP_US):
    """A PoseEstimator holding one frame per entry of *eulers*.

    ``eulers`` maps a frame index to a (wx, wy, wz) tuple in rad/s, or to
    None for a frame whose pose estimation failed.
    """
    estimator = PoseEstimator()
    for index, euler in enumerate(eulers):
        ts = index * step_us
        estimator._frames[ts] = FrameResult(
            timestamp_us=ts, frame_no=index, euler_angles=euler
        )
    return estimator


def timestamps(rotations):
    return [ts for ts, _ in rotations]


class TestMidpointStamping:
    def test_sample_lands_between_this_frame_and_the_next(self):
        """The motion between frames i and i+1 is attributed to the middle of
        that interval, not to frame i. An exact fixture (1 ms spacing) so the
        midpoints are exact integers and the assertion is not restating the
        rounding."""
        estimator = estimator_with([(0.1, 0.0, 0.0)] * 5, step_us=1000)
        assert timestamps(estimator.get_visual_rotations()) == [
            500, 1500, 2500, 3500, 4000,
        ]

    def test_the_step_between_samples_is_one_frame(self):
        """The stamp is a shift, not a rescaling: consecutive samples stay
        one frame apart, so the search sees the same rate it always did.
        Exactly one frame apart on an exact fixture; at 30 fps the midpoint
        lands on a half microsecond and rounds to either side, which is
        four orders of magnitude below the search's own resolution."""
        estimator = estimator_with([(0.1, 0.0, 0.0)] * 6, step_us=1000)
        stamped = timestamps(estimator.get_visual_rotations())
        deltas = [b - a for a, b in zip(stamped, stamped[1:])]
        assert deltas == [1000] * 4 + [500]

        # The final sample keeps its own timestamp instead of a midpoint, so
        # the last gap is half a frame wider by construction — compare only
        # the gaps between two stamped samples.
        real_rate = timestamps(estimator_with([(0.1, 0.0, 0.0)] * 6).get_visual_rotations())
        for a, b in zip(real_rate[:-2], real_rate[1:-1]):
            assert b - a == pytest.approx(STEP_US, abs=2)

    def test_property_for_a_realistic_frame_rate(self):
        """At 30 fps the midpoints fall on .5 µs, so pin the *rule* rather
        than hand-derived integers that depend on rounding mode."""
        estimator = estimator_with([(0.1, 0.0, 0.0)] * 5)
        stamped = timestamps(estimator.get_visual_rotations())
        source = sorted(estimator._frames)
        for index, ts in enumerate(stamped[:-1]):
            assert ts == round(source[index] + (source[index + 1] - source[index]) / 2)
        assert stamped[-1] == source[-1]

    def test_uneven_frame_spacing_uses_the_real_next_timestamp(self):
        """Variable frame rate: the midpoint is of the actual interval, not
        of the average frame duration."""
        estimator = PoseEstimator()
        for index, ts in enumerate([0, 10_000, 40_000]):
            estimator._frames[ts] = FrameResult(
                timestamp_us=ts, frame_no=index, euler_angles=(0.1, 0.0, 0.0)
            )
        assert timestamps(estimator.get_visual_rotations()) == [
            5_000, 25_000, 40_000,
        ]

    def test_single_frame_keeps_its_own_timestamp(self):
        estimator = estimator_with([(0.1, 0.0, 0.0)])
        assert timestamps(estimator.get_visual_rotations()) == [0]

    def test_results_are_ordered_by_timestamp_not_frame_number(self):
        """The search walks the signal as a time series, so ordering has to
        come from the timestamps. Here `frame_no` runs the other way round —
        a caller that numbered frames in arrival order — and the output must
        still be a time series."""
        estimator = PoseEstimator()
        source = [0, 1000, 2000, 3000]
        for ts in source:
            estimator._frames[ts] = FrameResult(
                timestamp_us=ts,
                frame_no=len(source) - 1 - ts // 1000,  # reversed
                euler_angles=(0.1, 0.0, 0.0),
            )
        assert timestamps(estimator.get_visual_rotations()) == [
            500, 1500, 2500, 3000,
        ]


class TestAxesAndUnits:
    def test_x_and_y_are_swapped_and_radians_become_degrees(self):
        """The camera axes are not the IMU axes, and the signal has to be in
        the same units as the gyro it is correlated against."""
        estimator = estimator_with([(1.0, 2.0, 3.0)] * 2)
        _, angular_velocity = estimator.get_visual_rotations()[0]
        degrees = 180.0 / np.pi
        # (wx, wy, wz) -> (wy, wx, wz), scaled rad/s -> deg/s.
        assert angular_velocity == pytest.approx([2.0 * degrees, 1.0 * degrees, 3.0 * degrees])

    def test_output_is_float64(self):
        estimator = estimator_with([(0.1, 0.2, 0.3)] * 2)
        for _, angular_velocity in estimator.get_visual_rotations():
            assert angular_velocity.dtype == np.float64
            assert angular_velocity.shape == (3,)


class TestGapInterpolation:
    def test_final_pass_fills_an_interior_gap(self):
        # Frame 2 failed; frames 1 and 3 read 1.0 and 3.0 rad/s.
        estimator = estimator_with(
            [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), None, (3.0, 3.0, 3.0),
             (3.0, 3.0, 3.0)]
        )
        rotations = estimator.get_visual_rotations(final_pass=True)
        assert len(rotations) == 5
        # The frame in the gap sits exactly half way between its neighbours.
        _, gap = rotations[2]
        assert gap == pytest.approx([2.0 * 180.0 / np.pi] * 3)

    def test_fast_pass_leaves_the_gap_out(self):
        """Upstream's cheap intermediate pass skips interpolation; the port's
        default is the final pass, which is the one that feeds the search."""
        estimator = estimator_with(
            [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), None, (3.0, 3.0, 3.0)]
        )
        assert len(estimator.get_visual_rotations(final_pass=False)) == 3
        assert len(estimator.get_visual_rotations(final_pass=True)) == 4

    def test_interpolation_is_weighted_by_position(self):
        estimator = estimator_with([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, None, (3.0, 3.0, 3.0)])
        rotations = estimator.get_visual_rotations(final_pass=True)
        assert len(rotations) == 5
        assert rotations[2][1] == pytest.approx([1.0 * 180.0 / np.pi] * 3)
        assert rotations[3][1] == pytest.approx([2.0 * 180.0 / np.pi] * 3)

    @pytest.mark.parametrize("gap_index", [0, 3])
    def test_a_gap_at_either_end_is_not_extrapolated(self, gap_index):
        """Filling an edge gap would mean inventing a magnitude past the data
        rather than blending between two measurements."""
        eulers = [(1.0, 1.0, 1.0)] * 4
        eulers[gap_index] = None
        estimator = estimator_with(eulers)
        assert len(estimator.get_visual_rotations(final_pass=True)) == 3

    def test_all_gaps_produces_nothing(self):
        estimator = estimator_with([None] * 4)
        assert estimator.get_visual_rotations() == []

    def test_no_gaps_is_unchanged_by_the_pass(self):
        eulers = [(0.1 * i, 0.2 * i, 0.3 * i) for i in range(5)]
        estimator = estimator_with(eulers)
        final = estimator.get_visual_rotations(final_pass=True)
        fast = estimator.get_visual_rotations(final_pass=False)
        assert len(final) == len(fast) == 5
        for (ts_a, a), (ts_b, b) in zip(final, fast):
            assert ts_a == ts_b
            assert a == pytest.approx(b)


class TestLowpass:
    def test_default_is_no_filtering(self):
        """Upstream's `lpf` starts at 0 and nothing in its CLI sets it."""
        estimator = estimator_with([(0.1 * i, 0.0, 0.0) for i in range(10)])
        assert estimator._lpf == 0.0
        assert len(estimator.get_visual_rotations()) == 10

    def test_frequency_is_quantised_to_hundredths(self):
        """Upstream stores it as an integer of hundredths of a Hz."""
        estimator = estimator_with([(0.1, 0.0, 0.0)] * 3)
        estimator.lowpass_filter(7.777, FPS)
        assert estimator._lpf == pytest.approx(7.77)

    def test_filtering_smooths_a_spike_and_keeps_the_timestamps(self):
        eulers = [(0.0, 0.0, 0.0)] * 20
        eulers[10] = (5.0, 5.0, 5.0)  # one-frame spike

        unfiltered = estimator_with(eulers)
        plain = unfiltered.get_visual_rotations()

        filtered = estimator_with(eulers)
        filtered.lowpass_filter(5.0, FPS)
        smooth = filtered.get_visual_rotations()

        assert timestamps(smooth) == timestamps(plain)
        spike_plain = float(np.abs(plain[10][1]).max())
        spike_smooth = float(np.abs(smooth[10][1]).max())
        assert spike_smooth < spike_plain * 0.6

    def test_zero_cutoff_is_a_pass_through(self):
        eulers = [(0.3 * i, -0.1 * i, 0.05 * i) for i in range(12)]
        estimator = estimator_with(eulers)
        before = estimator.get_visual_rotations()
        estimator.lowpass_filter(0.0, FPS)
        after = estimator.get_visual_rotations()
        for (_, a), (_, b) in zip(before, after):
            assert a == pytest.approx(b)

    def test_cutoff_above_nyquist_is_a_pass_through(self):
        """The filter itself refuses those; this pins that the caller does
        not end up with a different signal because of it."""
        eulers = [(0.3 * i, 0.0, 0.0) for i in range(12)]
        estimator = estimator_with(eulers)
        before = estimator.get_visual_rotations()
        estimator.lowpass_filter(FPS, FPS)  # == Nyquist
        after = estimator.get_visual_rotations()
        for (_, a), (_, b) in zip(before, after):
            assert a == pytest.approx(b)

    def test_process_level_passthrough(self):
        from pygyroflow.synchronization.autosync import AutosyncProcess

        process = AutosyncProcess(fps=FPS, offset_method=1)
        process.set_lpf(8.0)
        assert process.pose_estimator._lpf == pytest.approx(8.0)
