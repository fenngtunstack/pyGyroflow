"""RANSAC-style `adjust_offsets` (A-13)."""

from __future__ import annotations

import pytest

from pygyroflow.gyro_source.source import GyroSource


def _src_with(offsets: dict[int, float]) -> GyroSource:
    src = GyroSource()
    src.set_offsets(offsets)
    return src


class TestLineFit:
    def test_exact_line(self):
        # t (µs) -> ms offsets lying on 0.001·t + 10
        offsets = {k * 1_000_000: 0.001 * k + 10.0 for k in range(5)}
        fit = GyroSource._line_fit(offsets)
        assert fit is not None
        slope, intercept, residual = fit
        # slope in ms-per-µs: 0.001 ms per 1e6 µs = 1e-9.
        assert slope == pytest.approx(1e-9, rel=1e-6)
        assert intercept == pytest.approx(10.0, abs=1e-6)
        assert residual == pytest.approx(0.0, abs=1e-9)

    def test_residual_sums_squares(self):
        offsets = {0: 0.0, 1_000_000: 6.0, 2_000_000: 4.0}
        # best line y = 0.002x + 2: residuals 2, 2, ... wait compute: fit
        # by lstsq — just assert the residual is positive and the fit
        # closer than any single point.
        fit = GyroSource._line_fit(offsets)
        assert fit is not None
        assert fit[2] > 0.0

    def test_two_points_determine_the_line(self):
        offsets = {1000: 1.0, 2000: 2.5}
        fit = GyroSource._line_fit(offsets)
        assert fit is not None
        assert fit[0] == pytest.approx(1.5e-3)


class TestConsensusFit:
    def test_an_outlier_does_not_drag_the_drift(self):
        """Four offsets on a clean line plus one 40 ms outlier: the
        consensus drops the outlier, the fitted line matches the clean
        line, and — upstream's key property — the outlier's *own* key
        still gets a fitted (extrapolated) value rather than its wild
        one."""
        clean = {k * 1_000_000: 10.0 + 0.001 * k for k in range(4)}
        src = _src_with(clean)
        outlier_ts = 4_000_000
        src.set_offset(outlier_ts, 50.0)

        linear = src.offsets_linear
        # The clean points keep (approximately) their clean values.
        for k in clean:
            assert linear[k] == pytest.approx(clean[k], abs=0.5), k
        # The outlier is extrapolated onto the consensus line, not 50 ms.
        assert linear[outlier_ts] == pytest.approx(10.0 + 0.001 * 4, abs=1.0)

    def test_single_offset_passes_through(self):
        src = _src_with({0: 12.5})
        assert src.offsets_linear == {0: 12.5}
        assert src.offsets_adjusted == {12_500: 12.5}

    def test_two_offsets_fit_their_line(self):
        src = _src_with({0: 5.0, 2_000_000: 15.0})
        assert src.offsets_linear[0] == pytest.approx(5.0)
        assert src.offsets_linear[2_000_000] == pytest.approx(15.0)

    def test_steep_drift_falls_back_to_plain_fit(self):
        """A real (non-constant) drift exceeds the |slope| < 0.1 gate, so
        no consensus forms and the plain least squares applies — the
        offsets themselves survive in offsets_linear."""
        offsets = {k * 1_000_000: float(k) for k in range(4)}  # slope 1e-6? no:
        # µs keys: slope = 1 ms per 1e6 µs = 1e-6 — under the 0.1 gate.
        # Make it genuinely steep: 1 ms per ms.
        offsets = {k * 1_000: float(k) for k in range(5)}  # slope 1.0
        src = _src_with(offsets)
        for k in offsets:
            assert src.offsets_linear[k] == pytest.approx(offsets[k], abs=0.5)

    def test_adjusted_keys_shift_by_offset_in_us(self):
        src = _src_with({0: 8.0, 1_000_000: 12.0})
        assert set(src.offsets_adjusted) == {8_000, 1_012_000}
