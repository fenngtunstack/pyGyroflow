"""OptimSync — the spectrum fold and the low-motion branch.

Upstream does not take the FFT's modulus. It folds the full complex spectrum
against its own reverse and *adds* the pair:

    zip(cm.iter(), cm.iter().rev()).take(n/2).map(|(a, b)| a + b).norm()

Bin k is therefore ``cm[k] + cm[n-1-k]``, which for a real signal is
``cm[k] + conj(cm[k+1])`` — the sum of two adjacent bins, phase included.
That is materially different from ``|cm[k]|`` (bin 2 of the reference vector
below is 11.58 folded against 0.56 as a modulus), and every threshold
downstream — the rank gate, the 450/650 penalties, the 0.1 segment floor —
is calibrated against the folded form.

Reference values come from a rustfft 6 program fed the same input.
"""

import numpy as np
import pytest

from pygyroflow.synchronization.optimsync import OptimSync, _LOW_MOTION_MF_MAX

# Input chosen so the difference between fold and modulus is visible.
_REF_INPUT = np.array(
    [-0.5767908, 1.6307276, -3.3516407, 1.5041837,
     -0.20455225, -2.620809, 2.5539353, -3.055901],
    dtype=np.float32,
)
_REF_FOLD = [4.722471, 0.6208438, 11.575517, 12.163246]
_REF_MODULUS = [4.120847, 0.67404217, 0.5618743, 12.137311]


def _fold(signal):
    n = len(signal)
    spectrum = np.fft.fft(np.asarray(signal, dtype=np.float32))
    half = n // 2
    return np.abs(spectrum[:half] + spectrum[::-1][:half])


class TestSpectrumFold:
    def test_matches_the_rustfft_reference(self):
        assert np.allclose(_fold(_REF_INPUT), _REF_FOLD, atol=1e-5)

    def test_is_not_the_modulus(self):
        folded = _fold(_REF_INPUT)
        assert not np.allclose(folded, _REF_MODULUS, atol=1e-3)
        # and not 2*|Re| either — bin 2 alone disproves it
        assert folded[2] == pytest.approx(11.575517, abs=1e-5)

    def test_pairing_is_adjacent_bins_for_a_real_signal(self):
        """cm[k] + cm[n-1-k] == cm[k] + conj(cm[k+1])."""
        n = len(_REF_INPUT)
        spectrum = np.fft.fft(_REF_INPUT)
        for k in range(n // 2 - 1):
            assert (spectrum[k] + spectrum[n - 1 - k]) == pytest.approx(
                spectrum[k] + np.conj(spectrum[k + 1])
            )


class TestLowMotionBranch:
    """`mf_max < 50` switches the score to LF+MF.

    A slow pan keeps its energy under 2 Hz, exactly where the normal formula
    penalises it. Without the branch such a clip scores ~0 everywhere and
    yields no sync points at all.
    """

    @staticmethod
    def _run(signal, sr):
        ts = np.arange(len(signal)) / sr * 1000.0
        gyro = np.stack([signal, signal, signal * 0.5], axis=1)
        return OptimSync(ts, gyro).run(
            target_sync_points=5, trim_ranges_s=[(0.0, len(signal) / sr)]
        )

    def test_slow_pan_still_scores(self):
        sr = 200.0
        t = np.arange(int(sr * 20)) / sr
        # 0.3 Hz "hand drift" — firmly inside the low band
        slow = (20.0 * np.sin(2 * np.pi * 0.3 * t)).astype(np.float64)
        points, rank, _ = self._run(slow, sr)
        assert rank.max() > 0.0

    def test_threshold_constant(self):
        assert _LOW_MOTION_MF_MAX == 50.0
