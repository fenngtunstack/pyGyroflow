"""The keyframed-window half of dynamic zoom (D-09, zoom_dynamic).

When ``ZoomingSpeed`` is keyframed — or video speed feeds the window via
``video_speed_affects_zooming`` — every timestamp gets its own smoothing
window (``zoom_dynamic.rs:24-69``). The port only ever had the static
branch, so a keyframed zoom speed silently smoothed with the default
window instead.
"""

from __future__ import annotations

import pytest

from pygyroflow.keyframes.manager import KeyframeManager
from pygyroflow.keyframes.types import Easing, KeyframeType
from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.zooming.zoom_dynamic import (
    ZoomMethod,
    _envelope_follower,
    _min_rolling_dynamic,
    _DataPerTimestamp,
    compute,
)


def _params(window: float = 1.0, video_speed: float = 1.0,
            speed_affects: bool = False, keyframes: KeyframeManager | None = None,
            fps: float = 30.0) -> ComputeParams:
    return ComputeParams(
        width=1920, height=1080, output_width=1920, output_height=1080,
        scaled_fps=fps,
        adaptive_zoom_window=window,
        video_speed=video_speed,
        video_speed_affects_zooming=speed_affects,
        keyframes=keyframes if keyframes is not None else KeyframeManager(),
    )


def _timestamps(n: int, fps: float = 30.0) -> list[tuple[int, float]]:
    return [(k, k / fps * 1000.0) for k in range(n)]


def _plateau_signal(n: int = 121) -> list[float]:
    """Constant 1.0 with one deep notch — smoothing widens/shrinks the
    response to it."""
    vals = [1.0] * n
    vals[60] = 0.5
    return vals


class TestTheSwitch:
    def test_keyframed_speed_changes_the_result_vs_static(self):
        """Same signal, same nominal window: with ZoomingSpeed keyframed to a
        near-instant window early on, the early frames barely smooth; the
        static path smooths everything with the full window."""
        kf = KeyframeManager()
        kf.set_keyframe(
            KeyframeType.ZoomingSpeed, 0, 0.0001, easing=Easing.NoEasing
        )
        kf.set_keyframe(
            KeyframeType.ZoomingSpeed, 4000.0 * 1000.0, 1.0, easing=Easing.NoEasing
        )
        signal = _plateau_signal()

        static, _ = compute(
            _params(window=1.0), list(signal), _timestamps(len(signal)),
            ZoomMethod.GaussianFilter,
        )
        dynamic, _ = compute(
            _params(window=1.0, keyframes=kf), list(signal), _timestamps(len(signal)),
            ZoomMethod.GaussianFilter,
        )
        # Early: keyframed window ≈ 0 -> barely touched; static full window
        # lifts the 0.5 notch substantially.
        assert static[60] < 0.9
        assert dynamic[60] == pytest.approx(0.5)

    def test_video_speed_scales_the_envelope_window(self):
        """Half-speed footage halves the keyframed window → bigger envelope
        alpha → the follower tracks the notch sooner. (Upstream quirk, kept:
        on the *Gaussian* method the per-timestamp window changes nothing —
        the frame count and kernel come from the static parameter either
        way — so the speed effect only exists on the envelope.)"""
        signal = _plateau_signal()
        half, _ = compute(
            _params(window=1.0, video_speed=0.5, speed_affects=True),
            list(signal), _timestamps(len(signal)), ZoomMethod.EnvelopeFollower,
        )
        quarter, _ = compute(
            _params(window=1.0, video_speed=0.25, speed_affects=True),
            list(signal), _timestamps(len(signal)), ZoomMethod.EnvelopeFollower,
        )
        assert half != quarter
        # The notch *centre* is pinned at 0.5 by the min(); the direction
        # shows one frame out: the smaller window (bigger alpha) tracks the
        # notch more sharply, so its neighbours sit closer to their own x.
        assert quarter[59] > half[59]
        assert quarter[61] > half[61]

    def test_gaussian_keyframed_window_is_quirks_equal_to_static(self):
        """Upstream's per-timestamp Gaussian derives its frame count and
        kernel from the static parameter, so a uniformly keyframed speed
        reproduces the static result exactly. Pinned, because "fixing" it
        would silently diverge from Gyroflow."""
        kf = KeyframeManager()
        kf.set_keyframe(
            KeyframeType.ZoomingSpeed, 0, 1.0, easing=Easing.NoEasing
        )
        signal = _plateau_signal()
        through_kf, _ = compute(
            _params(window=1.0, keyframes=kf), list(signal),
            _timestamps(len(signal)), ZoomMethod.GaussianFilter,
        )
        static, _ = compute(
            _params(window=1.0), list(signal), _timestamps(len(signal)),
            ZoomMethod.GaussianFilter,
        )
        assert through_kf == pytest.approx(static)

    def test_video_speed_one_and_no_keyframes_is_the_static_path(self):
        signal = _plateau_signal()
        through_kf_branch, _ = compute(
            _params(window=1.0, video_speed=1.0, speed_affects=True),
            list(signal), _timestamps(len(signal)), ZoomMethod.GaussianFilter,
        )
        static, _ = compute(
            _params(window=1.0), list(signal), _timestamps(len(signal)),
            ZoomMethod.GaussianFilter,
        )
        assert through_kf_branch == pytest.approx(static)


class TestTheDynamicHelpers:
    def test_min_rolling_uses_per_timestamp_windows(self):
        """Every output cell reads its own window: a 3-frame window centred
        on index 1 sees indices 0..2 of the padded signal, not the static
        3-frame neighbourhood of the static path."""
        dpt = [_DataPerTimestamp(fps=30.0, window=0.1, frames=3) for _ in range(3)]
        padded = [1.0, 1.0, 0.5, 1.0, 1.0]
        got = _min_rolling_dynamic(padded, max_window_half=1, data_per_timestamp=dpt)
        assert got == pytest.approx([0.5, 0.5, 0.5])

    def test_envelope_per_sample_alpha_aligns_with_input_index(self):
        """Upstream pairs sample a[k] with alphas[k] in *both* passes
        (``.iter().rev().zip(&alphas)`` walks the samples backwards but the
        alphas forwards). alpha = 1 - exp(-(1/fps)/window): a huge window
        gives alpha≈0 (no response), a tiny one alpha≈1 (pass-through)."""
        dpt = [
            _DataPerTimestamp(fps=1.0, window=w, frames=1)
            for w in (1e9, 1e-3)  # alphas ≈ [0, 1]
        ]
        out = _envelope_follower([0.2, 1.0], dpt, None)
        # Reverse pass: a[1] with alpha≈1 passes 1.0 back; a[0] with alpha≈0
        # keeps the carried q=1.0 → min(0.2, 1.0) = 0.2. Forward pass:
        # out[0] = 0.2; out[1] uses alpha[1]≈1 → passes its own 1.0.
        # (Misaligned alphas would give out[1] = 0.2 instead.)
        assert out[0] == pytest.approx(0.2)
        assert out[1] == pytest.approx(1.0)

    def test_envelope_static_alpha_still_works(self):
        got = _envelope_follower([0.5, 1.0, 0.5], [], 1.0)
        # alpha=1: q = min(x, x) = x — the input comes back exactly.
        assert got == pytest.approx([0.5, 1.0, 0.5])


class TestEnvelopeFollowerKeyframedBranch:
    def test_keyframed_speed_envelope_path_runs(self):
        kf = KeyframeManager()
        kf.set_keyframe(
            KeyframeType.ZoomingSpeed, 0, 0.5, easing=Easing.NoEasing
        )
        signal = [1.0] * 30 + [0.5] * 30
        smoothed, minimal = compute(
            _params(window=1.0, keyframes=kf), list(signal),
            _timestamps(60), ZoomMethod.EnvelopeFollower,
        )
        assert minimal == pytest.approx(signal)
        # The envelope never goes below the input's plateau and converges
        # onto the notch.
        assert min(smoothed[30:]) >= 0.5
        assert smoothed[-1] == pytest.approx(0.5)


class TestFromIndex:
    def test_unknown_index_falls_back_with_error(self, caplog):
        """``zooming/mod.rs:20-27``: an unknown method index logs an error
        and falls back to GaussianFilter — a project from a newer Gyroflow
        must still render."""
        with caplog.at_level("ERROR"):
            assert ZoomMethod.from_index(99) == ZoomMethod.GaussianFilter
        assert any("Invalid zooming method: 99" in r.message
                   for r in caplog.records)

    def test_valid_indices_pass_through(self):
        assert ZoomMethod.from_index(0) == ZoomMethod.GaussianFilter
        assert ZoomMethod.from_index(1) == ZoomMethod.EnvelopeFollower
