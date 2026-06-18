"""Tests for KeyframeManager and easing functions."""

import math

import pytest
from numpy.testing import assert_allclose

from pygyroflow.keyframes import KeyframeManager, KeyframeType, Easing


class TestEasing:
    def test_no_easing_is_linear(self):
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert Easing.NoEasing.apply(alpha) == alpha

    def test_ease_in_monotonic(self):
        prev = 0.0
        for alpha in [i / 100.0 for i in range(101)]:
            val = Easing.EaseIn.apply(alpha)
            assert val >= prev - 1e-10
            prev = val

    def test_ease_out_monotonic(self):
        prev = 0.0
        for alpha in [i / 100.0 for i in range(101)]:
            val = Easing.EaseOut.apply(alpha)
            assert val >= prev - 1e-10
            prev = val

    def test_ease_in_out_monotonic(self):
        prev = 0.0
        for alpha in [i / 100.0 for i in range(101)]:
            val = Easing.EaseInOut.apply(alpha)
            assert val >= prev - 1e-10
            prev = val

    def test_boundary_values(self):
        for easing in [Easing.NoEasing, Easing.EaseIn, Easing.EaseOut, Easing.EaseInOut]:
            assert_allclose(easing.apply(0.0), 0.0, atol=1e-10)
            assert_allclose(easing.apply(1.0), 1.0, atol=1e-10)

    def test_interpolate_endpoints(self):
        a, b = 10.0, 20.0
        for easing_a in Easing:
            for easing_b in Easing:
                result_start = Easing.interpolate(easing_a, easing_b, a, b, 0.0)
                result_end = Easing.interpolate(easing_a, easing_b, a, b, 1.0)
                assert_allclose(result_start, a, atol=1e-10)
                assert_allclose(result_end, b, atol=1e-10)


class TestKeyframeManagerBasic:
    def test_empty_manager(self):
        km = KeyframeManager()
        assert not km.is_keyframed(KeyframeType.Fov)
        assert km.get_keyframes(KeyframeType.Fov) is None

    def test_set_and_get_keyframe(self):
        km = KeyframeManager()
        kf_id = km.set_keyframe(KeyframeType.Fov, 0, 1.5)
        assert kf_id > 0
        assert km.is_keyframed(KeyframeType.Fov)

        value = km.value_at_timestamp(KeyframeType.Fov, 0)
        assert value == 1.5

    def test_remove_keyframe(self):
        km = KeyframeManager()
        km.set_keyframe(KeyframeType.Fov, 0, 1.5)
        km.remove_keyframe(KeyframeType.Fov, 0)
        assert not km.is_keyframed_internally(KeyframeType.Fov)

    def test_single_keyframe_returns_value(self):
        km = KeyframeManager()
        km.set_keyframe(KeyframeType.Fov, 500000, 2.0)
        # Any timestamp lookup should return this single value
        assert km.value_at_timestamp(KeyframeType.Fov, 100000) == 2.0
        assert km.value_at_timestamp(KeyframeType.Fov, 500000) == 2.0
        assert km.value_at_timestamp(KeyframeType.Fov, 900000) == 2.0


class TestKeyframeInterpolation:
    def test_linear_interpolation_between_two_keyframes(self):
        km = KeyframeManager()
        km.set_keyframe(KeyframeType.Fov, 0, 1.0, Easing.NoEasing)
        km.set_keyframe(KeyframeType.Fov, 1000000, 2.0, Easing.NoEasing)

        # At midpoint (500ms = 500000us), value should be 1.5
        val = km.value_at_timestamp(KeyframeType.Fov, 500000)
        assert_allclose(val, 1.5, atol=1e-6)

    def test_interpolation_clamps_to_range(self):
        km = KeyframeManager()
        km.set_keyframe(KeyframeType.Fov, 100000, 1.0, Easing.NoEasing)
        km.set_keyframe(KeyframeType.Fov, 200000, 2.0, Easing.NoEasing)

        # Before first keyframe
        val = km.value_at_timestamp(KeyframeType.Fov, 50000)
        assert_allclose(val, 1.0, atol=1e-6)

        # After last keyframe
        val = km.value_at_timestamp(KeyframeType.Fov, 300000)
        assert_allclose(val, 2.0, atol=1e-6)

    def test_snap_to_closest_keyframe(self):
        """Keyframes within +/-1ms should snap together."""
        km = KeyframeManager()
        km.set_keyframe(KeyframeType.Fov, 100000, 1.0)
        # Set another within 1ms range — should update the same keyframe
        km.set_keyframe(KeyframeType.Fov, 100500, 1.5)

        kfs = km.get_keyframes(KeyframeType.Fov)
        assert len(kfs) == 1  # Only one keyframe (snapped)


class TestKeyframeSerialization:
    def test_roundtrip(self):
        km = KeyframeManager()
        km.set_keyframe(KeyframeType.Fov, 0, 1.0, Easing.NoEasing)
        km.set_keyframe(KeyframeType.Fov, 1000000, 2.0, Easing.EaseInOut)

        serialized = km.serialize()
        km2 = KeyframeManager()
        km2.deserialize(serialized)

        assert km2.is_keyframed(KeyframeType.Fov)
        assert_allclose(km2.value_at_timestamp(KeyframeType.Fov, 0), 1.0)
        assert_allclose(km2.value_at_timestamp(KeyframeType.Fov, 1000000), 2.0)

    def test_json_roundtrip(self):
        km = KeyframeManager()
        km.set_keyframe(KeyframeType.Fov, 0, 1.5)
        km.set_keyframe(KeyframeType.SmoothingParamSmoothness, 500000, 0.8)

        json_str = km.to_json()
        km2 = KeyframeManager()
        km2.from_json(json_str)

        assert km2.is_keyframed(KeyframeType.Fov)
        assert km2.is_keyframed(KeyframeType.SmoothingParamSmoothness)
