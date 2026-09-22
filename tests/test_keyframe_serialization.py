"""Keyframe serialization compatibility with upstream projects (D-17).

Upstream's serde writes the easing as the *variant-name string* and gives
every keyframe an id (random when absent, ``keyframes.rs:70``). The port
used to write the easing as an int and require the id — a project or lens
profile from upstream could not round-trip. The reader accepts both shapes;
the writer now emits the upstream shape.
"""

from __future__ import annotations

from pygyroflow.keyframes.manager import KeyframeManager
from pygyroflow.keyframes.types import Easing, KeyframeType


class TestUpstreamShapeIsWritten:
    def test_serialize_uses_the_variant_name_string(self):
        mgr = KeyframeManager()
        mgr.set_keyframe(KeyframeType.Fov, 1_000_000, 95.0, easing=Easing.EaseInOut)
        data = mgr.serialize()
        assert data["keyframes"]["Fov"]["1000000"]["easing"] == "EaseInOut"
        assert data["keyframes"]["Fov"]["1000000"]["id"] > 0

    def test_serialize_carries_the_wrapper_and_the_scale(self):
        mgr = KeyframeManager()
        mgr.timestamp_scale = 1.5
        mgr.update_gyro({0: 12.0})
        data = mgr.serialize()
        assert set(data) == {"keyframes", "gyro_offsets", "timestamp_scale"}
        assert data["timestamp_scale"] == 1.5
        assert data["gyro_offsets"] == {"0": 12.0}

    def test_roundtrip_preserves_values_and_easings(self):
        mgr = KeyframeManager()
        for ts, easing in (
            (1_000_000, Easing.NoEasing),
            (2_000_000, Easing.EaseIn),
            (3_000_000, Easing.EaseOut),
        ):
            mgr.set_keyframe(KeyframeType.Fov, ts, float(ts), easing=easing)
        fresh = KeyframeManager()
        fresh.deserialize(mgr.serialize())
        for ts in (1_000_000, 2_000_000, 3_000_000):
            kf = fresh._keyframes[KeyframeType.Fov][ts]
            assert kf.value == float(ts)
            assert kf.easing == mgr._keyframes[KeyframeType.Fov][ts].easing


class TestUpstreamShapeIsRead:
    def test_string_easing_and_missing_id(self):
        """The shape a real .gyroflow file carries: variant-name easing, no
        id (upstream's serde default makes it random)."""
        mgr = KeyframeManager()
        mgr.deserialize({
            "keyframes": {
                "Fov": {"1000000": {"value": 95.0, "easing": "EaseInOut"}},
            },
            "gyro_offsets": {},
            "timestamp_scale": None,
        })
        kf = mgr._keyframes[KeyframeType.Fov][1_000_000]
        assert kf.easing == Easing.EaseInOut
        assert 1 <= kf.id <= 2147483639  # random, within upstream's range

    def test_unknown_easing_string_falls_back_not_raises(self):
        """A newer Gyroflow adding an easing member must not break loading."""
        mgr = KeyframeManager()
        mgr.deserialize({
            "keyframes": {
                "Fov": {"1000000": {"value": 1.0, "easing": "EaseButFancier"}},
            },
        })
        assert mgr._keyframes[KeyframeType.Fov][1_000_000].easing == Easing.NoEasing

    def test_timestamp_scale_is_restored(self):
        mgr = KeyframeManager()
        mgr.deserialize({
            "keyframes": {},
            "gyro_offsets": {"5": 5.0},
            "timestamp_scale": 0.5,
        })
        assert mgr.timestamp_scale == 0.5
        assert mgr.gyro_offsets == {5: 5.0}


class TestLegacyShapeIsStillRead:
    def test_flat_map_with_integer_easings(self):
        """The port's previous format — keep it readable for old files."""
        mgr = KeyframeManager()
        mgr.deserialize({
            "Fov": {"1000000": {"id": 7, "value": 1.0, "easing": 3}},
        })
        kf = mgr._keyframes[KeyframeType.Fov][1_000_000]
        assert kf.easing == Easing.EaseInOut
        assert kf.id == 7


class TestTheProjectWiring:
    def test_apply_project_stabilization_reads_keyframes(self):
        from pygyroflow.manager import StabilizationManager

        mgr = StabilizationManager()
        mgr._apply_project_stabilization({
            "keyframes": {
                "keyframes": {
                    "Fov": {"1000000": {"value": 88.0, "easing": "EaseIn"}},
                },
                "gyro_offsets": {},
                "timestamp_scale": None,
            },
        })
        assert mgr.keyframes._keyframes[KeyframeType.Fov][1_000_000].value == 88.0

    def test_a_saved_project_roundtrips_its_keyframes(self, tmp_path):
        """save → load: the keyframes survive a real file, in the section
        upstream keeps them in. The offsets ride the gyro source (their
        authoritative home) and are mirrored back on load."""
        from pygyroflow.manager import StabilizationManager

        mgr = StabilizationManager()
        mgr.keyframes.set_keyframe(
            KeyframeType.Fov, 1_000_000, 95.0, easing=Easing.EaseInOut
        )
        mgr.set_gyro_offset(0, 42.0)
        path = tmp_path / "roundtrip.gyroflow"
        mgr.save_project(str(path))

        fresh = StabilizationManager()
        fresh.load_project(str(path))
        assert (
            fresh.keyframes._keyframes[KeyframeType.Fov][1_000_000].value == 95.0
        )
        assert fresh.keyframes.gyro_offsets == {0: 42.0}
