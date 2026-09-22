"""Load-time lens-profile fixup for the GoPro digital lenses (D-16).

Upstream's ``adjust_lens_profile`` runs on every profile parsed from the
lens database (``resolve_interpolations``, ``lens_profile_database.rs:202``):
a Superview or Hyperview profile authored against 4:3 / 8:7 pixels gets its
calibration width widened back to what the sensor actually captured, and
``lens_model`` is renamed unconditionally. Without it, a profile whose
calibration is stored on the cropped pixels keeps producing FOVs from the
cropped geometry — everything downstream (adaptive zoom, output FOV) is
computed from the wrong frame size.
"""

from __future__ import annotations

from pygyroflow.lens.profile import LensProfile
from pygyroflow.stabilization.distortion_models import from_name


def _profile(digital_lens: str | None, w: int, h: int) -> LensProfile:
    data: dict = {
        "name": "test",
        "calib_dimension": {"w": w, "h": h},
        "calib_data": {"f": 1000.0},
    }
    if digital_lens is not None:
        data["digital_lens"] = digital_lens
    return LensProfile.from_json(data)


class TestSuperviewWidening:
    def test_4_3_calibration_widens_to_16_9(self):
        p = _profile("gopro_superview", 1440, 1080)
        assert p.calib_dimension["w"] == 1920  # round(1440 * 1.3333333333333)
        assert p.calib_dimension["h"] == 1080

    def test_lens_model_is_renamed_unconditionally(self):
        """The rename sits outside the aspect check upstream — a non-4:3
        calibration still gets relabeled, only not widened."""
        p = _profile("gopro_superview", 1920, 1080)
        assert p.calib_dimension["w"] == 1920  # untouched: 169 != 133
        assert p.lens_model == "Superview"

    def test_gopro6_superview_behaves_the_same(self):
        p = _profile("gopro6_superview", 1440, 1080)
        assert p.calib_dimension["w"] == 1920
        assert p.lens_model == "Superview"


class TestHyperviewWidening:
    def test_8_7_calibration_widens(self):
        # 1568 x 1372 is 8:7 (114 in the upstream hundredths encoding).
        p = _profile("gopro_hyperview", 1568, 1372)
        assert p.calib_dimension["w"] == round(1568 * 1.55555555555)
        assert p.calib_dimension["h"] == 1372

    def test_non_8_7_only_renames(self):
        p = _profile("gopro_hyperview", 1920, 1080)
        assert p.calib_dimension["w"] == 1920
        assert p.lens_model == "Hyperview"


class TestTheDefaultAndTheHook:
    def test_models_without_a_fixup_are_no_ops(self):
        """The macro default upstream is an empty body; poly3/fisheye and
        friends never touch the profile."""
        profile = _profile(None, 1440, 1080)
        from_name("opencv_fisheye").adjust_lens_profile(profile)
        assert profile.calib_dimension["w"] == 1440
        assert profile.lens_model != "Superview"

    def test_from_json_runs_the_hook(self):
        p = _profile("gopro_superview", 1440, 1080)
        assert p.calib_dimension["w"] == 1920

    def test_from_json_without_digital_lens_untouched(self):
        p = _profile(None, 1440, 1080)
        assert p.calib_dimension["w"] == 1440
        assert p.lens_model != "Superview"
