"""The manager's upstream setter surface (C-14)."""

from __future__ import annotations

import numpy as np
import pytest

from pygyroflow.manager import StabilizationManager
from pygyroflow.types.enums import BackgroundMode, ReadoutDirection


@pytest.fixture()
def mgr():
    return StabilizationManager()


class TestSettersTouchParams:
    def test_stab_enabled(self, mgr):
        mgr.set_stab_enabled(False)
        assert mgr.params.stab_enabled is False

    def test_video_speed(self, mgr):
        mgr.set_video_speed(0.5)
        assert mgr.params.video_speed == 0.5

    def test_max_zoom_including_none(self, mgr):
        mgr.set_max_zoom(150.0)
        assert mgr.params.max_zoom == 150.0
        mgr.set_max_zoom(None)
        assert mgr.params.max_zoom is None

    def test_frame_offset(self, mgr):
        mgr.set_frame_offset(3)
        assert mgr.params.frame_offset == 3

    def test_frame_readout_direction(self, mgr):
        mgr.set_frame_readout_direction(ReadoutDirection.BottomToTop)
        assert mgr.params.frame_readout_direction == ReadoutDirection.BottomToTop

    def test_additional_rotation_axes(self, mgr):
        mgr.set_additional_rotation(1.0, 2.0, 3.0)
        assert mgr.params.additional_rotation == (1.0, 2.0, 3.0)
        mgr.set_additional_rotation_x(9.0)
        mgr.set_additional_rotation_y(8.0)
        mgr.set_additional_rotation_z(7.0)
        assert mgr.params.additional_rotation == (9.0, 8.0, 7.0)

    def test_additional_translation_axes(self, mgr):
        mgr.set_additional_translation(0.1, 0.2, 0.3)
        mgr.set_additional_translation_z(9.0)
        assert mgr.params.additional_translation == (0.1, 0.2, 9.0)

    def test_stretches_and_refraction(self, mgr):
        mgr.set_input_horizontal_stretch(1.1)
        mgr.set_input_vertical_stretch(0.9)
        mgr.set_light_refraction_coefficient(0.95)
        assert mgr.params.input_horizontal_stretch == 1.1
        assert mgr.params.input_vertical_stretch == 0.9
        assert mgr.params.light_refraction_coefficient == 0.95

    def test_background_family(self, mgr):
        mgr.set_background_mode(BackgroundMode.MirrorPixels)
        mgr.set_background_margin(0.2)
        mgr.set_background_margin_feather(0.05)
        mgr.set_background_color((0.1, 0.2, 0.3))
        assert mgr.params.background_mode == BackgroundMode.MirrorPixels
        assert mgr.params.background_margin == 0.2
        assert mgr.params.background_margin_feather == 0.05
        assert mgr.params.background.dtype == np.float32
        assert mgr.params.background[1] == pytest.approx(0.2)

    def test_horizon_lock(self, mgr):
        mgr.set_horizon_lock(50.0, 1.0, 2.0)
        lock = mgr.smoothing.horizon_lock
        assert lock.lock_enabled is True
        assert lock.horizonlockpercent == 50.0
        assert lock.horizonroll == 1.0
        assert lock.horizonpitch == 2.0
        mgr.set_horizon_lock(0.0)
        assert lock.lock_enabled is False

    def test_zooming_method(self, mgr):
        mgr.set_zooming_method(0)
        assert mgr.params.adaptive_zoom_method == 0

    def test_debug_overlays(self, mgr):
        mgr.set_show_detected_features(False)
        mgr.set_show_optical_flow(False)
        assert mgr.params.show_detected_features is False
        assert mgr.params.show_optical_flow is False


class TestDigitalLensSetters:
    """The digital lens lives on the lens profile (upstream lib.rs:1030,
    1038-1043), not on stabilization params."""

    def test_name(self, mgr):
        mgr.set_digital_lens_name("gopro_superview")
        assert mgr.lens.digital_lens == "gopro_superview"

    def test_params_default_to_zeros_on_first_write(self, mgr):
        mgr.lens.digital_lens_params = None
        mgr.set_digital_lens_param(2, 1.5)
        assert mgr.lens.digital_lens_params == [0.0, 0.0, 1.5, 0.0]

    def test_out_of_range_index_is_ignored(self, mgr):
        mgr.set_digital_lens_param(7, 9.0)
        assert mgr.lens.digital_lens_params == [0.0] * 4


class TestInvalidation:
    """Upstream's setters invalidate what they touch; the port's cache
    ids must move so the next compute_id comparison fails."""

    def _ids(self, mgr):
        return (mgr._compute_id, mgr._zooming_checksum,
                mgr._smoothing_checksum)

    def test_smoothing_setters_move_the_id(self, mgr):
        before = mgr._compute_id
        mgr.set_stab_enabled(False)
        mgr.set_video_speed(2.0)
        mgr.set_additional_rotation(1.0, 0.0, 0.0)
        assert mgr._compute_id > before

    def test_zooming_only_setters_reset_the_checksum(self, mgr):
        mgr.set_max_zoom(150.0)
        mgr.set_light_refraction_coefficient(0.9)
        assert mgr._zooming_checksum == 0
