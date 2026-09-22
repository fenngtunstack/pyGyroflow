"""Decoder frame rotation in the CPU sampler (D-13).

Upstream tags render buffers with the decoder's rotation metadata and the
CPU sampler re-maps coordinates into that geometry (``cpu_undistort.rs:
483-489``): the mapped coordinates are rotated about the stored frame's
centre onto the rotated frame's centre, and the effective frame size becomes
the rotated bounding box. Two quirks are deliberate and pinned:

* the background-mode clamps keep bounding against the **unrotated**
  ``params.width/height`` (``cpu_undistort.rs:492-500``);
* only the *sampling* bounds follow the rotated size — that is the shape of
  the frame array the sampler receives.
"""

from __future__ import annotations

import ctypes

import cv2
import numpy as np
import pytest

from pygyroflow.stabilization.cpu_undistort import cpu_undistort
from pygyroflow.stabilization.frame_transform import FrameTransform
from pygyroflow.types.kernel_params import KernelParams


def _kernel_params(width=1920, height=1080, input_rotation=0.0) -> KernelParams:
    kp = KernelParams()
    kp.width, kp.height = width, height
    kp.output_width, kp.output_height = width, height
    kp.f = (ctypes.c_float * 2)(1000.0, 1000.0)
    kp.c = (ctypes.c_float * 2)(width / 2.0, height / 2.0)
    kp.fov = 1.0
    kp.matrix_count = 1
    kp.background_mode = 0
    kp.input_rotation = float(input_rotation)
    kp.output_rotation = 0.0
    return kp


def _identity_transform(width=1920, height=1080, input_rotation=0.0) -> FrameTransform:
    kp = _kernel_params(width, height, input_rotation)
    m = np.zeros((1, 14), dtype=np.float32)
    inv = np.linalg.inv(
        np.array([[1000.0, 0, width / 2], [0, 1000.0, height / 2], [0, 0, 1.0]])
    )
    m[0, :9] = inv.ravel()
    return FrameTransform(
        matrices=m, kernel_params=kp, distortion_model_name="opencv_fisheye"
    )


def _capture_remap(monkeypatch):
    """Return the map_x/map_y the sampler feeds to cv2.remap."""
    captured = {}
    real = cv2.remap

    def spy(frame, map_x, map_y, *args, **kwargs):
        captured["x"] = map_x.copy()
        captured["y"] = map_y.copy()
        return real(frame, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # The sampler imports cv2 inside the function; patching the module
    # attribute reaches it.
    monkeypatch.setattr(cv2, "remap", spy)
    return captured


class TestTheCoordinateRotation:
    def test_ninety_degrees_maps_into_the_rotated_frame(self, monkeypatch):
        """rotation=90 on a 1920x1080 stored frame: output pixel (540, 960)
        maps to (120, 540) in the delivered 1080x1920 frame — rotate the
        offset from the old centre onto the new centre by hand."""
        captured = _capture_remap(monkeypatch)
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)  # delivered: rotated
        cpu_undistort(frame, _identity_transform(input_rotation=90.0))
        assert captured["y"][960, 540] == pytest.approx(540.0, abs=1e-3)
        assert captured["x"][960, 540] == pytest.approx(120.0, abs=1e-3)

    def test_one_eighty_degrees_maps_through_the_centre(self, monkeypatch):
        captured = _capture_remap(monkeypatch)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cpu_undistort(frame, _identity_transform(input_rotation=180.0))
        # (x, y) -> (1920 - x, 1080 - y): centre-symmetric.
        assert captured["x"][100, 200] == pytest.approx(1920 - 200, abs=1e-3)
        assert captured["y"][100, 200] == pytest.approx(1080 - 100, abs=1e-3)

    def test_zero_rotation_is_identity(self, monkeypatch):
        captured = _capture_remap(monkeypatch)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cpu_undistort(frame, _identity_transform())
        assert captured["x"][960, 540] == pytest.approx(540.0, abs=1e-3)
        assert captured["y"][960, 540] == pytest.approx(960.0, abs=1e-3)


class TestTheSamplingBounds:
    # Shared geometry: on a 1920x1080 stored frame rotated 90, output pixel
    # (x=0, y=1) maps to rotated coordinates (1079, 0) — the right edge of
    # the delivered 1080-wide frame:
    #   dx = 0-960 = -960, dy = 1-540 = -539
    #   rx = -sin*dy + 540 = 539 + 540 = 1079, ry = sin*dx + 960 = 0.

    def test_fringe_clamp_uses_the_rotated_width(self, monkeypatch):
        """A coordinate in [samp_w-1, samp_w) clamps to the rotated width's
        last pixel — the array is the rotated one."""
        captured = _capture_remap(monkeypatch)
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        cpu_undistort(frame, _identity_transform(input_rotation=90.0))
        assert captured["x"][1, 0] == pytest.approx(1079.0, abs=1e-2)

    def test_background_clamp_ceiling_stays_on_the_unrotated_size(self, monkeypatch):
        """Upstream bounds the edge-repeat clamps with params.width/height
        (cpu_undistort.rs:492-500), not the rotated frame_size: a rotated
        coordinate of 1079 exceeds the rotated ceiling (1080-3) but sits
        under the *stored* ceiling (1920-3), so it must survive unclamped."""
        captured = _capture_remap(monkeypatch)
        kp = _kernel_params(input_rotation=90.0)
        kp.background_mode = 1  # edge repeat
        m = np.zeros((1, 14), dtype=np.float32)
        inv = np.linalg.inv(
            np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1.0]])
        )
        m[0, :9] = inv.ravel()
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        cpu_undistort(frame, FrameTransform(
            matrices=m, kernel_params=kp, distortion_model_name="opencv_fisheye"
        ))
        # Clamped to the rotated ceiling it would read 1077; it reads 1079.
        assert captured["x"][1, 0] == pytest.approx(1079.0, abs=1e-2)


class TestTheKernelParamsPacking:
    def test_at_timestamp_packs_the_rotation(self):
        from pygyroflow.stabilization.compute_params import ComputeParams
        from pygyroflow.stabilization.frame_transform import FrameTransform as FT

        params = ComputeParams(
            width=1920, height=1080, output_width=1920, output_height=1080,
            camera_matrix=np.array(
                [[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1.0]]
            ),
        )
        ft = FT.at_timestamp(params, 0.0, 0, input_rotation=90.0)
        assert ft.kernel_params.input_rotation == pytest.approx(90.0)

        default = FT.at_timestamp(params, 0.0, 0)
        assert default.kernel_params.input_rotation == 0.0
        assert default.kernel_params.output_rotation == 0.0


class TestTheEndToEndValue:
    def test_output_pixel_shows_the_rotated_frames_pixel(self):
        """A gradient frame delivered rotated: the stabilised output at
        (540, 960) must show exactly the delivered frame's (120, 540)
        pixel — the mapping proven above, checked through real sampling."""
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        frame[:, :, 0] = np.tile(np.arange(1080, dtype=np.uint8), (1920, 1))
        frame[:, :, 1] = np.repeat(
            np.arange(1920, dtype=np.uint8)[:, None], 1080, axis=1
        )
        out = cpu_undistort(
            frame, _identity_transform(input_rotation=90.0), interpolation=0
        )
        assert tuple(out[960, 540]) == (int(frame[540, 120, 0]),
                                        int(frame[540, 120, 1]),
                                        int(frame[540, 120, 2]))
