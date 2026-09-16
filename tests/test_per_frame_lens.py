"""Per-timestamp lens data (gap item D-01/D-06, the doc's "root of zoom-lens
and fisheye sync precision").

A zoom lens changes focal length during the clip, so its calibration changes
with it. Upstream carries two independent per-frame channels and a nearest-key
lookup that is fussier than "closest wins":

* ``ClosestMap`` — the lookup, with a strict tie-break and a strict cap.
* ``lens_positions`` — one scalar per time (a focal length in mm, or a GoPro
  FOV-adaptation crop score) that indexes the profile's own interpolation
  table.
* ``lens_params`` — raw per-frame intrinsics that override the profile.

Where the maps are empty — a fixed-focal-length clip, which is every clip
this repo has except one Sony prime — every consumer stays on the static
path, and several tests here pin that.
"""

from __future__ import annotations

import pathlib
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pygyroflow.gyro_source.file_metadata import LensParams  # noqa: E402
from pygyroflow.lens import LensProfile  # noqa: E402
from pygyroflow.stabilization.compute_params import ComputeParams  # noqa: E402
from pygyroflow.stabilization.frame_transform import (  # noqa: E402
    _get_frame_readout_time,
    _get_lens_data_at_timestamp,
)
from pygyroflow.util import ClosestMap  # noqa: E402

_REFERENCE_CLIP = pathlib.Path(
    "/home/ft/workspace/testvideos/issue-44-08-C0841-a7s3-sony85mm.MP4"
)
_ZERO_ZOOM_CLIP = pathlib.Path(
    "/home/ft/workspace/testvideos/issue-44-33-sony-rx100-7-C0756---ois-only.MP4"
)


def profile(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, coeffs=None,
            calib=(1920, 1080), model="opencv_fisheye", crop=None,
            interpolations=None, focal_length=None):
    """A LensProfile with just enough filled in to exercise the lens paths."""
    return LensProfile(
        name="test",
        camera_matrix=[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        distortion_coeffs=list(coeffs if coeffs is not None else []),
        calib_dimension={"w": calib[0], "h": calib[1]},
        orig_dimension={"w": calib[0], "h": calib[1]},
        distortion_model=model,
        crop=crop,
        interpolations_raw=interpolations,
        focal_length=focal_length,
    )


def params_for(lens, width=1920, height=1080, **overrides):
    """A ComputeParams built the way `_build_compute_params` builds one."""
    matrix = lens.get_camera_matrix((width, height))
    values = dict(
        width=width,
        height=height,
        frame_count=10,
        scaled_fps=30.0,
        camera_matrix=matrix,
        distortion_coeffs=lens.get_distortion_coeffs(),
        distortion_model_name=lens.distortion_model or "opencv_fisheye",
        calib_width=lens.calib_dimension["w"],
        calib_height=lens.calib_dimension["h"],
        input_horizontal_stretch=1.0,
        input_vertical_stretch=1.0,
        focal_length=lens.focal_length,
        lens=lens,
    )
    values.update(overrides)
    return ComputeParams(**values)


class TestClosestMap:
    """The lookup itself, before anything uses it."""

    def test_empty(self):
        assert ClosestMap().get_closest(0, 100_000) is None
        assert ClosestMap({}).get_closest(1234, 100_000) is None
        assert not ClosestMap()

    def test_exact_key_wins(self):
        lookup = ClosestMap({100: "a", 200: "b"})
        assert lookup.get_closest(200, 100_000) == "b"
        assert len(lookup) == 2

    def test_strictly_closer_neighbour_wins(self):
        lookup = ClosestMap({100: "below", 300: "above"})
        assert lookup.get_closest(150, 100_000) == "below"   # 50 vs 150
        assert lookup.get_closest(280, 100_000) == "above"   # 20 vs 180

    def test_an_exact_tie_returns_nothing(self):
        """Upstream drops an equidistant lookup rather than picking a side;
        the caller then falls back to the static calibration. Choosing one
        would be arbitrary, and arbitrarily choosing is how a zoom lens ends
        up half a step off at the crossover."""
        lookup = ClosestMap({100: "below", 300: "above"})
        assert lookup.get_closest(200, 100_000) is None

    def test_the_cap_is_strict(self):
        lookup = ClosestMap({100: "only"})
        assert lookup.get_closest(100 + 100_000, 100_000) is None
        assert lookup.get_closest(100 + 99_999, 100_000) == "only"

    def test_one_sided_lookups(self):
        lookup = ClosestMap({100: "only"})
        assert lookup.get_closest(0, 100_000) == "only"       # nothing below
        assert lookup.get_closest(1_000_000, 100_000) is None  # too far above

    def test_a_neighbour_just_outside_the_cap_is_not_used(self):
        lookup = ClosestMap({0: "a"})
        assert lookup.get_closest(100_001, 100_000) is None
        assert lookup.get_closest(99_999, 100_000) == "a"

    def test_keys_are_sorted_once(self):
        """The callers walk a whole clip; a per-call sort would be
        quadratic. Order of insertion must not matter."""
        lookup = ClosestMap({300: "c", 100: "a", 200: "b"})
        assert lookup._keys == [100, 200, 300]
        assert lookup.get_closest(200, 100_000) == "b"

    def test_the_missing_key_sentinel_only_bites_for_negative_keys(self):
        """Upstream stands in for "no neighbour" with the key -99999 rather
        than infinity, so an absent side sits at a distance of about
        key + 99999. Reproduced exactly rather than "fixed", because the only
        way to tell the two apart is a negative key — and every caller keys
        these maps by a microsecond timestamp, which is never negative.

        Recorded so the next reader does not "simplify" the sentinel away and
        quietly change a comparison that is currently unreachable."""
        lookup = ClosestMap({-500_000: "far below"})
        # below distance 500000 vs the sentinel's 99999: the present side
        # loses, so nothing comes back even though the cap allows it.
        assert lookup.get_closest(0, 1_000_000) is None
        # The same shape with a non-negative key: the present side is always
        # closer than the sentinel, so the sentinel cannot matter.
        assert ClosestMap({0: "only"}).get_closest(500_000, 1_000_000) == "only"

    def test_mapping_with_real_values(self):
        lookup = ClosestMap({0: 85.0, 5_000_000: 200.0})
        assert lookup.get_closest(1_000, 100_000) == 85.0
        assert lookup.get_closest(4_999_000, 100_000) == 200.0


class TestStaticPathUnchanged:
    """With no per-frame data, nothing about the result may move."""

    def test_empty_maps_reproduce_the_old_behaviour(self):
        lens = profile(calib=(1920, 1080))
        params = params_for(lens)
        matrix, coeffs, limit, h_stretch, v_stretch, fl = (
            _get_lens_data_at_timestamp(params, 1000.0)
        )
        assert matrix == pytest.approx(np.asarray(lens.camera_matrix))
        assert coeffs == list(params.distortion_coeffs)
        assert limit == params.radial_distortion_limit
        assert (h_stretch, v_stretch) == (1.0, 1.0)
        assert fl == lens.focal_length

    def test_calibration_resolution_scaling_still_applies(self):
        """A calibration at half the video resolution is doubled up."""
        lens = profile(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, calib=(960, 540))
        params = params_for(lens, width=1920, height=1080)
        matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] == pytest.approx(2000.0)
        assert matrix[1, 1] == pytest.approx(2000.0)
        # The principal point starts at the calibration's centre (480, 270)
        # and is scaled by the same ratio, landing back on the video centre.
        assert matrix[0, 2] == pytest.approx(960.0)
        assert matrix[1, 2] == pytest.approx(540.0)

    def test_zero_stretch_becomes_one(self):
        """0.0 is the "unset" sentinel, not a degenerate stretch."""
        lens = profile()
        params = params_for(lens, input_horizontal_stretch=0.0,
                            input_vertical_stretch=0.0)
        _, _, _, h_stretch, v_stretch, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert (h_stretch, v_stretch) == (1.0, 1.0)


class TestLensPositions:
    """`lens_positions` selects an interpolated profile out of the table."""

    def _lens_with_zoom_table(self):
        return profile(
            fx=1000.0,
            fy=1000.0,
            calib=(1920, 1080),
            interpolations={
                "24.0": {"camera_matrix": [[800.0, 0.0, 960.0],
                                           [0.0, 800.0, 540.0],
                                           [0.0, 0.0, 1.0]],
                         "focal_length": 24.0},
                "48.0": {"camera_matrix": [[1600.0, 0.0, 960.0],
                                           [0.0, 1600.0, 540.0],
                                           [0.0, 0.0, 1.0]],
                         "focal_length": 48.0},
            },
        )

    def test_a_position_selects_the_interpolated_profile(self):
        lens = self._lens_with_zoom_table()
        lens.resolve_interpolations()
        params = params_for(lens)
        params.lens_positions = ClosestMap({0: 36.0})   # half way

        matrix, _, _, _, _, focal_length = _get_lens_data_at_timestamp(params, 0.0)
        # 36 mm is the midpoint of the 24 and 48 mm entries.
        assert matrix[0, 0] == pytest.approx(1200.0)
        assert focal_length == pytest.approx(36.0)

    def test_a_position_far_outside_the_table_still_interpolates(self):
        """Upstream clamps the lookup key to just inside the table's ends, so
        a focal length beyond the calibrated range uses the nearest entry
        rather than falling off."""
        lens = self._lens_with_zoom_table()
        lens.resolve_interpolations()
        params = params_for(lens)
        params.lens_positions = ClosestMap({0: 500.0})
        matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] > 1600.0

    def test_no_entry_within_the_window_uses_the_static_profile(self):
        lens = self._lens_with_zoom_table()
        lens.resolve_interpolations()
        params = params_for(lens)
        params.lens_positions = ClosestMap({500_000: 36.0})  # 500 ms away
        matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] == pytest.approx(1000.0)

    def test_a_constant_position_is_a_no_op(self):
        """A prime lens's single focal length still goes through the table
        lookup, and has to come out the same as not having one."""
        lens = profile(interpolations={"85.0": {"focal_length": 85.0}})
        lens.resolve_interpolations()
        plain = _get_lens_data_at_timestamp(params_for(lens), 0.0)

        with_position = params_for(lens)
        with_position.lens_positions = ClosestMap({0: 85.0, 33_000: 85.0})
        assert _get_lens_data_at_timestamp(with_position, 0.0)[0] == pytest.approx(
            plain[0]
        )


class TestLensParams:
    """`lens_params` overrides the intrinsics outright."""

    def _params(self, entries, **overrides):
        # Two coefficients, not four: upstream gates the whole override on
        # `distortion_coeffs.len() < 4`, and a real fisheye profile carries
        # four, so this branch is for profiles with a partial set.
        lens = profile(coeffs=[0.0, 0.0])
        params = params_for(lens, **overrides)
        params.lens_params = ClosestMap(entries)
        return params

    def test_pixel_focal_length_overrides_the_matrix(self):
        params = self._params({0: LensParams(pixel_focal_length=1234.0)})
        matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] == pytest.approx(1234.0)
        assert matrix[1, 1] == pytest.approx(1234.0)
        # The principal point is reset to the frame centre: the override is
        # already in video pixels, so the calibration's own centring no
        # longer describes it.
        assert matrix[0, 2] == pytest.approx(960.0)
        assert matrix[1, 2] == pytest.approx(540.0)

    def test_the_calibration_scaling_is_skipped_when_overridden(self):
        """Scaling an override that is already in video pixels would double
        it up — this is what upstream's `stretch_lens` flag guards."""
        lens = profile(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, calib=(960, 540),
                       coeffs=[0.0, 0.0])
        params = params_for(lens, width=1920, height=1080)
        params.lens_params = ClosestMap({0: LensParams(pixel_focal_length=1500.0)})
        matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] == pytest.approx(1500.0)  # not 3000

    def test_focal_length_and_pixel_pitch_compute_the_matrix(self):
        entry = LensParams(
            focal_length=50.0,               # mm
            pixel_pitch=(0, 4000),           # nanometres -> 4 µm
            capture_area_size=(0.0, 2160.0),  # pixels
        )
        params = self._params({0: entry})
        matrix, _, _, _, _, focal_length = _get_lens_data_at_timestamp(params, 0.0)
        # 50 mm / (0.004 mm * 2160) * 1080 = 6250 px
        assert matrix[0, 0] == pytest.approx(6250.0)
        assert focal_length == pytest.approx(50.0)

    def test_focal_length_without_pixel_pitch_leaves_the_matrix_alone(self):
        entry = LensParams(focal_length=50.0)  # no pitch / capture area
        params = self._params({0: entry})
        matrix, _, _, _, _, focal_length = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] == pytest.approx(1000.0)
        assert focal_length == pytest.approx(50.0)

    def test_distortion_coefficients_are_replaced(self):
        params = self._params(
            {0: LensParams(distortion_coefficients=[1.0, 2.0, 3.0, 4.0])}
        )
        _, coeffs, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert coeffs[:4] == [1.0, 2.0, 3.0, 4.0]

    def test_more_than_twelve_coefficients_are_refused(self):
        params = self._params(
            {0: LensParams(distortion_coefficients=[1.0] * 13)}
        )
        _, coeffs, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert coeffs == [0.0] * 12

    def test_a_profile_with_its_own_coefficients_wins(self):
        """Upstream gates the override on `distortion_coeffs.len() < 4`: a
        profile that carries per-frame coefficients of its own must not be
        overwritten by the file's."""
        lens = profile(coeffs=[9.0] * 12)
        params = params_for(lens)
        params.lens_params = ClosestMap(
            {0: LensParams(pixel_focal_length=1234.0,
                           distortion_coefficients=[1.0, 2.0, 3.0, 4.0])}
        )
        matrix, coeffs, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] == pytest.approx(1000.0)   # untouched
        assert coeffs[0] == pytest.approx(9.0)

    def test_no_entry_within_the_window_is_ignored(self):
        params = self._params({500_000: LensParams(pixel_focal_length=1234.0)})
        matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] == pytest.approx(1000.0)

    def test_an_exact_tie_falls_back_to_the_static_profile(self):
        """Two entries equidistant from the frame: the lookup yields nothing
        and the calibration stays put, rather than flipping a coin."""
        params = self._params(
            {
                0: LensParams(pixel_focal_length=1111.0),
                200_000: LensParams(pixel_focal_length=2222.0),
            }
        )
        matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(params, 100.0)
        assert matrix[0, 0] == pytest.approx(1000.0)


class TestDigitalZoom:
    def test_no_digital_zoom_leaves_the_matrix_alone(self):
        params = params_for(profile())
        assert params.digital_zoom is None
        matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] == pytest.approx(1000.0)

    def test_digital_zoom_scales_the_focal_lengths_only(self):
        params = params_for(profile(), digital_zoom=2.0)
        matrix, _, _, _, _, _ = _get_lens_data_at_timestamp(params, 0.0)
        assert matrix[0, 0] == pytest.approx(2000.0)
        assert matrix[1, 1] == pytest.approx(2000.0)
        assert matrix[0, 2] == pytest.approx(960.0)  # principal point unchanged


class TestReadoutTimeScale:
    """A partial-height readout finishes sooner, in proportion."""

    def test_no_lens_params_leaves_the_readout_alone(self):
        params = params_for(profile(), frame_readout_time=10.0)
        assert _get_frame_readout_time(params, 0.0) == pytest.approx(10.0)

    def test_a_partial_capture_area_scales_it(self):
        params = params_for(profile(), frame_readout_time=10.0)
        params.lens_params = ClosestMap(
            {0: LensParams(capture_area_size=(0.0, 1080.0),
                           sensor_size_px=(1920, 2160))}
        )
        assert _get_frame_readout_time(params, 0.0) == pytest.approx(5.0)

    def test_a_missing_entry_does_not_scale(self):
        params = params_for(profile(), frame_readout_time=10.0)
        params.lens_params = ClosestMap(
            {500_000: LensParams(capture_area_size=(0.0, 1080.0),
                                 sensor_size_px=(1920, 2160))}
        )
        assert _get_frame_readout_time(params, 0.0) == pytest.approx(10.0)

    def test_an_incomplete_entry_does_not_scale(self):
        params = params_for(profile(), frame_readout_time=10.0)
        params.lens_params = ClosestMap(
            {0: LensParams(capture_area_size=(0.0, 1080.0))}  # no sensor size
        )
        assert _get_frame_readout_time(params, 0.0) == pytest.approx(10.0)


class TestCalculateCameraFovs:
    def test_a_fixed_focal_length_gets_one_value(self):
        """No per-frame calibration, so computing it per frame would be
        `frame_count` identical lookups."""
        params = params_for(profile())
        params.frame_count = 100
        params.calculate_camera_fovs()
        assert len(params.camera_diagonal_fovs) == 1

    def test_the_single_value_matches_the_static_computation(self):
        params = params_for(profile(fx=1000.0, fy=1000.0))
        params.calculate_camera_fovs()
        diagonal = (1920**2 + 1080**2) ** 0.5
        expected = 2.0 * np.degrees(np.arctan(diagonal / (2.0 * 1000.0)))
        assert params.camera_diagonal_fovs[0] == pytest.approx(expected)

    def test_a_moving_calibration_gets_one_value_per_frame(self):
        params = params_for(profile())
        params.frame_count = 4
        params.lens_params = ClosestMap(
            {
                index * 100_000: LensParams(pixel_focal_length=1000.0 + index * 500.0)
                for index in range(4)
            }
        )
        params.calculate_camera_fovs()
        assert len(params.camera_diagonal_fovs) == 4
        # A longer focal length is a narrower field of view.
        assert params.camera_diagonal_fovs == sorted(
            params.camera_diagonal_fovs, reverse=True
        )
        assert params.camera_diagonal_fovs[0] > params.camera_diagonal_fovs[-1]

    def test_a_zero_focal_length_degrades_to_the_reference(self):
        params = params_for(profile())
        params.frame_count = 2
        params.lens_params = ClosestMap(
            {0: LensParams(pixel_focal_length=0.0),
             100_000: LensParams(pixel_focal_length=0.0)}
        )
        params.calculate_camera_fovs()
        assert params.camera_diagonal_fovs == [120.0, 120.0]


# ----------------------------------------------------------------------
# The parser side
# ----------------------------------------------------------------------


def _mp4_box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + fourcc + payload


def _sony_tlv(tag: int, payload: bytes) -> bytes:
    return struct.pack(">HH", tag, len(payload)) + payload


def build_rtmd_mp4(focal_per_packet, readings_per_packet=3, delta_ms=100.0):
    """A minimal mp4 with one rtmd track whose packets differ in 0x8005.

    `tests/test_e2e.py` has a similar builder, but it writes the same payload
    every packet — which is exactly the dimension these tests need to vary.
    """
    packets = []
    for focal in focal_per_packet:
        gyro = [(655, -482, -650)] * readings_per_packet
        tlv = b"".join([
            _sony_tlv(0xE439, struct.pack(">f", 65.5)),
            _sony_tlv(0xE43A, struct.pack(">H", 0x420)),
            _sony_tlv(0xE43B,
                      struct.pack(">ii", len(gyro), 6)
                      + struct.pack(f">{len(gyro) * 3}h",
                                    *[v for s in gyro for v in s])),
            _sony_tlv(0xE40E, struct.pack(">i", 14297)),
            # 0x8005 is the decimal f16 focal length in mm/1000.
            _sony_tlv(0x8005, struct.pack(">H", focal)),
        ])
        packets.append(b"\x00\x1c" + b"\x00" * 0x1A + tlv)

    payload = b"".join(packets)
    count = len(packets)

    def make_moov(offset: int) -> bytes:
        stsd = _mp4_box(
            b"stsd",
            struct.pack(">II", 0, 1) + struct.pack(">I", 16) + b"rtmd" + b"\x00" * 8,
        )
        stco = _mp4_box(
            b"stco",
            struct.pack(">II", 0, count)
            + b"".join(
                struct.pack(">I", offset + sum(len(p) for p in packets[:i]))
                for i in range(count)
            ),
        )
        stsz = _mp4_box(
            b"stsz",
            struct.pack(">III", 0, 0, count)
            + b"".join(struct.pack(">I", len(p)) for p in packets),
        )
        stsc = _mp4_box(b"stsc", struct.pack(">II", 0, 1) + struct.pack(">III", 1, 1, 1))
        stts = _mp4_box(
            b"stts", struct.pack(">II", 0, 1) + struct.pack(">II", count, int(delta_ms))
        )
        stbl = _mp4_box(b"stbl", stsd + stco + stsz + stsc + stts)
        minf = _mp4_box(b"minf", stbl)
        hdlr = _mp4_box(b"hdlr", b"\x00" * 8 + b"rtmd" + b"\x00" * 12)
        mdhd = _mp4_box(b"mdhd", struct.pack(">IIIII", 0, 0, 0, 1000, 30))
        mdia = _mp4_box(b"mdia", mdhd + hdlr + minf)
        trak = _mp4_box(b"trak", _mp4_box(b"tkhd", b"\x00" * 84) + mdia)
        return _mp4_box(b"moov", _mp4_box(b"mvhd", b"\x00" * 96) + trak)

    moov = make_moov(0)
    moov = make_moov(len(moov))
    return moov + payload


def focal_to_f16(mm: float) -> int:
    """Sony's decimal f16 for a focal length in millimetres.

    The tag decodes as `mantissa * 10^exp` and the parser then multiplies by
    1000, so the payload carries mm/1000 — a 24 mm lens is 0.024, i.e.
    mantissa 24 with exponent -3. The exponent nibble is the two's-complement
    of the exponent in the low four bits (rtmd_tags read_f16).
    """
    value = mm / 1000.0
    for exp in range(-8, 8):
        mantissa = round(value / (10.0**exp))
        if mantissa < 0 or mantissa > 0x0FFF:
            continue
        if abs(mantissa * 10.0**exp - value) > 1e-12:
            continue
        return ((exp if exp >= 0 else exp + 16) << 12) | mantissa
    raise ValueError(f"cannot encode {mm} mm")


class TestSonyFocalLengthPerSample:
    def test_a_zoom_writes_one_entry_per_imu_row(self, tmp_path):
        """The whole point: the focal length changes between packets, and
        each is keyed by its own sample time."""
        path = tmp_path / "zoom.mp4"
        path.write_bytes(
            build_rtmd_mp4(
                [focal_to_f16(24.0), focal_to_f16(48.0), focal_to_f16(70.0)],
                readings_per_packet=2,
            )
        )
        from pygyroflow.telemetry import parse_telemetry_file

        metadata = parse_telemetry_file(str(path), fps=30.0)
        assert len(metadata.raw_imu) == 6
        assert len(metadata.lens_positions) == 6
        # Two rows per packet, so the values come in pairs.
        values = [v for _, v in sorted(metadata.lens_positions.items())]
        assert values == pytest.approx([24.0, 24.0, 48.0, 48.0, 70.0, 70.0])

    def test_the_keys_line_up_with_the_imu_timeline(self, tmp_path):
        path = tmp_path / "zoom.mp4"
        path.write_bytes(
            build_rtmd_mp4([focal_to_f16(24.0), focal_to_f16(48.0)],
                           readings_per_packet=3)
        )
        from pygyroflow.telemetry import parse_telemetry_file

        metadata = parse_telemetry_file(str(path), fps=30.0)
        imu_us = [round(r.timestamp_ms * 1000.0) for r in metadata.raw_imu]
        assert sorted(metadata.lens_positions) == imu_us

    def test_no_focal_tag_leaves_the_map_empty(self, tmp_path):
        """The RX100 VII really does omit it; a clip without the tag must
        not produce a map of zeros."""
        path = tmp_path / "nofocal.mp4"
        data = build_rtmd_mp4([focal_to_f16(24.0)], readings_per_packet=2)
        # Strip the 0x8005 TLV (tag 2 bytes + len 2 bytes + payload 2 bytes).
        data = data.replace(_sony_tlv(0x8005, struct.pack(">H", focal_to_f16(24.0))), b"")
        path.write_bytes(data)
        from pygyroflow.telemetry import parse_telemetry_file

        metadata = parse_telemetry_file(str(path), fps=30.0)
        assert metadata.lens_positions == {}


@pytest.mark.skipif(not _REFERENCE_CLIP.is_file(), reason="reference clip missing")
class TestAgainstRealSonyClips:
    def test_the_prime_lens_clip_reports_a_constant_focal_length(self):
        """An 85 mm prime: the tag is present in every packet and never
        changes. The map is populated — so the lookup path is reachable —
        but the value is constant, so the interpolated profile is the base
        profile. There is no zooming Sony clip in this repo to show more."""
        from pygyroflow.telemetry import parse_telemetry_file

        metadata = parse_telemetry_file(str(_REFERENCE_CLIP), fps=30.0)
        assert metadata.detected_source == "Sony ILCE-7SM3"
        assert len(metadata.lens_positions) == len(metadata.raw_imu)
        assert set(metadata.lens_positions.values()) == {85.0}

    def test_the_zoom_clip_simply_has_no_focal_tag(self):
        """The one genuine zoom in the repo omits 0x8005 entirely. Recorded
        so a future reader does not mistake the empty map for a parse bug."""
        from pygyroflow.telemetry import parse_telemetry_file

        metadata = parse_telemetry_file(str(_ZERO_ZOOM_CLIP), fps=30.0)
        assert metadata.detected_source == "Sony DSC-RX100M7"
        assert metadata.raw_imu
        assert metadata.lens_positions == {}
