"""Port of ``opensource/gyroflow/src/core/gyro_source/sony.rs`` — the Sony
deep features: lens profile from LensDistortion, IBIS/OIS stabilization
splines, per-frame time offsets and mesh correction.

Upstream reads these from a ``GroupedTagMap``; here the per-sample Sony RTMD
tag dict built by ``telemetry.parser._sony_walk_tlv`` (tag id -> payload
bytes) plays that role, and the tag payloads decode per
``vendor/telemetry-parser/src/sony/rtmd_tags.rs``.

Deviations from bit-exactness (documented at each site):
- the degree-6 polynomial fit uses numpy's SVD least squares
  (``lstsq(rcond=1e-18)``) against nalgebra's ``SVD::solve(.., 1e-18)``
- the mesh inverse uses scipy's Nelder-Mead configured to upstream
  argmin's simplex and tolerances — same algorithm family, not the same
  floating-point trajectory.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

from pygyroflow.gyro_source.file_metadata import FileMetadata, LensParams
from pygyroflow.gyro_source.splines import (
    MAX_GRID_SIZE,
    BivariateSpline,
    CatmullRom,
)

# ---------------------------------------------------------------------------
# RTMD tag ids (rtmd_tags.rs)
# ---------------------------------------------------------------------------

T_SENSOR_SIZE_PX = 0xE405       # Imager SensorSizePixels, u32 x2
T_PIXEL_PITCH = 0xE407          # Imager PixelPitch, u32 x2 (nm)
T_CAPTURE_ORIGIN = 0xE409       # Imager CaptureAreaOrigin, f32 x2
T_CAPTURE_SIZE = 0xE40A         # Imager CaptureAreaSize, f32 x2
T_FIRST_FRAME_TS = 0xE40C       # Imager FirstFrameTimestamp, i32 µs -> ms
T_EXPOSURE_TIME = 0xE40D        # Imager ExposureTime, i32 µs -> ms
T_IBIS_DATA = 0xE40F            # IBIS table 1, i32 TimeVector3
T_IBIS_DATA2 = 0xE450           # IBIS table 2, i16 TimeVector3
T_OIS_DATA = 0xE416             # LensOSS table, i32 TimeVector3
T_LENS_DIST_ENABLED = 0xE420    # LensDistortion Enabled, u8
T_LENS_DIST_DATA = 0xE421       # LensDistortion table
T_FPD_ENABLED = 0xE422          # FocalPlaneDistortion Enabled, u8
T_FPD_DATA = 0xE423             # FocalPlaneDistortion table
T_MESH_DATA = 0xE42F            # MeshCorrection table
T_GYRO_FREQUENCY = 0xE435       # Gyroscope Frequency, i32
T_GYRO_SCALER = 0xE436          # Gyroscope sampling scaler, i32 (1e6)
T_GYRO_TIME_OFFSET = 0xE437     # Gyroscope TimeOffset, i32 µs -> ms
T_LENS_FOCAL = 0x8005           # Lens FocalLength, f16 mm x 1000
T_LENS_FOCUS = 0x8001           # Lens FocusDistance, f16 m
T_SENSOR_WIDTH_MM = 0x8104      # Default SensorWidth, u16/1000 mm
T_SENSOR_HEIGHT_MM = 0x8105     # Default SensorHeight, u16/1000 mm


# ---------------------------------------------------------------------------
# Small payload readers (GetWithType equivalents)
# ---------------------------------------------------------------------------

def read_f16(raw: bytes) -> float:
    """Sony decimal-float 16 (rtmd_tags.rs read_f16) — NOT IEEE half."""
    (num,) = struct.unpack(">h", raw[:2])
    exp = (num >> 12) & 0x0F
    if exp >= 8:
        exp = -((~exp & 0x7) + 1)
    return (num & 0x0FFF) * 10.0**exp


def _i32(tags: dict, tag: int) -> int | None:
    raw = tags.get(tag)
    if raw is None or len(raw) < 4:
        return None
    return struct.unpack(">i", raw[:4])[0]


def _i32_ms(tags: dict, tag: int) -> float | None:
    """i32 microseconds read as milliseconds (rtmd_tags `i32/1000.0`)."""
    v = _i32(tags, tag)
    return None if v is None else v / 1000.0


def _u32x2(tags: dict, tag: int) -> tuple[int, int] | None:
    """0xe405/0xe407: u16 BE pairs promoted to u32 (rtmd_tags.rs:389-400)."""
    raw = tags.get(tag)
    if raw is None or len(raw) < 4:
        return None
    a, b = struct.unpack(">HH", raw[:4])
    return a, b


def _f32x2(tags: dict, tag: int) -> tuple[float, float] | None:
    """0xe409/0xe40a: u32 BE read then cast to f32 — the payload holds
    integer pixel counts, not IEEE floats (rtmd_tags.rs:401-410)."""
    raw = tags.get(tag)
    if raw is None or len(raw) < 8:
        return None
    a, b = struct.unpack(">II", raw[:8])
    return float(a), float(b)


def _rust_round(x: float) -> float:
    """f64::round — half away from zero (Python's round is banker's)."""
    return math.copysign(math.floor(abs(x) + 0.5), x)


# ---------------------------------------------------------------------------
# Table decoders
# ---------------------------------------------------------------------------

def decode_ibis_table1(payload: bytes) -> list[tuple[int, int, int, int]] | None:
    """0xe40f: count i32, length i32 (must be 16), count x (t,x,y,z i32)."""
    if len(payload) < 8:
        return None
    count, length = struct.unpack(">ii", payload[:8])
    if length != 16:
        return None
    if count <= 0:
        return []
    need = 8 + count * 16
    if len(payload) < need:
        return None
    flat = struct.unpack(f">{count * 4}i", payload[8:need])
    return [tuple(flat[i * 4:i * 4 + 4]) for i in range(count)]


def decode_ibis_table2(payload: bytes) -> list[tuple[int, int, int, int]] | None:
    """0xe450: count i32, length i32 (must be 10), count x (t i32, xyz i16)."""
    if len(payload) < 8:
        return None
    count, length = struct.unpack(">ii", payload[:8])
    if length != 10:
        return None
    if count <= 0:
        return []
    out = []
    pos = 8
    for _ in range(count):
        if pos + 10 > len(payload):
            return None
        t = struct.unpack(">i", payload[pos:pos + 4])[0]
        x, y, z = struct.unpack(">hhh", payload[pos + 4:pos + 10])
        out.append((t, x, y, z))
        pos += 10
    return out


def decode_ois_table(payload: bytes) -> list[tuple[int, int, int, int]] | None:
    """0xe416: like 0xe40f but a negative count yields the (-1,-1,-1,-1)
    sentinel (the "unsupported lens" marker)."""
    if len(payload) < 8:
        return None
    count, length = struct.unpack(">ii", payload[:8])
    if length != 16:
        return None
    if count < 0:
        return [(-1, -1, -1, -1)]
    if count == 0:
        return []
    out = []
    pos = 8
    for _ in range(count):
        if pos + 16 > len(payload):
            return None
        out.append(struct.unpack(">4i", payload[pos:pos + 16]))
        pos += 16
    return out


def decode_lens_distortion(payload: bytes) -> dict | None:
    """0xe421 — the JSON-equivalent dict upstream's parser produces."""
    if len(payload) < 21:
        return None
    focal_length_nm, eff_sensor_height_nm = struct.unpack(">II", payload[:8])
    (unk1,) = struct.unpack(">B", payload[8:9])
    (coeff_scale,) = struct.unpack(">f", payload[9:13])
    elem_count, elem_size = struct.unpack(">II", payload[13:21])
    if elem_count == 0xFFFFFFFF:
        elem_count = 0
    coeffs: list[int] = []
    pos = 21
    for _ in range(elem_count):
        if pos + 2 > len(payload):
            return None
        coeffs.append(struct.unpack(">H", payload[pos:pos + 2])[0])
        pos += 2
    return {
        "focal_length_nm": focal_length_nm,
        "effective_sensor_height_nm": eff_sensor_height_nm,
        "unk1": unk1,
        "coeff_scale": coeff_scale,
        "coeffs": coeffs,
    }


def decode_focal_plane(payload: bytes) -> dict | None:
    """0xe423 — FocalPlaneDistortion table."""
    if len(payload) < 16:
        return None
    aa = struct.unpack(">i", payload[0:4])[0]
    bb = struct.unpack(">h", payload[4:6])[0]
    cc = struct.unpack(">h", payload[6:8])[0]
    elem_count, _elem_size = struct.unpack(">ii", payload[8:16])
    coords: list[tuple[int, int]] = []
    pos = 16
    for _ in range(elem_count):
        if pos + 4 > len(payload):
            return None
        x, y = struct.unpack(">hh", payload[pos:pos + 4])
        coords.append((x, y))
        pos += 4
    scale = 32768.0 / cc if cc != 0 else 1.0
    return {"unk1": aa, "unk2": bb, "scale": scale, "unk4": coords}


def decode_mesh(payload: bytes) -> dict | None:
    """0xe42f — MeshCorrection table (9-byte header grid, 81x i16 base
    grid, subdivision exponents and the derived mesh grid)."""
    if len(payload) < 180:
        return None
    (unk1,) = struct.unpack(">h", payload[0:2])
    offset_x, offset_y = struct.unpack(">ii", payload[2:10])
    size_x, size_y = struct.unpack(">HH", payload[10:14])
    xs = list(struct.unpack(">81h", payload[14:14 + 162]))
    ys = list(struct.unpack(">81h", payload[176:176 + 162]))
    div2d_x, div2d_y, div_x, div_y = struct.unpack(">4B", payload[338:342])
    divisions_x_2d = 2.0**div2d_x
    divisions_y_2d = 2.0**div2d_y
    total = div_x * div_y
    mesh_2d = []
    for y in range(div_y):
        for x in range(div_x):
            idx = total - 1 - (y * div_x + x)
            mesh_2d.append((
                (size_x / 8.0) * x + (xs[idx] / divisions_x_2d),
                (size_y / 8.0) * y + (ys[idx] / divisions_y_2d),
            ))
    return {
        "unk1": unk1,
        "offset": [offset_x, offset_y],
        "size": [size_x, size_y],
        "mesh": mesh_2d,
        "raw_mesh": list(zip(xs, ys)),
        "divisions_2d": [divisions_x_2d, divisions_y_2d],
        "divisions": [div_x, div_y],
    }


# ---------------------------------------------------------------------------
# CameraStabData
# ---------------------------------------------------------------------------

@dataclass
class CameraStabData:
    """One frame's IBIS/OIS stabilization data (file_metadata.rs:27)."""

    offset: float
    sensor_size: tuple[int, int]
    crop_area: tuple[float, float, float, float]
    pixel_pitch: tuple[int, int]
    ibis_spline: CatmullRom = field(default_factory=CatmullRom)
    ois_spline: CatmullRom = field(default_factory=CatmullRom)


# ---------------------------------------------------------------------------
# get_time_offset (sony.rs:212-230)
# ---------------------------------------------------------------------------

def get_time_offset(
    md: FileMetadata,
    tags: dict,
    sample_rate: float,
    camera_model: str | None = None,
) -> tuple[float, float] | None:
    """Per-frame time offset from Imager/Gyroscope tags.

    Returns (original_sample_rate, offset) or None when the required tags
    are absent.
    """
    model_offset = 1.5 if camera_model == "DSC-RX0M2" else 0.0

    first_frame_ts = _i32_ms(tags, T_FIRST_FRAME_TS)
    exposure_time = _i32_ms(tags, T_EXPOSURE_TIME)
    offset = _i32_ms(tags, T_GYRO_TIME_OFFSET)
    sampling_frequency = _i32(tags, T_GYRO_FREQUENCY)
    if first_frame_ts is None or exposure_time is None or offset is None \
            or sampling_frequency is None:
        return None
    sampling_frequency = float(sampling_frequency)
    scaler = float(_i32(tags, T_GYRO_SCALER) or 1000000)
    original_sample_rate = sampling_frequency

    rounded_offset = _rust_round(offset * 1000.0 * (1000000.0 / scaler))
    period = 1000000.0 / sampling_frequency
    offset_diff = _rust_round(
        rounded_offset - period * math.floor(rounded_offset / period)) / 1000.0

    frame_offset = (first_frame_ts - (exposure_time / 2.0)
                    + ((md.frame_readout_time or 0.0) / 2.0)
                    + model_offset + offset_diff - offset)

    return original_sample_rate, frame_offset / sampling_frequency * sample_rate


# ---------------------------------------------------------------------------
# stab_collect / stab_calc_splines (sony.rs:232-433)
# ---------------------------------------------------------------------------

@dataclass
class ISTemp:
    """Accumulated IBIS/OIS samples across frames (sony.rs:233)."""

    frame_interval: int = 0
    original_sample_rate: float = 0.0
    first_frame_ts: list[float] = field(default_factory=list)
    pixel_pitch: tuple[int, int] = (0, 0)
    sensor_size: tuple[int, int] = (0, 0)
    per_frame_exposure: list[float] = field(default_factory=list)
    per_frame_start_idx: list[int] = field(default_factory=list)
    per_frame_crop: list[tuple[float, float, float, float]] = field(default_factory=list)
    t: list[int] = field(default_factory=list)
    ibis_x: list[int] = field(default_factory=list)
    ibis_y: list[int] = field(default_factory=list)
    ibis_a: list[int] = field(default_factory=list)
    ois_x: list[int] = field(default_factory=list)
    ois_y: list[int] = field(default_factory=list)

    def calc_time_diff(self, i1: int, i2: int) -> int | None:
        a = max(min(i1, i2, len(self.t) - 1), 0)
        b = max(min(max(i1, i2), len(self.t) - 1), 0)
        dt = self.t[b] - self.t[a]
        if dt < 0:
            dt += self.frame_interval
        return dt

    def search_idx(self, frame: int, top_offset: float,
                   time_offset: float) -> tuple[int, float] | None:
        if frame >= len(self.per_frame_start_idx):
            return None
        start_idx = self.per_frame_start_idx[frame]
        index = start_idx
        if start_idx >= len(self.t):
            return None
        current_time = float(self.t[start_idx])
        if top_offset >= 0.0:
            while current_time <= time_offset and index < len(self.t) - 1:
                d = self.calc_time_diff(index, index + 1)
                if d is None:
                    return None
                current_time += float(d)
                index += 1
        else:
            while index > 0 and current_time > time_offset:
                d = self.calc_time_diff(index - 1, index)
                if d is None:
                    return None
                current_time -= float(d)
                index -= 1
        return index, current_time

    def search_top_idx2(self, frame: int,
                        top_offset: float) -> tuple[int, float] | None:
        r = self.search_idx(frame, top_offset, top_offset)
        if r is None:
            return None
        top_index, current_time = r
        adj = 2 if top_offset >= 0.0 else 1
        for _ in range(adj):
            if top_index > 0:
                d = self.calc_time_diff(top_index - 1, top_index)
                if d is None:
                    return None
                current_time -= float(d)
                top_index -= 1
        return top_index, current_time

    def search_bot_idx2(self, frame: int, top_offset: float,
                        bot_offset: float) -> tuple[int, float] | None:
        r = self.search_idx(frame, top_offset, bot_offset)
        if r is None:
            return None
        bot_index, current_time = r
        adj = 2 if bot_offset >= 0.0 else 1
        for _ in range(adj):
            if bot_index > 0:
                d = self.calc_time_diff(bot_index, bot_index + 1)
                if d is None:
                    return None
                current_time += float(d)
                bot_index += 1
        return bot_index, current_time

    def calc_ofs(self, idx: int) -> int | None:
        acc_time = 0
        for i in range(idx):
            d = self.calc_time_diff(i, i + 1)
            if d is None:
                return None
            acc_time += d
        return acc_time


def stab_collect(is_temp: ISTemp, tags: dict, frame_rate: float) -> bool:
    """Accumulate one frame's IBIS/OIS tables (sony.rs:310-368)."""
    frequency = _i32(tags, T_GYRO_FREQUENCY)
    first_frame_ts = _i32_ms(tags, T_FIRST_FRAME_TS)
    exposure_time = _i32_ms(tags, T_EXPOSURE_TIME)
    sensor_size = _u32x2(tags, T_SENSOR_SIZE_PX)
    pixel_pitch = _u32x2(tags, T_PIXEL_PITCH)
    crop_origin = _f32x2(tags, T_CAPTURE_ORIGIN)
    crop_size = _f32x2(tags, T_CAPTURE_SIZE)
    if None in (frequency, first_frame_ts, exposure_time, sensor_size,
                pixel_pitch, crop_origin, crop_size):
        return False

    start_idx = len(is_temp.t)

    ibis_raw = tags.get(T_IBIS_DATA)
    if ibis_raw is not None:
        shift = decode_ibis_table1(ibis_raw)
        angle = decode_ibis_table2(tags.get(T_IBIS_DATA2, b""))
        if shift is None or angle is None or len(shift) != len(angle):
            return False
        for s, a in zip(shift, angle):
            is_temp.t.append(s[0])
            is_temp.ibis_x.append(s[1])
            is_temp.ibis_y.append(s[2])
            is_temp.ibis_a.append(a[3])

    ois_raw = tags.get(T_OIS_DATA)
    if ois_raw is not None:
        shift = decode_ois_table(ois_raw)
        if shift is None:
            return False
        for s in shift:
            if not is_temp.ibis_x:  # OIS-only: `t` wasn't pushed by IBIS
                is_temp.t.append(s[0])
            is_temp.ois_x.append(s[1])
            is_temp.ois_y.append(s[2])

    is_temp.frame_interval = int(1000000.0 / frame_rate)
    is_temp.per_frame_exposure.append(exposure_time * 1000.0)
    is_temp.per_frame_start_idx.append(start_idx)
    is_temp.per_frame_crop.append(
        (crop_origin[0], crop_origin[1], crop_size[0], crop_size[1]))
    is_temp.original_sample_rate = float(frequency)
    is_temp.first_frame_ts.append(first_frame_ts * 1000.0)
    is_temp.pixel_pitch = pixel_pitch
    is_temp.sensor_size = sensor_size
    return True


def stab_calc_splines(
    md: FileMetadata, is_temp: ISTemp
) -> list[CameraStabData] | None:
    """Build per-frame CatmullRom splines (sony.rs:370-433)."""
    num_frames = len(is_temp.per_frame_exposure)
    readout_time = max((md.frame_readout_time or 0.0) * 1000.0, 1.0)

    per_frame_data: list[CameraStabData] = []
    for frame in range(num_frames):
        crop_area = is_temp.per_frame_crop[frame]
        exposuretime = is_temp.per_frame_exposure[frame]
        first_timestamp = is_temp.first_frame_ts[frame]
        top_offset = first_timestamp - exposuretime / 2.0
        bot_offset = top_offset + readout_time
        entry_rate = is_temp.sensor_size[1] / readout_time

        top = is_temp.search_top_idx2(frame, top_offset)
        if top is None:
            continue
        top_index, time = top
        bot = is_temp.search_bot_idx2(frame, top_offset, bot_offset)
        if bot is None:
            continue
        n_entries = bot[0] - top_index + 1

        ofs_rows = int(abs(time - top_offset) * entry_rate)

        ibis_spline = CatmullRom()
        ois_spline = CatmullRom()
        frame_failed = False

        for i in range(n_entries):
            ofs = is_temp.calc_ofs(i)
            if ofs is None:
                # upstream `?`: this frame is dropped from the result
                frame_failed = True
                break
            ts = ofs * entry_rate
            if top_index + i < len(is_temp.ibis_x):
                ibis_spline.add_point(ts, np.array([
                    is_temp.ibis_x[top_index + i],
                    is_temp.ibis_y[top_index + i],
                    is_temp.ibis_a[top_index + i],
                ], dtype=np.float64))
            if top_index + i < len(is_temp.ois_x):
                ois_spline.add_point(ts, np.array([
                    is_temp.ois_x[top_index + i],
                    is_temp.ois_y[top_index + i],
                    0.0,
                ], dtype=np.float64))
        if frame_failed:
            continue

        per_frame_data.append(CameraStabData(
            offset=float(ofs_rows),
            sensor_size=is_temp.sensor_size,
            crop_area=crop_area,
            pixel_pitch=is_temp.pixel_pitch,
            ibis_spline=ibis_spline,
            ois_spline=ois_spline,
        ))

    if not per_frame_data:
        return None
    assert len(per_frame_data) == num_frames
    return per_frame_data


# ---------------------------------------------------------------------------
# init_lens_profile (sony.rs:11-209)
# ---------------------------------------------------------------------------

def _a2y(a: float, params: np.ndarray) -> float:
    return float(sum(a ** (i + 1) * params[i] for i in range(6)))


def _a2y_diff(a: float, params: np.ndarray) -> float:
    return float(sum((i + 1.0) * a**i * params[i] for i in range(6)))


def _y2a(y: float, params: np.ndarray) -> float:
    x = 0.01
    for _ in range(50):
        x = x - (_a2y(x, params) - y) / _a2y_diff(x, params)
    return x


def _sync_settings() -> dict:
    return {
        "initial_offset": 0,
        "initial_offset_inv": False,
        "search_size": 0.3,
        "max_sync_points": 5,
        "every_nth_frame": 1,
        "time_per_syncpoint": 0.5,
        "do_autosync": False,
    }


def init_lens_profile(
    md: FileMetadata,
    tags: dict,
    size: tuple[int, int],
    timestamp_ms: float,
    camera_model: str | None = None,
    lens_display_name: str | None = None,
) -> None:
    """Build md.lens_profile / update md.lens_params from LensDistortion.

    ``video_rotation`` is treated as 0 (landscape): upstream reads it from
    the sample info, which the vendored RTMD parser does not provide for
    MP4 sources.
    """
    lmd_raw = tags.get(T_LENS_DIST_DATA)
    if lmd_raw is None:
        return
    lmd = decode_lens_distortion(lmd_raw)
    if lmd is None:
        return

    pixel_pitch = _u32x2(tags, T_PIXEL_PITCH)
    crop_size = _f32x2(tags, T_CAPTURE_SIZE)
    lens_compensation_enabled = bool(_i32(tags, T_LENS_DIST_ENABLED))

    try:
        if pixel_pitch is None or crop_size is None:
            return

        video_rotation = 0.0
        is_vertical = video_rotation == 90 or video_rotation == 270

        focal_mm_from_tag = read_f16(tags[T_LENS_FOCAL]) * 1000.0 \
            if T_LENS_FOCAL in tags and len(tags[T_LENS_FOCAL]) >= 2 else None
        focal_length_str = f"{focal_mm_from_tag:.2f} mm" \
            if focal_mm_from_tag is not None else None

        focal_length_mm = float(lmd["focal_length_nm"]) / 1000000.0
        approx_focal_length_mm = focal_mm_from_tag \
            if focal_mm_from_tag is not None else focal_length_mm

        ratio = approx_focal_length_mm / max(focal_length_mm, 0.000001)
        is_bad_focal_length = abs(ratio - 1.0) > 0.5

        sensor_height = float(lmd["effective_sensor_height_nm"]) / 1e9
        coeff_scale = float(lmd["coeff_scale"])
        lens_in_ray_angle = [
            float(c) / max(coeff_scale, 1.0) / 180.0 * math.pi
            for c in lmd["coeffs"]
        ]
        if not lens_in_ray_angle or sensor_height == 0.0 or is_bad_focal_length:
            sensor_size_px = _u32x2(tags, T_SENSOR_SIZE_PX)
            if sensor_size_px is None:
                return

            fallback_focal_mm = approx_focal_length_mm if is_bad_focal_length \
                else float(lmd["focal_length_nm"]) / 1000000.0
            sws = crop_size[0] / max(sensor_size_px[0], 1.0)
            shs = crop_size[1] / max(sensor_size_px[1], 1.0)

            sw = (_f32_mm(tags, T_SENSOR_WIDTH_MM) or 0.0) * sws
            sh = (_f32_mm(tags, T_SENSOR_HEIGHT_MM) or 0.0) * shs

            # Fallback lens profile: focal length only, no distortion.
            if fallback_focal_mm > 0.0 and sw > 0.0 and sh > 0.0:
                fx = fallback_focal_mm / sw * size[0]
                fy = fallback_focal_mm / sh * size[1]
                timestamp_us = int(_rust_round(timestamp_ms * 1000.0))
                lp = md.lens_params.get(timestamp_us)
                if lp is not None:
                    lp.focal_length = fallback_focal_mm
                    lp.pixel_focal_length = fx
                if md.lens_profile is None:
                    lens_model = _lens_model_name(
                        lens_display_name, focal_length_str)
                    md.lens_profile = {
                        "calibrated_by": "Not calibrated",
                        "camera_brand": "Sony",
                        "camera_model": camera_model or "",
                        "lens_model": lens_model,
                        "calib_dimension": {"w": size[0], "h": size[1]},
                        "orig_dimension": {"w": size[0], "h": size[1]},
                        "output_dimension": {
                            "w": size[1] if is_vertical else size[0],
                            "h": size[0] if is_vertical else size[1],
                        },
                        "frame_readout_time": md.frame_readout_time,
                        "official": False,
                        "asymmetrical": False,
                        "note": f"Distortion comp.: "
                                f"{'On' if lens_compensation_enabled else 'Off'}",
                        "fisheye_params": {
                            "camera_matrix": [
                                [fx, 0.0, size[0] / 2],
                                [0.0, fy, size[1] / 2],
                                [0.0, 0.0, 1.0],
                            ],
                            "distortion_coeffs": [],
                        },
                        "sync_settings": {},
                        "calibrator_version": "---",
                    }
            return

        lens_in_ray_angle.insert(0, 0.0)
        lens_out_radius = np.array([
            (i / 10.0) * sensor_height for i in range(11)], dtype=np.float64)

        matrix = np.array([
            [angle ** (power + 1) for power in range(6)]
            for angle in lens_in_ray_angle
        ], dtype=np.float64)
        # nalgebra SVD::new(m, true, true).solve(&b, 1e-18)
        poly_coeffs, _, _, _ = np.linalg.lstsq(matrix, lens_out_radius,
                                               rcond=1e-18)
        if poly_coeffs.size != 6:
            return

        sensor_crop_px = np.array([crop_size[0], crop_size[1]])
        pitch_m = np.array([pixel_pitch[0], pixel_pitch[1]]) / 1e9
        video_res_px = np.array([float(size[0]), float(size[1])])

        # Rust float division is silent (0/0 → inf/NaN); mirror that
        # instead of warning — downstream try_block-style guards decide.
        with np.errstate(divide="ignore", invalid="ignore"):
            sensor_crop = pitch_m * sensor_crop_px
            pixel_pitch_scaled = sensor_crop / video_res_px

        fov_hor = _y2a(sensor_crop[0] / 2.0, poly_coeffs)
        fov_vert = _y2a(sensor_crop[1] / 2.0, poly_coeffs)
        fov_diag = _y2a(float(np.linalg.norm(sensor_crop)) / 2.0, poly_coeffs)

        with np.errstate(divide="ignore", invalid="ignore"):
            focal_length = max(
                video_res_px[0] / math.tan(fov_hor),
                video_res_px[1] / math.tan(fov_vert),
                float(np.linalg.norm(video_res_px)) / math.tan(fov_diag),
            ) / 2.0
            post_scale = [
                1.0 / pixel_pitch_scaled[0] / focal_length,
                1.0 / pixel_pitch_scaled[1] / focal_length,
            ]
        fx = focal_length

        timestamp_us = int(_rust_round(timestamp_ms * 1000.0))
        lp = md.lens_params.get(timestamp_us)
        if lp is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                lp.focal_length = focal_length * sensor_height / size[1] * 1000.0
            lp.pixel_focal_length = focal_length
            lp.distortion_coefficients = \
                list(poly_coeffs) + list(post_scale)

        if md.lens_profile is None:
            lens_model = _lens_model_name(lens_display_name, focal_length_str)
            md.lens_profile = {
                "calibrated_by": "Sony",
                "camera_brand": "Sony",
                "camera_model": camera_model or "",
                "lens_model": lens_model,
                "calib_dimension": {"w": size[0], "h": size[1]},
                "orig_dimension": {"w": size[0], "h": size[1]},
                "output_dimension": {
                    "w": size[1] if is_vertical else size[0],
                    "h": size[0] if is_vertical else size[1],
                },
                "frame_readout_time": md.frame_readout_time,
                "official": True,
                "asymmetrical": False,
                "note": f"Distortion comp.: "
                        f"{'On' if lens_compensation_enabled else 'Off'}",
                "fisheye_params": {
                    "camera_matrix": [
                        [fx, 0.0, size[0] / 2],
                        [0.0, fx, size[1] / 2],
                        [0.0, 0.0, 1.0],
                    ],
                    "distortion_coeffs": [],
                },
                "distortion_model": "sony",
                "sync_settings": _sync_settings(),
                "calibrator_version": "---",
            }
    except (ArithmeticError, ValueError, OverflowError):
        # telemetry_parser::try_block! — upstream swallows the failure too
        return


def _f32_mm(tags: dict, tag: int) -> float | None:
    """0x8104/0x8105: u16/1000 mm."""
    raw = tags.get(tag)
    if raw is None or len(raw) < 2:
        return None
    return struct.unpack(">H", raw[:2])[0] / 1000.0


def _lens_model_name(display_name: str | None,
                     focal_length_str: str | None) -> str:
    if display_name and focal_length_str:
        return f"{display_name} ({focal_length_str})"
    if display_name:
        return display_name
    return focal_length_str or ""


def collect_lens_params(
    lens_state: LensParams, tags: dict, timestamp_us: int
) -> LensParams | None:
    """The mod.rs:194-243 Imager/Lens walk that fills md.lens_params.

    *lens_state* accumulates across samples (upstream keeps one LensParams
    and mutates it per sample); when it has pitch + crop area + a focal
    length, (timestamp_us, clone) is emitted.
    """
    if tags.get(T_PIXEL_PITCH) is not None:
        pp = _u32x2(tags, T_PIXEL_PITCH)
        if pp is not None:
            lens_state.pixel_pitch = pp
    ca_size = _f32x2(tags, T_CAPTURE_SIZE)
    if ca_size is not None:
        lens_state.capture_area_size = ca_size
        lens_state.capture_area_origin = (0.0, 0.0)
    ca_origin = _f32x2(tags, T_CAPTURE_ORIGIN)
    if ca_origin is not None:
        lens_state.capture_area_origin = ca_origin
    sp = _u32x2(tags, T_SENSOR_SIZE_PX)
    if sp is not None:
        lens_state.sensor_size_px = sp
    if T_LENS_FOCAL in tags and len(tags[T_LENS_FOCAL]) >= 2:
        lens_state.focal_length = read_f16(tags[T_LENS_FOCAL]) * 1000.0
    if T_LENS_FOCUS in tags and len(tags[T_LENS_FOCUS]) >= 2:
        lens_state.focus_distance = read_f16(tags[T_LENS_FOCUS])

    if lens_state.pixel_pitch is not None \
            and lens_state.capture_area_size is not None \
            and (lens_state.pixel_focal_length is not None
                 or lens_state.focal_length is not None):
        return LensParams(**vars(lens_state))
    return None


# ---------------------------------------------------------------------------
# get_mesh_correction (sony.rs:435-579)
# ---------------------------------------------------------------------------

def _mesh_jsonish(mesh_data: dict | None, focal_plane_data: dict | None,
                  crop_origin: tuple[float, float],
                  crop_size: tuple[float, float]) -> str:
    """The canonical serialization the upstream cache keys its CRC on."""

    def enc(v):
        if v is None:
            return "null"
        if isinstance(v, dict):
            return "{" + ",".join(
                f'"{k}":{enc(x)}' for k, x in v.items()) + "}"
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(enc(x) for x in v) + "]"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return repr(v)
        return str(v)

    return enc([mesh_data, focal_plane_data,
                float(crop_origin[0]), float(crop_origin[1]),
                float(crop_size[0]), float(crop_size[1])])


def get_mesh_correction(
    tags: dict,
    cache: dict[int, tuple[list[float], list[float]]],
) -> tuple[list[float], list[float]] | None:
    """Forward + inverse mesh correction buffers from MeshCorrection and
    FocalPlaneDistortion tags."""
    crop_origin = _f32x2(tags, T_CAPTURE_ORIGIN)
    crop_size = _f32x2(tags, T_CAPTURE_SIZE)
    if crop_origin is None or crop_size is None:
        return None

    mesh_data = decode_mesh(tags[T_MESH_DATA]) if T_MESH_DATA in tags else None
    focal_plane_data = decode_focal_plane(tags[T_FPD_DATA]) \
        if T_FPD_DATA in tags else None

    crc = zlib.crc32(_mesh_jsonish(mesh_data, focal_plane_data,
                                   crop_origin, crop_size).encode())
    if crc in cache:
        return cache[crc]

    has_any_mesh_value = False
    has_any_focal_plane_value = False
    if mesh_data is not None:
        for coord in mesh_data["raw_mesh"]:
            if coord[0] != 0 or coord[1] != 0:
                has_any_mesh_value = True
                break
    fp_coords: list[float] = []
    if focal_plane_data is not None:
        unk1 = float(focal_plane_data["unk1"])
        unk2 = float(focal_plane_data["unk2"])
        scale = float(focal_plane_data["scale"])
        fp_coords = [float(len(focal_plane_data["unk4"])), unk1, unk2, scale]
        for coord in focal_plane_data["unk4"]:
            has_any_focal_plane_value = True
            fp_coords.append(coord[0] / 32768.0)
            fp_coords.append(coord[1] / 32768.0)
        if len(fp_coords) == 4:
            fp_coords = [0.0]
        elif fp_coords[0] != 8.0:
            fp_coords = [0.0]
    else:
        fp_coords = [0.0]

    if not has_any_mesh_value and not has_any_focal_plane_value:
        return None

    size = ((mesh_data["size"][0], mesh_data["size"][1])
            if mesh_data is not None else (0.0, 0.0))
    divisions = (tuple(mesh_data["divisions"]) if mesh_data is not None
                 else (0, 0))

    mesh = _build_mesh_buffer(
        mesh_data, divisions, size, crop_origin, crop_size,
        fp_coords, has_any_mesh_value, forward_mesh=None)
    inv_mesh = _build_mesh_buffer(
        mesh_data, divisions, size, crop_origin, crop_size,
        fp_coords, has_any_mesh_value, forward_mesh=mesh)

    result = (mesh, inv_mesh)
    cache[crc] = result
    return result


def _build_mesh_buffer(
    mesh_data: dict | None,
    divisions: tuple[int, int],
    size: tuple[float, float],
    crop_origin: tuple[float, float],
    crop_size: tuple[float, float],
    fp_coords: list[float],
    has_any_mesh_value: bool,
    forward_mesh: list[float] | None,
) -> list[float]:
    # Upstream pushes the whole fixed-size coefficient arrays
    # ([f64; MAX_GRID_SIZE] each) per row — MAX_GRID_SIZE is 9, so every
    # (component, row) block is 4*9 floats wide, matching the layout
    # BivariateSpline::interpolate reads back (block = grid_h * 4 at the
    # only legal grid width).
    a = [0.0] * MAX_GRID_SIZE
    b = [0.0] * MAX_GRID_SIZE
    c = [0.0] * MAX_GRID_SIZE
    d = [0.0] * MAX_GRID_SIZE
    alpha = [0.0] * (MAX_GRID_SIZE - 1)
    mu = [0.0] * MAX_GRID_SIZE
    z = [0.0] * MAX_GRID_SIZE

    out: list[float] = [0.0,
                        float(divisions[0]), float(divisions[1]),
                        float(size[0]), float(size[1]),
                        float(crop_origin[0]), float(crop_origin[1]),
                        float(crop_size[0]), float(crop_size[1])]

    if has_any_mesh_value and mesh_data is not None:
        if forward_mesh is not None:
            # Inverse buffer: sample the COMPLETE forward mesh at each grid
            # point and solve for the pre-image (sony.rs:553-556).
            step = (size[0] / (divisions[0] - 1.0),
                    size[1] / (divisions[1] - 1.0))
            grid = [(x, y) for y in range(divisions[1])
                    for x in range(divisions[0])]
            for (gx, gy) in grid:
                target = (step[0] * gx, step[1] * gy)
                inv = _inverse_interpolate_mesh(
                    target[0], target[1], size, forward_mesh)
                out.extend([inv[0], inv[1]])
        else:
            for coord in mesh_data["mesh"]:
                out.append(float(coord[0]))
                out.append(float(coord[1]))

        for mesh_offset in range(2):
            for j in range(divisions[1]):
                BivariateSpline._cubic_spline_coefficients(
                    out[9 + mesh_offset:], 2, j * divisions[0],
                    size[0], divisions[0], a, b, c, d, alpha, mu, z)
                out.extend(a)
                out.extend(b)
                out.extend(c)
                out.extend(d)

    out[0] = float(len(out))
    out.extend(fp_coords)
    return out


def _inverse_interpolate_mesh(
    x_prime: float, y_prime: float, size: tuple[float, float], mesh: list[float]
) -> tuple[float, float]:
    from scipy.optimize import minimize

    def cost(p):
        interp = BivariateSpline(int(mesh[1]), int(mesh[2]))
        r = (interp.interpolate(size[0], size[1], mesh, 0, p[0], p[1]) - x_prime,
             interp.interpolate(size[0], size[1], mesh, 1, p[0], p[1]) - y_prime)
        return r[0] * r[0] + r[1] * r[1]

    x0 = [x_prime, y_prime]
    result = minimize(
        cost, x0, method="Nelder-Mead",
        options={
            "initial_simplex": [
                x0,
                [x_prime + 0.0003, y_prime],
                [x_prime, y_prime + 0.0003],
            ],
            "xatol": 1e-10,
            "fatol": 1e-10,
            "maxiter": 400,
        },
    )
    return float(result.x[0]), float(result.x[1])
