"""Gyroflow Protobuf telemetry — port of ``vendor/telemetry-parser
src/gyroflow/binary.rs`` over the wire schema in ``gyroflow.proto``.

The upstream converter emits typed tag groups that gyro_source/mod.rs then
maps onto FileMetadata. As with the other parsers in this package, those two
stages are fused here: each decoded ``Main`` message fills FileMetadata
directly, keeping binary.rs's semantics (exposure precedence, per-frame
readout authority, camera-clock timestamps, IBIS sign flip, quats_rotation
conjugation, EIS-quat → image_orientations quantization).

Deviations from bit-exactness: none intended; floats are round-tripped
through the protobuf wire format exactly.
"""

from __future__ import annotations

import logging
import struct

import numpy as np

from pygyroflow.gyro_source.file_metadata import FileMetadata, LensParams
from pygyroflow.gyro_source.sony import ISTemp, stab_calc_splines
from pygyroflow.types.enums import ReadoutDirection
from pygyroflow.types.time_types import TimeIMU

log = logging.getLogger(__name__)

_MAGIC = b"GyroflowProtobuf"

# ReadoutDirection enum values (gyroflow.proto clip_metadata)
_READOUT_DIRECTIONS = {
    0: ReadoutDirection.TopToBottom,
    1: ReadoutDirection.BottomToTop,
    2: ReadoutDirection.LeftToRight,
    3: ReadoutDirection.RightToLeft,
}


# ---------------------------------------------------------------------------
# Generic protobuf wire decoding
# ---------------------------------------------------------------------------

def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _read_fields(data: bytes) -> dict[int, list[tuple[int, object]]]:
    """Wire-type-aware field scan: field -> [(wire_type, value), ...].

    Values are ints for varint, raw bytes for length-delimited, and decoded
    floats for the fixed 64/32-bit types.
    """
    fields: dict[int, list[tuple[int, object]]] = {}
    pos = 0
    end = len(data)
    while pos < end:
        key, pos = _read_varint(data, pos)
        field_num = key >> 3
        wire_type = key & 7
        if wire_type == 0:
            val, pos = _read_varint(data, pos)
        elif wire_type == 1:
            if pos + 8 > end:
                break
            val = struct.unpack("<d", data[pos:pos + 8])[0]
            pos += 8
        elif wire_type == 5:
            if pos + 4 > end:
                break
            val = struct.unpack("<f", data[pos:pos + 4])[0]
            pos += 4
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            if pos + length > end:
                break
            val = data[pos:pos + length]
            pos += length
        else:
            break  # groups (3/4) — not used by this schema
        fields.setdefault(field_num, []).append((wire_type, val))
    return fields


def _first(msg: dict, n: int) -> tuple[int, object] | None:
    entries = msg.get(n)
    return entries[0] if entries else None


def _as_i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def get_u32(msg: dict, n: int, default: int = 0) -> int:
    e = _first(msg, n)
    return int(e[1]) if e else default


def get_i32(msg: dict, n: int, default: int = 0) -> int:
    e = _first(msg, n)
    return _as_i32(int(e[1])) if e else default


def get_bool(msg: dict, n: int) -> bool:
    e = _first(msg, n)
    return bool(e[1]) if e else False


def get_f32(msg: dict, n: int) -> float | None:
    e = _first(msg, n)
    if e is None:
        return None
    return float(e[1]) if e[0] == 5 else struct.unpack("<f", struct.pack("<I", int(e[1]) & 0xFFFFFFFF))[0]


def get_f64(msg: dict, n: int) -> float | None:
    e = _first(msg, n)
    if e is None:
        return None
    if e[0] == 1:
        return float(e[1])
    if e[0] == 0:  # some producers varint-encode doubles? not per spec
        return float(e[1])
    return None


def get_str(msg: dict, n: int) -> str:
    e = _first(msg, n)
    if e is None or e[0] != 2:
        return ""
    return bytes(e[1]).decode("utf-8", "replace")


def get_msg(msg: dict, n: int) -> dict | None:
    e = _first(msg, n)
    if e is None or e[0] != 2:
        return None
    return _read_fields(bytes(e[1]))


def get_msgs(msg: dict, n: int) -> list[dict]:
    return [_read_fields(bytes(v)) for wt, v in msg.get(n, []) if wt == 2]


def get_packed_f32(msg: dict, n: int) -> list[float]:
    """Repeated float: packed (wire type 2) or unpacked (wire type 5)."""
    out: list[float] = []
    for wt, v in msg.get(n, []):
        if wt == 2:
            out.extend(struct.unpack(f"<{len(v) // 4}f", v[:len(v) // 4 * 4]))
        elif wt == 5:
            out.append(float(v))
    return out


# ---------------------------------------------------------------------------
# Message parsers (gyroflow.proto schema)
# ---------------------------------------------------------------------------

def parse_main(data: bytes) -> dict | None:
    """Main: magic=1, protocol_version=2, header=3, frame=4."""
    msg = _read_fields(data)
    if _first(msg, 1) is None and _first(msg, 3) is None and _first(msg, 4) is None:
        return None
    return msg


def parse_camera(msg: dict) -> dict:
    return {
        "brand": get_str(msg, 1),
        "model": get_str(msg, 2),
        "serial": get_str(msg, 3),
        "lens_brand": get_str(msg, 5),
        "lens_model": get_str(msg, 6),
        "pp_x": get_u32(msg, 7),
        "pp_y": get_u32(msg, 8),
        "sensor_w": get_u32(msg, 9),
        "sensor_h": get_u32(msg, 10),
        "crop_factor": get_f32(msg, 11),
        "lens_profile_pref": get_str(msg, 12),
        "imu_orientation": get_str(msg, 13),
        "imu_rotation": get_msg(msg, 14),
        "quats_rotation": get_msg(msg, 15),
        "additional_data": get_str(msg, 16),
    }


def parse_clip(msg: dict) -> dict:
    return {
        "frame_w": get_u32(msg, 1),
        "frame_h": get_u32(msg, 2),
        "record_fps": get_f32(msg, 4) or 0.0,
        "sensor_fps": get_f32(msg, 5) or 0.0,
        "rotation": get_i32(msg, 7),
        "imu_sample_rate": get_u32(msg, 8),
        "pixel_aspect": get_f32(msg, 10) or 0.0,
        "frame_readout_time_us": get_f64(msg, 11) or 0.0,
        "readout_direction": get_i32(msg, 12),
    }


def parse_quat(msg: dict | None) -> tuple[float, float, float, float] | None:
    if msg is None:
        return None
    return (get_f32(msg, 1) or 0.0, get_f32(msg, 2) or 0.0,
            get_f32(msg, 3) or 0.0, get_f32(msg, 4) or 0.0)


def parse_frame(msg: dict) -> dict:
    lens = []
    for lmsg in get_msgs(msg, 16):
        dist = None
        for n, name in ((6, "opencv_fisheye"), (7, "opencv_standard"),
                        (8, "poly3"), (9, "poly5"), (10, "ptlens"),
                        (11, "generic_polynomial")):
            sub = get_msg(lmsg, n)
            if sub is not None:
                dist = (name, get_packed_f32(sub, 1))
                break
        if get_msg(lmsg, 5) is not None:
            dist = ("no_distortion", [])
        lens.append({
            "intrinsic": get_packed_f32(lmsg, 1),
            "focal_length_mm": get_f32(lmsg, 2),
            "f_number": get_f32(lmsg, 3),
            "focus_distance_mm": get_f32(lmsg, 4),
            "distortion": dist,
        })
    imu = [{
        "ts": get_f64(m, 1),
        "gyro": (get_f32(m, 2) or 0.0, get_f32(m, 3) or 0.0, get_f32(m, 4) or 0.0),
        "acc": (get_f32(m, 5) or 0.0, get_f32(m, 6) or 0.0, get_f32(m, 7) or 0.0),
        "mag": (get_f32(m, 8), get_f32(m, 9), get_f32(m, 10)),
    } for m in get_msgs(msg, 17)]
    quats = [{
        "ts": get_f64(m, 1),
        "quat": parse_quat(get_msg(m, 2)),
    } for m in get_msgs(msg, 18)]
    ois = [{
        "ts": get_f64(m, 1),
        "x": get_f32(m, 2) or 0.0,
        "y": get_f32(m, 3) or 0.0,
    } for m in get_msgs(msg, 19)]
    ibis = [{
        "ts": get_f64(m, 1),
        "x": get_f32(m, 2) or 0.0,
        "y": get_f32(m, 3) or 0.0,
        "roll": get_f32(m, 4) or 0.0,
    } for m in get_msgs(msg, 20)]
    eis = []
    for m in get_msgs(msg, 21):
        entry: dict = {"ts": get_f64(m, 1)}
        q = parse_quat(get_msg(m, 2))
        if q is not None:
            entry["quaternion"] = q
        mw = get_msg(m, 3)
        if mw is not None:
            gw, gh = get_u32(mw, 1), get_u32(mw, 2)
            warped = get_packed_f32(mw, 5)
            entry["mesh_warp"] = {
                "grid_w": gw, "grid_h": gh,
                "region_w": get_f32(mw, 3) or 0.0,
                "region_h": get_f32(mw, 4) or 0.0,
                "warped_xy": warped,
            }
        eis.append(entry)
    return {
        "start_ts": get_f64(msg, 1) or 0.0,
        "end_ts": get_f64(msg, 2) or 0.0,
        "iso": get_u32(msg, 4),
        "exposure_us": get_f64(msg, 5),
        "digital_zoom": get_f32(msg, 8),
        "shutter_num": get_i32(msg, 9),
        "shutter_den": get_i32(msg, 10),
        "shutter_angle": get_f32(msg, 11),
        "crop_x": get_f32(msg, 12),
        "crop_y": get_f32(msg, 13),
        "crop_w": get_f32(msg, 14),
        "crop_h": get_f32(msg, 15),
        "lens": lens,
        "imu": imu,
        "quaternions": quats,
        "ois": ois,
        "ibis": ibis,
        "eis": eis,
    }


# ---------------------------------------------------------------------------
# binary.rs helpers
# ---------------------------------------------------------------------------

def pack_readout_time_ms(us: float, direction: ReadoutDirection) -> float:
    """Pack the scan direction into the readout magnitude (binary.rs:108)."""
    ms = us / 1000.0
    if direction is ReadoutDirection.BottomToTop:
        return -ms
    if direction is ReadoutDirection.LeftToRight:
        return ms + 10000.0
    if direction is ReadoutDirection.RightToLeft:
        return -(ms + 10000.0)
    return ms


def orient_vec3(v: tuple[float, float, float], io: bytes) -> tuple[float, float, float]:
    def m(o: int, comp: float) -> float:
        c = chr(o)
        if c == "X":
            return v[0]
        if c == "x":
            return -v[0]
        if c == "Y":
            return v[1]
        if c == "y":
            return -v[1]
        if c == "Z":
            return v[2]
        if c == "z":
            return -v[2]
        return 0.0
    return (m(io[0], v[0]), m(io[1], v[1]), m(io[2], v[2]))


def rotate_vec3_by_quat(v: tuple[float, float, float],
                        q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Hamilton rotation v' = q·v·q⁻¹, expanded (binary.rs:69)."""
    qw, qx, qy, qz = q
    vx, vy, vz = v
    dot = qx * vx + qy * vy + qz * vz
    w2_uu = qw * qw - (qx * qx + qy * qy + qz * qz)
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx
    return (
        2.0 * dot * qx + w2_uu * vx + 2.0 * qw * cx,
        2.0 * dot * qy + w2_uu * vy + 2.0 * qw * cy,
        2.0 * dot * qz + w2_uu * vz + 2.0 * qw * cz,
    )


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def conjugate_quat_by(q, r):
    """q' = r · q · r⁻¹ (binary.rs:100)."""
    r_inv = (r[0], -r[1], -r[2], -r[3])
    return quat_mul(quat_mul(r, q), r_inv)


# ---------------------------------------------------------------------------
# Tag-dict payload synthesis (feeds gyro_source.sony's verified decoders)
# ---------------------------------------------------------------------------

def _encode_vec_timevector3_i32(entries) -> bytes:
    """The 0xe40f/0xe416 byte layout: count i32, length 16, entries."""
    return struct.pack(">ii", len(entries), 16) + b"".join(
        struct.pack(">4i", *e) for e in entries)


def _encode_vec_timevector3_i16z(entries) -> bytes:
    """The 0xe450 layout: count i32, length 10, (t i32, x/y/z i16)."""
    return struct.pack(">ii", len(entries), 10) + b"".join(
        struct.pack(">i3h", *e) for e in entries)


def _synth_sony_tags(ibis_shifts, ibis_angles, ois_shifts,
                     pitch, sensor, origin, size,
                     first_frame_ts_us: float, exposure_us: float,
                     frequency: int) -> dict:
    """Byte-layout tag dict in the shape gyro_source.sony's decoders read."""
    tags: dict = {
        0xE405: struct.pack(">HH", sensor[0], sensor[1]),
        0xE407: struct.pack(">HH", pitch[0], pitch[1]),
        0xE409: struct.pack(">II", int(origin[0]), int(origin[1])),
        0xE40A: struct.pack(">II", int(size[0]), int(size[1])),
        0xE40C: struct.pack(">i", int(first_frame_ts_us)),
        0xE40D: struct.pack(">i", int(exposure_us)),
        0xE435: struct.pack(">i", frequency),
    }
    if ibis_shifts:
        tags[0xE40F] = _encode_vec_timevector3_i32(ibis_shifts)
        tags[0xE450] = _encode_vec_timevector3_i16z(ibis_angles)
    if ois_shifts:
        tags[0xE416] = _encode_vec_timevector3_i32(ois_shifts)
    return tags


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

class _State:
    """Cross-message state (binary.rs GyroflowProtobuf struct)."""

    def __init__(self) -> None:
        self.camera: dict | None = None
        self.clip: dict | None = None
        self.imu_orientation = "XYZ"
        self.imu_rotation: tuple | None = None
        self.quats_rotation: tuple | None = None
        self.vendor = "Gyroflow"
        self.model: str | None = None
        self.frame_readout_time: float | None = None
        self.frame_readout_time_us = 0.0
        self.readout_direction = ReadoutDirection.TopToBottom
        self.distortion_model_name: str | None = None
        self.lens_profile_emitted = False
        self.first_start_ts_us: float | None = None
        # EIS-quat i16 payloads across frames → image_orientations (A-05
        # mod.rs:337-343 zip happens once at the end)
        self.eis_iori: list = []

    def nominal_frame_interval_us(self) -> float:
        c = self.clip
        if c and c["record_fps"] > 0.0:
            return 1.0e6 / c["record_fps"]
        if c and c["sensor_fps"] > 0.0:
            return 1.0e6 / c["sensor_fps"]
        return 1.0e6 / 60.0

    def resolve_exposure_us(self, frame: dict) -> float:
        if frame["exposure_us"] is not None:
            return max(frame["exposure_us"], 0.0)
        num, den = frame["shutter_num"], frame["shutter_den"]
        if num and den:
            return abs(num / den) * 1.0e6
        angle = frame["shutter_angle"]
        if angle is not None:
            c = self.clip
            if c and c["sensor_fps"] > 0.0:
                rate = c["sensor_fps"]
            elif c and c["record_fps"] > 0.0:
                rate = c["record_fps"]
            else:
                rate = 0.0
            if rate > 0.0:
                return (angle / 360.0) * 1.0e6 / rate
        return 0.0


def _process_header(state: _State, header: dict, md: FileMetadata) -> None:
    cam = get_msg(header, 1)
    if cam is not None:
        c = parse_camera(cam)
        state.camera = c
        if c["imu_orientation"]:
            state.imu_orientation = c["imu_orientation"]
        state.imu_rotation = parse_quat(c["imu_rotation"])
        state.quats_rotation = parse_quat(c["quats_rotation"])
        if c["brand"]:
            state.vendor = c["brand"]
        state.model = c["model"] or None

    clip = get_msg(header, 2)
    if clip is not None:
        cl = parse_clip(clip)
        state.clip = cl
        state.frame_readout_time_us = cl["frame_readout_time_us"] or 0.0
        state.readout_direction = _READOUT_DIRECTIONS.get(
            cl["readout_direction"], ReadoutDirection.TopToBottom)
        state.frame_readout_time = pack_readout_time_ms(
            state.frame_readout_time_us, state.readout_direction)
        if cl["record_fps"] > 0:
            md.frame_rate = cl["record_fps"]


def _build_lens_profile(state: _State) -> dict | None:
    clip, cam = state.clip, state.camera
    if clip is None or cam is None:
        return None
    if clip["frame_w"] == 0 or clip["frame_h"] == 0:
        return None
    dist_model = state.distortion_model_name or "opencv_fisheye"
    is_vertical = abs(clip["rotation"]) in (90, 270)
    out_w = clip["frame_h"] if is_vertical else clip["frame_w"]
    out_h = clip["frame_w"] if is_vertical else clip["frame_h"]
    cx, cy = clip["frame_w"] / 2.0, clip["frame_h"] / 2.0

    lens_model_full = (f"{cam['lens_brand']} {cam['lens_model']}"
                       if cam["lens_brand"] and cam["lens_model"]
                       else cam["lens_model"])
    frt = state.frame_readout_time_us / 1000.0 if state.frame_readout_time_us > 0.0 else 0.0
    profile = {
        "calibrated_by": "Gyroflow Protobuf",
        "camera_brand": cam["brand"],
        "camera_model": cam["model"],
        "lens_model": lens_model_full,
        "calib_dimension": {"w": clip["frame_w"], "h": clip["frame_h"]},
        "orig_dimension": {"w": clip["frame_w"], "h": clip["frame_h"]},
        "output_dimension": {"w": out_w, "h": out_h},
        "frame_readout_time": frt,
        "official": True,
        "asymmetrical": False,
        "fisheye_params": {
            "camera_matrix": [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]],
            "distortion_coeffs": [],
        },
        "distortion_model": dist_model,
        "fps": clip["record_fps"] if clip["record_fps"] > 0.0 else clip["sensor_fps"],
        "input_horizontal_stretch": clip["pixel_aspect"] if clip["pixel_aspect"] > 0.0 else 1.0,
        "input_vertical_stretch": 1.0,
        "sync_settings": {
            "initial_offset": 0,
            "initial_offset_inv": False,
            "search_size": 0.3,
            "max_sync_points": 5,
            "every_nth_frame": 1,
            "time_per_syncpoint": 0.5,
            "do_autosync": False,
        },
        "calibrator_version": "---",
    }
    if cam["crop_factor"] is not None:
        profile["crop_factor"] = cam["crop_factor"]
    return profile


_DISTORTION_MAP = {
    "no_distortion": "opencv_fisheye",
    "opencv_fisheye": "opencv_fisheye",
    "opencv_standard": "opencv_standard",
    "poly3": "poly3",
    "poly5": "poly5",
    "ptlens": "ptlens",
}


def _mesh_to_sony_dict(mw: dict) -> dict | None:
    """binary.rs build_mesh_correction_json: the sony.rs consumer shape."""
    gw, gh = mw["grid_w"], mw["grid_h"]
    warped = mw["warped_xy"]
    if gw < 2 or gh < 2 or len(warped) != 2 * gw * gh:
        return None
    step_x = mw["region_w"] / (gw - 1.0)
    step_y = mw["region_h"] / (gh - 1.0)
    mesh_arr, raw_mesh = [], []
    for j in range(gh):
        for i in range(gw):
            k = j * gw + i
            wx, wy = warped[2 * k], warped[2 * k + 1]
            ax, ay = step_x * i, step_y * j
            mesh_arr.append((wx, wy))
            raw_mesh.append((wx - ax, wy - ay))
    return {
        "size": [mw["region_w"], mw["region_h"]],
        "divisions": [gw, gh],
        "mesh": mesh_arr,
        "raw_mesh": raw_mesh,
        "divisions_2d": [1, 1],
    }


def _process_frame(
    state: _State,
    frame: dict,
    md: FileMetadata,
    lens_state: LensParams,
    is_temp: ISTemp | None,
    mesh_acc: list,
    mesh_cache: dict,
    ts_ms: float | None,
) -> None:
    cam, clip = state.camera, state.clip
    sensor_w = cam["sensor_w"] if cam else 0
    sensor_h = cam["sensor_h"] if cam else 0
    raw_pp_x = cam["pp_x"] if cam else 0
    raw_pp_y = cam["pp_y"] if cam else 0
    pp_x = raw_pp_x if raw_pp_x > 0 else raw_pp_y
    pp_y = raw_pp_y if raw_pp_y > 0 else raw_pp_x
    frame_w = clip["frame_w"] if clip else 0
    frame_h = clip["frame_h"] if clip else 0

    crop_origin = (frame["crop_x"] or 0.0, frame["crop_y"] or 0.0)
    crop_size = (frame["crop_w"] if frame["crop_w"] is not None else float(sensor_w),
                 frame["crop_h"] if frame["crop_h"] is not None else float(sensor_h))

    # mod.rs:194-243-equivalent lens_params accumulation
    if pp_x > 0 and pp_y > 0:
        lens_state.pixel_pitch = (pp_x, pp_y)
    if sensor_w > 0 and sensor_h > 0:
        lens_state.sensor_size_px = (sensor_w, sensor_h)
        lens_state.capture_area_origin = crop_origin
        lens_state.capture_area_size = crop_size

    # ---- per-frame exposure / lens / zoom ----
    exposure_us = state.resolve_exposure_us(frame)

    lens = frame["lens"][0] if frame["lens"] else None
    if lens is not None:
        if lens["focal_length_mm"] is not None:
            lens_state.focal_length = lens["focal_length_mm"]
        if lens["intrinsic"] and len(lens["intrinsic"]) >= 9:
            f_x = lens["intrinsic"][0]
            f_y = lens["intrinsic"][4]
            if f_x > 0.0 and f_y > 0.0:
                lens_state.pixel_focal_length = f_x
        elif lens["focal_length_mm"] is not None and pp_x > 0 and pp_y > 0 \
                and crop_size[0] > 0 and crop_size[1] > 0 \
                and frame_w > 0 and frame_h > 0:
            sensor_w_mm = (pp_x * crop_size[0]) / 1.0e6
            sensor_h_mm = (pp_y * crop_size[1]) / 1.0e6
            if sensor_w_mm > 0.0 and sensor_h_mm > 0.0:
                lens_state.pixel_focal_length = \
                    (lens["focal_length_mm"] / sensor_w_mm) * frame_w
        if lens["distortion"] is not None:
            name, _coeffs = lens["distortion"]
            model_name = _DISTORTION_MAP.get(name)
            if model_name and state.distortion_model_name is None:
                state.distortion_model_name = model_name

    if lens_state.pixel_pitch is not None \
            and lens_state.capture_area_size is not None \
            and (lens_state.pixel_focal_length is not None
                 or lens_state.focal_length is not None):
        ts_us = int(round((ts_ms if ts_ms is not None
                           else frame["start_ts"] / 1000.0) * 1000.0))
        md.lens_params[ts_us] = LensParams(**vars(lens_state))

    zoom = frame["digital_zoom"]
    if zoom is not None and zoom > 1.000001:
        md.digital_zoom = float(zoom)

    # ---- per-frame readout (authoritative doubles; binary.rs:427-451) ----
    per_frame_readout_us = max(frame["end_ts"] - frame["start_ts"], 0.0)
    if state.first_start_ts_us == frame["start_ts"] and per_frame_readout_us > 0.0 \
            and state.frame_readout_time_us <= 0.0:
        state.frame_readout_time_us = per_frame_readout_us
        state.frame_readout_time = pack_readout_time_ms(
            per_frame_readout_us, state.readout_direction)
        md.frame_readout_time = state.frame_readout_time
        md.frame_readout_direction = state.readout_direction

    # ---- raw IMU: camera-clock timestamps, physical units ----
    apply_rotation = state.imu_rotation is not None
    orient_bytes = (state.imu_orientation.encode()
                    if apply_rotation and len(state.imu_orientation) == 3
                    else b"XYZ")
    for imu in frame["imu"]:
        t_abs_us = imu["ts"] if imu["ts"] is not None else frame["start_ts"]
        t_ms = t_abs_us / 1.0e3
        gx, gy, gz = imu["gyro"]
        ax, ay, az = imu["acc"]
        if apply_rotation:
            g_o = orient_vec3((gx, gy, gz), orient_bytes)
            a_o = orient_vec3((ax, ay, az), orient_bytes)
            gx, gy, gz = rotate_vec3_by_quat(g_o, state.imu_rotation)
            ax, ay, az = rotate_vec3_by_quat(a_o, state.imu_rotation)
        mag = None
        if imu["mag"][0] is not None:
            mag = np.array(imu["mag"], dtype=np.float64)
            if apply_rotation:
                mag = np.array(rotate_vec3_by_quat(
                    orient_vec3((mag[0], mag[1], mag[2]), orient_bytes),
                    state.imu_rotation))
        md.raw_imu.append(TimeIMU(
            timestamp_ms=t_ms,
            gyro=np.array([gx, gy, gz], dtype=np.float64),
            accl=np.array([ax, ay, az], dtype=np.float64),
        ))

    # ---- quaternions: camera clock, conjugation applied ----
    for q in frame["quaternions"]:
        if q["quat"] is None:
            continue
        quat = q["quat"]
        if state.quats_rotation is not None:
            quat = conjugate_quat_by(quat, state.quats_rotation)
        t_abs_us = q["ts"] if q["ts"] is not None else frame["start_ts"]
        md.quaternions[int(round(t_abs_us))] = _to_quat64(quat)

    # ---- EIS: mesh warp → mesh_correction; quaternions → image_orientations
    eis_quat_i16: list[tuple[float, float, float, float]] = []
    for eis in frame["eis"]:
        if "mesh_warp" in eis:
            sony_mesh = _mesh_to_sony_dict(eis["mesh_warp"])
            if sony_mesh is not None:
                from pygyroflow.gyro_source.sony import get_mesh_correction_from

                mesh = get_mesh_correction_from(
                    sony_mesh, None, crop_origin, crop_size, mesh_cache)
                if mesh is not None:
                    mesh_acc.append(mesh)
        if "quaternion" in eis:
            w, x, y, z = eis["quaternion"]
            s = 32767.0
            eis_quat_i16.append((
                max(-1.0, min(1.0, w)) * s,
                max(-1.0, min(1.0, x)) * s,
                max(-1.0, min(1.0, y)) * s,
                max(-1.0, min(1.0, z)) * s,
            ))
    if eis_quat_i16:
        # mod.rs:314-326 rearrangement: payload (w,x,y,z) i16 → stored quat
        # (x, y, z, w) / scale (the upstream Vector4::new quirk). The zip
        # against quaternion timestamps happens ONCE at finalize, matching
        # mod.rs:337-343's accumulate-then-zip.
        state.eis_iori.extend(
            [v / 32767.0 for v in (q[1], q[2], q[3], q[0])]
            for q in eis_quat_i16)

    # ---- IBIS / OIS → Sony deep pass (only when the vendor is Sony) ----
    if is_temp is not None:
        interval_us = state.nominal_frame_interval_us()
        ibis_shifts, ibis_angles, ois_shifts = [], [], []
        combined = sorted(
            ((s["ts"] if s["ts"] is not None else frame["start_ts"], s)
             for s in frame["ibis"]), key=lambda p: p[0])
        for t_abs_us, s in combined:
            dt = t_abs_us - frame["start_ts"]
            if dt < 0.0 or dt >= interval_us:
                continue
            t_rel = int(round(dt))
            # SIGN FLIP: proto reports image-content displacement; the
            # Sony-shaped pipeline carries sensor displacement (binary.rs).
            ibis_shifts.append((t_rel, int(round(-s["x"])),
                                int(round(-s["y"])), 0))
            ibis_angles.append((t_rel, 0, 0, int(round(s["roll"] * 1000.0))))
        for t_abs_us, s in sorted(
                ((s["ts"] if s["ts"] is not None else frame["start_ts"], s)
                 for s in frame["ois"]), key=lambda p: p[0]):
            dt = t_abs_us - frame["start_ts"]
            if dt < 0.0 or dt >= interval_us:
                continue
            ois_shifts.append((int(round(dt)), int(round(s["x"])),
                               int(round(s["y"])), 0))

        if ibis_shifts or ois_shifts:
            tags = _synth_sony_tags(
                ibis_shifts, ibis_angles, ois_shifts,
                (pp_x, pp_y), (sensor_w, sensor_h), crop_origin, crop_size,
                frame["start_ts"], exposure_us,
                int(clip["imu_sample_rate"]) if clip else 0)
            from pygyroflow.gyro_source import sony as _sony
            _sony.stab_collect(is_temp, tags, 1.0e6 / interval_us)


def _to_quat64(q):
    from pygyroflow.types.quaternion import Quat64
    return Quat64.from_quaternion(np.array(q, dtype=np.float64))


def parse_gyroflow_proto(
    messages: list[bytes],
    fps: float,
    video_size: tuple[int, int] = (0, 0),
    sample_ts_ms: list[float] | None = None,
) -> FileMetadata:
    """Parse Main protobuf messages (one per metadata sample) into
    FileMetadata. *messages* are the per-sample payloads in stream order;
    *sample_ts_ms* optionally carries each sample's video-timeline timestamp
    (used for lens_params keys, mirroring mod.rs's info.timestamp_ms).

    The Sony-shaped IBIS/OIS deep pass runs only when the declared camera
    brand is "Sony" (mod.rs gates stab_calc_splines on camera_type()).
    """
    metadata = FileMetadata(detected_source="Gyroflow")
    state = _State()
    lens_state = LensParams()
    is_temp: ISTemp | None = None
    mesh_acc: list = []
    mesh_cache: dict = {}

    for idx, payload in enumerate(messages):
        if _MAGIC not in payload:
            log.warning("Gyroflow proto sample without magic; skipping")
            continue
        main = parse_main(payload)
        if main is None:
            log.warning("Undecodable Gyroflow proto sample; skipping")
            continue
        header = get_msg(main, 3)
        if header is not None:
            _process_header(state, header, metadata)
        frame_msg = get_msg(main, 4)
        if frame_msg is not None:
            frame = parse_frame(frame_msg)
            if state.first_start_ts_us is None:
                state.first_start_ts_us = frame["start_ts"]
            if state.vendor == "Sony" and is_temp is None:
                is_temp = ISTemp()
            ts_ms = sample_ts_ms[idx] if sample_ts_ms and idx < len(sample_ts_ms) \
                else None
            _process_frame(state, frame, metadata, lens_state, is_temp,
                           mesh_acc, mesh_cache, ts_ms)

    # finalize
    if state.camera:
        metadata.imu_orientation = state.imu_orientation
    if state.frame_readout_time is not None:
        metadata.frame_readout_time = state.frame_readout_time
        metadata.frame_readout_direction = state.readout_direction
    metadata.has_accurate_timestamps = True
    if state.vendor != "Gyroflow" or state.model:
        metadata.detected_source = (
            f"{state.vendor} {state.model}" if state.model else state.vendor)

    # mod.rs:337-343: zip image orientations onto quaternion timestamps once,
    # after everything is accumulated
    if state.eis_iori and metadata.quaternions:
        metadata.image_orientations = {
            ts: _to_quat64(q)
            for ts, q in zip(metadata.quaternions.keys(), state.eis_iori)
        }

    if state.distortion_model_name is not None and state.camera \
            and state.camera["brand"] and state.clip:
        profile = _build_lens_profile(state)
        if profile is not None and metadata.lens_profile is None:
            metadata.lens_profile = profile

    if is_temp is not None:
        stab = stab_calc_splines(metadata, is_temp)
        if stab is not None:
            metadata.camera_stab_data = stab
    if mesh_acc:
        metadata.mesh_correction = mesh_acc

    return metadata
