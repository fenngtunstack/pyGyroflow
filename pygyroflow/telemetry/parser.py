"""Telemetry parser -- extracts gyro data from video files.

Tries the PyO3 native bridge (telemetry_parser_bridge built with maturin)
first, then falls back to a pure-Python parser for GoPro GPMF and DJI
protobuf telemetry formats.
"""

from __future__ import annotations

import logging
import os
import struct

from pygyroflow.gyro_source.file_metadata import FileMetadata
from pygyroflow.types.enums import ReadoutDirection
from pygyroflow.types.errors import TelemetryParseError

log = logging.getLogger(__name__)


def parse_telemetry_file(
    path: str,
    sample_index: int | None = None,
    video_size: tuple[int, int] = (0, 0),
    fps: float = 0.0,
) -> FileMetadata:
    """Parse telemetry from a video file.

    First tries the PyO3 bridge (telemetry_parser_bridge), falls back to a
    basic GPMF parser for GoPro files if the bridge is not available.

    Args:
        path: Path to the video file.
        sample_index: Optional sample index for multi-stream files.
        video_size: (width, height) of the video.
        fps: Video frame rate.

    Returns:
        FileMetadata with parsed gyro data.

    Raises:
        TelemetryParseError: If parsing fails.
    """
    if not os.path.isfile(path):
        raise TelemetryParseError(f"File not found: {path}")

    # Try native bridge first
    try:
        from pygyroflow.telemetry._native import parse_telemetry

        result = parse_telemetry(path, sample_index)
        if result is not None:
            return result
    except ImportError:
        log.info("telemetry_parser_bridge not available, using fallback parser")
    except Exception as exc:
        log.warning("Native telemetry parser failed: %s", exc)

    # Fallback: try to parse based on file extension
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".mov"):
        return _fallback_parse(path, sample_index, video_size, fps)

    raise TelemetryParseError(f"Unsupported file format: {ext}")


def _fallback_parse(
    path: str,
    sample_index: int | None = None,
    video_size: tuple[int, int] = (0, 0),
    fps: float = 0.0,
) -> FileMetadata:
    """Fallback parser when telemetry_parser_bridge is not available.

    Detects the camera brand from the file content and dispatches to the
    appropriate parser (GoPro GPMF or DJI protobuf).
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        log.warning("Cannot read file %s: %s", path, exc)
        return FileMetadata(detected_source="Unknown")

    # Detect GoPro: look for 'gpmd' FourCC tag in MP4 box structure
    if _detect_gopro(data):
        return _parse_gopro(data, fps)

    # Detect DJI: look for 'djmd' tag and "CAM meta" handler
    if _detect_dji(data):
        return _parse_dji(data, fps)

    log.warning("Unrecognized telemetry format in %s", path)
    return FileMetadata(detected_source="Unknown")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_gopro(data: bytes) -> bool:
    """Detect GoPro GPMF stream by looking for 'gpmd' codec tag."""
    return data.find(b"gpmd") >= 0


def _detect_dji(data: bytes) -> bool:
    """Detect DJI metadata stream by looking for 'djmd' codec tag."""
    return data.find(b"djmd") >= 0 or (data.find(b"dvtm") >= 0 and data.find(b"DJI") >= 0)


# ---------------------------------------------------------------------------
# MP4 box parser (minimal, for data track extraction)
# ---------------------------------------------------------------------------

def _read_mp4_box(data: bytes, pos: int, end: int):
    """Read an MP4 box header.

    Returns (fourcc, box_start, header_size, box_size) or None.
    """
    if pos + 8 > end:
        return None
    size = struct.unpack(">I", data[pos:pos + 4])[0]
    fourcc = data[pos + 4:pos + 8].decode("ascii", errors="replace")
    header_size = 8
    if size == 1:  # 64-bit extended size
        if pos + 16 > end:
            return None
        size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
        header_size = 16
    elif size == 0:  # Box extends to end of file
        size = end - pos
    return fourcc, pos, header_size, size


def _mp4_find_data_track_samples(
    data: bytes, target_codec: str
) -> list[tuple[int, int]]:
    """Find data track samples in an MP4 file by codec tag.

    Parses the moov/trak/mdia/minf/stbl box hierarchy to locate the track
    with the specified codec tag (e.g. 'gpmd', 'djmd'), then reads stco/co64
    for chunk offsets and stsz for sample sizes.

    Returns list of (offset, size) pairs for each sample.
    """
    end = len(data)

    # Find moov box
    moov_start = moov_end = None
    pos = 0
    while pos < end:
        box = _read_mp4_box(data, pos, end)
        if box is None:
            break
        fourcc, _, hs, bs = box
        if fourcc == "moov":
            moov_start = pos + hs
            moov_end = pos + bs
            break
        pos += bs
    if moov_start is None:
        return []

    # Walk trak boxes inside moov
    pos = moov_start
    while pos < moov_end:
        box = _read_mp4_box(data, pos, moov_end)
        if box is None:
            break
        fourcc, _, hs, bs = box

        if fourcc == "trak":
            result = _mp4_parse_trak(data, pos + hs, pos + bs, target_codec)
            if result is not None:
                return result

        pos += bs
    return []


def _mp4_parse_trak(
    data: bytes, trak_start: int, trak_end: int, target_codec: str
) -> list[tuple[int, int]] | None:
    """Parse a single trak box looking for *target_codec*.

    Returns list of (offset, size) sample pairs, or None if this track
    doesn't match.
    """
    hdlr_type = None
    stsd_codec = None
    chunk_offsets = []
    sample_sizes = []

    pos = trak_start
    while pos < trak_end:
        box = _read_mp4_box(data, pos, trak_end)
        if box is None:
            break
        fourcc, _, hs, bs = box

        if fourcc == "mdia":
            mdia_pos = pos + hs
            mdia_end = pos + bs
            while mdia_pos < mdia_end:
                mdia_box = _read_mp4_box(data, mdia_pos, mdia_end)
                if mdia_box is None:
                    break
                m_fourcc, _, m_hs, m_bs = mdia_box

                if m_fourcc == "hdlr":
                    # hdlr: version(4) + flags(4) + handler_type(4)
                    off = mdia_pos + m_hs + 8
                    if off + 4 <= mdia_pos + m_bs:
                        hdlr_type = data[off:off + 4].decode("ascii", errors="replace")

                elif m_fourcc == "minf":
                    _mp4_parse_stbl(
                        data, mdia_pos + m_hs, mdia_pos + m_bs,
                        target_codec,
                    )
                    # We need to capture the results, so call inline
                    minf_pos = mdia_pos + m_hs
                    minf_end = mdia_pos + m_bs
                    while minf_pos < minf_end:
                        minf_box = _read_mp4_box(data, minf_pos, minf_end)
                        if minf_box is None:
                            break
                        mi_fourcc, _, mi_hs, mi_bs = minf_box

                        if mi_fourcc == "stbl":
                            _r = _mp4_parse_stbl(
                                data, minf_pos + mi_hs, minf_pos + mi_bs,
                                target_codec,
                            )
                            if _r is not None:
                                return _r

                        minf_pos += mi_bs

                mdia_pos += m_bs

        pos += bs
    return None


def _mp4_parse_stbl(
    data: bytes, stbl_start: int, stbl_end: int, target_codec: str
) -> list[tuple[int, int]] | None:
    """Parse stbl box to find sample offsets and sizes for *target_codec*."""
    stsd_codec = None
    chunk_offsets: list[int] = []
    sample_sizes: list[int] = []
    stsc_entries: list[tuple[int, int, int]] = []  # (first_chunk, samples_per_chunk, sample_desc_idx)

    pos = stbl_start
    while pos < stbl_end:
        box = _read_mp4_box(data, pos, stbl_end)
        if box is None:
            break
        fourcc, _, hs, bs = box
        box_payload = pos + hs
        box_end = pos + bs

        if fourcc == "stsd":
            # version(4) + entry_count(4) + entries
            if box_payload + 8 <= box_end:
                entry_count = struct.unpack(">I", data[box_payload + 4:box_payload + 8])[0]
                entry_pos = box_payload + 8
                for _ in range(entry_count):
                    if entry_pos + 8 > box_end:
                        break
                    entry_size = struct.unpack(">I", data[entry_pos:entry_pos + 4])[0]
                    codec = data[entry_pos + 4:entry_pos + 8].decode("ascii", errors="replace")
                    if codec == target_codec:
                        stsd_codec = codec
                    entry_pos += max(entry_size, 8)

        elif fourcc in ("stco", "co64"):
            if box_payload + 8 <= box_end:
                entry_count = struct.unpack(">I", data[box_payload + 4:box_payload + 8])[0]
                for i in range(entry_count):
                    if fourcc == "stco":
                        off_pos = box_payload + 8 + i * 4
                        if off_pos + 4 <= box_end:
                            chunk_offsets.append(struct.unpack(">I", data[off_pos:off_pos + 4])[0])
                    else:  # co64
                        off_pos = box_payload + 8 + i * 8
                        if off_pos + 8 <= box_end:
                            chunk_offsets.append(struct.unpack(">Q", data[off_pos:off_pos + 8])[0])

        elif fourcc == "stsz":
            if box_payload + 12 <= box_end:
                default_size = struct.unpack(">I", data[box_payload + 4:box_payload + 8])[0]
                count = struct.unpack(">I", data[box_payload + 8:box_payload + 12])[0]
                if default_size == 0 and count > 0:
                    for i in range(count):
                        sz_pos = box_payload + 12 + i * 4
                        if sz_pos + 4 <= box_end:
                            sample_sizes.append(struct.unpack(">I", data[sz_pos:sz_pos + 4])[0])
                elif default_size > 0:
                    sample_sizes = [default_size] * count

        elif fourcc == "stsc":
            if box_payload + 8 <= box_end:
                entry_count = struct.unpack(">I", data[box_payload + 4:box_payload + 8])[0]
                for i in range(entry_count):
                    sc_pos = box_payload + 8 + i * 12
                    if sc_pos + 12 <= box_end:
                        first_chunk = struct.unpack(">I", data[sc_pos:sc_pos + 4])[0]
                        spc = struct.unpack(">I", data[sc_pos + 4:sc_pos + 8])[0]
                        sdi = struct.unpack(">I", data[sc_pos + 8:sc_pos + 12])[0]
                        stsc_entries.append((first_chunk, spc, sdi))

        pos += bs

    # Check if this is the right codec
    if stsd_codec != target_codec:
        return None

    # Build sample (offset, size) list using stsc to map chunks to samples
    if not chunk_offsets:
        return []

    # If stsc is trivial (1 sample per chunk), use direct mapping
    if not stsc_entries or (
        len(stsc_entries) == 1
        and stsc_entries[0][0] == 1
        and stsc_entries[0][1] == 1
    ):
        # One sample per chunk
        result = []
        size_iter = iter(sample_sizes)
        for off in chunk_offsets:
            try:
                sz = next(size_iter)
            except StopIteration:
                sz = 0
            result.append((off, sz))
        return result

    # General stsc handling: expand chunk -> sample mapping
    samples = []
    stsc_sorted = sorted(stsc_entries, key=lambda x: x[0])
    for idx, chunk_off in enumerate(chunk_offsets):
        chunk_num = idx + 1  # 1-based
        # Find which stsc entry applies
        spc = 1
        for si in range(len(stsc_sorted) - 1, -1, -1):
            if chunk_num >= stsc_sorted[si][0]:
                spc = stsc_sorted[si][1]
                break
        for _ in range(spc):
            if not sample_sizes:
                break
            sz = sample_sizes.pop(0)
            samples.append((chunk_off, sz))
            chunk_off += sz

    return samples if samples else None


# ---------------------------------------------------------------------------
# GPMF KLV parser (GoPro)
# ---------------------------------------------------------------------------

_GPMF_HEADER_SIZE = 8  # FourCC(4) + Type(1) + StructSize(1) + Repeat(2)


def _gpmf_read_klv(data: bytes, pos: int, end: int):
    """Read one GPMF KLV element at *pos*.

    GPMF format: FourCC(4) + Type(1) + StructSize(1) + Repeat(2 BE).
    Total payload = StructSize * Repeat, padded to 4-byte alignment.

    Returns (fourcc_str, type_byte, struct_size, repeat, payload_start,
             payload_bytes, next_pos) or None if not enough data.
    """
    if pos + _GPMF_HEADER_SIZE > end:
        return None
    fourcc = data[pos:pos + 4].decode("ascii", errors="replace")
    type_byte = data[pos + 4]
    struct_size = data[pos + 5]
    repeat = struct.unpack(">H", data[pos + 6:pos + 8])[0]
    payload_bytes = struct_size * repeat
    payload_start = pos + _GPMF_HEADER_SIZE
    next_pos = (payload_start + payload_bytes + 3) & ~3
    return fourcc, type_byte, struct_size, repeat, payload_start, payload_bytes, next_pos


def _parse_gopro(data: bytes, fps: float) -> FileMetadata:
    """Parse GoPro GPMF telemetry from raw MP4 file data.

    Extracts gyroscope (GYRO) and accelerometer (ACCL) samples, converts
    them to physical units using the SCAL factor, and builds TimeIMU records.

    The ORIN tag gives axis ordering (e.g. 'ZXY'), SIUN gives units
    (e.g. 'rad/s', 'm/s2'). SCAL is a divisor: physical = raw / SCAL.
    """
    import numpy as np

    from pygyroflow.types.time_types import TimeIMU

    metadata = FileMetadata(detected_source="GoPro")

    # Find gpmd track samples via MP4 box parsing
    samples = _mp4_find_data_track_samples(data, "gpmd")
    if not samples:
        log.warning("No gpmd track found in GoPro file")
        return metadata

    raw_imu: list[TimeIMU] = []

    for offset, size in samples:
        if offset + size > len(data):
            continue
        gpmf_chunk = data[offset:offset + size]
        _parse_gpmf_chunk(gpmf_chunk, fps, raw_imu)

    metadata.raw_imu = raw_imu
    # GoPro uses ZXY orientation (ORIN tag in GPMF)
    metadata.imu_orientation = "ZXY"
    metadata.frame_rate = fps if fps > 0 else None
    metadata.has_accurate_timestamps = True

    return metadata


def _parse_gpmf_chunk(
    gpmf_data: bytes,
    fps: float,
    raw_imu: list,
) -> None:
    """Parse one gpmd track sample (contains one or more DEVC blocks).

    Extracts ACCL and GYRO streams from each DEVC/STRM, computes
    timestamps from STMP and sample counts, and appends TimeIMU entries
    to *raw_imu*.
    """
    import numpy as np

    from pygyroflow.types.time_types import TimeIMU

    RAD_TO_DEG = 180.0 / 3.141592653589793

    pos = 0
    end = len(gpmf_data)

    while pos < end - _GPMF_HEADER_SIZE:
        klv = _gpmf_read_klv(gpmf_data, pos, end)
        if klv is None:
            break
        fourcc, type_byte, _, _, payload_start, payload_bytes, next_pos = klv
        if fourcc != "DEVC" or type_byte != 0:
            pos = next_pos
            continue

        devc_end = next_pos

        # Per-DEVC accumulators
        accl_samples = None
        accl_scale = 1.0
        accl_stamp_us = 0
        accl_count = 0
        accl_struct_size = 6

        gyro_samples = None
        gyro_scale = 1.0
        gyro_stamp_us = 0
        gyro_count = 0
        gyro_struct_size = 6

        # Parse DEVC children
        dpos = payload_start
        while dpos < devc_end - _GPMF_HEADER_SIZE:
            child = _gpmf_read_klv(gpmf_data, dpos, devc_end)
            if child is None:
                break
            c_fourcc, c_type, _, _, _, _, c_np = child

            if c_fourcc == "STRM" and c_type == 0:
                # Parse STRM contents for ACCL/GYRO
                _parse_gpmf_strm(
                    gpmf_data, c_np,  # c_ps is at child[4], c_np is end
                    child[4],  # payload_start of STRM
                    RAD_TO_DEG,
                    # Mutable accumulators via closure
                    accl_ref := [accl_samples, accl_scale, accl_stamp_us, accl_count, accl_struct_size],
                    gyro_ref := [gyro_samples, gyro_scale, gyro_stamp_us, gyro_count, gyro_struct_size],
                )
                # Read back updated values
                accl_samples, accl_scale, accl_stamp_us, accl_count, accl_struct_size = accl_ref
                gyro_samples, gyro_scale, gyro_stamp_us, gyro_count, gyro_struct_size = gyro_ref

            dpos = c_np

        # Build TimeIMU entries
        if accl_count == 0 and gyro_count == 0:
            pos = next_pos
            continue

        n_samples = max(accl_count, gyro_count)

        # GoPro IMU sample rate is typically ~200Hz.
        # Each DEVC block spans approximately 1 second with n_samples samples.
        # So sample_interval = 1_000_000 / n_samples microseconds.
        if n_samples > 1:
            sample_interval_us = 1_000_000.0 / n_samples
        else:
            sample_interval_us = 5_000.0  # 5ms default

        base_stamp_us = accl_stamp_us if accl_stamp_us else gyro_stamp_us

        for i in range(n_samples):
            ts_ms = (base_stamp_us + i * sample_interval_us) / 1000.0

            gyro = None
            if gyro_samples is not None and i < gyro_count:
                n_axes = gyro_struct_size // 2
                offset = i * n_axes
                raw_vals = gyro_samples[offset:offset + n_axes]
                gyro = np.array(
                    [v / gyro_scale * RAD_TO_DEG for v in raw_vals],
                    dtype=np.float64,
                )

            accl = None
            if accl_samples is not None and i < accl_count:
                n_axes = accl_struct_size // 2
                offset = i * n_axes
                raw_vals = accl_samples[offset:offset + n_axes]
                accl = np.array(
                    [v / accl_scale for v in raw_vals],
                    dtype=np.float64,
                )

            raw_imu.append(TimeIMU(timestamp_ms=ts_ms, gyro=gyro, accl=accl))

        pos = next_pos


def _parse_gpmf_strm(
    gpmf_data: bytes,
    strm_end: int,
    strm_ps: int,
    rad_to_deg: float,
    accl_ref: list,
    gyro_ref: list,
) -> None:
    """Parse a STRM block, updating accl_ref / gyro_ref in place."""
    spos = strm_ps
    stream_name = ""
    stream_stamp = 0
    stream_orin = "XYZ"
    stream_scal = 1.0

    # First pass: collect metadata (STNM, STMP, ORIN, SIUN, SCAL)
    while spos < strm_end - _GPMF_HEADER_SIZE:
        tag_klv = _gpmf_read_klv(gpmf_data, spos, strm_end)
        if tag_klv is None:
            break
        t_fourcc, t_type, t_ss, t_rep, t_ps, t_pb, t_np = tag_klv

        if t_fourcc == "STNM" and t_type == ord("c"):
            stream_name = gpmf_data[t_ps:t_ps + t_pb].decode("ascii", errors="replace").rstrip("\x00")
        elif t_fourcc == "STMP" and t_type == ord("J") and t_pb >= 8:
            stream_stamp = struct.unpack(">q", gpmf_data[t_ps:t_ps + 8])[0]
        elif t_fourcc == "ORIN" and t_type == ord("c"):
            stream_orin = gpmf_data[t_ps:t_ps + t_pb].decode("ascii", errors="replace").rstrip("\x00")
        elif t_fourcc == "SCAL" and t_type == ord("s") and t_pb >= 2:
            stream_scal = float(struct.unpack(">h", gpmf_data[t_ps:t_ps + 2])[0])

        spos = t_np

    # Second pass: extract ACCL/GYRO using the collected metadata
    spos = strm_ps
    while spos < strm_end - _GPMF_HEADER_SIZE:
        tag_klv = _gpmf_read_klv(gpmf_data, spos, strm_end)
        if tag_klv is None:
            break
        t_fourcc, t_type, t_ss, t_rep, t_ps, t_pb, t_np = tag_klv

        if t_fourcc == "ACCL" and t_type == ord("s"):
            n_axes = t_ss // 2
            accl_ref[0] = struct.unpack(f">{n_axes * t_rep}h", gpmf_data[t_ps:t_ps + t_pb])
            accl_ref[1] = stream_scal if stream_scal != 1.0 else 1.0
            accl_ref[2] = stream_stamp
            accl_ref[3] = t_rep
            accl_ref[4] = t_ss

        elif t_fourcc == "GYRO" and t_type == ord("s"):
            n_axes = t_ss // 2
            gyro_ref[0] = struct.unpack(f">{n_axes * t_rep}h", gpmf_data[t_ps:t_ps + t_pb])
            gyro_ref[1] = stream_scal if stream_scal != 1.0 else 1.0
            gyro_ref[2] = stream_stamp
            gyro_ref[3] = t_rep
            gyro_ref[4] = t_ss

        spos = t_np


# ---------------------------------------------------------------------------
# DJI protobuf parser
# ---------------------------------------------------------------------------

def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Read a protobuf-style varint, return (value, new_pos)."""
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


def _parse_dji(data: bytes, fps: float) -> FileMetadata:
    """Parse DJI telemetry from raw MP4 file data.

    DJI stores telemetry in a 'djmd' data track as Protocol Buffer messages.
    The protobuf schema (ProductMeta) contains:
      - clip_meta (field 1): camera model, lens info
      - stream_meta (field 2): video stream info
      - frame_meta (field 3): per-frame IMU quaternions

    We extract Quaternion arrays from frame_meta.imu_frame_meta and store
    them in FileMetadata.quaternions. The quaternion rotation follows the
    same convention as the Rust telemetry-parser: multiply by (0.5,-0.5,-0.5,0.5)
    then (0,0,1,0).
    """
    import numpy as np

    from pygyroflow.types.quaternion import Quat64

    metadata = FileMetadata(detected_source="DJI")

    # Find djmd track samples via MP4 box parsing
    samples = _mp4_find_data_track_samples(data, "djmd")
    if not samples:
        log.warning("No djmd track found in DJI file")
        return metadata

    first_timestamp = None
    quats: dict[int, Quat64] = {}
    frame_rate = fps if fps > 0 else 29.97

    for pkt_idx, (offset, size) in enumerate(samples):
        if offset + size > len(data) or size == 0:
            continue

        buf = data[offset:offset + size]

        # ProductMeta { clip_meta=1, stream_meta=2, frame_meta=3 }
        frame_meta = _pb_extract_field(buf, 3)
        if frame_meta is None:
            continue

        # FrameMeta { frame_meta_header=1, camera_frame_meta=2, imu_frame_meta=3 }
        imu_frame = _pb_extract_field(frame_meta, 3)
        if imu_frame is None:
            continue

        # FrameMetaOfIMU { IMU_dev_header=1, attitude_after_fusion=2, vsync_pos=3, single_attitude=4 }
        # Try field 2 (IMU_attitude_after_fusion) for wm169-like devices
        attitude_data = _pb_extract_field(imu_frame, 2)

        if attitude_data is not None:
            # Could be DeviceMultiAttitude { current_frame=1, prev_frame=2, next_frame=3 }
            # Try extracting field 1 (current_frame) which is a DeviceAttitude
            inner = _pb_extract_field(attitude_data, 1)
            if inner is not None:
                # Check if inner contains quaternions (field 3 repeated)
                test_quats = _pb_extract_repeated_quaternions(inner, field_num=3)
                if test_quats:
                    attitude_data = inner

        if attitude_data is None:
            # Try field 4 (IMU_single_attitude_after_fusion) for eagle4_wa530-like
            attitude_data = _pb_extract_field(imu_frame, 4)

        if attitude_data is None:
            continue

        # Extract frame timestamp from FrameMetaHeader.
        # FrameMetaHeader: field 1 = frame_index, field 2 = timestamp (microseconds).
        # The first djmd packet may be a config packet without a valid frame_index.
        frame_ts = 0
        frame_idx = None
        frame_ts_data = _pb_extract_field(frame_meta, 1)
        if frame_ts_data is not None:
            frame_idx, _ = _pb_read_field0(frame_ts_data)
            frame_ts = _pb_read_varint_field(frame_ts_data, 2)

        if first_timestamp is None:
            first_timestamp = frame_ts
        relative_ts = frame_ts - first_timestamp

        # Extract quaternion array from DeviceAttitude (field 3)
        quaternions = _pb_extract_repeated_quaternions(attitude_data, field_num=3)
        if not quaternions:
            continue

        n_quats = len(quaternions)
        frame_ts_ms = relative_ts / 1000.0  # microseconds -> milliseconds

        # Detect model name from first packet's clip_meta
        if pkt_idx == 0:
            clip_meta = _pb_extract_field(buf, 1)
            if clip_meta is not None:
                header = _pb_extract_field(clip_meta, 1)
                if header is not None:
                    name_field = _pb_extract_field(header, 10)
                    if name_field is not None:
                        model = name_field.decode("utf-8", errors="replace").rstrip("\x00")
                        if model:
                            # Name may already include "DJI" prefix
                            if model.startswith("DJI "):
                                metadata.detected_source = model
                            else:
                                metadata.detected_source = f"DJI {model}"

        for qi, (w, x, y, z) in enumerate(quaternions):
            if w != w or x != x or y != y or z != z:  # NaN check
                continue

            # Apply coordinate transform matching Rust telemetry-parser
            qw, qx, qy, qz = _multiply_quat(w, x, y, z, 0.5, -0.5, -0.5, 0.5)
            qw, qx, qy, qz = _multiply_quat(qw, qx, qy, qz, 0.0, 0.0, 1.0, 0.0)

            if qw == 0.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:
                continue

            # Distribute quaternions evenly across the frame interval
            vsync_duration_ms = 1000.0 / frame_rate
            q_ts_ms = frame_ts_ms + (qi / max(n_quats, 1)) * vsync_duration_ms
            ts_us = int(q_ts_ms * 1000)

            q_obj = Quat64.from_quaternion(np.array([qw, qx, qy, qz]))
            quats[ts_us] = q_obj

    metadata.quaternions = quats
    metadata.has_accurate_timestamps = True
    metadata.frame_rate = fps if fps > 0 else None

    if quats:
        log.info("DJI: extracted %d quaternion samples", len(quats))
    else:
        log.warning("DJI: no quaternion data extracted")

    return metadata


# ---------------------------------------------------------------------------
# Protobuf helpers (minimal decoder without compiled schema)
# ---------------------------------------------------------------------------

def _pb_read_varint_field(data: bytes, target_field: int) -> int:
    """Read a varint field (wire type 0) by field number from protobuf data.

    Returns the value of the first occurrence, or 0 if not found.
    """
    pos = 0
    while pos < len(data):
        tag, npos = _read_varint(data, pos)
        if tag == 0:
            break
        field_num = tag >> 3
        wire_type = tag & 7

        if wire_type == 0:
            val, npos = _read_varint(data, npos)
            if field_num == target_field:
                return val
        elif wire_type == 1:
            npos += 8
        elif wire_type == 2:
            length, npos = _read_varint(data, npos)
            npos += length
        elif wire_type == 5:
            npos += 4
        else:
            break
        pos = npos
    return 0


def _pb_extract_field(data: bytes, target_field: int) -> bytes | None:
    """Extract a length-delimited field from a protobuf message.

    Scans the protobuf data for the specified field number (wire type 2)
    and returns its payload bytes.
    """
    pos = 0
    while pos < len(data):
        tag, npos = _read_varint(data, pos)
        if tag == 0:
            break
        field_num = tag >> 3
        wire_type = tag & 7

        if wire_type == 0:  # varint
            _, npos = _read_varint(data, npos)
        elif wire_type == 1:  # 64-bit
            npos += 8
        elif wire_type == 2:  # length-delimited
            length, npos = _read_varint(data, npos)
            if field_num == target_field:
                return data[npos:npos + length]
            npos += length
        elif wire_type == 5:  # 32-bit
            npos += 4
        else:
            break
        pos = npos
    return None


def _pb_read_field0(data: bytes) -> tuple[int | None, int]:
    """Read the first varint field (field 1, wire type 0) from protobuf data."""
    pos = 0
    while pos < len(data):
        tag, npos = _read_varint(data, pos)
        if tag == 0:
            break
        field_num = tag >> 3
        wire_type = tag & 7

        if wire_type == 0:
            val, npos = _read_varint(data, npos)
            if field_num == 1:
                return val, npos
        elif wire_type == 1:
            npos += 8
        elif wire_type == 2:
            length, npos = _read_varint(data, npos)
            npos += length
        elif wire_type == 5:
            npos += 4
        else:
            break
        pos = npos
    return None, pos


def _pb_extract_repeated_quaternions(
    data: bytes, field_num: int
) -> list[tuple[float, float, float, float]]:
    """Extract repeated Quaternion messages from protobuf data.

    Quaternion message has fields: w=1, x=2, y=3, z=4 (all float, wire type 5).
    Returns list of (w, x, y, z) tuples.
    """
    quats: list[tuple[float, float, float, float]] = []
    pos = 0
    while pos < len(data):
        tag, npos = _read_varint(data, pos)
        if tag == 0:
            break
        fn = tag >> 3
        wt = tag & 7

        if wt == 0:
            _, npos = _read_varint(data, npos)
        elif wt == 1:
            npos += 8
        elif wt == 2:
            length, npos = _read_varint(data, npos)
            if fn == field_num:
                q = _parse_quaternion(data[npos:npos + length])
                if q is not None:
                    quats.append(q)
            npos += length
        elif wt == 5:
            npos += 4
        else:
            break
        pos = npos
    return quats


def _parse_quaternion(data: bytes) -> tuple[float, float, float, float] | None:
    """Parse a single Quaternion protobuf message.

    Fields: w=1(float), x=2(float), y=3(float), z=4(float).
    Float fields use wire type 5 (32-bit fixed, little-endian).
    """
    w = x = y = z = 0.0
    found = False
    pos = 0
    while pos < len(data):
        tag, npos = _read_varint(data, pos)
        if tag == 0:
            break
        fn = tag >> 3
        wt = tag & 7

        if wt == 5:  # 32-bit (float)
            if npos + 4 <= len(data):
                val = struct.unpack("<f", data[npos:npos + 4])[0]
                npos += 4
                if fn == 1:
                    w = val; found = True
                elif fn == 2:
                    x = val; found = True
                elif fn == 3:
                    y = val; found = True
                elif fn == 4:
                    z = val; found = True
            else:
                break
        elif wt == 0:
            _, npos = _read_varint(data, npos)
        elif wt == 1:
            npos += 8
        elif wt == 2:
            length, npos = _read_varint(data, npos)
            npos += length
        else:
            break
        pos = npos

    return (w, x, y, z) if found else None


def _multiply_quat(
    w1: float, x1: float, y1: float, z1: float,
    w2: float, x2: float, y2: float, z2: float,
) -> tuple[float, float, float, float]:
    """Multiply two quaternions q1 * q2 in [w, x, y, z] format."""
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
