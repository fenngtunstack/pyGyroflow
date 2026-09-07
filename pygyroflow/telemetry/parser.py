"""Telemetry parser -- extracts gyro data from video files.

Pure-Python parser for GoPro GPMF, DJI protobuf and Sony RTMD telemetry
formats. (A PyO3 bridge crate ``telemetry_parser_bridge`` once existed but
was an empty scaffold and has been removed; this parser is the sole path.)
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

    Detects the camera brand from the file content and dispatches to the
    appropriate parser (GoPro GPMF or DJI protobuf).

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

    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".mov", ".insv"):
        return _parse_embedded(path, sample_index, video_size, fps)

    raise TelemetryParseError(f"Unsupported file format: {ext}")


def _parse_embedded(
    path: str,
    sample_index: int | None = None,
    video_size: tuple[int, int] = (0, 0),
    fps: float = 0.0,
) -> FileMetadata:
    """Parse embedded telemetry (GoPro GPMF or DJI protobuf).

    Detects the camera brand from the file content and dispatches to the
    appropriate parser.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        log.warning("Cannot read file %s: %s", path, exc)
        return FileMetadata(detected_source="Unknown")

    # Detect GoPro: look for 'gpmd' FourCC tag in MP4 box structure
    if _detect_gopro(data):
        return _parse_gopro(data, fps, video_size)

    # Detect DJI: look for 'djmd' tag and "CAM meta" handler
    if _detect_dji(data):
        return _parse_dji(data, fps, video_size)

    # Detect Sony: RTMD metadata track ('meta' handler + samples starting
    # with 00 1C, or the Sony XML manufacturer tag)
    if _detect_sony(data):
        return _parse_sony(data, fps, video_size)

    # Detect Insta360: extra-info trailer magic at the end of the file
    if _detect_insta360(data):
        return _parse_insta360(data, fps, video_size)

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


def _detect_sony(data: bytes) -> bool:
    """Detect Sony RTMD telemetry.

    Mirrors upstream telemetry-parser's Sony::detect: either the XML
    manufacturer tag, or an MP4 metadata track whose samples begin with
    00 1C (the RTMD length prefix).
    """
    if data.find(b'manufacturer="Sony"') >= 0:
        return True
    samples = _mp4_find_data_track_samples(data, "rtmd")
    return bool(samples) and all(
        size > 0x1C and data[offset : offset + 2] == b"\x00\x1c"
        for offset, size in samples[:4]
    )


_INSTA360_MAGIC = b"8db42d694ccc418790edff439fe026bf"
_INSTA360_HEADER_SIZE = 32 + 4 + 4 + 32  # padding(32) + size(4) + version(4) + magic(32)


def _detect_insta360(data: bytes) -> bool:
    """Detect Insta360 extra-info trailer (magic at end of file)."""
    return data[-len(_INSTA360_MAGIC):] == _INSTA360_MAGIC


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


def _mp4_find_timed_data_track_samples(
    data: bytes, target_codec: str
) -> list[tuple[int, int, float]]:
    """Like :func:`_mp4_find_data_track_samples` but also returns each
    sample's duration in milliseconds (from the track's stts run-length
    table and mdhd timescale).

    Returns list of (offset, size, duration_ms).
    """
    result = _mp4_find_data_track_samples_with_durations(data, target_codec)
    if result is None:
        return []
    samples, durations_ms = result
    if not durations_ms:
        durations_ms = [0.0] * len(samples)
    # Pad/trim to sample count (defensive; lengths should match)
    if len(durations_ms) < len(samples):
        durations_ms = durations_ms + [0.0] * (len(samples) - len(durations_ms))
    return [
        (off, size, dur)
        for (off, size), dur in zip(samples, durations_ms)
    ]


def _mp4_find_data_track_samples(
    data: bytes, target_codec: str
) -> list[tuple[int, int]]:
    """Find data track samples in an MP4 file by codec tag.

    Parses the moov/trak/mdia/minf/stbl box hierarchy to locate the track
    with the specified codec tag (e.g. 'gpmd', 'djmd'), then reads stco/co64
    for chunk offsets and stsz for sample sizes.

    Returns list of (offset, size) pairs for each sample.
    """
    result = _mp4_find_data_track_samples_with_durations(data, target_codec)
    if result is None:
        return []
    return result[0]


def _mp4_find_data_track_samples_with_durations(
    data: bytes, target_codec: str
) -> tuple[list[tuple[int, int]], list[float]] | None:
    """Find data track samples plus per-sample durations in ms.

    Returns (samples, durations_ms) for the first track matching
    *target_codec*, or None when no track matches.
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
        return None

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
    return None


def _mp4_parse_trak(
    data: bytes, trak_start: int, trak_end: int, target_codec: str
) -> tuple[list[tuple[int, int]], list[float]] | None:
    """Parse a single trak box looking for *target_codec*.

    Returns (samples, durations_ms) for the matching track, or None if this
    track doesn't match. Sample durations come from the stts run-length
    table scaled by the mdhd timescale (empty when stts is absent).
    """
    timescale = None

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

                if m_fourcc == "mdhd":
                    # version(1)+flags(3) [+ ctime(4|8) + mtime(4|8)] +
                    # timescale(4) + duration(4)
                    off = mdia_pos + m_hs
                    ver = data[off + 4] if off + 8 <= mdia_pos + m_bs else 0
                    ts_off = off + (20 if ver == 1 else 12)
                    if ts_off + 4 <= mdia_pos + m_bs:
                        timescale = struct.unpack(
                            ">I", data[ts_off:ts_off + 4]
                        )[0]

                elif m_fourcc == "minf":
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
                                timescale=timescale or 1000,
                            )
                            if _r is not None:
                                return _r

                        minf_pos += mi_bs

                mdia_pos += m_bs

        pos += bs
    return None


def _mp4_parse_stbl(
    data: bytes,
    stbl_start: int,
    stbl_end: int,
    target_codec: str,
    timescale: int = 1000,
) -> tuple[list[tuple[int, int]], list[float]] | None:
    """Parse stbl box to find sample offsets and sizes for *target_codec*.

    Returns (samples, durations_ms): the (offset, size) pairs plus each
    sample's duration in milliseconds expanded from the stts run-length
    table (empty list when stts is absent). None when the codec doesn't
    match.
    """
    stsd_codec = None
    chunk_offsets: list[int] = []
    sample_sizes: list[int] = []
    stsc_entries: list[tuple[int, int, int]] = []  # (first_chunk, samples_per_chunk, sample_desc_idx)
    stts_deltas: list[int] = []  # per-sample delta in track ticks

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

        elif fourcc == "stts":
            # Run-length (sample_count, sample_delta) pairs in track ticks
            if box_payload + 8 <= box_end:
                entry_count = struct.unpack(">I", data[box_payload + 4:box_payload + 8])[0]
                for i in range(entry_count):
                    tt_pos = box_payload + 8 + i * 8
                    if tt_pos + 8 <= box_end:
                        count, delta = struct.unpack(">II", data[tt_pos:tt_pos + 8])
                        stts_deltas.extend([delta] * count)

        pos += bs

    # Check if this is the right codec
    if stsd_codec != target_codec:
        return None

    durations_ms = [
        d / timescale * 1000.0 for d in stts_deltas
    ] if timescale > 0 else []

    # Build sample (offset, size) list using stsc to map chunks to samples
    if not chunk_offsets:
        return [], durations_ms

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
        return result, durations_ms

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

    return (samples, durations_ms) if samples else None


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


def _parse_gopro(data: bytes, fps: float, video_size: tuple[int, int] = (0, 0)) -> FileMetadata:
    """Parse GoPro GPMF telemetry from raw MP4 file data.

    Extracts gyroscope (GYRO) and accelerometer (ACCL) samples, converts
    them to physical units using the SCAL factor, and builds TimeIMU records.

    For Hero9+ files the per-frame CORI (CameraOrientation) and IORI
    (ImageOrientation) quaternion streams take priority: they are combined
    (CORI * IORI) into FileMetadata.quaternions exactly like upstream
    telemetry-parser's process_samples(), with frame-aligned timestamps
    from the video fps. Upstream Gyroflow stabilizes from these directly
    (integration_method = 0), bypassing raw-IMU integration entirely.

    Axis orientation follows upstream telemetry-parser: derived from
    ORIN+ORIO -> MTRX -> IMUO when the tags exist, with model-specific
    fallbacks (HERO6 -> "ZyX", HERO7 Silver -> "YXz"). Files without
    those tags (Hero8+, e.g. our test footage) get NO orientation remap
    (None) — Gyroflow guesses the orientation at runtime
    (``guess_imu_orientation``). Hardcoding "ZXY" was a porting bug that
    scrambled the axes on modern GoPro files.
    """
    import numpy as np

    from pygyroflow.types.quaternion import Quat64
    from pygyroflow.types.time_types import TimeIMU

    metadata = FileMetadata(detected_source="GoPro")

    # Find gpmd track samples via MP4 box parsing (with stts durations)
    timed_samples = _mp4_find_timed_data_track_samples(data, "gpmd")
    if not timed_samples:
        log.warning("No gpmd track found in GoPro file")
        return metadata

    stream_info: dict[str, dict] = {}  # stream name -> {orin, orio, mtrx}
    camera_tags: dict = {}  # EISA/EISE/VFOV/ZFOV/PRJT (first occurrence wins)
    model: str | None = None
    quats: dict[int, Quat64] = {}
    frame_ms = 1000.0 / fps if fps > 0 else 0.0
    frame_idx = 0

    # Raw IMU timestamps follow upstream telemetry-parser's
    # util::normalized_imu + GoPro::get_avg_sample_duration: GoPro gyro/accel
    # readings are laid on a UNIFORM grid t_i = i * avg_diff across the whole
    # file, where avg_diff prefers the GYRO stream's first/last STMP span and
    # otherwise falls back to the MP4 sample-table total duration / reading
    # count. STMP absolute values are never used as per-packet bases: older
    # files (Hero5/6) carry no STMP at all, and every packet would otherwise
    # collapse onto the first ~1 s of the timeline.
    rows: list[tuple] = []  # (gyro_row, accl_row), entries np arrays or None
    total_duration_ms = 0.0
    total_gyro_count = 0
    last_gyro_packet_count = 0
    stmp_first_us: int | None = None
    stmp_last_us: int | None = None

    for offset, size, duration_ms in timed_samples:
        if offset + size > len(data):
            continue
        gpmf_chunk = data[offset:offset + size]
        pkt_model, gyro_rows, accl_rows, pkt_gyro_count, pkt_stmp_us = (
            _parse_gpmf_chunk(gpmf_chunk, stream_info, camera_tags)
        )
        model = pkt_model or model
        rows.extend(zip(gyro_rows, accl_rows))
        total_duration_ms += duration_ms

        if pkt_gyro_count > 0:
            total_gyro_count += pkt_gyro_count
            last_gyro_packet_count = pkt_gyro_count
        if pkt_stmp_us:
            if stmp_first_us is None:
                stmp_first_us = pkt_stmp_us
            stmp_last_us = pkt_stmp_us

        cori, iori = _parse_gpmf_orientation_chunk(gpmf_chunk)
        # Upstream: only emit when both streams are present and equal length
        if cori and len(cori) == len(iori):
            for c, i in zip(cori, iori):
                w, x, y, z = _multiply_quat(c[0], c[1], c[2], c[3], i[0], i[1], i[2], i[3])
                ts_us = int(round(frame_idx * frame_ms * 1000.0))
                quats[ts_us] = Quat64.from_quaternion(np.array([w, x, y, z]))
                frame_idx += 1

    # Modern GoPros (verified Hero8/10) write the camera-identification DEVC
    # (VFOV/EISA/EISE/ZFOV/PRJT) at the very end of mdat, AFTER the last
    # sample-table entry. Recover it with a bounded tail scan — camera tags
    # only; any IMU rows found there are discarded (the sample table, not
    # the tail, defines the telemetry stream).
    if not camera_tags and timed_samples:
        tail_start = timed_samples[-1][0] + timed_samples[-1][1]
        tail = data[tail_start : tail_start + 65536]
        p = 0
        while p < len(tail) - _GPMF_HEADER_SIZE:
            i = tail.find(b"DEVC", p)
            if i < 0 or i + _GPMF_HEADER_SIZE > len(tail):
                break
            klv = _gpmf_read_klv(tail, i, len(tail))
            if klv is None:
                break
            devc_end = min(klv[6], len(tail))
            _parse_gpmf_chunk(tail[i:devc_end], None, camera_tags)
            if camera_tags:
                break
            p = klv[6]

    raw_imu: list[TimeIMU] = []
    if rows:
        if (
            stmp_first_us is not None
            and stmp_last_us is not None
            and stmp_last_us > stmp_first_us
            and total_gyro_count > 0
        ):
            denom = max(1, total_gyro_count - last_gyro_packet_count)
            avg_diff_ms = (stmp_last_us - stmp_first_us) / 1000.0 / denom
        elif total_gyro_count > 0 and total_duration_ms > 0:
            avg_diff_ms = total_duration_ms / total_gyro_count
        elif total_gyro_count > 0:
            avg_diff_ms = 5.0  # no timing at all: assume ~200 Hz IMU
        else:
            avg_diff_ms = 0.0
        for i, (gyro, accl) in enumerate(rows):
            raw_imu.append(TimeIMU(timestamp_ms=i * avg_diff_ms, gyro=gyro, accl=accl))
        log.info(
            "GoPro: %d raw IMU readings on uniform grid, avg step %.3f ms "
            "(span %.1f ms, %d gpmd packets)",
            len(rows), avg_diff_ms, len(rows) * avg_diff_ms, len(timed_samples),
        )
    metadata.raw_imu = raw_imu

    if quats:
        metadata.quaternions = quats
        log.info("GoPro: extracted %d CORI*IORI quaternions", len(quats))

    metadata.frame_rate = fps if fps > 0 else None
    metadata.has_accurate_timestamps = True

    # Rolling-shutter readout time from the SROT tag (f32 ms). Upstream
    # reads it via the Unknown(0x53524F54) tag; in real files it appears
    # as a nested 'SROT' + type(f) + size + repeat + f32 payload.
    srot_pos = data.find(b"SROT")
    if srot_pos > 0 and srot_pos + 12 <= len(data):
        if data[srot_pos + 4] == ord("f"):
            (readout_ms,) = struct.unpack(">f", data[srot_pos + 8 : srot_pos + 12])
            if 0.1 < readout_ms < 100.0:
                metadata.frame_readout_time = float(readout_ms)
                log.info("GoPro: SROT frame_readout_time = %.2f ms", readout_ms)

    # Camera model from the MINF tag (e.g. "HERO12 Black") for automatic
    # lens-profile matching. Searched at file level (the tag lives outside
    # the sampled gpmd track in some firmwares).
    if not model:
        minf_pos = data.find(b"MINF")
        if minf_pos > 0 and minf_pos + 12 <= len(data):
            try:
                tb = data[minf_pos + 4]
                ss = data[minf_pos + 5]
                rep = struct.unpack(">H", data[minf_pos + 6 : minf_pos + 8])[0]
                n = ss * rep
                if tb == ord("c") and 4 <= n <= 64:
                    model = data[minf_pos + 8 : minf_pos + 8 + n].split(b"\x00")[0].decode("ascii", "replace").strip()
            except (IndexError, ValueError):
                pass
    if model:
        metadata.detected_source = f"GoPro {model}"

    if camera_tags:
        metadata.additional_data["camera_tags"] = {"Default": camera_tags}

    # Orientation derivation must run after the model fallbacks: the
    # model-specific routes (HERO6 -> "ZyX", HERO7 Silver -> "YXz") are the
    # only source of orientation for files whose streams carry no
    # ORIN/ORIO/MTRX tags.
    metadata.imu_orientation = _gopro_derive_orientation(stream_info, model)

    return metadata


def _parse_gpmf_orientation_chunk(gpmf_data: bytes) -> tuple[list, list]:
    """Extract CORI / IORI quaternion streams from one gpmd sample.

    Mirrors upstream telemetry-parser GoPro::process_samples(): each
    CameraOrientation/ImageOrientation STRM carries an i16 quaternion
    array scaled by SCAL (default 32767), converted with the SAME sign
    convention as upstream: (w/s, -x/s, y/s, z/s).

    Returns (cori_list, iori_list) of [w, x, y, z] float lists.
    """
    cori: list[list[float]] = []
    iori: list[list[float]] = []

    pos = 0
    end = len(gpmf_data)
    while pos < end - _GPMF_HEADER_SIZE:
        klv = _gpmf_read_klv(gpmf_data, pos, end)
        if klv is None:
            break
        fourcc, type_byte, _, _, payload_start, _, next_pos = klv
        if fourcc != "DEVC" or type_byte != 0:
            pos = next_pos
            continue

        dpos = payload_start
        devc_end = next_pos
        while dpos < devc_end - _GPMF_HEADER_SIZE:
            child = _gpmf_read_klv(gpmf_data, dpos, devc_end)
            if child is None:
                break
            c_fourcc, c_type, _, _, _, _, c_np = child
            if c_fourcc == "STRM" and c_type == 0:
                spos = child[4]
                stream_name = ""
                stream_scal = 32767.0
                cori_payload = None
                iori_payload = None
                while spos < c_np - _GPMF_HEADER_SIZE:
                    tk = _gpmf_read_klv(gpmf_data, spos, c_np)
                    if tk is None:
                        break
                    t_fourcc, t_type, t_ss, t_rep, t_ps, t_pb, t_np = tk
                    if t_fourcc == "STNM" and t_type == ord("c"):
                        stream_name = gpmf_data[t_ps:t_ps + t_pb].decode("ascii", errors="replace").rstrip("\x00")
                    elif t_fourcc == "SCAL" and t_type == ord("s") and t_pb >= 2:
                        stream_scal = float(struct.unpack(">h", gpmf_data[t_ps:t_ps + 2])[0])
                    elif t_fourcc == "CORI" and t_type == ord("s"):
                        cori_payload = (t_ps, t_pb)
                    elif t_fourcc == "IORI" and t_type == ord("s"):
                        iori_payload = (t_ps, t_pb)
                    spos = t_np

                target = None
                payload = None
                if stream_name == "CameraOrientation" and cori_payload:
                    target, payload = cori, cori_payload
                elif stream_name == "ImageOrientation" and iori_payload:
                    target, payload = iori, iori_payload
                if target is not None and payload is not None:
                    ps, pb = payload
                    n = pb // 8  # 4 x i16 per sample
                    vals = struct.unpack(f">{n * 4}h", gpmf_data[ps:ps + n * 8])
                    for k in range(n):
                        w, x, y, z = vals[k * 4:k * 4 + 4]
                        target.append([w / stream_scal, -x / stream_scal, y / stream_scal, z / stream_scal])
            dpos = c_np
        break  # only the first DEVC per chunk carries these streams

    return cori, iori


def _gopro_orientations_to_matrix(orin: str, orio: str) -> list[float] | None:
    """Port of upstream KLV::orientations_to_matrix (ORIN + ORIO -> 3x3)."""
    if not orin or len(orin) != len(orio):
        return None
    out: list[float] = []
    for o in orio:
        for i in orin:
            if i == o:
                out.append(1.0)
            elif i.lower() == o.lower():
                out.append(-1.0)
            else:
                out.append(0.0)
    return out


def _gopro_mtrx_to_orientation(mtrx: list[float]) -> str | None:
    """Port of upstream GoPro::mtrx_to_orientation (3x3 -> 'XyZ')."""
    if len(mtrx) != 9:
        return None
    chars = []
    for r in range(3):
        row = mtrx[r * 3 : r * 3 + 3]
        if row[0] > 0.5:
            chars.append("X")
        elif row[0] < -0.5:
            chars.append("x")
        elif row[1] > 0.5:
            chars.append("Y")
        elif row[1] < -0.5:
            chars.append("y")
        elif row[2] > 0.5:
            chars.append("Z")
        elif row[2] < -0.5:
            chars.append("z")
        else:
            return None
    return "".join(chars)


def _gopro_derive_orientation(stream_info: dict[str, dict], model: str | None) -> str | None:
    """Derive the IMU orientation the way upstream telemetry-parser does."""
    # Prefer the gyroscope stream's tags, fall back to any stream
    candidates = [stream_info.get("Gyroscope"), stream_info.get("Accelerometer")]
    candidates += [v for k, v in stream_info.items() if k not in ("Gyroscope", "Accelerometer")]

    for info in candidates:
        if not info:
            continue
        mtrx = info.get("mtrx")
        if mtrx:
            o = _gopro_mtrx_to_orientation(mtrx)
            if o:
                return o
        orin, orio = info.get("orin"), info.get("orio")
        if orin and orio:
            m = _gopro_orientations_to_matrix(orin, orio)
            if m:
                o = _gopro_mtrx_to_orientation(m)
                if o:
                    return o

    if model:
        if "HERO6" in model:
            return "ZyX"
        if "HERO7 Silver" in model:
            return "YXz"
    return None


def _parse_gpmf_chunk(
    gpmf_data: bytes,
    stream_info: dict[str, dict] | None = None,
    camera_tags: dict | None = None,
) -> tuple[str | None, list, list, int, int]:
    """Parse one gpmd track sample (contains one or more DEVC blocks).

    Extracts ACCL and GYRO streams from each DEVC/STRM and returns physical-
    unit rows for the whole packet:

    Returns (model, gyro_rows, accl_rows, gyro_count, last_stamp_us).
    Timestamps are NOT assigned here — upstream telemetry-parser lays GoPro
    IMU readings on a uniform whole-file grid (see _parse_gopro), so per-
    packet STMP values are only reported for the grid-step estimate.

    DEVC-level camera tags (EISA/EISE/VFOV/ZFOV/PRJT) are collected into
    *camera_tags* (first occurrence wins) for lens-profile auto-loading.
    """
    import numpy as np

    RAD_TO_DEG = 180.0 / 3.141592653589793

    pos = 0
    end = len(gpmf_data)
    model: str | None = None
    gyro_rows: list = []
    accl_rows: list = []
    chunk_gyro_count = 0
    last_stamp_us = 0

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

            if c_fourcc == "DMNL" or c_fourcc == "MINF":
                # model name (for orientation fallbacks)
                model = gpmf_data[child[4] : child[4] + child[5]].decode("ascii", errors="replace").rstrip("\x00")
            elif c_fourcc in ("EISA", "EISE", "VFOV", "ZFOV", "PRJT") and camera_tags is not None:
                # Camera/lens identification tags (upstream reads them from
                # GroupId::Default for CameraIdentifier::from_telemetry_parser)
                val: object = None
                if c_type == ord("c"):
                    val = gpmf_data[child[4] : child[4] + child[5]].decode("ascii", errors="replace").rstrip("\x00")
                elif c_type == ord("f") and child[5] >= 4:
                    val = struct.unpack(">f", gpmf_data[child[4] : child[4] + 4])[0]
                if val is not None:
                    camera_tags.setdefault(c_fourcc, val)
            elif c_fourcc == "STRM" and c_type == 0:
                # Parse STRM contents for ACCL/GYRO (+ orientation tags)
                _parse_gpmf_strm(
                    gpmf_data, c_np,  # c_ps is at child[4], c_np is end
                    child[4],  # payload_start of STRM
                    RAD_TO_DEG,
                    # Mutable accumulators via closure
                    accl_ref := [accl_samples, accl_scale, 0, accl_count, accl_struct_size],
                    gyro_ref := [gyro_samples, gyro_scale, gyro_stamp_us, gyro_count, gyro_struct_size],
                    stream_info,
                )
                # Read back updated values
                accl_samples, accl_scale, _accl_stamp, accl_count, accl_struct_size = accl_ref
                gyro_samples, gyro_scale, gyro_stamp_us, gyro_count, gyro_struct_size = gyro_ref
                if gyro_stamp_us:
                    last_stamp_us = gyro_stamp_us

            dpos = c_np

        # Emit physical-unit rows for this DEVC
        if accl_count == 0 and gyro_count == 0:
            pos = next_pos
            continue

        n_samples = max(accl_count, gyro_count)
        for i in range(n_samples):
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

            gyro_rows.append(gyro)
            accl_rows.append(accl)

        chunk_gyro_count += gyro_count
        pos = next_pos

    return model, gyro_rows, accl_rows, chunk_gyro_count, last_stamp_us


def _parse_gpmf_strm(
    gpmf_data: bytes,
    strm_end: int,
    strm_ps: int,
    rad_to_deg: float,
    accl_ref: list,
    gyro_ref: list,
    stream_info: dict[str, dict] | None = None,
) -> None:
    """Parse a STRM block, updating accl_ref / gyro_ref in place."""
    spos = strm_ps
    stream_name = ""
    stream_stamp = 0
    stream_orin = "XYZ"
    stream_orio = ""
    stream_mtrx: list[float] | None = None
    stream_scal = 1.0

    # First pass: collect metadata (STNM, STMP, ORIN, ORIO, MTRX, SIUN, SCAL)
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
        elif t_fourcc == "ORIO" and t_type == ord("c"):
            stream_orio = gpmf_data[t_ps:t_ps + t_pb].decode("ascii", errors="replace").rstrip("\x00")
        elif t_fourcc == "MTRX" and t_type == ord("f") and t_pb >= 36:
            stream_mtrx = list(struct.unpack(">9f", gpmf_data[t_ps:t_ps + 36]))
        elif t_fourcc == "SCAL" and t_type == ord("s") and t_pb >= 2:
            stream_scal = float(struct.unpack(">h", gpmf_data[t_ps:t_ps + 2])[0])

        spos = t_np

    if stream_info is not None and stream_name:
        info = stream_info.setdefault(stream_name, {})
        info.setdefault("orin", stream_orin)
        info.setdefault("orio", stream_orio)
        if stream_mtrx and "mtrx" not in info:
            info["mtrx"] = stream_mtrx

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


def _parse_dji(data: bytes, fps: float, video_size: tuple[int, int] = (0, 0)) -> FileMetadata:
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
    sensor_fps = 0.0

    for pkt_idx, (offset, size) in enumerate(samples):
        if offset + size > len(data) or size == 0:
            continue

        buf = data[offset:offset + size]

        # ProductMeta { clip_meta=1, stream_meta=2, frame_meta=3 }
        # From clip_meta (first sample only) extract the lens profile inputs
        # exactly like upstream telemetry-parser's handle_parsed!():
        #   readout = sensor_readout_time(4).readout_time(1) / 1e6 / (fps/sensor_fps)
        #   focal  = digital_focal_length(8).focal_length(1)
        #   coeffs = distortion_coefficients(3).coeffients(1, repeated float)
        if pkt_idx == 0:
            clip_meta = _pb_extract_field(buf, 1)
            if clip_meta is not None:
                readout_us = _pb_read_varint_field(
                    _pb_extract_field(clip_meta, 4) or b"", 1)
                focal_len = _pb_read_f32_field(
                    _pb_extract_field(clip_meta, 8) or b"", 1)
                coeffs = _pb_read_repeated_f32_field(
                    _pb_extract_field(clip_meta, 3) or b"", 1)
                sensor_fps = _pb_read_f32_field(
                    _pb_extract_field(clip_meta, 11) or b"", 1)

                stream_meta = _pb_extract_field(buf, 2)
                vfr = None
                if stream_meta is not None:
                    vsm = _pb_extract_field(stream_meta, 3)
                    if vsm is not None:
                        vfr = _pb_read_f32_field(vsm, 3)
                if vfr and vfr > 1.0:
                    frame_rate = vfr

                if readout_us:
                    rt = readout_us / 1e6
                    if sensor_fps and sensor_fps > 1.0 and frame_rate > 1.0:
                        rt /= frame_rate / sensor_fps
                    metadata.frame_readout_time = rt

                if focal_len and focal_len > 1.0:
                    w, h = video_size if video_size and video_size[0] > 0 else (1920, 1080)
                    ow, oh = w, h
                    if round(w / h * 100) == 133:  # 4:3 -> 16:9 like upstream
                        oh = round(w / 1.7777777777777)
                    metadata.lens_profile = {
                        "calibrated_by": "DJI",
                        "camera_brand": "DJI",
                        "camera_model": (metadata.detected_source or "DJI").replace("DJI ", ""),
                        "calib_dimension": {"w": w, "h": h},
                        "orig_dimension": {"w": w, "h": h},
                        "output_dimension": {"w": ow, "h": oh},
                        "frame_readout_time": metadata.frame_readout_time,
                        "official": True,
                        "fisheye_params": {
                            "camera_matrix": [
                                [focal_len, 0.0, w / 2.0],
                                [0.0, focal_len, h / 2.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "distortion_coeffs": coeffs or [0.0] * 12,
                        },
                        "calibrator_version": "---",
                    }
                    log.info(
                        "DJI: lens profile from metadata: focal=%.1f coeffs=%s readout=%.2fms",
                        focal_len, [round(c, 4) for c in (coeffs or [])][:4],
                        metadata.frame_readout_time or 0.0,
                    )

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
        # offset (field 4): time offset between first sensor row and the
        # first sample — used in the upstream timestamp model below.
        quaternions = _pb_extract_repeated_quaternions(attitude_data, field_num=3)
        if not quaternions:
            continue
        att_offset = _pb_read_f32_field(attitude_data, 4)

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

        prev_quat = None
        inv = False
        for qi, (w, x, y, z) in enumerate(quaternions):
            if w != w or x != x or y != y or z != z:  # NaN check
                continue

            # Apply coordinate transform matching Rust telemetry-parser:
            # quat = (0,0,1,0) * (raw * (0.5,-0.5,-0.5,0.5)). Multiply
            # ORDER matters — the previous (raw*m1)*z180 flipped y/z signs
            # versus the upstream reference.
            qw, qx, qy, qz = _multiply_quat(w, x, y, z, 0.5, -0.5, -0.5, 0.5)
            # Rotate Y axis 180 deg for horizon lock
            qw, qx, qy, qz = _multiply_quat(0.0, 0.0, 1.0, 0.0, qw, qx, qy, qz)

            if qw == 0.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:
                continue

            # Quaternion double-cover continuity: flip the sign whenever the
            # jump to the previous sample exceeds 1.5, mirroring the Rust
            # telemetry-parser (`inv = !inv`). Without this the stream
            # alternates between q and -q at discontinuities, producing
            # phantom 180-deg rotations in the per-sample deltas.
            if prev_quat is not None:
                dq = (
                    (prev_quat[0] - qw) ** 2
                    + (prev_quat[1] - qx) ** 2
                    + (prev_quat[2] - qy) ** 2
                    + (prev_quat[3] - qz) ** 2
                ) ** 0.5
                if dq > 1.5:
                    inv = not inv
            prev_quat = (qw, qx, qy, qz)

            if inv:
                qw, qx, qy, qz = -qw, -qx, -qy, -qz

            # Upstream timestamp model (dji/mod.rs, non-commented code):
            #   quat_ts = frame_ms + ((i - offset) / len) * (1000 / sensor_fps)
            #   ts_ms   = quat_ts / fps_ratio      (fps_ratio = fps / sensor_fps)
            # The previous "evenly across video-frame interval" model drifted
            # ~0.33s over a 25s clip (sample dt 0.998ms vs true 1.011ms).
            vsync_ms = 1000.0 / sensor_fps if sensor_fps > 1.0 else 1000.0 / frame_rate
            fps_ratio = frame_rate / sensor_fps if sensor_fps > 1.0 else 1.0
            quat_ts_ms = frame_ts_ms + ((qi - att_offset) / max(n_quats, 1)) * vsync_ms
            q_ts_ms = quat_ts_ms / fps_ratio if fps_ratio > 0.0 else quat_ts_ms
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
# Sony RTMD parser
# ---------------------------------------------------------------------------

def _sony_read_f16(raw: bytes) -> float:
    """Decode Sony's decimal-float 16 (NOT IEEE 754 half).

    Upstream rtmd_tags read_f16: 4-bit exponent (sign-extended when >= 8)
    with a 12-bit mantissa scaled by 10^exp — used by the lens tags
    (focal length etc.). Returned in the tag's own unit (mm *before* the
    x1000 applied by upstream; see _parse_sony).
    """
    (num,) = struct.unpack(">h", raw[:2])
    exp = (num >> 12) & 0x0F
    if exp >= 8:
        exp = -((~exp & 0x7) + 1)
    return (num & 0x0FFF) * 10.0**exp


def _sony_orientation(payload: bytes) -> str | None:
    """Decode the 3-nibble IMU orientation (upstream rtmd_tags read_orientation).

    e.g. 0x152 -> 'Yzx' (RX100 VII), 0x420 -> 'XYZ' (A7S III), 0x241 -> 'xZY' (RX0 II).
    """
    chars = "XxYyZz"
    (num,) = struct.unpack(">H", payload[:2])
    try:
        return "".join(chars[num & 0xF] + chars[(num >> 4) & 0xF] + chars[(num >> 8) & 0xF])
    except IndexError:
        return None


def _sony_normalize_orientation(v: str) -> str:
    """Sony's normalize_imu_orientation: swap X/Y, invert Z case."""
    chars = list(v)
    chars[0], chars[1] = chars[1], chars[0]
    chars[2] = chars[2].swapcase()
    return "".join(chars)


def _sony_walk_tlv(buf: bytes, pos: int, end: int, tags: dict[int, bytes]) -> None:
    """Flatten one Sony RTMD TLV block into *tags* (later entries win).

    Layout per upstream Sony::parse_metadata: u16 tag + u16 len + payload;
    tag 0x060e is a 16-byte UUID (skip); 0x8300 is a nested container.
    """
    while pos + 4 <= end:
        tag, length = struct.unpack(">HH", buf[pos : pos + 4])
        if tag == 0x060e:
            pos += 2 + 14
            continue
        if tag in (0, 0xFFFF):
            break
        payload_start = pos + 4
        if payload_start + length > end:
            break
        payload = buf[payload_start : payload_start + length]
        tags[tag] = payload
        if tag == 0x8300:
            _sony_walk_tlv(payload, 0, len(payload), tags)
        pos = payload_start + length


def _parse_sony(data: bytes, fps: float, video_size: tuple[int, int] = (0, 0)) -> FileMetadata:
    """Parse Sony RTMD telemetry from a raw MP4 file.

    Mirrors upstream telemetry-parser sony/mod.rs + util.rs
    normalized_imu_interpolated:
    - gyro (0xe43b) / accel (0xe44b) blocks: [i32 count][i32 len=6][count x 3 i16]
    - values: raw / scale (0xe439/0xe449); accel additionally x9.80665
      (upstream inserts Unit "g" for the accelerometer group)
    - raw rows keep stream axis order (orientation "XYZ"); the decoded
      stream orientation (0xe43a, normalized) is returned via
      FileMetadata.imu_orientation for GyroSource to apply
    - timestamps: uniform grid i * total_duration / reading_count
    """
    import numpy as np

    from pygyroflow.types.time_types import TimeIMU

    metadata = FileMetadata(detected_source="Sony")

    timed_samples = _mp4_find_timed_data_track_samples(data, "rtmd")
    if not timed_samples:
        log.warning("No rtmd track found in Sony file")
        return metadata

    # Model from the optional XML urn:x-canon manufacturer block
    model: str | None = None
    lens_name: str | None = None
    xml_pos = data.find(b'manufacturer="Sony"')
    if xml_pos >= 0:
        window = data[xml_pos : xml_pos + 1024]
        m = _find_between(window, b'modelName="', b'"')
        if m:
            model = m.decode("ascii", errors="replace")
        m = _find_between(window, b'Lens modelName="', b'"')
        if m:
            lens_name = m.decode("ascii", errors="replace")

    G_TO_MS2 = 9.80665
    rows: list[tuple] = []
    total_duration_ms = 0.0
    total_gyro_count = 0
    orientation: str | None = None
    focal_length: float | None = None
    readout_ms: float | None = None

    for sample_idx, (offset, size, duration_ms) in enumerate(timed_samples):
        chunk = data[offset : offset + size]
        if len(chunk) <= 0x1C or chunk[:2] != b"\x00\x1c":
            continue
        tags: dict[int, bytes] = {}
        _sony_walk_tlv(chunk, 0x1C, len(chunk), tags)

        if sample_idx == 0:
            if 0x8005 in tags and len(tags[0x8005]) >= 2:
                focal_length = _sony_read_f16(tags[0x8005]) * 1000.0
            if 0xe43a in tags:
                raw_orient = _sony_orientation(tags[0xe43a])
                if raw_orient:
                    orientation = _sony_normalize_orientation(raw_orient)
            if 0xe40e in tags and len(tags[0xe40e]) >= 4:
                rt = struct.unpack(">i", tags[0xe40e][:4])[0] / 1000.0
                if abs(rt) > 0.1:
                    readout_ms = rt

        gyro_vals = None
        gyro_count = 0
        g = tags.get(0xE43B)
        if g is not None and len(g) >= 8:
            count, length = struct.unpack(">ii", g[:8])
            if count > 0 and length == 6 and len(g) >= 8 + count * 6:
                scale = struct.unpack(">f", tags[0xE439][:4])[0] if 0xE439 in tags and len(tags[0xE439]) >= 4 else 1.0
                raw = np.frombuffer(g[8 : 8 + count * 6], dtype=">i2").astype(np.float64)
                gyro_vals = raw.reshape(count, 3) / (scale if scale else 1.0)
                gyro_count = count

        accl_vals = None
        accl_count = 0
        a = tags.get(0xE44B)
        if a is not None and len(a) >= 8:
            count, length = struct.unpack(">ii", a[:8])
            if count > 0 and length == 6 and len(a) >= 8 + count * 6:
                scale = struct.unpack(">f", tags[0xE449][:4])[0] if 0xE449 in tags and len(tags[0xE449]) >= 4 else 1.0
                raw = np.frombuffer(a[8 : 8 + count * 6], dtype=">i2").astype(np.float64)
                accl_vals = raw.reshape(count, 3) / (scale if scale else 1.0) * G_TO_MS2
                accl_count = count

        n = max(gyro_count, accl_count)
        for i in range(n):
            rows.append((
                gyro_vals[i] if gyro_vals is not None and i < gyro_count else None,
                accl_vals[i] if accl_vals is not None and i < accl_count else None,
            ))
        total_duration_ms += duration_ms
        total_gyro_count += gyro_count

    raw_imu: list[TimeIMU] = []
    if rows and total_gyro_count > 0 and total_duration_ms > 0:
        avg_diff_ms = total_duration_ms / total_gyro_count
        for i, (gyro, accl) in enumerate(rows):
            raw_imu.append(TimeIMU(timestamp_ms=i * avg_diff_ms, gyro=gyro, accl=accl))
        log.info(
            "Sony: %d raw IMU readings, avg step %.4f ms (span %.1f ms, %d rtmd packets)",
            len(rows), avg_diff_ms, len(rows) * avg_diff_ms, len(timed_samples),
        )

    metadata.raw_imu = raw_imu
    metadata.imu_orientation = orientation
    metadata.frame_readout_time = readout_ms
    metadata.frame_rate = fps if fps > 0 else None
    metadata.has_accurate_timestamps = True
    if model:
        metadata.detected_source = f"Sony {model}"

    # Camera identifier inputs (focal length drives the Sony lens-profile
    # autoload identifier "xx.xx mm")
    lens_tags: dict = {}
    if focal_length:
        lens_tags["FocalLength"] = focal_length
    if lens_name:
        lens_tags["DisplayName"] = lens_name
    if lens_tags:
        metadata.additional_data["camera_tags"] = {"Lens": lens_tags}

    return metadata


def _find_between(haystack: bytes, start: bytes, end: bytes) -> bytes | None:
    """Return the bytes between *start* and *end* (first occurrence)."""
    i = haystack.find(start)
    if i < 0:
        return None
    j = haystack.find(end, i + len(start))
    if j < 0:
        return None
    return haystack[i + len(start) : j]


# ---------------------------------------------------------------------------
# Insta360 extra-info parser
# ---------------------------------------------------------------------------

def _pb_read_f64_field(data: bytes, target_field: int) -> float:
    """Read a 64-bit double field (wire type 1) by field number."""
    pos = 0
    while pos < len(data):
        tag, npos = _read_varint(data, pos)
        if tag == 0:
            break
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:
            _, npos = _read_varint(data, npos)
        elif wire_type == 1:
            if field_num == target_field and npos + 8 <= len(data):
                return struct.unpack("<d", data[npos:npos + 8])[0]
            npos += 8
        elif wire_type == 2:
            length, npos = _read_varint(data, npos)
            npos += length
        elif wire_type == 5:
            npos += 4
        else:
            break
        pos = npos
    return 0.0


def _pb_read_string_field(data: bytes, target_field: int) -> str:
    """Read a string field (wire type 2) by field number."""
    chunk = _pb_extract_field(data, target_field)
    if chunk is None:
        return ""
    return chunk.decode("utf-8", errors="replace")


def _insta360_collect_records(data: bytes) -> list[tuple[int, int, bytes]]:
    """Collect (record_id, format, payload) from the trailing extra-info blob.

    Mirrors upstream Insta360::parse_file: records are found either through
    an Offsets index record (modern files) or by walking backwards from the
    72-byte header (legacy layout). Each record is [payload][format u8]
    [id u8][size u32 LE].
    """
    hdr = data[-_INSTA360_HEADER_SIZE:]
    extra_size = struct.unpack("<I", hdr[32:36])[0]
    extra_start = len(data) - extra_size

    records: list[tuple[int, int, bytes]] = []

    # Fast path: Offsets index (first_id byte just before the header)
    trailer = len(data) - (_INSTA360_HEADER_SIZE + 6)
    if trailer >= 0 and data[trailer + 1] == 0:  # record id == Offsets
        (tbl_size,) = struct.unpack("<I", data[trailer + 2 : trailer + 6])
        tbl_start = trailer - tbl_size
        if tbl_start >= 0:
            tbl = data[tbl_start:trailer]
            entries: dict[int, tuple[int, int]] = {}
            p = 0
            while p + 10 <= len(tbl):
                rid, rfmt = tbl[p], tbl[p + 1]
                rsize, roff = struct.unpack("<II", tbl[p + 2 : p + 10])
                if rid > 0:
                    entries[rid] = (roff, rsize)
                p += 10
            for rid, (roff, rsize) in entries.items():
                base = extra_start + roff
                if base < 0 or base + rsize + 6 > len(data):
                    continue
                payload = data[base : base + rsize]
                rfmt = data[base + rsize]
                id2 = data[base + rsize + 1]
                (size2,) = struct.unpack("<I", data[base + rsize + 2 : base + rsize + 6])
                if size2 == rsize and id2 == rid:
                    records.append((rid, rfmt, payload))
            if records:
                return records

    # Legacy fallback: walk records backwards
    offset = _INSTA360_HEADER_SIZE + 6
    while offset < extra_size:
        pos = len(data) - offset
        if pos + 6 > len(data):
            break
        rfmt = data[pos]
        rid = data[pos + 1]
        (rsize,) = struct.unpack("<I", data[pos + 2 : pos + 6])
        payload_start = pos - rsize
        if payload_start < 0:
            break
        records.append((rid, rfmt, data[payload_start:pos]))
        offset += rsize + 6
    return records


def _parse_insta360(data: bytes, fps: float, video_size: tuple[int, int] = (0, 0)) -> FileMetadata:
    """Parse Insta360 extra-info telemetry (gyro/accel + inline lens profile).

    Port of upstream telemetry-parser insta360/{mod,record,extra_info}.rs:
    - trailing 72-byte header + record chain (Offsets index or backward walk)
    - Metadata record (protobuf): camera model, ranges, first-frame timestamp,
      rolling-shutter time, offset_v3 (lens intrinsics)
    - Gyro record: [u64 ts][6 x (u16|f64)] = acc xyz + gyro xyz per sample;
      raw files scale by 32768/range, non-raw gyro is rad/s and accel in g
    - timestamps: t -= first_frame_timestamp/1000; raw: t /= 1000;
      t -= gyro_timestamp/1000 (kept in seconds, matching upstream's
      TimeVector3 consumption which multiplies by 1000 for ms)
    """
    import numpy as np

    from pygyroflow.types.time_types import TimeIMU

    metadata = FileMetadata(detected_source="Insta360")

    records = _insta360_collect_records(data)
    if not records:
        log.warning("Insta360: no extra-info records found")
        return metadata

    # ---- Metadata record (id 1, protobuf) ----
    model: str | None = None
    is_raw_gyro = False
    gyro_range = 0.0
    acc_range = 0.0
    first_frame_ts = 0.0
    rolling_shutter_ms = None
    gyro_timestamp = 0.0
    dimension = None
    crop_info = None
    offset_v3: list[float] = []

    for rid, _fmt, payload in records:
        if rid != 1:
            continue
        model = _pb_read_string_field(payload, 2) or None  # camera_type
        is_raw_gyro = _pb_read_varint_field(payload, 62) != 0
        cfg = _pb_extract_field(payload, 65)  # GyroConfigInfo
        if cfg is not None:
            acc_range = float(_pb_read_varint_field(cfg, 1))
            gyro_range = float(_pb_read_varint_field(cfg, 2))
        first_frame_ts = float(_pb_read_varint_field(payload, 24))  # i64
        rst = _pb_read_f64_field(payload, 25)  # rolling_shutter_time
        if abs(rst) > 0.0:
            rolling_shutter_ms = rst
        if _pb_read_varint_field(payload, 29) != 0:  # is_has_gyro_timestamp
            gyro_timestamp = _pb_read_f64_field(payload, 28)
        dim = _pb_extract_field(payload, 19)  # Vector2 dimension
        if dim is not None:
            dimension = (
                _pb_read_varint_field(dim, 1),
                _pb_read_varint_field(dim, 2),
            )
        crop = _pb_extract_field(payload, 27)  # WindowCropInfo
        if crop is not None:
            crop_info = (
                _pb_read_varint_field(crop, 1),
                _pb_read_varint_field(crop, 2),
                _pb_read_varint_field(crop, 3),
                _pb_read_varint_field(crop, 4),
            )
        off_v3_str = _pb_read_string_field(payload, 54)
        if off_v3_str:
            try:
                offset_v3 = [float(v) for v in off_v3_str.split("_")]
            except ValueError:
                offset_v3 = []
        break

    # ---- IMU orientation (upstream model table) ----
    has_offset_v3 = len(offset_v3) >= 20
    if has_offset_v3:
        imu_orientation = {
            "Insta360 GO 2": "XYZ", "Insta360 GO 3": "XYZ",
            "Insta360 GO 3S": "yXZ", "Insta360 GO Ultra": "YxZ",
            "Insta360 OneR": "Xyz", "Insta360 OneRS": "Xyz",
            "Insta360 X4": "yzX", "Insta360 X5": "yzX",
        }.get(model or "", "Xyz")
    else:
        imu_orientation = {
            "Insta360 Go": "xyZ", "Insta360 GO 2": "yXZ",
            "Insta360 OneR": "yXZ", "Insta360 OneRS": "yxz",
            "Insta360 ONE X2": "xZy",
        }.get(model or "", "yXZ")

    # ---- Gyro record (id 3) ----
    RAD_TO_DEG = 180.0 / 3.141592653589793
    G_TO_MS2 = 9.80665
    gyro_scale = 32768.0 / (gyro_range if gyro_range > 0 else 2000.0)
    accl_scale = 32768.0 / (acc_range if acc_range > 0 else 16.0)
    fft = first_frame_ts / 1000.0
    gt = gyro_timestamp / 1000.0

    raw_imu: list[TimeIMU] = []
    for rid, _fmt, payload in records:
        if rid != 3:
            continue
        item_size = 8 + 6 * (2 if is_raw_gyro else 8)
        # Tolerate a trailing partial item: real files exist with a stray
        # byte after the last sample (upstream's strict loop errors out on
        # them and silently drops ALL telemetry via .ok()).
        n = len(payload) // item_size
        for i in range(n):
            base = i * item_size
            (ts,) = struct.unpack("<Q", payload[base : base + 8])
            t = ts / 1000.0
            d = base + 8
            if not is_raw_gyro:
                vals = struct.unpack("<6d", payload[d : d + 48])
                acc = np.array(vals[0:3], dtype=np.float64) * G_TO_MS2
                gyro = np.array(vals[3:6], dtype=np.float64) * RAD_TO_DEG
            else:
                vals = struct.unpack("<6H", payload[d : d + 12])
                acc = (np.array(vals[0:3], dtype=np.float64) - 32768.0) / accl_scale * G_TO_MS2
                gyro = (np.array(vals[3:6], dtype=np.float64) - 32768.0) / gyro_scale

            # Timestamp model from upstream process_map (result in seconds;
            # GyroSource consumes ms, hence * 1000 below)
            t -= fft
            if is_raw_gyro:
                t /= 1000.0
            t -= gt
            raw_imu.append(TimeIMU(timestamp_ms=t * 1000.0, gyro=gyro, accl=acc))

        log.info(
            "Insta360: %d gyro samples (%s format, ranges %s/%s)",
            n, "raw" if is_raw_gyro else "double", gyro_range, acc_range,
        )

    metadata.raw_imu = raw_imu
    metadata.imu_orientation = imu_orientation
    metadata.frame_readout_time = rolling_shutter_ms
    metadata.frame_rate = fps if fps > 0 else None
    metadata.has_accurate_timestamps = True
    if model:
        metadata.detected_source = model if model.startswith("Insta360") else f"Insta360 {model}"

    # ---- Inline lens profile from offset_v3 (upstream insert_lens_profile) ----
    if dimension and crop_info and len(offset_v3) >= 21:
        (w, h) = dimension
        (src_w, src_h, dst_w, dst_h) = crop_info
        (_num, xi, fx, fy, cx, cy, yaw, pitch, roll,
         _tx, _ty, _tz, k1, k2, k3, p1, p2,
         lens_width, lens_height, _lens_type, _flag) = offset_v3[:21]

        bare_model = (model or "").replace("Insta360 ", "")
        cx_fix = 2.0 if bare_model in ("X4", "X5") else 1.0
        c_ratio = (w / lens_width * cx_fix, h / lens_height)
        f_ratio = (dst_w / w, dst_h / h)

        def _out_size(width: int, height: int) -> tuple[int, int]:
            aspect = int(width / height * 100)
            if aspect in (133, 100):
                return width, round(width / 1.7777777777777)
            return width, height

        ow, oh = _out_size(w, h)
        metadata.lens_profile = {
            "calibrated_by": "Insta360",
            "camera_brand": "Insta360",
            "camera_model": bare_model,
            "calib_dimension": {"w": w, "h": h},
            "orig_dimension": {"w": w, "h": h},
            "output_dimension": {"w": ow, "h": oh},
            "frame_readout_time": rolling_shutter_ms,
            "official": True,
            "asymmetrical": True,
            "fisheye_params": {
                "camera_matrix": [
                    [fx / f_ratio[0], 0.0, cx * c_ratio[0]],
                    [0.0, fy / f_ratio[1], cy * c_ratio[1]],
                    [0.0, 0.0, 1.0],
                ],
                "distortion_coeffs": [k1, k2, k3, p1, p2, xi],
            },
            "distortion_model": "insta360",
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

        # Rotate IMU vectors by the mounting angles from offset_v3
        if abs(pitch) > 0.0 or abs(roll) > 0.0 or abs(yaw) > 0.0:
            DEG2RAD = 3.141592653589793 / 180.0
            ry, rp, rr = yaw * DEG2RAD, pitch * DEG2RAD, roll * DEG2RAD
            sry, cry = np.sin(ry), np.cos(ry)
            srp, crp = np.sin(rp), np.cos(rp)
            srr, crr = np.sin(rr), np.cos(rr)
            mat = np.array([
                [crr * crp, crr * srp * sry - srr * cry, crr * srp * cry + srr * sry],
                [srr * crp, srr * srp * sry + cry * cry * 0 + crr * cry, srr * srp * cry - crr * sry],
                [-srp, crp * sry, crp * cry],
            ])
            # upstream row 2 middle term: sy*sp*sr + cy*cr
            mat[1][1] = srr * srp * sry + crr * cry
            for x in raw_imu:
                if x.gyro is not None:
                    x.gyro = mat @ x.gyro
                if x.accl is not None:
                    x.accl = mat @ x.accl

    if raw_imu:
        ts = [x.timestamp_ms for x in raw_imu]
        log.info(
            "Insta360: %d raw IMU readings, span %.1f ms, readout %s ms",
            len(raw_imu), ts[-1] - ts[0] if len(ts) > 1 else 0.0, rolling_shutter_ms,
        )

    return metadata


# ---------------------------------------------------------------------------
# Protobuf helpers (minimal decoder without compiled schema)
# ---------------------------------------------------------------------------

def _pb_read_f32_field(data: bytes, target_field: int) -> float:
    """Read a 32-bit float field (wire type 5) by field number.

    Returns the value of the first occurrence, or 0.0 if not found.
    """
    import struct as _struct

    pos = 0
    while pos < len(data):
        tag, npos = _read_varint(data, pos)
        if tag == 0:
            break
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:
            _, npos = _read_varint(data, npos)
        elif wire_type == 1:
            npos += 8
        elif wire_type == 2:
            length, npos = _read_varint(data, npos)
            npos += length
        elif wire_type == 5:
            if field_num == target_field and npos + 4 <= len(data):
                return _struct.unpack("<f", data[npos:npos + 4])[0]
            npos += 4
        else:
            break
        pos = npos
    return 0.0


def _pb_read_repeated_f32_field(data: bytes, target_field: int) -> list[float]:
    """Read a repeated float field (packed wire type 2, or unpacked type 5)."""
    import struct as _struct

    values: list[float] = []
    pos = 0
    while pos < len(data):
        tag, npos = _read_varint(data, pos)
        if tag == 0:
            break
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:
            _, npos = _read_varint(data, npos)
        elif wire_type == 1:
            npos += 8
        elif wire_type == 2:
            length, npos = _read_varint(data, npos)
            if field_num == target_field:
                chunk = data[npos:npos + length]
                for k in range(0, len(chunk) - 3, 4):
                    values.append(_struct.unpack("<f", chunk[k:k + 4])[0])
            npos += length
        elif wire_type == 5:
            if field_num == target_field and npos + 4 <= len(data):
                values.append(_struct.unpack("<f", data[npos:npos + 4])[0])
            npos += 4
        else:
            break
        pos = npos
    return values


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
