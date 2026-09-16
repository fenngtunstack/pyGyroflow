"""Container-level video metadata that FFmpeg exposes but PyAV does not.

Currently just the display matrix: a phone or action camera records one way
and stores a rotation in the track header, and the player applies it at
playback. Ignoring it renders portrait footage on its side.

Upstream reads it with ``av_display_rotation_get`` over the ``tkhd`` matrix
(ffmpeg_processor.rs) and folds it into ``video_rotation`` as
``(360 - rotation) % 360`` (render_queue.rs).
"""

from __future__ import annotations

import logging
import math
import struct

log = logging.getLogger(__name__)

# ISO/IEC 14496-12 tkhd. Version 0 fields before the matrix are
# version/flags + creation + modification + track_ID + reserved + duration
# + reserved[2] + layer/alternate_group + volume/reserved = 10 int32.
# Version 1 widens the two timestamps and the duration to 64 bits, adding
# three int32.
_MATRIX_OFFSET = {0: 40, 1: 52}
_MATRIX_BYTES = 36  # nine 16.16 fixed-point values
_FIXED_POINT = 65536.0


def _read_box(data: bytes, pos: int, end: int):
    """(fourcc, header_size, box_size) at *pos*, or None."""
    if pos + 8 > end:
        return None
    size = struct.unpack(">I", data[pos : pos + 4])[0]
    fourcc = data[pos + 4 : pos + 8]
    header = 8
    if size == 1:
        if pos + 16 > end:
            return None
        size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
        header = 16
    elif size == 0:
        size = end - pos
    if size < header or pos + size > end:
        return None
    return fourcc, header, size


def _iter_boxes(data: bytes, start: int, end: int):
    pos = start
    while pos < end:
        box = _read_box(data, pos, end)
        if box is None:
            return
        fourcc, header, size = box
        yield fourcc, pos + header, pos + size
        pos += size


def _find_box(data: bytes, path: list[bytes], start: int, end: int):
    """First box matching the fourcc *path*, as (body_start, body_end)."""
    if not path:
        return (start, end)
    for fourcc, body, box_end in _iter_boxes(data, start, end):
        if fourcc == path[0]:
            found = _find_box(data, path[1:], body, box_end)
            if found is not None:
                return found
    return None


def _matrix_rotation(matrix: list[int]) -> float:
    """``av_display_rotation_get`` over nine 16.16 values."""
    def conv(value: int) -> float:
        return value / _FIXED_POINT

    scale_x = math.hypot(conv(matrix[0]), conv(matrix[3]))
    scale_y = math.hypot(conv(matrix[1]), conv(matrix[4]))
    if scale_x == 0.0 or scale_y == 0.0:
        return 0.0
    return -math.degrees(
        math.atan2(conv(matrix[1]) / scale_y, conv(matrix[0]) / scale_x)
    )


def _tkhd_rotation(data: bytes, tkhd_body: int, tkhd_end: int) -> float:
    if tkhd_body + 4 > tkhd_end:
        return 0.0
    version = data[tkhd_body]
    offset = _MATRIX_OFFSET.get(version)
    if offset is None or tkhd_body + offset + _MATRIX_BYTES > tkhd_end:
        return 0.0
    matrix = list(
        struct.unpack(
            ">9i", data[tkhd_body + offset : tkhd_body + offset + _MATRIX_BYTES]
        )
    )
    if matrix == [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824]:
        return 0.0  # identity — skip the trig
    return _matrix_rotation(matrix)


def read_display_rotation(path: str) -> float:
    """Rotation in degrees stored in the video track's display matrix.

    Returns 0.0 when there is no matrix, the file is unreadable, or the
    matrix is the identity. Only the first few hundred KB are read: ``moov``
    sits at the front or the back of an MP4, never in the middle.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        log.debug("Cannot read %s for display rotation: %s", path, exc)
        return 0.0

    try:
        moov = _find_box(data, [b"moov"], 0, len(data))
        if moov is None:
            return 0.0

        fallback = None
        for fourcc, trak_body, trak_end in _iter_boxes(data, *moov):
            if fourcc != b"trak":
                continue
            tkhd = _find_box(data, [b"tkhd"], trak_body, trak_end)
            if tkhd is None:
                continue
            rotation = _tkhd_rotation(data, *tkhd)
            if fallback is None:
                fallback = rotation

            # Prefer the video track: audio tracks carry an identity matrix
            # and would answer 0 for a genuinely rotated file.
            hdlr = _find_box(data, [b"mdia", b"hdlr"], trak_body, trak_end)
            if hdlr is not None and data[hdlr[0] + 8 : hdlr[0] + 12] == b"vide":
                return rotation
        return fallback or 0.0
    except Exception:  # malformed container: rotation is optional
        log.debug("Malformed container while reading rotation: %s", path)
        return 0.0
