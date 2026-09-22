"""The Gyroflow GCSV text-IMU format (A-02's text half).

``.gcsv`` is *the* external-IMU interchange format — a camera without
built-in telemetry, logged by a phone or a dedicated logger, lands here.
Port of ``telemetry-parser/src/gyroflow/gcsv.rs``:

* the header block (``key,value`` rows until the ``t,...`` data header);
* ``tscale`` for the time column, ``gscale``/``ascale``/``mscale`` as
  **divisors** (upstream multiplies by their reciprocals — 1/gscale —
  and the gyro additionally by π/180);
* the readout-direction encoding into a signed
  ``frame_readout_time`` (the ``+10000`` sentinel marks horizontal
  readouts, which the consumer splits back out);
* gyro columns 1-3, accelerometer 4-6, magnetometer 7-9, all optional
  by row width.
"""

from __future__ import annotations

import csv
import io
import math
import logging

from pygyroflow.gyro_source.file_metadata import FileMetadata
from pygyroflow.types.enums import ReadoutDirection
from pygyroflow.types.time_types import TimeIMU

logger = logging.getLogger(__name__)

_GCSV_EXTENSIONS = ("gcsv", "csv", "txt", "bin")

# Upstream gcsv.rs:63-76 — the signed readout-time encoding.
_READOUT_DIRECTION_BY_CODE = {
    ("0", "TopToBottom"): (1.0, ReadoutDirection.TopToBottom),
    ("1", "180", "BottomToTop"): (-1.0, ReadoutDirection.BottomToTop),
    ("2", "270", "LeftToRight"): (1.0, ReadoutDirection.LeftToRight),
    ("3", "90", "RightToLeft"): (-1.0, ReadoutDirection.RightToLeft),
}


def detect_gcsv(buffer: bytes) -> bool:
    """The ``GYROFLOW IMU LOG`` / ``CAMERA IMU LOG`` first line
    (``gcsv.rs:33-37``)."""
    return buffer.startswith(b"GYROFLOW IMU LOG") or buffer.startswith(
        b"CAMERA IMU LOG"
    )


def _parse_readout_time(header: dict[str, str]):
    """gcsv.rs:50-63: absent → None; the sentinel arithmetic stays in the
    value (the +10000 marks horizontal) with the direction split out."""
    if "frame_readout_time" not in header:
        return None, ReadoutDirection.TopToBottom
    direction = header.pop("frame_readout_direction", "0")
    readout = float(header.pop("frame_readout_time", "0.0") or 0.0)
    for codes, (sign, enum_value) in _READOUT_DIRECTION_BY_CODE.items():
        if direction in codes:
            horizontal = 10000.0 if enum_value.is_horizontal() else 0.0
            return sign * (readout + horizontal), enum_value
    return None, ReadoutDirection.TopToBottom


def parse_gcsv(data: bytes) -> FileMetadata:
    """Parse a complete ``.gcsv`` payload into :class:`FileMetadata`."""
    header: dict[str, str] = {}
    gyro: list[tuple[float, float, float, float]] = []
    accl: list[tuple[float, float, float, float]] = []
    magn: list[tuple[float, float, float, float]] = []

    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), skipinitialspace=True)
    time_scale = 0.001  # default to millisecond
    passed_header = False

    for row in reader:
        if not row:
            continue
        stripped = [c.strip() for c in row]
        if len(stripped) == 1:
            continue  # the first line
        if len(stripped) == 2 and not passed_header:
            header[stripped[0]] = stripped[1]
            continue
        if stripped[0] in ("t", "time") and not passed_header:
            passed_header = True
            try:
                time_scale = float(header.pop("tscale", "0.001"))
            except ValueError:
                time_scale = 0.001
            continue
        try:
            t = float(stripped[0]) * time_scale
        except ValueError:
            logger.debug("gcsv: bad time column %r", stripped[0])
            continue
        if len(stripped) >= 4:
            gyro.append((t, _f(stripped, 1), _f(stripped, 2), _f(stripped, 3)))
        if len(stripped) >= 7:
            accl.append((t, _f(stripped, 4), _f(stripped, 5), _f(stripped, 6)))
        if len(stripped) >= 10:
            magn.append((t, _f(stripped, 7), _f(stripped, 8), _f(stripped, 9)))

    def scale_of(key: str, default: str = "1.0") -> float:
        try:
            return float(header.pop(key, default))
        except ValueError:
            return float(default)

    gyro_scale = 1.0 / scale_of("gscale") * math.pi / 180.0
    accl_scale = 1.0 / scale_of("ascale")
    mag_scale = 100.0 / scale_of("mscale")  # Gauss to microtesla
    imu_orientation = header.pop("orientation", "xzY")
    lensprofile = header.pop("lensprofile", None)

    meta = FileMetadata(
        detected_source="GCSV",
        imu_orientation=imu_orientation,
        additional_data={"vendor": header.pop("vendor", "gcsv"),
                         "extra": dict(header)},
    )
    readout_time, direction = _parse_readout_time(header)
    if readout_time is not None:
        meta.frame_readout_time = readout_time
        meta.frame_readout_direction = direction
    if lensprofile:
        meta.lens_profile = lensprofile

    import numpy as np

    raw_imu: list[TimeIMU] = []
    a_by_t = {t: (x, y, z) for t, x, y, z in accl}
    for t, x, y, z in gyro:
        # gcsv time is seconds (after tscale); TimeIMU is milliseconds.
        raw_imu.append(TimeIMU(
            timestamp_ms=t * 1000.0,
            gyro=np.array([x * gyro_scale, y * gyro_scale, z * gyro_scale]),
            accl=(np.array(a_by_t[t]) * accl_scale
                  if t in a_by_t else None),
        ))
    meta.raw_imu = raw_imu
    if magn:
        meta.additional_data["magnetometer"] = [
            (t * 1000.0, x * mag_scale, y * mag_scale, z * mag_scale)
            for t, x, y, z in magn
        ]
    return meta


def _f(row: list[str], index: int) -> float:
    try:
        return float(row[index])
    except (ValueError, IndexError):
        return 0.0
