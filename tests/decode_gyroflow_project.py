# -*- coding: utf-8 -*-
"""Decode Gyroflow .gyroflow project fields (base91+zlib+bincode/cbor)."""

from __future__ import annotations

import json
import struct
import zlib

B91_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!#$%&()*+,./:;<=>?@[]^_`{|}~\""
)
B91_DECODE = {c: i for i, c in enumerate(B91_ALPHABET)}


def b91decode(data: str) -> bytes:
    v = -1
    b = 0
    n = 0
    out = bytearray()
    for ch in data:
        if ch not in B91_DECODE:
            continue
        c = B91_DECODE[ch]
        if v < 0:
            v = c
        else:
            v += c * 91
            b |= v << n
            n += 13 if (v & 8191) > 88 else 14
            while n >= 8:
                out.append(b & 255)
                b >>= 8
                n -= 8
            v = -1
    if v >= 0:
        out.append((b | v << n) & 255)
    return bytes(out)


def decode_field(s: str):
    """Returns decompressed bytes of a 'q:'-prefixed field."""
    if s.startswith("q:"):
        s = s[2:]
    return zlib.decompress(b91decode(s))


def parse_quat_list_bincode(b: bytes):
    """bincode legacy Vec<(i64, [f64; 4])> -> [(ts_us, w, x, y, z), ...]"""
    (n,) = struct.unpack_from("<Q", b, 0)
    off = 8
    out = []
    for _ in range(n):
        ts, w, x, y, z = struct.unpack_from("<qdddd", b, off)
        out.append((ts, w, x, y, z))
        off += 8 + 32
    return out


def parse_f64_list_bincode(b: bytes):
    (n,) = struct.unpack_from("<Q", b, 0)
    off = 8
    out = []
    for _ in range(n):
        (v,) = struct.unpack_from("<d", b, off)
        out.append(v)
        off += 8
    return out


def main() -> int:
    d = json.load(open(r'..\DJI_20260507160359_0005_D.gyroflow', encoding='utf-8'))
    gs = d["gyro_source"]

    sq = parse_quat_list_bincode(decode_field(gs["smoothed_quaternions"]))
    iq = parse_quat_list_bincode(decode_field(gs["integrated_quaternions"]))
    try:
        fovs = parse_f64_list_bincode(decode_field(gs["adaptive_zoom_fovs"]))
    except Exception:
        fovs = []
    ts_sync = decode_field(gs["synced_imu_timestamps"])

    print(f"smoothed n={len(sq)}   first={sq[0]}  last={sq[-1]}")
    print(f"integrated n={len(iq)} first={iq[0]}")
    print(f"fovs n={len(fovs)} first={fovs[:5]}")

    # save as npz for downstream comparison
    import numpy as np
    np.savez(
        r'..\official_processed.npz',
        smoothed_ts=np.array([r[0] for r in sq]),
        smoothed_q=np.array([[r[1], r[2], r[3], r[4]] for r in sq]),
        integrated_ts=np.array([r[0] for r in iq]),
        integrated_q=np.array([[r[1], r[2], r[3], r[4]] for r in iq]),
        fovs=np.array(fovs),
        ts_sync_raw=np.frombuffer(ts_sync, dtype=np.uint8),
    )
    print("saved ../official_processed.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
