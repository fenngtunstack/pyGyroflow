# -*- coding: utf-8 -*-
"""Swap our correction quaternions into the official .gyroflow project and
render with the official Gyroflow binary — closing the differential loop.

official renderer + official correction = 0.12px  (known)
our renderer       + official correction = 2.5px   (measured, offq3)
official renderer + OUR correction       = this experiment
"""

from __future__ import annotations

import json
import subprocess
import sys
import zlib

import numpy as np

from tests.decode_gyroflow_project import B91_ALPHABET, b91decode


def b91encode(data: bytes) -> str:
    b = 0
    n = 0
    out = []
    for byte in data:
        b |= byte << n
        n += 8
        if n > 13:
            v = b & 8191
            if v > 88:
                b >>= 13
                n -= 13
            else:
                v = b & 16383
                b >>= 14
                n -= 14
            out.append(B91_ALPHABET[v % 91] + B91_ALPHABET[v // 91])
    if n:
        out.append(B91_ALPHABET[b % 91])
        if n > 7 or b > 90:
            out.append(B91_ALPHABET[b // 91])
    return "".join(out)


def encode_cbor_field(value) -> str:
    import cbor2 as cbor

    data = cbor.dumps(value)
    comp = zlib.compress(data, 9)
    return b91encode(comp)


def main() -> int:
    import cbor2 as cbor
    from pygyroflow.manager import StabilizationManager

    # our correction with Plain 2.04 (same as offq3 state)
    m = StabilizationManager()
    m.load_video("../DJI_20260507160359_0005_D.MP4")
    m.set_smoothing_method(2)
    m.set_smoothing_param("time_constant", 2.04)
    m.set_adaptive_zoom(4.0)
    m.recompute_blocking()

    our_corr = m.gyro.smoothed_quaternions
    keys = sorted(our_corr.keys())
    print("our correction:", len(keys), "entries")
    payload = {}
    for k in keys:
        q = our_corr[k].quaternion()  # wxyz
        payload[int(k)] = [q[1], q[2], q[3], q[0]]  # xyzw like official
    enc = encode_cbor_field(payload)
    print("encoded len:", len(enc))

    # verify round trip
    back = cbor.loads(zlib.decompress(b91decode(enc)))
    k0 = keys[0]
    assert list(back[str(k0)] if isinstance(list(back.keys())[0], str) else back[keys[0]]) == payload[keys[0]], "roundtrip mismatch"
    print("roundtrip OK")

    d = json.load(open(r'..\DJI_20260507160359_0005_D.gyroflow', encoding='utf-8'))
    d['gyro_source']['smoothed_quaternions'] = enc
    # point at the same input, output to a new suffix via project output filename
    d['output']['output_filename'] = 'DJI_20260507160359_0005_D_ourcorr.mp4'
    json.dump(d, open(r'..\DJI_swap.gyroflow', 'w', encoding='utf-8'))
    print("wrote ../DJI_swap.gyroflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
