"""Quick GoPro GPMF parser — extract GYRO/ACCL/CORI for pipeline verification."""
import struct
import numpy as np
import sys

DEG2RAD = np.pi / 180.0

def extract_gpmf_tag(data, tag, expected_struct_size, sample_fmt):
    samples = []
    tag_bytes = tag.encode("ascii")
    pos = 0
    while True:
        idx = data.find(tag_bytes, pos)
        if idx < 0 or idx + 12 > len(data):
            break
        type_byte = data[idx + 4]
        struct_size = data[idx + 5] | (data[idx + 6] << 8) | (data[idx + 7] << 16)
        repeat = struct.unpack(">I", data[idx + 8 : idx + 12])[0]
        if struct_size == expected_struct_size and type_byte == 0x73:  # int16
            for i in range(repeat):
                off = idx + 12 + i * struct_size
                if off + struct_size <= len(data):
                    vals = struct.unpack(sample_fmt, data[off : off + struct_size])
                    samples.append(vals)
        pos = idx + 4
    return samples


def find_scal(data, tag):
    tag_bytes = tag.encode("ascii")
    idx = data.find(tag_bytes)
    if idx < 0:
        return None
    search_start = max(0, idx - 1024)
    chunk = data[search_start:idx]
    scal_idx = chunk.rfind(b"SCAL")
    if scal_idx < 0:
        return None
    abs_idx = search_start + scal_idx
    if abs_idx + 12 > len(data):
        return None
    type_byte = data[abs_idx + 4]
    struct_size = data[abs_idx + 5] | (data[abs_idx + 6] << 8) | (data[abs_idx + 7] << 16)
    repeat = struct.unpack(">I", data[abs_idx + 8 : abs_idx + 12])[0]
    if type_byte == 0x6C:  # int32
        end = abs_idx + 12 + repeat * 4
        if end <= len(data):
            return struct.unpack(f">{repeat}i", data[abs_idx + 12 : end])
    return None


def parse_gopro_telemetry(path):
    with open(path, "rb") as f:
        data = f.read()

    gyro_raw = extract_gpmf_tag(data, "GYRO", 6, ">hhh")
    accl_raw = extract_gpmf_tag(data, "ACCL", 6, ">hhh")
    gyro_scal = find_scal(data, "GYRO")
    accl_scal = find_scal(data, "ACCL")

    print(f"GYRO: {len(gyro_raw)} raw samples")
    print(f"ACCL: {len(accl_raw)} raw samples")
    print(f"GYRO SCAL: {gyro_scal}")
    print(f"ACCL SCAL: {accl_scal}")

    if not gyro_raw or not gyro_scal:
        print("No GYRO data found")
        return None, None

    gyro = np.array(gyro_raw, dtype=np.float64)
    gyro[:, 0] *= gyro_scal[0]
    gyro[:, 1] *= gyro_scal[1]
    gyro[:, 2] *= gyro_scal[2]

    accl = np.zeros_like(gyro)
    if accl_raw and accl_scal:
        accl = np.array(accl_raw, dtype=np.float64)
        accl[:, 0] *= accl_scal[0]
        accl[:, 1] *= accl_scal[1]
        accl[:, 2] *= accl_scal[2]

    print(f"GYRO (deg/s) range: x=[{gyro[:,0].min():.1f}, {gyro[:,0].max():.1f}]")
    print(f"ACCL (m/s²) first 3: {accl[:3]}")
    return gyro, accl


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/ft/workspace/PreReserach/msGyroFlow/GX010045.MP4"
    gyro, accl = parse_gopro_telemetry(path)
    if gyro is not None:
        print(f"\nParsed OK: {len(gyro)} gyro samples")
