"""Parse GoPro GPMF - find the DEVC containing GYRO and extract samples."""
import struct
import numpy as np
import sys

def main(path):
    with open(path, "rb") as f:
        data = f.read()

    # Strategy: find GYRO tags, then walk backward to find the enclosing DEVC/STRM
    # Extract all GYRO and ACCL data from all GPMF blocks in the file

    # First, find all GPMF top-level boxes (DEVC containers)
    # GoPro stores one DEVC per video frame in mdat

    # Find all GYRO occurrences
    gyro_offsets = []
    pos = 0
    while True:
        idx = data.find(b"GYRO", pos)
        if idx < 0:
            break
        gyro_offsets.append(idx)
        pos = idx + 4

    print(f"Found {len(gyro_offsets)} GYRO tags")

    # For each GYRO, parse the KLV right after the tag
    # GPMF KLV: FourCC(4) + Type(1) + StructSize(3) + Repeat(4) + Payload
    all_gyro = []
    all_accl = []
    gyro_scale = None
    accl_scale = None

    for gyro_off in gyro_offsets[:3]:  # Debug first 3
        # The GYRO tag
        tag = data[gyro_off:gyro_off+4]
        type_byte = data[gyro_off+4]
        struct_size = data[gyro_off+5] | (data[gyro_off+6] << 8) | (data[gyro_off+7] << 16)
        repeat = struct.unpack(">I", data[gyro_off+8:gyro_off+12])[0]
        payload_size = struct_size * repeat
        print(f"\nGYRO at {gyro_off}: type={type_byte:#04x} struct_size={struct_size} repeat={repeat} payload={payload_size}")

        # Parse GYRO samples if struct_size matches int16x3 = 6 bytes
        if struct_size == 6 and type_byte == 0x73:
            for i in range(min(repeat, 5)):
                off = gyro_off + 12 + i * 6
                x, y, z = struct.unpack(">hhh", data[off:off+6])
                print(f"  raw[{i}]: ({x}, {y}, {z})")

        # Find SCAL for this GYRO - search backward up to 512 bytes
        search_start = max(0, gyro_off - 512)
        chunk = data[search_start:gyro_off]
        scal_pos = chunk.rfind(b"SCAL")
        if scal_pos >= 0:
            abs_scal = search_start + scal_pos
            st = data[abs_scal+5] | (data[abs_scal+6] << 8) | (data[abs_scal+7] << 16)
            rp = struct.unpack(">I", data[abs_scal+8:abs_scal+12])[0]
            tp = data[abs_scal+4]
            print(f"  SCAL at {abs_scal}: type={tp:#04x} struct_size={st} repeat={rp}")
            if tp == 0x6c:  # int32
                n = min(rp, 4)
                vals = struct.unpack(f">{n}i", data[abs_scal+12:abs_scal+12+n*4])
                print(f"  SCAL values: {vals}")
                if gyro_scale is None:
                    gyro_scale = vals

    # Now extract all samples properly
    all_gyro_samples = []
    for off in gyro_offsets:
        type_byte = data[off+4]
        struct_size = data[off+5] | (data[off+6] << 8) | (data[off+7] << 16)
        repeat = struct.unpack(">I", data[off+8:off+12])[0]
        if struct_size == 6 and type_byte == 0x73:
            for i in range(repeat):
                sample_off = off + 12 + i * 6
                if sample_off + 6 <= len(data):
                    x, y, z = struct.unpack(">hhh", data[sample_off:sample_off+6])
                    all_gyro_samples.append((x, y, z))

    print(f"\nTotal GYRO samples: {len(all_gyro_samples)}")

    if all_gyro_samples and gyro_scale:
        gyro = np.array(all_gyro_samples, dtype=np.float64)
        gyro[:, 0] *= gyro_scale[0]
        gyro[:, 1] *= gyro_scale[1]
        gyro[:, 2] *= gyro_scale[2]
        print(f"GYRO (deg/s) first 5: {gyro[:5]}")
        print(f"GYRO shape: {gyro.shape}")
        print(f"GYRO range: x=[{gyro[:,0].min():.1f}, {gyro[:,0].max():.1f}] y=[{gyro[:,1].min():.1f}, {gyro[:,1].max():.1f}] z=[{gyro[:,2].min():.1f}, {gyro[:,2].max():.1f}]")
        return gyro
    return None

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/ft/workspace/PreReserach/msGyroFlow/GX010045.MP4"
    result = main(path)
    if result is not None:
        print(f"\nOK: {len(result)} gyro samples extracted")
