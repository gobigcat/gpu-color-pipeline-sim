import numpy as np
import time
import pandas as pd

def generate_identity_lut(size=17):
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    for r in range(size):
        for g in range(size):
            for b in range(size):
                lut[r, g, b] = [r / (size - 1), g / (size - 1), b / (size - 1)]
    return lut

def generate_flat_lut_from_identity(lut):
    size = lut.shape[0]
    flat_lut = []
    for level in range(2 * size):
        for r in range(size):
            for g in range(size):
                b = level - r - g
                if 0 <= b < size:
                    flat_lut.append(lut[r, g, b])
    return np.array(flat_lut, dtype=np.float32)

def apply_3dlut_standard(rgb_inputs, lut):
    size = lut.shape[0]
    results = []
    for rgb in rgb_inputs:
        r_idx, g_idx, b_idx = (np.clip(rgb, 0, 1) * (size - 1)).astype(int)
        results.append(lut[r_idx, g_idx, b_idx])
    return results

def apply_3dlut_flat(rgb_inputs, lut, size=17):
    results = []
    for rgb in rgb_inputs:
        r, g, b = (np.clip(rgb, 0, 1) * (size - 1)).astype(int)
        linear_index = r * size * size + g * size + b
        linear_index = min(linear_index, len(lut) - 1)
        results.append(lut[linear_index])
    return results

def benchmark_lut_throughput(size=65, num_pixels=100_000):
    rgb_data = np.random.rand(num_pixels, 3).astype(np.float32)
    identity_lut = generate_identity_lut(size)
    flat_lut = generate_flat_lut_from_identity(identity_lut)

    start_std = time.time()
    _ = apply_3dlut_standard(rgb_data, identity_lut)
    end_std = time.time()
    throughput_std = num_pixels / (end_std - start_std)

    start_flat = time.time()
    _ = apply_3dlut_flat(rgb_data, flat_lut, size)
    end_flat = time.time()
    throughput_flat = num_pixels / (end_flat - start_flat)

    df = pd.DataFrame({
        "LUT Type": ["Standard 3D LUT", "Spatially-Local Flat LUT"],
        "Throughput (pixels/sec)": [throughput_std, throughput_flat],
        "Time (s)": [end_std - start_std, end_flat - start_flat]
    })

    print(df)

if __name__ == "__main__":
    benchmark_lut_throughput()
