import numpy as np
import time
import pandas as pd

def generate_identity_lut(size=65):
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    for r in range(size):
        for g in range(size):
            for b in range(size):
                lut[r, g, b] = [r / (size - 1), g / (size - 1), b / (size - 1)]
    return lut

def flatten_lut_row_major(lut):
    return lut.reshape(-1, 3)

def apply_3dlut_standard(rgb_inputs, lut):
    size = lut.shape[0]
    results = []
    for rgb in rgb_inputs:
        r_idx, g_idx, b_idx = (np.clip(rgb, 0, 1) * (size - 1)).astype(int)
        results.append(lut[r_idx, g_idx, b_idx])
    return results

def apply_flat_lut_row_major(rgb_inputs, lut_flat, size=65):
    results = []
    for rgb in rgb_inputs:
        r, g, b = (np.clip(rgb, 0, 1) * (size - 1)).astype(int)
        linear_index = r * size * size + g * size + b
        results.append(lut_flat[linear_index])
    return results

def benchmark_lut_throughput(size=33, num_pixels=100_000):
    rgb_data = np.random.rand(num_pixels, 3).astype(np.float32)
    identity_lut = generate_identity_lut(size)
    flat_lut = flatten_lut_row_major(identity_lut)

    start_std = time.time()
    _ = apply_3dlut_standard(rgb_data, identity_lut)
    end_std = time.time()
    throughput_std = num_pixels / (end_std - start_std)

    start_flat = time.time()
    _ = apply_flat_lut_row_major(rgb_data, flat_lut, size)
    end_flat = time.time()
    throughput_flat = num_pixels / (end_flat - start_flat)

    df = pd.DataFrame({
        "LUT Type": ["Standard 3D LUT", "Row-Major Flat LUT"],
        "Throughput (pixels/sec)": [throughput_std, throughput_flat],
        "Time (s)": [end_std - start_std, end_flat - start_flat]
    })

    print(df)

if __name__ == "__main__":
    benchmark_lut_throughput()
