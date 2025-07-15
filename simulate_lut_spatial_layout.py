import numpy as np

def generate_spatially_local_3dlut(size=17):
    # Create a 3DLUT and flatten it using a Z-order curve (Morton order) approximation
    # to simulate spatial locality: nearby RGB values map to nearby memory
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    for r in range(size):
        for g in range(size):
            for b in range(size):
                lut[r, g, b] = [r / (size - 1), g / (size - 1), b / (size - 1)]
    # Return flattened LUT reshaped by z-order-like traversal
    flat_lut = []
    for level in range(2 * size):  # simple locality-aware traversal
        for r in range(size):
            for g in range(size):
                b = level - r - g
                if 0 <= b < size:
                    flat_lut.append(lut[r, g, b])
    return np.array(flat_lut, dtype=np.float32)

def apply_flat_lut(rgb_input, lut, size=17):
    rgb_scaled = np.clip(rgb_input, 0, 1) * (size - 1)
    r, g, b = rgb_scaled.astype(int)
    # Compute approximate linear index using priority to low RGB values
    linear_index = r * size * size + g * size + b
    linear_index = np.clip(linear_index, 0, len(lut) - 1)
    return lut[linear_index]

# Example usage
if __name__ == "__main__":
    size = 17
    lut = generate_spatially_local_3dlut(size)
    test_pixel = np.array([0.5, 0.25, 0.75])  # RGB input in [0, 1]
    output_pixel = apply_flat_lut(test_pixel, lut, size)
    print(f"Input RGB: {test_pixel}")
    print(f"Output RGB after spatially-local 3DLUT: {output_pixel}")
