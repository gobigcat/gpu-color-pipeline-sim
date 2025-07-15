# GPU Color Pipeline Simulation

This project simulates a simplified GPU color processing pipeline with an emphasis on 3DLUT (3D Look-Up Table) based tone mapping and color correction techniques.

Inspired by real-world performance bottlenecks in color processing systems, this simulation demonstrates how LUT layout and access pattern optimization can affect throughput.

## 🔍 Features

- Simulated 3DLUT application for pixel color transformation
- LUT input/output formats (supporting FP16/FP32 range)
- Configurable data layout (linear vs interleaved)
- Throughput measurement: simulate clock-based timing
- Visual pipeline diagram (see `pipeline_diagram.png`)
- Clean, modular Python structure with utilities

## 📊 Goal

Illustrate the effect of memory layout and cache behavior on GPU color processing throughput. In one setup, we aim to improve LUT access throughput from `0.6 pixel/clk` to `0.8 pixel/clk` by restructuring data layout.

## 💡 Future Plans

- Add signal modeling and noise injection
- Extend to tone mapping simulation (HDR10 → SDR)
- Compare 1DLUT vs 3DLUT behavior

## 🛠 Requirements

- Python 3.8+
- numpy, matplotlib (for visualization)

## 📄 License

MIT License
