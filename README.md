# KRUMA - High-Performance CPU AI Core

![Version](https://img.shields.io/badge/version-0.3.1-blue)
![Rust](https://img.shields.io/badge/Made%20With-Rust-informational)
![License](https://img.shields.io/badge/License-MIT-success)

**KRUMA** is a low-level, high-performance **CPU-based AI core** built in Rust. Its primary goal is to provide N-dimensional tensor operations, stateful neural network abstractions, and automatic differentiation algorithms from scratch, without relying on external BLAS libraries or GPU acceleration. 

## Core Architecture & Features (v0.3.1)

### Hardware Acceleration (backend/cpu.rs)
- **Cache-Aware Matrix Multiplication**: Utilizes parallel transposition of matrix B to prevent cache misses, combined with **4x4 Register Blocking** to minimize redundant memory loads.
- **SIMD Vectorization**: Heavy inner loops are unsafely cast to `f32x4` slices, processing 4 floating-point operations per CPU cycle via the `wide` crate.
- **Concurrency**: Parallel operations are dynamically distributed across all logical cores using the `rayon` thread pool.

### Deep Learning Modules (nn, optim, loss)
- **Network Abstractions (nn)**:
  - `Sequential` container mapping the multivariate Chain Rule for seamless forward/backward passes.
  - `Linear` fully connected layers with batch-aware gradient calculation and He-inspired (Kaiming) variance scaling.
  - **Regularization**: `Dropout` layer utilizing an inverted dropout technique and a custom LCG pseudo-RNG for deterministic training limits.
  - **Activations**: `Tanh` (stateful layer caching outputs for backward pass), `Sigmoid`, and `ReLU`.
- **Optimization (optim)**:
  - **Adam Optimizer**: Stable implementation tracking exponential moving averages of gradients and variances. Features algorithmic optimizations by factoring out bias correction from the inner loop, in-place state updates (`update_first_moment_inplace`), and strict gradient clamping (`[-1.0, 1.0]`) to prevent exploding parameters.
- **Loss Functions (loss)**:
  - **`CrossEntropyLoss`**: Implements the numerical **max-shift trick** (`log_softmax`) to prevent `f32` overflow/underflow. Utilizes fused parallel sums to avoid intermediate tensor allocations.
  - **`MSELoss`**: Features fused zero-allocation parallel reduction for high-efficiency continuous regression tasks.

### Tensor Engine (tensor.rs)
- N-Dimensional coordinate mapping to contiguous 1D heap allocations (`DeviceBuffer`).
- O(1) memory space broadcasting by artificially manipulating spatial `strides` to 0.

## Comparison between KRUMA & PyTorch

The repository includes a Jupyter Notebook (`experiments/exp_pytorch.ipynb`) designed to run comparative analytics between KRUMA's Rust binaries and identical PyTorch routines. 

The notebook captures standard output from `cargo run` and generates plots via `matplotlib` for two primary benchmarks:
1. **Matrix Multiplication Benchmark (1024x1024)**: Compares compute time (in seconds) between KRUMA's vectorized CPU backend and PyTorch's matrix multiplication.
2. **XOR Challenge Convergence**: Tracks and plots the MSE loss over 500 epochs to verify that KRUMA's automatic differentiation and Adam optimizer mathematically converge identically to PyTorch's implementation.

## Experiments

**1. SIMD Matrix Multiplication Benchmark**
Compares KRUMA's vectorized CPU backend against a naive triple-loop implementation on 1024x1024 matrices, calculating GFLOPS and verifying floating-point precision tolerance.
```
cargo run --release --bin matmul
```

**2. Simple XOR**
```
cargo run --release --bin xor
```

**3. Multi-Class Classification**
Trains a 3-layer neural network with Dropout and Tanh activations on a synthetic dataset.
```
cargo run --release --bin main
```