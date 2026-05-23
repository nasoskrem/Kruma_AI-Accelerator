# KRUMA - High-Performance CPU AI Core

![Version](https://img.shields.io/badge/version-0.3-blue)
![Rust](https://img.shields.io/badge/Made%20With-Rust-informational)
![License](https://img.shields.io/badge/License-MIT-success)

**KRUMA** is a low-level, high-performance **CPU-based AI core** built in Rust. Its primary goal is to provide N-dimensional tensor operations, stateful neural network abstractions, and automatic differentiation algorithms from scratch, without relying on external BLAS libraries or GPU acceleration. 

By leveraging **SIMD vectorization (128-bit `f32x4`)**, **cache-aware memory access**, and **thread-level parallelism**, KRUMA achieves speedups over naive CPU implementations.


## Core Architecture & Features (v0.3)

### Hardware Acceleration (backend/cpu.rs)
- **Cache-Aware Matrix Multiplication**: Utilizes parallel transposition of matrix B to prevent cache misses, combined with **4x4 Register Blocking** to minimize memory loads.
- **SIMD Vectorization**: Heavy inner loops are unsafely cast to `f32x4` slices, processing 4 floating-point operations per CPU cycle via the `wide` crate.
- **Fearless Concurrency**: Embarrassingly parallel operations (element-wise functions, row-reductions) are dynamically distributed across all logical cores using the `rayon` work-stealing thread pool.

### Deep Learning Modules (nn, optim, loss)
- **Network Abstractions (nn)**:
  - `Sequential` container mapping the multivariate Chain Rule for seamless forward/backward passes.
  - `Linear` fully connected layers with batch-aware gradient calculation and He-inspired variance scaling.
  - **Regularization**: `Dropout` layer utilizing an inverted dropout technique and a custom LCG pseudo-RNG for deterministic training limits.
  - **Activations**: `Tanh`, `Sigmoid`, and `ReLU`.
- **Optimization (optim)**:
  - **Adam Optimizer**: Stable implementation tracking exponential moving averages of gradients (Momentum) and squared gradients (Variance). Features out-of-loop bias correction and strict gradient clamping (`[-1.0, 1.0]`) to prevent exploding parameters.
- **Loss Functions (loss)**:
  - **`CrossEntropyLoss`**: Implements the numerical **max-shift trick** (`log_softmax`) to prevent `f32` overflow/underflow during multi-class probability normalization.
  - **`MSELoss`**: Mean Squared Error for continuous regression and binary tasks.

### Tensor Engine (tensor.rs)
- N-Dimensional coordinate mapping to contiguous 1D heap allocations (`DeviceBuffer`).
- $O(1)$ memory broadcasting by artificially manipulating spatial `strides` to 0.


## Experiments

KRUMA includes three functional proofs of the engine's capabilities, defined as separate binaries in the `Cargo.toml`.

**1. SIMD Matrix Multiplication Benchmark**
Compares KRUMA's vectorized CPU backend against a naive triple-loop implementation on $1024 \times 1024$ matrices, calculating GFLOPS and verifying precision.
```
cargo run --release --bin matmul
```

**2. Simple XOR**
```
cargo run --release --bin xor
```

**3. Multi-Class Classification**
```
cargo run --release --bin classes
```