# KRUMA — High-Performance CPU AI Core

![Version](https://img.shields.io/badge/version-0.2.0-blue)
![Rust](https://img.shields.io/badge/Made%20With-Rust-informational)

**KRUMA** is a low-level, high-performance **CPU-based AI core** built in Rust. Its primary goal is to provide fundamental tensor operations and network abstractions similar to larger frameworks, optimized specifically for standard multi-core CPUs. It leverages **SIMD vectorization** and **multithreading** to achieve significant performance gains.

---

## ✨ Features (v0.2.0)

### 🚀 Core Acceleration (`backend/cpu.rs`)
- **Matrix Multiplication (`matmul`)**: Optimized with **Parallel Transposition**, **4x4 Register Blocking**, and **SIMD (f32x4)** for efficient compute on large tensors.
- **Parallelism**: Multithreaded execution via `rayon` is utilized across all key tensor and backend operations.
- **Key Operations**: Stable, parallelized implementations of `relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, and `sum_columns` (reduction).

### 🧠 Deep Learning Modules (`nn`, `optim`, `loss`)
- **Network Abstractions (`nn`)**:
  - `Sequential` container for creating modular, stacked network architectures.
  - `Linear` Layer with batch-aware gradient calculation using `sum_axis0`.
  - **Regularization**: `Dropout` layer (uses pseudo-RNG for training).
  - **Activations**: `Tanh`, `Sigmoid`, and `ReLU`.
- **Optimization (`optim`)**:
  - **Adam** (Adaptive Moment Estimation) optimizer for adaptive learning rate control.
- **Loss Functions (`loss`)**:
  - **`CrossEntropyLoss`** (for stable multi-class classification).
  - `MSELoss` (Mean Squared Error).

### 📐 Tensor Utilities
- Fundamental ops: `add`, `sub`, `mul`, `transpose`, `sum_axis0`.
- Utility: `argmax` (for extracting class predictions).

---

## 📦 Usage Example (Multi-Class Classification)

This demonstrates a full training loop using Sequential, Adam, and CrossEntropyLoss.

```rust
use kruma::tensor::Tensor;
use kruma::nn::{Linear, Tanh, Sequential, Dropout};
use kruma::optim::Adam;
use kruma::loss::CrossEntropyLoss;

// 1. Define Model Architecture
let mut model = Sequential::new(vec![
    Box::new(Linear::new(3, 8)),
    Box::new(Tanh::new()),
    Box::new(Dropout::new(0.2)), 
    Box::new(Linear::new(8, 3)),
]);

// 2. Setup Optimizer & Data
let mut optimizer = Adam::new(0.01);
let criterion = CrossEntropyLoss;
let x_train = Tensor::from_data([6, 3], /* ... data ... */);
let y_train = Tensor::from_data([6, 3], /* ... targets ... */);

// 3. Training Step
let logits = model.forward(&x_train);
let loss = criterion.forward(&logits, &y_train);

optimizer.zero_grad(&mut model);
let grad = criterion.backward(&logits, &y_train);
model.backward(&grad);
optimizer.step(&mut model);

println!("Loss: {}", loss);