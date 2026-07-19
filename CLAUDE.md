# KRUMA — High-Performance CPU AI Core

KRUMA is a low-level, from-scratch **CPU-based AI core in Rust**: N-dimensional
tensors, stateful neural-network layers, and reverse-mode autodiff, with **no
external BLAS and no GPU**. Performance comes from cache-aware matmul, SIMD, and
`rayon` parallelism.

- Crate name (library): `kruma` (see `src/lib.rs`)
- Package name: `Kruma_AI-Accelerator`
- Rust edition: **2024**
- Current version: **0.3.1**

## Build / run / test commands

```bash
cargo build                       # debug build (fast, for iteration)
cargo build --release             # optimized build (use for any benchmark)
cargo test                        # compile everything + run tests
cargo run --release --bin main    # multi-class classification demo (500 epochs)
cargo run --release --bin xor     # XOR convergence demo
cargo run --release --bin matmul  # SIMD matmul benchmark vs naive triple-loop
cargo run --release --bin classes # synthetic multi-class classification
```

Always use `--release` for anything performance-related — the SIMD/blocking work
is meaningless in a debug build.

> Note: there are currently no `#[test]` functions in the tree, so `cargo test`
> today mainly acts as a full compile check across the library **and all four
> experiment binaries**. As tests are added they run automatically.

## Layout

| Path | What lives here |
|---|---|
| `src/lib.rs` | Crate root; re-exports the modules below |
| `src/tensor.rs` | N-D `Tensor` over a contiguous 1D `DeviceBuffer`; stride-based broadcasting (strides set to 0 for O(1) broadcast) |
| `src/backend/cpu.rs` | `CpuBackend`: cache-aware matmul (parallel transpose of B + 4×4 register blocking) |
| `src/backend/simd.rs` | `f32x4` (via `wide`) vectorized inner loops |
| `src/backend/traits.rs` | Backend trait definitions |
| `src/backend/mod.rs` | Re-exports `CpuBackend` and traits |
| `src/nn.rs` | `Module` trait; `Sequential`, `Linear`, `Tanh`, `Sigmoid`, `ReLU`, `Dropout` |
| `src/optim.rs` | `Adam` optimizer (bias correction factored out of the inner loop, in-place moment updates, grad clamping to `[-1, 1]`) |
| `src/loss.rs` | `CrossEntropyLoss` (log-softmax max-shift trick), `MSELoss` (fused zero-alloc reduction) |
| `src/utils.rs` | Shared helpers |
| `experiments/` | Standalone `bin` targets + `exp_pytorch.ipynb` (KRUMA vs PyTorch comparison) |

Binary targets are declared explicitly in `Cargo.toml` (`main`, `matmul`, `xor`,
`classes`) — the experiment sources live under `experiments/`, not `src/bin/`.

## Conventions & gotchas

- **`f32` throughout.** Numerical tricks (log-softmax max-shift, grad clamping)
  exist to keep `f32` stable — preserve them when refactoring.
- **`unsafe` in the SIMD path.** `backend/simd.rs` casts slices to `f32x4`.
  Any change there must keep length/alignment assumptions intact.
- **Determinism.** `Dropout` uses a custom LCG pseudo-RNG so training is
  reproducible; don't swap in a nondeterministic RNG without a reason.
- **Autodiff.** `Sequential` chains forward/backward via the multivariate chain
  rule; stateful layers (e.g. `Tanh`) cache their forward output for `backward`.
- **Dependencies are intentionally minimal:** `rayon` (threads), `wide` (SIMD),
  `cc` (build). Adding a heavy dep (esp. a BLAS/GPU crate) defeats the project's
  purpose — discuss before introducing one.

## CI

`.github/workflows/build.yml` runs a **SonarQube** scan on pushes to `main`
(self-hosted runner). It does not currently gate on `cargo test`.
