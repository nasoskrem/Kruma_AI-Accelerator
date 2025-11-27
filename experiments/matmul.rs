use kruma::tensor::Tensor;
use kruma::tensor::ops::TensorOps;
use std::time::Instant;

// STANDARD AI VERIFICATION: Relative Tolerance check.
fn is_close(a: f32, b: f32, rel_tol: f32, abs_tol: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= abs_tol { return true; }
    let largest = a.abs().max(b.abs());
    diff <= largest * rel_tol
}

fn naive_matmul(a: &Tensor<f32, 2>, b: &Tensor<f32, 2>) -> Tensor<f32, 2> {
    let [m, k] = a.shape;
    let [k2, n] = b.shape;
    assert_eq!(k, k2, "Inner dims must match");

    let mut out = Tensor::<f32, 2>::new([m, n]);
    // Pure Rust triple loop without SIMD or parallelism
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for x in 0..k {
                sum += a[[i, x]] * b[[x, j]];
            }
            out[[i, j]] = sum;
        }
    }
    out
}

fn main() {
    println!("KRUMA: Matrix Multiplication Benchmark");
    println!("----------------------------------------------");

    // 1024x1024 is the sweet spot:
    // - Large enough to freeze  a naive implementation for a few seconds.
    // - Small enough to run instantly on Kruma.
    let size = 1024; 
    println!("Initializing {}x{} Matrices...", size, size);

    let mut data = vec![0.0; size * size];
    for i in 0..size {
        for j in 0..size {
            data[i * size + j] = ((i + j) % 100) as f32 * 0.01;
        }
    }

    let a = Tensor::from_data([size, size], data.clone());
    let b = Tensor::from_data([size, size], data);

    // --- BENCHMARK 1: Naive Rust ---
    println!("\n[Baseline] Running Naive Triple-Loop (1024x1024)...");
    println!("   (Please wait... This runs on 1 core without SIMD)");
    
    let start_naive = Instant::now();
    let naive_res = naive_matmul(&a, &b);
    let dur_naive = start_naive.elapsed();
    
    println!("Naive Time:      {:.4}s", dur_naive.as_secs_f32());

    // --- BENCHMARK 2: KRUMA Engine ---
    println!("\n[KRUMA] Running Matrix Mul (1024x1024)...");
    
    let start_kruma = Instant::now();
    let kruma_res = a.matmul(&b);
    let dur_kruma = start_kruma.elapsed();
    
    // Calculate Stats
    let ops = 2.0 * (size as f64).powi(3);
    let gflops = (ops / dur_kruma.as_secs_f64()) / 1e9;
    let speedup = dur_naive.as_secs_f32() / dur_kruma.as_secs_f32();
    
    println!("KRUMA Time:      {:.4}s", dur_kruma.as_secs_f32());
    println!("PERFORMANCE:     {:.2} GFLOPS", gflops);
    println!("SPEEDUP:         {:.2}x FASTER", speedup);

    // --- VERIFICATION ---
    println!("\n[Verification] Checking Accuracy...");
    let mut max_diff = 0.0f32;
    let mut errors = 0;
    
    // Verify 100 random points
    for i in 0..100 {
        let r = (i * 12345) % size;
        let c = (i * 67890) % size;
        
        let val_naive = naive_res[[r, c]];
        let val_kruma = kruma_res[[r, c]];
        let diff = (val_naive - val_kruma).abs();
        
        if diff > max_diff { max_diff = diff; }

        // Use standard float tolerance
        if !is_close(val_naive, val_kruma, 1e-4, 1e-2) {
            println!("❌ Mismatch at [{},{}]: Naive={} KRUMA={}", r, c, val_naive, val_kruma);
            errors += 1;
        }
    }

    if errors == 0 {
        println!("✅ Correctness Check: PASSED (Max Diff: {})", max_diff);
    } else {
        println!("❌ Correctness Check: FAILED with {} errors", errors);
    }
}