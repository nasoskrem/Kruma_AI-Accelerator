use crate::backend::HardwareBackend;
use crate::tensor::DeviceBuffer;
use rayon::prelude::*;
use wide::f32x4;

pub struct CpuBackend;

impl HardwareBackend for CpuBackend {
    fn matmul(&self, a: &DeviceBuffer, b: &DeviceBuffer, c: &mut DeviceBuffer, m: usize, k: usize, n: usize) {
        let a_slice = a.as_slice();
        let b_slice = b.as_slice();
        let c_slice = c.as_mut_slice();

        assert_eq!(a_slice.len(), m * k);
        assert_eq!(b_slice.len(), k * n);
        assert_eq!(c_slice.len(), m * n);

        // Parallel Transpose B to ensure sequential memory access
        let b_transposed = self.transpose_b(b_slice, k, n);

        // pre calculate SIMD boundaries
        let simd_k = k / 4;
        let remainder = k % 4;

        // Parallelize over rows of C
        c_slice
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, c_row)| {

                // Slices for the current row of A
                let a_row = &a_slice[i * k .. (i + 1) * k];

                // Cast to SIMD slice
                let a_simd = unsafe { 
                    std::slice::from_raw_parts(a_row.as_ptr() as *const f32x4, simd_k) 
                };

                for j in 0..n {
                    let b_row = &b_transposed[j * k .. (j + 1) * k];
                    
                    let b_simd = unsafe { 
                        std::slice::from_raw_parts(b_row.as_ptr() as *const f32x4, simd_k) 
                    };

                    // Instruction-Level Parallelis
                    let mut sum0 = f32x4::ZERO;
                    let mut sum1 = f32x4::ZERO;
                    let mut sum2 = f32x4::ZERO;
                    let mut sum3 = f32x4::ZERO;

                    let mut idx = 0;

                    // Process 16 floats (4 SIMD vectors) per loop iteration
                    while idx + 3 < simd_k {
                        sum0 += a_simd[idx] * b_simd[idx];
                        sum1 += a_simd[idx + 1] * b_simd[idx + 1];
                        sum2 += a_simd[idx + 2] * b_simd[idx + 2];
                        sum3 += a_simd[idx + 3] * b_simd[idx + 3];
                        idx += 4;
                    }

                    // Handle remaining SIMD vectors (if simd_k is not a multiple of 4)
                    while idx < simd_k {
                        sum0 += a_simd[idx] * b_simd[idx];
                        idx += 1;
                    }

                    // Single horizontal addition of the vectors
                    let mut total = (sum0 + sum1 + sum2 + sum3).reduce_add();

                    // SCALAR CLEANUP LOOP
                    // Handle the remaining 1, 2, or 3 floats if k is not a clean multiple of 4
                    for x in 0..remainder {
                        let scalar_idx = (simd_k * 4) + x;
                        total += a_row[scalar_idx] * b_row[scalar_idx];
                    }

                    c_row[j] = total;
                }
            });
    }

    fn relu(&self, input: &DeviceBuffer, output: &mut DeviceBuffer) {
        input
            .as_slice()
            .par_iter()
            .zip(output.as_mut_slice().par_iter_mut())
            .for_each(|(&x, o)| *o = x.max(0.0));
    }

    fn relu_inplace(&self, data: &mut DeviceBuffer) {
        data.as_mut_slice()
            .par_chunks_mut(1024)
            .for_each(|chunk| {
                for x in chunk {
                    *x = x.max(0.0);
                }
            });
    }

    fn softmax(&self, input: &DeviceBuffer, output: &mut DeviceBuffer, m: usize, n: usize) {
        let in_slice = input.as_slice();
        let out_slice = output.as_mut_slice();
        assert_eq!(in_slice.len(), m * n);
        out_slice
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row_out): (usize, &mut [f32])| {
                let row_in = &in_slice[i * n..(i + 1) * n];
                let max_val = row_in.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0;
                for (j, &x) in row_in.iter().enumerate() {
                    let ex = (x - max_val).exp();
                    row_out[j] = ex;
                    sum_exp += ex;
                }
                let inv_sum = 1.0 / sum_exp;
                for x in row_out.iter_mut() {
                    *x *= inv_sum;
                }
            });
    }

    fn log_softmax(&self, input: &DeviceBuffer, output: &mut DeviceBuffer, m: usize, n: usize) {
        let in_slice = input.as_slice();
        let out_slice = output.as_mut_slice();
        assert_eq!(in_slice.len(), m * n);
        out_slice
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row_out): (usize, &mut [f32])| {
                let row_in = &in_slice[i * n..(i + 1) * n];
                let max_val = row_in.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let sum_exp: f32 = row_in.iter().map(|&x| (x - max_val).exp()).sum();
                let log_sum_exp = sum_exp.ln();
                for (j, &x) in row_in.iter().enumerate() {
                    row_out[j] = x - max_val - log_sum_exp;
                }
            });
    }

    fn sigmoid(&self, input: &DeviceBuffer, output: &mut DeviceBuffer) {
        input
            .as_slice()
            .par_iter()
            .zip(output.as_mut_slice().par_iter_mut())
            .for_each(|(&x, o)| *o = 1.0 / (1.0 + (-x).exp()));
    }

    fn tanh(&self, input: &DeviceBuffer, output: &mut DeviceBuffer) {
        input
            .as_slice()
            .par_iter()
            .zip(output.as_mut_slice().par_iter_mut())
            .for_each(|(&x, o)| *o = x.tanh());
    }

    fn sum_columns(&self, input: &DeviceBuffer, output: &mut DeviceBuffer, m: usize, n: usize) {
        let in_slice = input.as_slice();
        let out_slice = output.as_mut_slice();
        out_slice.fill(0.0);
        out_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, out_val): (usize, &mut f32)| {
                let mut sum = 0.0;
                for i in 0..m {
                    sum += in_slice[i * n + j];
                }
                *out_val = sum;
            });
    }
}

impl CpuBackend {
    fn transpose_b(&self, b_slice: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut b_t = vec![0.0; n * k];
        b_t.par_chunks_mut(k).enumerate().for_each(|(j, b_col)| {
            for i in 0..k {
                b_col[i] = b_slice[i * n + j];
            }
        });
        b_t
    }
}