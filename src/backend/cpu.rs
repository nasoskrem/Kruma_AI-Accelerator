use crate::backend::HardwareBackend;
use crate::tensor::DeviceBuffer;
use rayon::prelude::*;
use wide::f32x4;

pub struct CpuBackend;

impl HardwareBackend for CpuBackend {
    fn matmul(&self, a: &DeviceBuffer, b: &DeviceBuffer, c: &mut DeviceBuffer, m: usize, k: usize, n: usize) {
        let (a_slice, b_slice, c_slice) = (a.as_slice(), b.as_slice(), c.as_mut_slice());
        
        assert_eq!((a_slice.len(), b_slice.len(), c_slice.len()), (m * k, k * n, m * n));

        let b_transposed = self.transpose_b(b_slice, k, n);

        c_slice.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            let a_row = &a_slice[i * k .. (i + 1) * k];
            self.process_row(c_row, a_row, &b_transposed, k, n);
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
    
    #[inline]
    fn process_row(&self, c_row: &mut [f32], a_row: &[f32], b_transposed: &[f32], k: usize, n: usize) {
        let simd_k = k / 4;
        let a_simd = unsafe { std::slice::from_raw_parts(a_row.as_ptr() as *const f32x4, simd_k) };

        for j in 0..n {
            let b_row = &b_transposed[j * k .. (j + 1) * k];
            let b_simd = unsafe { std::slice::from_raw_parts(b_row.as_ptr() as *const f32x4, simd_k) };
            
            c_row[j] = self.compute_dot(a_row, b_row, a_simd, b_simd, k);
        }
    }

    #[inline]
    fn compute_dot(&self, a_row: &[f32], b_row: &[f32], a_simd: &[f32x4], b_simd: &[f32x4], k: usize) -> f32 {
        let mut total = self.simd_ilp_sum(a_simd, b_simd);
        let remainder = k % 4;
        let simd_offset = (k / 4) * 4;
        
        for x in 0..remainder {
            let idx = simd_offset + x;
            total += a_row[idx] * b_row[idx];
        }
        
        total
    }

    #[inline]
    fn simd_ilp_sum(&self, a_simd: &[f32x4], b_simd: &[f32x4]) -> f32 {
        let (mut s0, mut s1, mut s2, mut s3) = (f32x4::ZERO, f32x4::ZERO, f32x4::ZERO, f32x4::ZERO);
        let mut idx = 0;

        while idx + 3 < a_simd.len() {
            s0 += a_simd[idx] * b_simd[idx];     s1 += a_simd[idx+1] * b_simd[idx+1];
            s2 += a_simd[idx+2] * b_simd[idx+2]; s3 += a_simd[idx+3] * b_simd[idx+3];
            idx += 4;
        }

        while idx < a_simd.len() {
            s0 += a_simd[idx] * b_simd[idx];
            idx += 1;
        }

        (s0 + s1 + s2 + s3).reduce_add()
    }
}