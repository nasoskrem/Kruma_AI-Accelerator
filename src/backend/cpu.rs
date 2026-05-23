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

        let b_transposed = self.transpose_b(b_slice, k, n);
        self.matmul_blocked(a_slice, &b_transposed, c_slice, m, k, n);
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

    fn matmul_blocked(&self, a_slice: &[f32], b_transposed: &[f32], c_slice: &mut [f32], m: usize, k: usize, n: usize) {
        c_slice
            .par_chunks_mut(n * 4)
            .enumerate()
            .for_each(|(block_idx, c_block)| {
                let start_row = block_idx * 4;
                let rows_in_block = (start_row + 4).min(m) - start_row;
                let a_row_offsets: Vec<usize> = (0..rows_in_block)
                    .map(|r| (start_row + r) * k)
                    .collect();

                for j in 0..n {
                    let b_row_ptr = unsafe { b_transposed.as_ptr().add(j * k) };
                    let mut sums = [0.0; 4];

                    let mut k_idx = 0;
                    while k_idx + 4 <= k {
                        unsafe {
                            let b_vec = f32x4::new([
                                *b_row_ptr.add(k_idx),
                                *b_row_ptr.add(k_idx + 1),
                                *b_row_ptr.add(k_idx + 2),
                                *b_row_ptr.add(k_idx + 3),
                            ]);
                            for r in 0..rows_in_block {
                                let a_ptr = a_slice.as_ptr().add(a_row_offsets[r] + k_idx);
                                let a_vec = f32x4::new([
                                    *a_ptr,
                                    *a_ptr.add(1),
                                    *a_ptr.add(2),
                                    *a_ptr.add(3),
                                ]);
                                sums[r] += (a_vec * b_vec).reduce_add();
                            }
                        }
                        k_idx += 4;
                    }

                    while k_idx < k {
                        unsafe {
                            let b_val = *b_row_ptr.add(k_idx);
                            for r in 0..rows_in_block {
                                sums[r] += *a_slice.get_unchecked(a_row_offsets[r] + k_idx) * b_val;
                            }
                        }
                        k_idx += 1;
                    }

                    for r in 0..rows_in_block {
                        c_block[r * n + j] = sums[r];
                    }
                }
            });
    }
}