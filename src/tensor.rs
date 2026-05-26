use std::ops::{Index, IndexMut};
use wide::f32x4;
use crate::backend::{CpuBackend, HardwareBackend};

/// A contiguous, 1-dimensional memory buffer used to interface with the hardware backend.
pub struct DeviceBuffer {
    data: Vec<f32>,
}

impl DeviceBuffer {

    /// Creates a new buffer of the specified size, initialized entirely with zeros.
    pub fn new(size: usize) -> Self {
        let mut data = Vec::with_capacity(size);
        data.resize(size, 0.0);
        Self { data }
    }

    /// Creates a new buffer by copying data from an existing slice of floats.
    pub fn from_slice(slice: &[f32]) -> Self {
        Self { data: slice.to_vec() }
    }

    /// Consumes the buffer and returns the underlying `Vec<f32>`.
    pub fn into_vec(self) -> Vec<f32> { self.data }

    /// Returns a read-only slice representing the buffer's data.
    pub fn as_slice(&self) -> &[f32] { &self.data }

    /// Returns a mutable slice representing the buffer's data.
    pub fn as_mut_slice(&mut self) -> &mut [f32] { &mut self.data }

    /// Reinterprets the float data as chunks of 4 (`f32x4`) for SIMD ops.
    pub fn as_simd(&self) -> &[f32x4] {
        assert_eq!(self.data.len() % 4, 0);
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32x4, self.data.len() / 4) }
    }

    /// Reinterprets the float data as mut chunks of 4 (`f32x4`) for SIMD ops.
    pub fn as_mut_simd(&mut self) -> &mut [f32x4] {
        assert_eq!(self.data.len() % 4, 0);
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f32x4, self.data.len() / 4) }
    }
}

/// # A Tensor.
/// 
///  `T` is the data type and `D` is the number of dimensions.
#[derive(Debug, Clone)]
pub struct Tensor<T, const D: usize> {
    /// The data stored in a flat Vector.
    pub data: Vec<T>,
    /// The size of the tensor (each dimension).
    pub shape: [usize; D],
    /// Number of elements to skip in memory to move to the next index along each dimension.
    pub strides: [usize; D],
}


impl<T: Default + Clone, const D: usize> Tensor<T, D> {

    /// Creates a new Tensor with the given shape, filled with the default value.
    pub fn new(shape: [usize; D]) -> Self {
        let strides = Self::compute_strides(&shape);
        let size = shape.iter().product();
        Self { data: vec![T::default(); size], shape, strides }
    }

    /// Creates a Tensor from an existing flat vector of data.
    pub fn from_data(shape: [usize; D], data: Vec<T>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        let strides = Self::compute_strides(&shape);
        Self { data, shape, strides }
    }

    /// Computes the memory strides for a C-contiguous layout.
    pub fn compute_strides(shape: &[usize; D]) -> [usize; D] {
        let mut strides = [0; D];
        let mut stride = 1;
        // Loop backwards, inner dimension has stride = 1, outer dimensions multiply outwards.
        for i in (0..D).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        strides
    }

    /// Converts a D dimensional index (e.g., `[row, col]`) into its actual 1-dimensional memory index.
    pub fn index_flat(&self, idx: [usize; D]) -> usize {
        idx.iter().zip(&self.strides).map(|(i, s)| i * s).sum()
    }

    pub fn numel(&self) -> usize { self.data.len() }
}

/// Van read a Tensor element using brackets.
impl<T: Default + Clone, const D: usize> Index<[usize; D]> for Tensor<T, D> {
    type Output = T;
    fn index(&self, index: [usize; D]) -> &Self::Output { &self.data[self.index_flat(index)] }
}

/// Can modify a Tensor element using brackets.
impl<T: Default + Clone, const D: usize> IndexMut<[usize; D]> for Tensor<T, D> {
    fn index_mut(&mut self, index: [usize; D]) -> &mut Self::Output {
        let idx = self.index_flat(index);
        &mut self.data[idx]
    }
}

/// N‑dimensional operations.
impl<const D: usize> Tensor<f32, D> {
    pub fn add(&self, other: &Self) -> Self { self.apply_elementwise(other, |a, b| a + b) }
    pub fn sub(&self, other: &Self) -> Self { self.apply_elementwise(other, |a, b| a - b) }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Self) -> Self { self.apply_elementwise(other, |a, b| a * b) }

    /// Multiplies every element in the Tensor by a scalar float.
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Tensor { data, shape: self.shape, strides: self.strides }
    }

    /// Element-wise ops between `self` and `other`.
    /// Supports broadcasting.
    fn apply_elementwise<F>(&self, other: &Self, op: F) -> Self
    where F: Fn(f32, f32) -> f32 {
        let (out_shape, s_strides, o_strides) = Self::broadcast_strides(self, other);
        let out_size = out_shape.iter().product();
        let mut data = Vec::with_capacity(out_size);

        // Matching shapes
        if self.shape == other.shape {
            for i in 0..self.data.len() { data.push(op(self.data[i], other.data[i])); }
            return Tensor { data, shape: out_shape, strides: Self::compute_strides(&out_shape) };
        }

        // Broadcasting
        let mut idx = [0; D];
        for _ in 0..out_size {
            let flat_self = (0..D).map(|i| idx[i] * s_strides[i]).sum::<usize>();
            let flat_other = (0..D).map(|i| idx[i] * o_strides[i]).sum::<usize>();
            data.push(op(self.data[flat_self], other.data[flat_other]));

            // Advance the multi-dimensional digital counter
            for i in (0..D).rev() {
                idx[i] += 1;
                if idx[i] < out_shape[i] { break; }
                idx[i] = 0;
            }
        }
        Tensor { data, shape: out_shape, strides: Self::compute_strides(&out_shape) }
    }

    /// The resulting shape and the virtual strides needed for broadcasting two tensors.
    fn broadcast_strides(a: &Self, b: &Self) -> ([usize; D], [usize; D], [usize; D]) {
        let mut out_shape = [0; D];
        let mut a_strides = [0; D];
        let mut b_strides = [0; D];
        for i in 0..D {
            let s1 = a.shape[i];
            let s2 = b.shape[i];
            if s1 != s2 && s1 != 1 && s2 != 1 {
                panic!("Shape mismatch for broadcasting: {:?} and {:?}", a.shape, b.shape);
            }
            out_shape[i] = s1.max(s2);
            a_strides[i] = if s1 == 1 { 0 } else { a.strides[i] };
            b_strides[i] = if s2 == 1 { 0 } else { b.strides[i] };
        }
        (out_shape, a_strides, b_strides)
    }

    /// ReLU `max(0, x)`.
    /// 
    /// Computation is in CPU backend.
    pub fn relu(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.relu(&input, &mut out);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }

    /// Sigmoid `1 / (1 + exp(-x))`.
    /// 
    /// Computation is in CPU backend.
    pub fn sigmoid(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.sigmoid(&input, &mut out);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }

    /// Tanh.
    /// 
    /// Computation is in CPU backend.
    pub fn tanh(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.tanh(&input, &mut out);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }
}

// 2‑dimensional operations (matmul, softmax, etc.)
impl Tensor<f32, 2> {

    /// Matrix Multiplication between two 2D Tensors.
    pub fn matmul(&self, other: &Self) -> Self {
        let [m, k1] = self.shape;
        let [k2, n] = other.shape;
        assert_eq!(k1, k2);
        let mut out = Tensor::new([m, n]);
        let a_buf = DeviceBuffer::from_slice(&self.data);
        let b_buf = DeviceBuffer::from_slice(&other.data);
        let mut c_buf = DeviceBuffer::new(out.data.len());
        CpuBackend.matmul(&a_buf, &b_buf, &mut c_buf, m, k1, n);
        out.data = c_buf.into_vec();
        out
    }

    /// Softmax function row-by-row.
    pub fn softmax(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.softmax(&input, &mut out, self.shape[0], self.shape[1]);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }

    /// Log-Softmax function row-by-row.
    pub fn log_softmax(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.log_softmax(&input, &mut out, self.shape[0], self.shape[1]);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }

    /// Sum the values of the Matrix along the 0th axis.
    /// 
    /// Used in Backpropagation for Bias gradients.
    pub fn sum_axis0(&self) -> Self {
        let [m, n] = self.shape;
        let mut out = Tensor::new([1, n]);
        let in_buf = DeviceBuffer::from_slice(&self.data);
        let mut out_buf = DeviceBuffer::new(n);
        CpuBackend.sum_columns(&in_buf, &mut out_buf, m, n);
        out.data = out_buf.into_vec();
        out
    }

    /// Returns the transposed version of the Matrix.
    pub fn transpose(&self) -> Self {
        let [m, n] = self.shape;
        let mut data = vec![0.0; self.data.len()];
        for i in 0..m {
            for j in 0..n {
                data[j * m + i] = self[[i, j]];
            }
        }
        Tensor::from_data([n, m], data)
    }

    /// Finds the index of the maximum value in each row.
    /// 
    /// Used to extract the final class prediction from a neural network's output layer.
    pub fn argmax(&self) -> Vec<usize> {
        let [m, n] = self.shape;
        let mut indices = Vec::with_capacity(m);
        for i in 0..m {
            let start = i * n;
            let row = &self.data[start..start + n];
            let (max_idx, _) = row.iter().enumerate()
                .fold((0, f32::NEG_INFINITY), |(i_max, val_max), (i, &val)| {
                    if val > val_max { (i, val) } else { (i_max, val_max) }
                });
            indices.push(max_idx);
        }
        indices
    }
}