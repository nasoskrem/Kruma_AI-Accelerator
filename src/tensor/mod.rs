use std::ops::{Index, IndexMut};
use wide::f32x4;
pub mod ops;

pub struct DeviceBuffer {
    data: Vec<f32>,
}

impl DeviceBuffer {
    pub fn new(size: usize) -> Self {
        let mut data: Vec<f32> = Vec::with_capacity(size);
        // Initialize with zeros to avoid garbage data
        data.resize(size, 0.0);
        Self { data }
    }

    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn as_simd(&self) -> &[f32x4] {
        assert!(self.data.len() % 4 == 0, "Data length must be multiple of 4");
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32x4,
                self.data.len() / 4,
            )
        }
    }

    pub fn as_mut_simd(&mut self) -> &mut [f32x4] {
        assert!(self.data.len() % 4 == 0, "Data length must be multiple of 4");
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut f32x4,
                self.data.len() / 4,
            )
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<T, const D: usize> {
    pub data: Vec<T>,
    pub shape: [usize; D],
    pub strides: [usize; D],
}

impl<T: Default + Clone, const D: usize> Tensor<T, D> {
    
    pub fn new(shape: [usize; D]) -> Self {
        let strides = Self::compute_strides(&shape);
        let size = shape.iter().product();
        Self {
            data: vec![T::default(); size],
            shape,
            strides,
        }
    }

    pub fn from_data(shape: [usize; D], data: Vec<T>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "Shape does not match data length");
        let strides = Self::compute_strides(&shape);
        Self { data, shape, strides }
    }
    
    pub fn compute_strides(shape: &[usize; D]) -> [usize; D] {
        let mut strides = [0; D];
        let mut stride = 1;
        for i in (0..D).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        strides
    }

    pub fn index_flat(&self, idx: [usize; D]) -> usize {
        idx.iter().zip(&self.strides).map(|(i, s)| i * s).sum()
    }
    
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

impl<T: Default + Clone, const D: usize> Index<[usize; D]> for Tensor<T, D> {
    type Output = T;
    fn index(&self, index: [usize; D]) -> &Self::Output {
        &self.data[self.index_flat(index)]
    }
}

impl<T: Default + Clone, const D: usize> IndexMut<[usize; D]> for Tensor<T, D> {
    fn index_mut(&mut self, index: [usize; D]) -> &mut Self::Output {
        let flat_idx = self.index_flat(index);
        &mut self.data[flat_idx]
    }
}