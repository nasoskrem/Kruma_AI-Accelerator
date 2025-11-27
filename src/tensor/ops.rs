use crate::tensor::Tensor;
use crate::backend::{CpuBackend, HardwareBackend};
use crate::tensor::DeviceBuffer;

pub trait TensorOps<T> {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn matmul(&self, other: &Self) -> Self;
    fn relu(&self) -> Self;
    fn sigmoid(&self) -> Self;
    fn tanh(&self) -> Self;
    fn softmax(&self) -> Self;
    fn log_softmax(&self) -> Self;
    fn sum_axis0(&self) -> Self;
    fn transpose(&self) -> Self;
    fn argmax(&self) -> Vec<usize>;
}

impl TensorOps<f32> for Tensor<f32, 2> {
    fn add(&self, other: &Self) -> Self {
        let [m1, n1] = self.shape;
        let [m2, n2] = other.shape;
        if m1 == m2 && n1 == n2 {
            let data = self.data.iter().zip(&other.data).map(|(a,b)| a+b).collect();
            return Tensor {data, shape: self.shape, strides: self.strides };
        }
        if m2 == 1 && n1 == n2 { // Broadcast
            let mut data = Vec::with_capacity(self.data.len());
            for i in 0..m1 {
                let start = i * n1;
                for j in 0..n1 { data.push(self.data[start + j] + other.data[j]); }
            }
            return Tensor {data, shape: self.shape, strides: self.strides };
        }
        if m1 == 1 && n1 == n2 { // Reverse broadcast
             let mut data = Vec::with_capacity(other.data.len());
             for i in 0..m2 {
                 let start = i * n2;
                 for j in 0..n2 { data.push(self.data[j] + other.data[start + j]); }
             }
             return Tensor {data, shape: other.shape, strides: other.strides };
        }
        panic!("Shape mismatch add");
    }

    fn sub(&self, other: &Self) -> Self {
        let [m1, n1] = self.shape;
        let [m2, n2] = other.shape;
        if m1 == m2 && n1 == n2 {
            let data = self.data.iter().zip(&other.data).map(|(a,b)| a-b).collect();
            return Tensor {data, shape: self.shape, strides: self.strides };
        }
        if m2 == 1 && n1 == n2 {
            let mut data = Vec::with_capacity(self.data.len());
            for i in 0..m1 {
                let start = i * n1;
                for j in 0..n1 { data.push(self.data[start + j] - other.data[j]); }
            }
            return Tensor {data, shape: self.shape, strides: self.strides };
        }
        panic!("Shape mismatch sub");
    }

    fn mul(&self, other: &Self) -> Self {
        let data = self.data.iter().zip(&other.data).map(|(a,b)| a*b).collect();
        Tensor {data, shape: self.shape, strides: self.strides }
    }

    fn matmul(&self, other: &Self) -> Self {
        let [m, k1] = self.shape;
        let [k2, n] = other.shape;
        assert_eq!(k1, k2);
        let mut out = Tensor::<f32, 2>::new([m, n]);
        let a_buf = DeviceBuffer::from_slice(&self.data);
        let b_buf = DeviceBuffer::from_slice(&other.data);
        let mut c_buf = DeviceBuffer::new(out.data.len());
        CpuBackend.matmul(&a_buf, &b_buf, &mut c_buf, m, k1, n);
        out.data = c_buf.into_vec();
        out
    }

    fn relu(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.relu(&input, &mut out);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }
    fn sigmoid(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.sigmoid(&input, &mut out);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }
    fn tanh(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.tanh(&input, &mut out);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }
    fn softmax(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.softmax(&input, &mut out, self.shape[0], self.shape[1]);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }
    fn log_softmax(&self) -> Self {
        let input = DeviceBuffer::from_slice(&self.data);
        let mut out = DeviceBuffer::new(self.data.len());
        CpuBackend.log_softmax(&input, &mut out, self.shape[0], self.shape[1]);
        Tensor { data: out.into_vec(), shape: self.shape, strides: self.strides }
    }
    fn sum_axis0(&self) -> Self {
        let [m, n] = self.shape;
        let mut out = Tensor::<f32, 2>::new([1, n]);
        let in_buf = DeviceBuffer::from_slice(&self.data);
        let mut out_buf = DeviceBuffer::new(n);
        CpuBackend.sum_columns(&in_buf, &mut out_buf, m, n);
        out.data = out_buf.into_vec();
        out
    }
    fn transpose(&self) -> Self {
        let [m, n] = self.shape;
        let mut data = vec![0.0; self.data.len()];
        for i in 0..m {
            for j in 0..n { data[j * m + i] = self[[i, j]]; }
        }
        Tensor::from_data([n, m], data)
    }
    fn argmax(&self) -> Vec<usize> {
        let [m, n] = self.shape;
        let mut indices = Vec::with_capacity(m);
        for i in 0..m {
            let start = i * n;
            let end = start + n;
            let row = &self.data[start..end];
            let (max_idx, _) = row.iter().enumerate()
                .fold((0, f32::NEG_INFINITY), |(i_max, val_max), (i, &val)| {
                    if val > val_max { (i, val) } else { (i_max, val_max) }
                });
            indices.push(max_idx);
        }
        indices
    }
}