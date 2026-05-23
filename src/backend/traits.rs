use crate::tensor::DeviceBuffer;

pub trait HardwareBackend {
    fn matmul(&self, a: &DeviceBuffer, b: &DeviceBuffer, c: &mut DeviceBuffer, m: usize, k: usize, n: usize);
    fn relu(&self, input: &DeviceBuffer, output: &mut DeviceBuffer);
    fn relu_inplace(&self, data: &mut DeviceBuffer);
    fn softmax(&self, input: &DeviceBuffer, output: &mut DeviceBuffer, m: usize, n: usize);
    fn sigmoid(&self, input: &DeviceBuffer, output: &mut DeviceBuffer);
    fn tanh(&self, input: &DeviceBuffer, output: &mut DeviceBuffer);
    fn log_softmax(&self, input: &DeviceBuffer, output: &mut DeviceBuffer, m: usize, n: usize);
    fn sum_columns(&self, input: &DeviceBuffer, output: &mut DeviceBuffer, m: usize, n: usize);
}