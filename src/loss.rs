use crate::tensor::Tensor;
use crate::tensor::ops::TensorOps;

pub struct MSELoss;
impl MSELoss {
    pub fn forward(&self, pred: &Tensor<f32, 2>, target: &Tensor<f32, 2>) -> f32 {
        let neg_one = Tensor::from_data(target.shape, vec![-1.0; target.data.len()]);
        let diff = pred.add(&target.mul(&neg_one));
        let sq_diff = diff.mul(&diff);
        sq_diff.data.iter().sum::<f32>() / pred.data.len() as f32
    }
    pub fn backward(&self, pred: &Tensor<f32, 2>, target: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let n = pred.data.len() as f32;
        let scale = 2.0 / n;
        let neg_one = Tensor::from_data(target.shape, vec![-1.0; target.data.len()]);
        let diff = pred.add(&target.mul(&neg_one));
        let scale_tensor = Tensor::from_data(diff.shape, vec![scale; diff.data.len()]);
        diff.mul(&scale_tensor)
    }
}

pub struct CrossEntropyLoss;
impl CrossEntropyLoss {
    pub fn forward(&self, logits: &Tensor<f32, 2>, target: &Tensor<f32, 2>) -> f32 {
        let log_probs = logits.log_softmax();
        let n = logits.shape[0] as f32;
        let loss_sum: f32 = log_probs.mul(target).data.iter().sum();
        -loss_sum / n
    }
    pub fn backward(&self, logits: &Tensor<f32, 2>, target: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let n = logits.shape[0] as f32;
        let probs = logits.softmax();
        let diff = probs.sub(target);
        let scale = Tensor::from_data(diff.shape, vec![1.0/n; diff.data.len()]);
        diff.mul(&scale)
    }
}