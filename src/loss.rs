use crate::tensor::Tensor;

pub struct MSELoss;

impl MSELoss {
    pub fn forward<const D: usize>(&self, pred: &Tensor<f32, D>, target: &Tensor<f32, D>) -> f32 {
        let diff = pred.sub(target);
        let sq_diff = diff.mul(&diff);
        sq_diff.data.iter().sum::<f32>() / pred.data.len() as f32
    }

    pub fn backward<const D: usize>(&self, pred: &Tensor<f32, D>, target: &Tensor<f32, D>) -> Tensor<f32, D> {
        let n = pred.data.len() as f32;
        let scale = 2.0 / n;
        let diff = pred.sub(target);
        diff.mul_scalar(scale)
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
        diff.mul_scalar(1.0 / n)
    }
}