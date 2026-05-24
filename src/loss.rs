use crate::tensor::Tensor;
use rayon::prelude::*;

pub struct MSELoss;

impl MSELoss {
    pub fn forward<const D: usize>(&self, pred: &Tensor<f32, D>, target: &Tensor<f32, D>) -> f32 {
        let n = pred.data.len() as f32;
        
        // Fused zero-allocation parallel reduction
        let sum_sq: f32 = pred.data.par_iter()
            .zip(target.data.par_iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                diff * diff
            })
            .sum();
            
        sum_sq / n
    }

    pub fn backward<const D: usize>(&self, pred: &Tensor<f32, D>, target: &Tensor<f32, D>) -> Tensor<f32, D> {
        let n = pred.data.len() as f32;
        let scale = 2.0 / n;
        
        // Compute gradients directly into a single parallel collection 
        // without creating an intermediate 'diff' tensor.
        let grad_data: Vec<f32> = pred.data.par_iter()
            .zip(target.data.par_iter())
            .map(|(&p, &t)| (p - t) * scale)
            .collect();
            
        Tensor::from_data(pred.shape, grad_data)
    }
}

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn forward(&self, logits: &Tensor<f32, 2>, target: &Tensor<f32, 2>) -> f32 {
        let log_probs = logits.log_softmax();
        let n = logits.shape[0] as f32;
        
        // Fused parallel sum instead of log_probs.mul(target).data.iter().sum()
        let loss_sum: f32 = log_probs.data.par_iter()
            .zip(target.data.par_iter())
            .map(|(&lp, &t)| lp * t)
            .sum();
            
        -loss_sum / n
    }

    pub fn backward(&self, logits: &Tensor<f32, 2>, target: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let n = logits.shape[0] as f32;
        let scale = 1.0 / n;
        let probs = logits.softmax();
        
        // Fused parallel subtraction and scalar multiplication
        let grad_data: Vec<f32> = probs.data.par_iter()
            .zip(target.data.par_iter())
            .map(|(&p, &t)| (p - t) * scale)
            .collect();
            
        Tensor::from_data(logits.shape, grad_data)
    }
}