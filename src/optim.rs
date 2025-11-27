use crate::tensor::Tensor;
use crate::nn::Module;
use crate::tensor::ops::TensorOps;

pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    state: Vec<Option<(Tensor<f32, 2>, Tensor<f32, 2>)>>,
    t: i32,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, state: Vec::new(), t: 0 }
    }

    pub fn step(&mut self, module: &mut dyn Module) {
        self.t += 1;
        let pairs = module.params_and_grads();
        if self.state.len() != pairs.len() { self.state = (0..pairs.len()).map(|_| None).collect(); }

        for (i, (param, grad)) in pairs.into_iter().enumerate() {
            if self.state[i].is_none() {
                self.state[i] = Some((Tensor::new(param.shape), Tensor::new(param.shape)));
            }
            let (m, v) = self.state[i].as_mut().unwrap();

            // m = beta1 * m + (1-beta1) * grad
            let b1 = Tensor::from_data(grad.shape, vec![self.beta1; grad.data.len()]);
            let inv_b1 = Tensor::from_data(grad.shape, vec![1.0 - self.beta1; grad.data.len()]);
            let new_m = m.mul(&b1).add(&grad.mul(&inv_b1));
            *m = new_m;

            // v = beta2 * v + (1-beta2) * grad^2
            let b2 = Tensor::from_data(grad.shape, vec![self.beta2; grad.data.len()]);
            let inv_b2 = Tensor::from_data(grad.shape, vec![1.0 - self.beta2; grad.data.len()]);
            let g_sq = grad.mul(grad);
            let new_v = v.mul(&b2).add(&g_sq.mul(&inv_b2));
            *v = new_v;

            // Update
            let bias_c1 = 1.0 - self.beta1.powi(self.t);
            let bias_c2 = 1.0 - self.beta2.powi(self.t);
            let eff_lr = self.lr * (bias_c2.sqrt() / bias_c1);

            let v_sqrt_data: Vec<f32> = v.data.iter().map(|x| 1.0 / (x.sqrt() + self.epsilon)).collect();
            let inv_denom = Tensor::from_data(v.shape, v_sqrt_data);
            let step = m.mul(&inv_denom);
            
            let lr_tsr = Tensor::from_data(step.shape, vec![-eff_lr; step.data.len()]);
            *param = param.add(&step.mul(&lr_tsr));
        }
    }

    pub fn zero_grad(&self, module: &mut dyn Module) {
        for grad in module.grads_mut() {
            for x in &mut grad.data { *x = 0.0; }
        }
    }
}