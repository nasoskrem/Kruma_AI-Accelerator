use crate::tensor::Tensor;
use crate::nn::Module;

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
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            state: Vec::new(),
            t: 0,
        }
    }

    pub fn step(&mut self, module: &mut dyn Module<2>) {
            self.t += 1;
            let pairs = module.params_and_grads();
            self.ensure_state_initialized(&pairs);

            for (i, (param, grad)) in pairs.into_iter().enumerate() {
                let (m, v) = self.state[i].as_mut().unwrap();
                
                // Call separate update functions
                Self::update_first_moment_inplace(m, self.beta1, grad);
                Self::update_second_moment_inplace(v, self.beta2, grad);
                
                let step = Self::compute_step(m, v, self.lr, self.beta1, self.beta2, self.epsilon, self.t);
                self.apply_step_inplace(param, &step);
            }
    }

    pub fn zero_grad(&self, module: &mut dyn Module<2>) {
        for grad in module.grads_mut() {
            for x in &mut grad.data {
                *x = 0.0;
            }
        }
    }

    fn ensure_state_initialized(&mut self, pairs: &[(&mut Tensor<f32, 2>, &Tensor<f32, 2>)]) {
        if self.state.len() != pairs.len() {
            self.state = (0..pairs.len()).map(|_| None).collect();
        }
        for (i, (param, _)) in pairs.iter().enumerate() {
            if self.state[i].is_none() {
                self.state[i] = Some((Tensor::new(param.shape), Tensor::new(param.shape)));
            }
        }
    }

    // m = beta * m + (1-beta) * grad
    fn update_first_moment_inplace(m: &mut Tensor<f32, 2>, beta: f32, grad: &Tensor<f32, 2>) {
        for i in 0..m.data.len() {
            m.data[i] = beta * m.data[i] + (1.0 - beta) * grad.data[i];
        }
    }

    // v = beta * v + (1-beta) * (grad * grad)
    fn update_second_moment_inplace(v: &mut Tensor<f32, 2>, beta: f32, grad: &Tensor<f32, 2>) {
        for i in 0..v.data.len() {
            // Square the gradient here
            v.data[i] = beta * v.data[i] + (1.0 - beta) * (grad.data[i] * grad.data[i]);
        }
    }

    fn compute_step(
        m: &Tensor<f32, 2>,
        v: &Tensor<f32, 2>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        t: i32,
    ) -> Tensor<f32, 2> {
        let bias_c1 = 1.0 - beta1.powi(t);
        let bias_c2 = 1.0 - beta2.powi(t);
        let eff_lr = lr * (bias_c2.sqrt() / bias_c1);
        let mut step_data = Vec::with_capacity(m.data.len());
        for i in 0..m.data.len() {
            let denom = v.data[i].sqrt() + eps;
            let step_val = -eff_lr * m.data[i] / denom;
            step_data.push(step_val);
        }
        Tensor::from_data(m.shape, step_data)
    }

    fn apply_step_inplace(&self, param: &mut Tensor<f32, 2>, step: &Tensor<f32, 2>) {
        for i in 0..param.data.len() {
            let clipped = step.data[i].clamp(-1.0, 1.0); // ← clamp μεταξύ -1 και 1
            param.data[i] += clipped;
        }
    }
}