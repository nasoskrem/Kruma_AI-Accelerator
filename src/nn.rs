use crate::tensor::Tensor;
use crate::tensor::ops::TensorOps;

// 1. MODULE TRAIT
pub trait Module {
    fn forward(&mut self, input: &Tensor<f32, 2>) -> Tensor<f32, 2>;
    fn backward(&mut self, grad_output: &Tensor<f32, 2>) -> Tensor<f32, 2>;
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor<f32, 2>, &Tensor<f32, 2>)>;
    fn grads_mut(&mut self) -> Vec<&mut Tensor<f32, 2>>;
}

// 2. SEQUENTIAL CONTAINER
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}
impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self { Self { layers } }
}
impl Module for Sequential {
    fn forward(&mut self, input: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let mut x = input.clone();
        for layer in &mut self.layers { x = layer.forward(&x); }
        x
    }
    fn backward(&mut self, grad_output: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() { grad = layer.backward(&grad); }
        grad
    }
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor<f32, 2>, &Tensor<f32, 2>)> {
        let mut all = Vec::new();
        for layer in &mut self.layers { all.extend(layer.params_and_grads()); }
        all
    }
    fn grads_mut(&mut self) -> Vec<&mut Tensor<f32, 2>> {
        let mut all = Vec::new();
        for layer in &mut self.layers { all.extend(layer.grads_mut()); }
        all
    }
}

// 3. LINEAR LAYER
pub struct Linear {
    pub weights: Tensor<f32, 2>,
    pub bias: Tensor<f32, 2>,
    pub d_weights: Tensor<f32, 2>,
    pub d_bias: Tensor<f32, 2>,
    pub input_cache: Option<Tensor<f32, 2>>,
}
impl Linear {
    pub fn new(in_f: usize, out_f: usize) -> Self {
        let scale = (2.0 / in_f as f32).sqrt();
        let mut w_data = Vec::with_capacity(in_f * out_f);
        // Simple pseudo-random init
        for i in 0..(in_f * out_f) {
            let val = ((i as f32 * 12.9898).sin() * 43758.5453).fract(); 
            w_data.push(val * scale);
        }
        Self {
            weights: Tensor::from_data([in_f, out_f], w_data),
            bias: Tensor::new([1, out_f]),
            d_weights: Tensor::new([in_f, out_f]),
            d_bias: Tensor::new([1, out_f]),
            input_cache: None,
        }
    }
}
impl Module for Linear {
    fn forward(&mut self, input: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        self.input_cache = Some(input.clone());
        input.matmul(&self.weights).add(&self.bias)
    }
    fn backward(&mut self, grad_output: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let input = self.input_cache.as_ref().unwrap();
        self.d_weights = input.transpose().matmul(grad_output);
        self.d_bias = grad_output.sum_axis0(); // Correctly reduces batch gradient
        grad_output.matmul(&self.weights.transpose())
    }
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor<f32, 2>, &Tensor<f32, 2>)> {
        vec![(&mut self.weights, &self.d_weights), (&mut self.bias, &self.d_bias)]
    }
    fn grads_mut(&mut self) -> Vec<&mut Tensor<f32, 2>> {
        // FIXED: Removed parentheses here
        vec![&mut self.d_weights, &mut self.d_bias]
    }
}

// 4. ACTIVATIONS & DROPOUT
pub struct Tanh { output_cache: Option<Tensor<f32, 2>> }
impl Tanh { pub fn new() -> Self { Self { output_cache: None } } }
impl Module for Tanh {
    fn forward(&mut self, input: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let out = input.tanh();
        self.output_cache = Some(out.clone());
        out
    }
    fn backward(&mut self, grad_output: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let out = self.output_cache.as_ref().unwrap();
        let one = Tensor::from_data(out.shape, vec![1.0; out.data.len()]);
        let sq = out.mul(out);
        grad_output.mul(&one.sub(&sq))
    }
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor<f32, 2>, &Tensor<f32, 2>)> { vec![] }
    fn grads_mut(&mut self) -> Vec<&mut Tensor<f32, 2>> { vec![] }
}

pub struct Dropout {
    pub p: f32,
    mask: Option<Tensor<f32, 2>>,
}
impl Dropout { pub fn new(p: f32) -> Self { Self { p, mask: None } } }
impl Module for Dropout {
    fn forward(&mut self, input: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let scale = 1.0 / (1.0 - self.p);
        let mut mask_data = Vec::with_capacity(input.data.len());
        let mut seed = 123456789; 
        for _ in 0..input.data.len() {
             seed = (1103515245 * seed + 12345) % 2147483647;
             let val = if (seed as f32 / 2147483648.0) > self.p { 1.0 } else { 0.0 };
             mask_data.push(val * scale);
        }
        let mask = Tensor::from_data(input.shape, mask_data);
        let out = input.mul(&mask);
        self.mask = Some(mask);
        out
    }
    fn backward(&mut self, grad_output: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        if let Some(mask) = &self.mask { grad_output.mul(mask) } else { grad_output.clone() }
    }
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor<f32, 2>, &Tensor<f32, 2>)> { vec![] }
    fn grads_mut(&mut self) -> Vec<&mut Tensor<f32, 2>> { vec![] }
}