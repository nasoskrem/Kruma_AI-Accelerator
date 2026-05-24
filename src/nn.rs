use crate::tensor::Tensor;
use crate::utils::PseudoRng;

pub trait Module<const D: usize = 2> {
    fn forward(&mut self, input: &Tensor<f32, D>) -> Tensor<f32, D>;
    fn backward(&mut self, grad_output: &Tensor<f32, D>) -> Tensor<f32, D>;
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor<f32, 2>, &Tensor<f32, 2>)>;
    fn grads_mut(&mut self) -> Vec<&mut Tensor<f32, 2>>;
}

pub struct Sequential<const D: usize = 2> {
    layers: Vec<Box<dyn Module<D>>>,
}

impl<const D: usize> Sequential<D> {
    pub fn new(layers: Vec<Box<dyn Module<D>>>) -> Self { Self { layers } }
}

impl<const D: usize> Module<D> for Sequential<D> {
    fn forward(&mut self, input: &Tensor<f32, D>) -> Tensor<f32, D> {
        let mut x = input.clone();
        for layer in &mut self.layers { x = layer.forward(&x); }
        x
    }
    fn backward(&mut self, grad_output: &Tensor<f32, D>) -> Tensor<f32, D> {
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
        let size = in_f * out_f;
        let mut w_data = Vec::with_capacity(size);
        for i in 0..size { w_data.push(PseudoRng::generate_weight(i, scale)); }
        Self {
            weights: Tensor::from_data([in_f, out_f], w_data),
            bias: Tensor::new([1, out_f]),
            d_weights: Tensor::new([in_f, out_f]),
            d_bias: Tensor::new([1, out_f]),
            input_cache: None,
        }
    }
}

impl Module<2> for Linear {
    fn forward(&mut self, input: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        self.input_cache = Some(input.clone());
        input.matmul(&self.weights).add(&self.bias)
    }
    fn backward(&mut self, grad_output: &Tensor<f32, 2>) -> Tensor<f32, 2> {
        let input = self.input_cache.as_ref().expect("Forward not called");
        self.d_weights = input.transpose().matmul(grad_output);
        self.d_bias = grad_output.sum_axis0();
        grad_output.matmul(&self.weights.transpose())
    }
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor<f32, 2>, &Tensor<f32, 2>)> {
        vec![(&mut self.weights, &self.d_weights), (&mut self.bias, &self.d_bias)]
    }
    fn grads_mut(&mut self) -> Vec<&mut Tensor<f32, 2>> { vec![&mut self.d_weights, &mut self.d_bias] }
}

pub struct Tanh<const D: usize = 2> { output_cache: Option<Tensor<f32, D>> }
impl<const D: usize> Default for Tanh<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> Tanh<D> { pub fn new() -> Self { Self { output_cache: None } } }
impl<const D: usize> Module<D> for Tanh<D> {
    fn forward(&mut self, input: &Tensor<f32, D>) -> Tensor<f32, D> {
        let out = input.tanh();
        self.output_cache = Some(out.clone());
        out
    }
    fn backward(&mut self, grad_output: &Tensor<f32, D>) -> Tensor<f32, D> {
        let out = self.output_cache.as_ref().unwrap();
        let one = Tensor::from_data(out.shape, vec![1.0; out.data.len()]);
        let sq = out.mul(out);
        grad_output.mul(&one.sub(&sq))
    }
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor<f32, 2>, &Tensor<f32, 2>)> { vec![] }
    fn grads_mut(&mut self) -> Vec<&mut Tensor<f32, 2>> { vec![] }
}

pub struct Dropout<const D: usize = 2> {
    pub p: f32,
    mask: Option<Tensor<f32, D>>,
    seed: u32,  
}

impl<const D: usize> Dropout<D> {
    pub fn new(p: f32) -> Self {
        Self {
            p,
            mask: None,
            seed: 123456789,
        }
    }
}


impl<const D: usize> Module<D> for Dropout<D> {
    fn forward(&mut self, input: &Tensor<f32, D>) -> Tensor<f32, D> {
        let mask_data = PseudoRng::generate_mask(input.data.len(), self.p, &mut self.seed);
        let mask = Tensor::from_data(input.shape, mask_data);
        self.mask = Some(mask.clone());
        input.mul(&mask)
    }

    fn backward(&mut self, grad_output: &Tensor<f32, D>) -> Tensor<f32, D> {
        match &self.mask {
            Some(mask) => grad_output.mul(mask),
            None => grad_output.clone(),
        }
    }
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor<f32, 2>, &Tensor<f32, 2>)> { vec![] }
    fn grads_mut(&mut self) -> Vec<&mut Tensor<f32, 2>> { vec![] }
}