pub struct PseudoRng;

impl PseudoRng {
    const SINE_FREQ: f32 = 12.9898;
    const AMPLITUDE: f32 = 43758.5453;
    const LCG_A: u32 = 1103515245;
    const LCG_C: u32 = 12345;
    const LCG_M: f32 = 2147483648.0;

    pub fn generate_weight(index: usize, scale: f32) -> f32 {
        let val = ((index as f32 * Self::SINE_FREQ).sin() * Self::AMPLITUDE).fract();
        val * scale
    }

    pub fn generate_mask(size: usize, p: f32, seed: &mut u32) -> Vec<f32> {
        let scale = 1.0 / (1.0 - p);
        let mut mask_data = Vec::with_capacity(size);
        for _ in 0..size {
            *seed = (Self::LCG_A.wrapping_mul(*seed).wrapping_add(Self::LCG_C)) & 0x7FFFFFFF;
            let val = if (*seed as f32 / Self::LCG_M) > p { 1.0 } else { 0.0 };
            mask_data.push(val * scale);
        }
        mask_data
    }
}