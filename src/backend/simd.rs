use crate::backend::SimdSliceExt;
use wide::f32x4;

impl SimdSliceExt for [f32] {
    #[inline]
    fn as_simd(&self) -> &[f32x4] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const f32x4, self.len() / 4) }
    }

    #[inline]
    fn as_mut_simd(&mut self) -> &mut [f32x4] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr() as *mut f32x4, self.len() / 4) }
    }
}