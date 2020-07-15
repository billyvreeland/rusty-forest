use ndarray::Array1;

pub trait ArrayExt<T> {
    fn sample_var(&self) -> f64;
}

impl ArrayExt<f64> for Array1<f64> {
    fn sample_var(&self) -> f64 {
        let len = self.len();
        if len == 0 {
            0.
        } else {
            let avg = self.mean().unwrap();
            let squared_diffs = self.mapv(|a| (a - avg).powi(2));
            squared_diffs.sum() / ((len - 1) as f64)
        }
    }
}




