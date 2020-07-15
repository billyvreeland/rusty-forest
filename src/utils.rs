
use ndarray::prelude::*;

pub fn rmse(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let square_diff = (a - b).mapv(|x| x.powi(2));
    let mean_square_diff = square_diff.sum() / square_diff.len() as f64;
    mean_square_diff.sqrt()
}

pub fn nrmse(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let r = rmse(a, b);
    let a_bar = a.sum() / a.len() as f64;
    r / a_bar.abs()
}

pub fn x_y_split(data: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    // Splits 2D array into 2D array and column vector, assuming target value is in last column
    let target_col = data.ncols()-1;
    let x = data.slice(s![.., ..target_col]).to_owned();
    let y = data.column(target_col).to_owned();
    (x, y)

}

pub fn train_test_split(data: &Array2<f64>, train_frac: f64) -> (Array2<f64>, Array2<f64>) {
    // Need to add shuffling, currently doing prior to reading data
    let train_rows = (train_frac * data.dim().0 as f64).round() as usize;
    let train = data.slice(s![..train_rows, ..]).to_owned();
    let test = data.slice(s![train_rows.., ..]).to_owned();
    (train, test)

}
