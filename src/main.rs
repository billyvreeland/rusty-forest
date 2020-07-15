/// Driver program to demonstrate usage of random forest
/// 

use std::error::Error;
use std::time::Instant;
use ndarray::prelude::*;

mod array_ext;
// use array_ext::*;

mod csv_io;
use csv_io::*;

mod forest;
use forest::*;

mod utils;
use utils::{nrmse, train_test_split, x_y_split};

fn main() -> Result<(), Box<dyn Error>> {

    // Read in example data, assumes target is last column in csv file
    // let data = read_csv_2d("./data/motor_profile_4.csv");
    let data = read_csv_2d("./data/medium.csv");

    // Set up train, validation, and test data with 60/20/20 splits
    let train_frac = 0.6;
    let (train, val_test) = train_test_split(&data, train_frac);
    let (x_train, y_train) = x_y_split(&train);
    
    let val_frac = 0.5; // half of val_test set
    let (val, test) = train_test_split(&val_test, val_frac);
    let (x_val, y_val) = x_y_split(&val);
    let (x_test, y_test) = x_y_split(&test);

    // Loop over different numbers of trees, keeping track of one with lowest validation error
    let n_trees = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let mut best_num_trees = 0;
    let mut best_val_error = f64::INFINITY;
    let mut test_error = f64::INFINITY;

    for n in &n_trees {
        // Set up random forest
        let mut rf = RandomForest {
            n_trees: *n,
            min_samples_to_split: 5,
            max_depth: 10,
            num_cols_to_use: 3,
            trees: Vec::new()
            // trees: vec![]
        };

        // Train
        let now = Instant::now();
        rf.fit(&x_train, &y_train);
        let train_time = now.elapsed();
        println!("*** Training duration for {} trees: {} milliseconds", rf.n_trees, train_time.as_millis());

        // Check training and validation errors
        let train_preds = rf.predict(&x_train);
        let train_error = nrmse(&y_train, &train_preds);
        println!("Training set Normalized RMSE:   {:.2}%", 100.*train_error);
        let val_preds = rf.predict(&x_val);
        let val_error = nrmse(&y_val, &val_preds);
        println!("Validation set Normalized RMSE: {:.2}%", 100.*val_error);
        println!("val_preds.head(): {:.4?}", val_preds.slice(s![0..5]));
        let val_error = nrmse(&y_val, &val_preds);
        if val_error < best_val_error {
            best_val_error = val_error;
            best_num_trees = rf.n_trees;
            let test_preds = rf.predict(&x_test);
            test_error = nrmse(&y_test, &test_preds);
        }
        println!();
    }

    println!("Best number of trees: {}", best_num_trees);
    println!("Test set Normalized RMSE: {:.2}%", 100.*test_error);

    Ok(())
}