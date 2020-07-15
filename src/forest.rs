
// use std::collections::HashMap;

use ndarray::{Array, Array1, Array2};
// use rayon::prelude::*;

// use crate::array_ext::ArrayExt;

mod tree;
use tree::*;



#[derive(Debug)]
pub struct RandomForest {
    pub n_trees: usize,
    pub min_samples_to_split: usize,
    pub max_depth: u8,
    pub num_cols_to_use: usize,
    pub trees: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        for _ in 0..self.n_trees {
            let mut tree = DecisionTree {
                root: None,
                min_samples_to_split: self.min_samples_to_split,
                max_depth: self.max_depth,
                num_cols_to_use: self.num_cols_to_use,
                num_total_cols: None
            };
            tree.fit(x, y);
            self.trees.push(tree);
        }
    }

    pub fn predict(&mut self, x: &Array2<f64>) -> Array1<f64> {
        let mut preds = Array::from(vec![0.; x.nrows()]);
        for tree in &self.trees {
            preds = preds + tree.predict(x);
        }
        preds = preds / self.n_trees as f64;
        preds
    }
}