
use rand::thread_rng;
use rand::seq::SliceRandom;

use ndarray::prelude::*;
use ndarray::{Array, Array1, Array2};

use crate::array_ext::ArrayExt;

#[derive(Debug)]
pub struct Node {
    split_col: Option<usize>,
    split_val: Option<f64>,
    pred_val: Option<f64>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>
}

impl Node {
    pub fn new_node(split_col: usize, split_val: f64) -> Self {
        Node {
            split_col: Some(split_col),
            split_val: Some(split_val),
            pred_val: None,
            left: None,
            right: None
        }
    }

    pub fn new_leaf(pred_val: Option<f64>) -> Self {
        Node {
            split_col: None,
            split_val: None,
            pred_val: pred_val,
            left: None,
            right: None
        }
    }

    pub fn is_leaf(&self) -> bool {
        if self.left.is_none() && self.right.is_none() {
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub struct DecisionTree {
    pub root: Option<Node>,
    pub min_samples_to_split: usize,
    pub max_depth: u8,
    pub num_cols_to_use: usize,
    pub num_total_cols: Option<usize>
}

impl DecisionTree {
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        self.num_total_cols = Some(x.ncols());
        let depth = 0;
        
        self.root = Some(self.build_tree(x, y, depth));
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut preds = vec![0.; x.nrows()];
        let mut idx = 0;
        for row in x.genrows() {
            preds[idx] = self.predict_one_obs(&row.to_owned());
            idx += 1;
        }
        Array1::from(preds)
    }

    fn build_tree(&mut self, x: &Array2<f64>, y: &Array1<f64>, mut depth: u8) -> Node {
        if x.nrows() < self.min_samples_to_split || depth == self.max_depth {
            let pred_val = y.mean();
            Node::new_leaf(pred_val)
        } else {
            let best_split = self.find_best_split(x, y);
            if best_split.is_none() {
                let pred_val = y.mean();
                Node::new_leaf(pred_val)
            } else {
                let bs = best_split.unwrap();
                let mut parent = Node::new_node(bs.col, bs.split_val);
                depth += 1;
                parent.left = Some(Box::new(self.build_tree(&bs.x_left, &bs.y_left, depth)));
                parent.right = Some(Box::new(self.build_tree(&bs.x_right, &bs.y_right, depth)));
                parent
            }
        }
    }

    fn find_best_split(&self, x: &Array2<f64>, y: &Array1<f64>)-> Option<SplitResult> {
        let mut best_split: Option<SplitResult> = None;
        let mut best_gain = f64::NEG_INFINITY;
        let y_len = y.len() as f64;
        let y_var = y.sample_var();
        for col in &self.pick_cols() {
            let col_mean = get_col_mean(x, col);
            let split_result = make_split(x, y, col, col_mean);
            if !split_result.is_none() {
                let sr = split_result.unwrap();
                let gain = var_reduction(y_len, y_var, &sr.y_left, &sr.y_right);
                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some(sr);
                }
            }
        }
        best_split
    }

    fn pick_cols(&self) -> Vec<usize> {
        // try making cols property of tree
        let mut cols: Vec<usize> = (0..self.num_total_cols.unwrap()).collect();
        cols.shuffle(&mut thread_rng());
        cols[0..self.num_cols_to_use].to_vec()
    }

    fn predict_one_obs(&self, x: &Array1<f64>) -> f64 {
        let mut current_node = self.root.as_ref().unwrap();
        loop {
            if current_node.is_leaf() {
                return current_node.pred_val.unwrap();
            } else {
                if x[current_node.split_col.unwrap()] <= current_node.split_val.unwrap() {
                    let new_node = &*(current_node.left.as_ref().unwrap());
                    current_node = new_node;
                } else {
                    let new_node = &*(current_node.right.as_ref().unwrap());
                    current_node = new_node;
                }
            }
        }
    }
}

pub struct SplitResult {
    pub col: usize,
    pub split_val: f64,
    pub x_left: Array2<f64>,
    pub y_left: Array1<f64>,
    pub x_right: Array2<f64>,
    pub y_right: Array1<f64>
}

pub fn make_split(x: &Array2<f64>, y: &Array1<f64>, split_col: &usize, split_val: f64) -> Option<SplitResult> {

    let rows = x.nrows();
    let cols = x.ncols();

    let mut x_left_vec = vec![0.; rows * cols];
    let mut y_left_vec = vec![0.; rows];
    let mut x_right_vec = vec![0.; rows * cols];
    let mut y_right_vec = vec![0.; rows];

    let mut row_idx = 0;
    let mut left_rows = 0;
    let mut left_pos = 0;
    let mut right_rows = 0;
    let mut right_pos = 0;

    for row in x.genrows() {
        if row[*split_col] <= split_val {
            y_left_vec[left_rows] = y[row_idx];
            x_left_vec[left_pos..left_pos+cols].copy_from_slice(&row.to_vec());
            left_rows += 1;
            left_pos += cols;
        } else {
            y_right_vec[right_rows] = y[row_idx];
            x_right_vec[right_pos..right_pos+cols].copy_from_slice(&row.to_vec());
            right_rows += 1;
            right_pos += cols;
        }
        row_idx += 1;
    }

    if left_pos == 0 || right_pos == 0 {
        None
    } else {
        let x_left_all = Array::from(x_left_vec.clone()).into_shape((rows, cols)).unwrap();
        let y_left_all = Array::from(y_left_vec);
    
        let x_right_all = Array::from(x_right_vec.clone()).into_shape((rows, cols)).unwrap();
        let y_right_all = Array::from(y_right_vec);
    
        Some(SplitResult{
            col: *split_col,
            split_val: split_val,
            x_left: x_left_all.slice(s![..left_rows, ..]).to_owned(),
            y_left: y_left_all.slice(s![..left_rows]).to_owned(),
            x_right: x_right_all.slice(s![..right_rows, ..]).to_owned(),
            y_right: y_right_all.slice(s![..right_rows]).to_owned()
        })
    }
}

pub fn var_reduction(parent_len: f64, parent_var: f64, child_a: &Array1<f64>, child_b: &Array1<f64>) -> f64 {
    let child_var = (child_a.len() as f64 * child_a.sample_var() + child_b.len() as f64 * child_b.sample_var()) / parent_len;
    parent_var - child_var
}

pub fn get_col_mean(x: &Array2<f64>, col_num: &usize) -> f64 {
    let c = x.column(*col_num);
    c.sum() / c.len() as f64
}
