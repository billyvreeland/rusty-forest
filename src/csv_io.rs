/// Helper functions for reading and writing csv files

use csv::{ReaderBuilder}; //, WriterBuilder};
use ndarray::{Array2};
use ndarray_csv::{Array2Reader}; // , Array2Writer};
use std::fs::File;

pub fn read_csv_2d(file_name: &str) -> Array2<f64> {
    let file = File::open(file_name).unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    reader.deserialize_array2_dynamic().unwrap()
}

/*
    // Write the array into the file.
    {
        let file = File::create("test.csv")?;
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        writer.serialize_array2(&array)?;
    }
    
*/