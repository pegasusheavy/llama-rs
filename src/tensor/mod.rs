//! Tensor module for llama-rs
//!
//! This module provides tensor types for representing multi-dimensional arrays
//! with support for various data types including quantized formats.

mod core;
mod dtype;
mod error;
mod storage;

pub use core::{compute_strides, Tensor};
pub use dtype::DType;
pub use error::TensorError;
pub use storage::TensorStorage;
