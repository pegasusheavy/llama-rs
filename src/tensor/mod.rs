//! Tensor module for llama-rs
//!
//! This module provides tensor types for representing multi-dimensional arrays
//! with support for various data types including quantized formats.

mod dtype;
mod error;
mod storage;
mod tensor;

pub use dtype::DType;
pub use error::TensorError;
pub use storage::TensorStorage;
pub use tensor::{compute_strides, Tensor};
