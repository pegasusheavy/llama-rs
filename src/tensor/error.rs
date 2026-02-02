//! Tensor error types

#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("Size mismatch: expected {expected} bytes, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    #[error("Invalid dtype for operation")]
    InvalidDType,

    #[error("Tensor is not contiguous")]
    NotContiguous,
}
