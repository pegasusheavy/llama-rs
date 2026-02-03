use crate::tensor::DType;

#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("Backend not available: {0}")]
    NotAvailable(String),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("DType mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDType(DType),

    #[error("Operation not supported: {0}")]
    Unsupported(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Tensor error: {0}")]
    Tensor(#[from] crate::tensor::TensorError),
}
