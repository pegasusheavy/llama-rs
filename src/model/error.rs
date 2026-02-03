//! Model-related error types

use crate::tensor::DType;

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Tensor shape mismatch for {name}: expected {expected:?}, got {got:?}")]
    TensorShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Tensor dtype mismatch for {name}: expected {expected:?}, got {got:?}")]
    TensorDTypeMismatch {
        name: String,
        expected: DType,
        got: DType,
    },

    #[error("Missing metadata: {0}")]
    MissingMetadata(String),

    #[error("Invalid metadata value for {key}: {message}")]
    InvalidMetadata { key: String, message: String },

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Context length exceeded: {current} > {max}")]
    ContextLengthExceeded { current: usize, max: usize },

    #[error("KV cache error: {0}")]
    KVCacheError(String),

    #[error("Backend error: {0}")]
    Backend(#[from] crate::backend::BackendError),

    #[error("GGUF error: {0}")]
    Gguf(#[from] crate::gguf::GgufError),

    #[error("Tensor error: {0}")]
    Tensor(#[from] crate::tensor::TensorError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type ModelResult<T> = Result<T, ModelError>;
