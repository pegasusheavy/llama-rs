//! RAG (Retrieval-Augmented Generation) support with pgvector
//!
//! This module provides integration with PostgreSQL + pgvector for
//! semantic search and context retrieval during inference.
//!
//! # Setup
//!
//! 1. Install pgvector extension in PostgreSQL:
//!    ```sql
//!    CREATE EXTENSION vector;
//!    ```
//!
//! 2. Create embeddings table:
//!    ```sql
//!    CREATE TABLE embeddings (
//!        id SERIAL PRIMARY KEY,
//!        content TEXT NOT NULL,
//!        embedding vector(384),  -- Adjust dimension for your model
//!        metadata JSONB,
//!        created_at TIMESTAMPTZ DEFAULT NOW()
//!    );
//!    CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);
//!    ```
//!
//! # Example
//!
//! ```ignore
//! use llama_gguf::rag::{RagConfig, RagStore};
//!
//! let config = RagConfig::new("postgres://user:pass@localhost/db");
//! let store = RagStore::connect(config).await?;
//!
//! // Search for relevant context
//! let results = store.search(&query_embedding, 5).await?;
//!
//! // Build augmented prompt
//! let context = results.iter().map(|r| r.content.as_str()).collect::<Vec<_>>().join("\n");
//! let prompt = format!("Context:\n{}\n\nQuestion: {}", context, user_query);
//! ```

#[cfg(feature = "rag")]
mod store;
#[cfg(feature = "rag")]
mod config;
#[cfg(feature = "rag")]
mod embedding;

#[cfg(feature = "rag")]
pub use store::*;
#[cfg(feature = "rag")]
pub use config::*;
#[cfg(feature = "rag")]
pub use embedding::*;
#[cfg(feature = "rag")]
pub use config::example_config;

use thiserror::Error;

/// RAG-related errors
#[derive(Error, Debug)]
pub enum RagError {
    #[error("Database connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Query failed: {0}")]
    QueryFailed(String),
    
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub type RagResult<T> = Result<T, RagError>;
