//! HTTP server with OpenAI-compatible API
//!
//! This module provides an HTTP server that implements the OpenAI API format
//! for chat completions and text completions.

mod api;
pub mod batch;
mod handlers;
mod types;

pub use api::{run_server, ServerConfig};
pub use batch::{
    BatchConfig, BatchScheduler, FinishReason, GenerationEvent, GenerationRequest, RequestId,
    SharedBatchScheduler, new_batch_scheduler,
};
pub use types::*;
