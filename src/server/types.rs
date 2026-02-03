//! OpenAI API compatible types

use serde::{Deserialize, Serialize};

/// Chat message role
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// Chat completion request
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model to use (ignored, uses loaded model)
    #[serde(default)]
    pub model: String,
    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-P sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: f32,
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chat completion chunk
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

/// Streaming choice delta
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: usize,
    pub delta: ChatCompletionDelta,
    pub finish_reason: Option<String>,
}

/// Content delta for streaming
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Text completion request
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model to use
    #[serde(default)]
    pub model: String,
    /// Prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-P sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Whether to stream
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// Text completion choice
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

/// Text completion response
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

/// Model info for /v1/models endpoint
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Models list response
#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Health check response
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub context_size: usize,
}

/// Error response
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                r#type: error_type.into(),
                code: None,
            },
        }
    }
}
