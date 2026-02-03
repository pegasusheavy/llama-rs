//! HTTP request handlers

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, Stream};
use tokio::sync::Mutex;

use crate::model::{InferenceContext, ModelConfig};
use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use crate::Model;

use super::types::*;

/// Shared application state
pub struct AppState {
    pub model: Arc<dyn Model>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: ModelConfig,
    pub model_name: String,
    /// Mutex to serialize inference requests (single-threaded for now)
    pub inference_lock: Mutex<()>,
}

/// Health check endpoint
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_name.clone(),
        context_size: state.config.max_seq_len,
    })
}

/// List models endpoint
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created,
            owned_by: "llama-rs".to_string(),
        }],
    })
}

/// Chat completions endpoint
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Acquire inference lock
    let _lock = state.inference_lock.lock().await;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let request_id = format!("chatcmpl-{}", created);

    // Format messages into prompt
    let prompt = format_chat_messages(&request.messages);

    // Create sampler
    let sampler_config = SamplerConfig {
        temperature: request.temperature,
        top_p: request.top_p,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        ..Default::default()
    };

    // Generate response
    match generate_response(
        &state,
        &prompt,
        request.max_tokens,
        sampler_config,
        request.stop.as_deref(),
    )
    .await
    {
        Ok((response_text, prompt_tokens, completion_tokens)) => {
            if request.stream {
                // Streaming response
                let stream = create_chat_stream(
                    request_id,
                    state.model_name.clone(),
                    created,
                    response_text,
                );
                Sse::new(stream).into_response()
            } else {
                // Non-streaming response
                let response = ChatCompletionResponse {
                    id: request_id,
                    object: "chat.completion".to_string(),
                    created,
                    model: state.model_name.clone(),
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatMessage {
                            role: Role::Assistant,
                            content: response_text,
                        },
                        finish_reason: "stop".to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                };
                Json(response).into_response()
            }
        }
        Err(e) => {
            let error = ErrorResponse::new(e.to_string(), "server_error");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
    }
}

/// Text completions endpoint
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    let _lock = state.inference_lock.lock().await;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let request_id = format!("cmpl-{}", created);

    let sampler_config = SamplerConfig {
        temperature: request.temperature,
        top_p: request.top_p,
        ..Default::default()
    };

    match generate_response(
        &state,
        &request.prompt,
        request.max_tokens,
        sampler_config,
        request.stop.as_deref(),
    )
    .await
    {
        Ok((response_text, prompt_tokens, completion_tokens)) => {
            let response = CompletionResponse {
                id: request_id,
                object: "text_completion".to_string(),
                created,
                model: state.model_name.clone(),
                choices: vec![CompletionChoice {
                    text: response_text,
                    index: 0,
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            Json(response).into_response()
        }
        Err(e) => {
            let error = ErrorResponse::new(e.to_string(), "server_error");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
    }
}

/// Format chat messages into a prompt string
fn format_chat_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for (i, msg) in messages.iter().enumerate() {
        match msg.role {
            Role::System => {
                prompt.push_str(&format!(
                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n",
                    msg.content
                ));
            }
            Role::User => {
                if i == 0 || matches!(messages.get(i - 1).map(|m| &m.role), Some(Role::System)) {
                    prompt.push_str(&format!("{} [/INST]", msg.content));
                } else {
                    prompt.push_str(&format!(" [INST] {} [/INST]", msg.content));
                }
            }
            Role::Assistant => {
                prompt.push_str(&format!(" {}", msg.content));
            }
        }
    }

    prompt
}

/// Generate text response
async fn generate_response(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
    sampler_config: SamplerConfig,
    _stop_sequences: Option<&[String]>,
) -> Result<(String, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    // Create a new context for this request
    let backend: Arc<dyn crate::Backend> =
        Arc::new(crate::backend::cpu::CpuBackend::new());
    let mut ctx = InferenceContext::new(&state.config, backend);
    let mut sampler = Sampler::new(sampler_config, state.config.vocab_size);

    // Encode prompt
    let prompt_tokens = state.tokenizer.encode(prompt, true)?;
    let prompt_len = prompt_tokens.len();

    let mut all_tokens = prompt_tokens.clone();

    // Process prompt tokens
    for (i, &token) in prompt_tokens.iter().enumerate() {
        if i < state.config.max_seq_len {
            let _ = state.model.forward(&[token], &mut ctx);
        }
    }

    // Generate response tokens
    let mut response_text = String::new();
    let mut completion_tokens = 0;

    for _ in 0..max_tokens {
        let last_token = *all_tokens.last().unwrap_or(&state.tokenizer.special_tokens.bos_token_id);

        // Forward pass
        let logits = state.model.forward(&[last_token], &mut ctx)?;

        // Sample next token
        let next_token = sampler.sample(&logits, &all_tokens);

        // Check for EOS
        if next_token == state.tokenizer.special_tokens.eos_token_id {
            break;
        }

        // Decode token
        if let Ok(text) = state.tokenizer.decode(&[next_token]) {
            // Check for stop patterns
            if text.contains("[INST]") || text.contains("</s>") {
                break;
            }
            response_text.push_str(&text);
        }

        all_tokens.push(next_token);
        completion_tokens += 1;
    }

    Ok((response_text, prompt_len, completion_tokens))
}

/// Create streaming response for chat completions
fn create_chat_stream(
    request_id: String,
    model: String,
    created: u64,
    response_text: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    // For simplicity, we send the whole response as a single chunk
    // A proper implementation would stream token by token
    let chunks = vec![
        // Initial chunk with role
        ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: Some(Role::Assistant),
                    content: None,
                },
                finish_reason: None,
            }],
        },
        // Content chunk
        ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: None,
                    content: Some(response_text),
                },
                finish_reason: None,
            }],
        },
        // Final chunk
        ChatCompletionChunk {
            id: request_id,
            object: "chat.completion.chunk".to_string(),
            created,
            model,
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        },
    ];

    stream::iter(chunks.into_iter().map(|chunk| {
        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok(Event::default().data(data))
    }))
}
