//! HTTP server setup and routing

use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

use crate::gguf::GgufFile;
use crate::model::ModelLoader;
use crate::tokenizer::Tokenizer;

use super::handlers::{self, AppState};

/// Server configuration
pub struct ServerConfig {
    /// Host address to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Path to the GGUF model file
    pub model_path: String,
}

/// Run the HTTP server
pub async fn run_server(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model from: {}", config.model_path);

    // Load GGUF file and tokenizer
    let gguf = GgufFile::open(&config.model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    eprintln!("Tokenizer loaded: {} tokens", tokenizer.vocab_size);

    // Load model
    let loader = ModelLoader::load(&config.model_path)?;
    let model_config = loader.config().clone();
    eprintln!(
        "Model config: {} layers, {} heads, {} dim",
        model_config.num_layers, model_config.num_heads, model_config.hidden_size
    );

    let model = loader.build_model()?;
    eprintln!("Model loaded successfully");

    // Extract model name from path
    let model_name = std::path::Path::new(&config.model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("llama")
        .to_string();

    // Create shared state
    let state = Arc::new(AppState {
        model: Arc::new(model),
        tokenizer: Arc::new(tokenizer),
        config: model_config,
        model_name,
        inference_lock: Mutex::new(()),
    });

    // Setup CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/models", get(handlers::list_models))
        // Health and status
        .route("/health", get(handlers::health))
        .route("/", get(|| async { "llama-rs server" }))
        .layer(cors)
        .with_state(state);

    let addr = format!("{}:{}", config.host, config.port);
    let socket_addr: SocketAddr = addr.parse()?;

    eprintln!();
    eprintln!("╭─────────────────────────────────────────────────────────────────╮");
    eprintln!("│                     llama-rs Server                              │");
    eprintln!("├─────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Listening on: http://{:<44} │", addr);
    eprintln!("├─────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Endpoints:                                                       │");
    eprintln!("│   POST /v1/chat/completions  - Chat completions (OpenAI API)     │");
    eprintln!("│   POST /v1/completions       - Text completions (OpenAI API)     │");
    eprintln!("│   GET  /v1/models            - List models                       │");
    eprintln!("│   GET  /health               - Health check                      │");
    eprintln!("╰─────────────────────────────────────────────────────────────────╯");
    eprintln!();

    let listener = tokio::net::TcpListener::bind(socket_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
