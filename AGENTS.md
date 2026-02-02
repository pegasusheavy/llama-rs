# AGENTS.md

Guidelines for AI coding assistants working on llama-rs.

## Project Overview

llama-rs is a Rust implementation of llama.cpp - a high-performance LLM inference engine. The goal is full feature parity with llama.cpp while providing idiomatic Rust APIs suitable for ecosystem contribution.

## Architecture

Monolithic library with feature flags. Key modules:

| Module | Purpose |
|--------|---------|
| `gguf/` | GGUF file format parsing and writing |
| `tensor/` | Tensor types, quantization, operations |
| `backend/` | Hardware backends (CPU, CUDA, Vulkan, Metal, ROCm) |
| `model/` | Model architectures (LLaMA, Mistral, Qwen, etc.) |
| `sampling/` | Token sampling strategies |
| `tokenizer/` | BPE, SentencePiece tokenizers |
| `server/` | HTTP server with OpenAI-compatible API |

## Code Style

- Follow standard Rust conventions (`cargo fmt`, `cargo clippy`)
- Use `thiserror` for error types
- Use `tracing` for logging, not `println!`
- Prefer `Result<T>` over panics
- Document public APIs with doc comments
- Use `#[repr(C)]` for types that must match llama.cpp memory layout

## Performance Guidelines

- Hot paths must avoid allocations - use scratch buffers
- Quantized operations work on blocks, not individual values
- SIMD code uses `pulp` for runtime dispatch (AVX/AVX2/AVX-512/NEON)
- Memory-map GGUF files, don't load entirely into RAM
- KV cache is pre-allocated, not grown dynamically

## Testing

- Unit tests for quantization roundtrips
- Integration tests load real GGUF files (small test models)
- Benchmark critical paths with `criterion`
- Test GPU backends with feature flags: `cargo test --features cuda`

## Common Tasks

### Adding a new model architecture

1. Add variant to `Architecture` enum in `model/mod.rs`
2. Create `model/<name>.rs` implementing the `Model` trait
3. Add architecture detection in `model/loader.rs` based on GGUF metadata
4. Add tests with a quantized test model

### Adding a new quantization format

1. Add variant to `DType` enum in `tensor/mod.rs`
2. Define block struct in `tensor/quantization.rs` with `#[repr(C)]`
3. Implement `quantize()` and `dequantize()` functions
4. Add SIMD-optimized versions in each backend
5. Add roundtrip tests

### Adding a new backend

1. Create `backend/<name>/` directory
2. Implement `Backend` trait
3. Add feature flag in `Cargo.toml`
4. Gate all backend code with `#[cfg(feature = "...")]`

## Reference Materials

- llama.cpp source: https://github.com/ggerganov/llama.cpp
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Design doc: `docs/plans/2026-02-02-llama-rs-design.md`

## Do Not

- Break GGUF compatibility with llama.cpp
- Add external tokenizer dependencies (load from GGUF only)
- Use unsafe without clear justification and safety comments
- Commit large model files to the repo
- Skip clippy warnings without `#[allow(...)]` with explanation
