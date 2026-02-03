# AGENTS.md

Guidelines for AI coding assistants working on llama-gguf.

## Project Overview

llama-gguf is a Rust implementation of llama.cpp - a high-performance LLM inference engine. The goal is full feature parity with llama.cpp while providing idiomatic Rust APIs suitable for ecosystem contribution.

## Architecture

Monolithic library with feature flags. Key modules:

| Module | Purpose |
|--------|---------|
| `gguf/` | GGUF file format parsing and writing |
| `tensor/` | Tensor types, quantization (Q2_K through Q8_0), operations |
| `backend/` | Hardware backends (CPU with SIMD, Vulkan planned) |
| `model/` | Model architectures (LLaMA, Mistral, Qwen2, TinyLlama, DeepSeek) |
| `sampling/` | Token sampling strategies (greedy, top-k, top-p, temperature) |
| `tokenizer/` | BPE tokenizer loaded from GGUF metadata |
| `server/` | HTTP server with OpenAI-compatible API |
| `huggingface.rs` | HuggingFace Hub model downloading |

## Supported Models

Models verified to work correctly:

| Model | RoPE Type | Notes |
|-------|-----------|-------|
| Qwen2/Qwen2.5 | NeoX | Uses attention biases |
| TinyLlama | Normal | GQA with 4 KV heads |
| Mistral | Normal | Requires instruction format `[INST]...[/INST]` |
| DeepSeek-Coder | Normal | Uses `scale_linear=4.0` for RoPE |
| LLaMA/LLaMA2/LLaMA3 | Normal | Standard architecture |

**Not supported** (different architecture):
- Phi-2/GPT-NeoX (combined QKV tensors)
- Gemma2 (extra norm layers, logit softcapping)

See `docs/MODEL_COMPATIBILITY.md` for full details.

## Code Style

- Follow standard Rust conventions (`cargo fmt`, `cargo clippy`)
- Use `thiserror` for error types
- Use `tracing` for logging, not `println!` or `eprintln!`
- Prefer `Result<T>` over panics
- Document public APIs with doc comments
- Use `#[repr(C)]` for types that must match llama.cpp memory layout

## Performance Guidelines

- Hot paths must avoid allocations - use scratch buffers
- Quantized operations work on blocks, not individual values
- SIMD code uses runtime feature detection (AVX2/AVX-512/NEON)
- Memory-map GGUF files, don't load entirely into RAM
- KV cache is pre-allocated, not grown dynamically

## Testing

- Unit tests for quantization roundtrips
- Integration tests load real GGUF files (small test models)
- Benchmark critical paths with `criterion` (v0.5+)
- Run all tests: `cargo test --release`
- Test with a model: `cargo run --release -- run <model.gguf> -p "test" -n 10`

## Common Tasks

### Adding a new model architecture

1. Add variant to `Architecture` enum in `model/architecture.rs`
2. Update `is_llama_like()` if architecture uses LLaMA tensor naming
3. Set correct `RopeType` in `model/loader.rs` (Normal vs NeoX)
4. Handle any architecture-specific config (biases, GQA, etc.)
5. Test with `cargo run --release -- info <model.gguf>` then inference

### RoPE Configuration

Two RoPE styles are supported:

- **Normal** (type 0): Consecutive pairs `(x[2i], x[2i+1])` - LLaMA, Mistral, TinyLlama
- **NeoX** (type 2): Split pairs `(x[i], x[i+d/2])` - Qwen2

Set in `model/loader.rs` based on architecture. Also handle:
- `freq_base`: Varies by model (10000, 100000, 1000000)
- `freq_scale`: Linear scaling factor (positions divided by this)

### Adding a new quantization format

1. Add variant to `DType` enum in `tensor/dtype.rs`
2. Define block struct in `tensor/quant/blocks.rs` with `#[repr(C)]`
3. Implement `dequantize_*` function in `tensor/quant/dequant.rs`
4. Add to match statements in `tensor/quant/mod.rs`
5. Add SIMD-optimized version in `backend/cpu/simd.rs`
6. Add roundtrip tests

### Adding a new backend

1. Create `backend/<name>/` directory
2. Implement `Backend` trait from `backend/mod.rs`
3. Add feature flag in `Cargo.toml`
4. Gate all backend code with `#[cfg(feature = "...")]`

## Debugging Models

When a model produces incorrect output:

1. Check architecture: `cargo run --release -- info <model.gguf>`
2. Verify RoPE type matches llama.cpp (use `llama-cpp-python` verbose mode)
3. Compare hidden states layer-by-layer using `examples/trace_all_layers.rs`
4. Check tensor shapes match expected dimensions
5. Verify dequantization produces reasonable values

## Reference Materials

- llama.cpp source: https://github.com/ggerganov/llama.cpp
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Model compatibility: `docs/MODEL_COMPATIBILITY.md`

## Do Not

- Break GGUF compatibility with llama.cpp
- Add external tokenizer dependencies (load from GGUF only)
- Use unsafe without clear justification and safety comments
- Commit large model files (*.gguf, *.bin) to the repo
- Skip clippy warnings without `#[allow(...)]` with explanation
- Use `println!`/`eprintln!` for debug output in library code
