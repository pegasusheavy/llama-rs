# llama-rs Design Document

A full Rust implementation of llama.cpp with feature parity.

## Goals

- Full llama.cpp feature parity
- All 50+ model architectures
- All hardware backends (CPU, CUDA, Vulkan, Metal, ROCm, SYCL)
- All quantization formats (basic, K-quants, IQ)
- Ecosystem contribution: clean APIs, good docs, contributor-friendly

## Project Structure

```
llama-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API re-exports
│   ├── main.rs             # CLI entry point
│   │
│   ├── gguf/               # GGUF file format
│   │   ├── mod.rs
│   │   ├── reader.rs       # Parse GGUF files
│   │   ├── writer.rs       # Write GGUF files
│   │   └── metadata.rs     # Key-value metadata handling
│   │
│   ├── tensor/             # Tensor operations
│   │   ├── mod.rs
│   │   ├── tensor.rs       # Core tensor type
│   │   ├── quantization.rs # Q4_0, Q8_0, K-quants, IQ formats
│   │   └── ops.rs          # Matmul, softmax, RoPE, etc.
│   │
│   ├── model/              # Model architectures
│   │   ├── mod.rs
│   │   ├── loader.rs       # Load weights from GGUF
│   │   ├── llama.rs        # LLaMA architecture
│   │   ├── mistral.rs      # Mistral architecture
│   │   └── ...             # Other architectures
│   │
│   ├── backend/            # Hardware backends
│   │   ├── mod.rs          # Backend trait
│   │   ├── cpu/            # CPU (AVX, NEON)
│   │   ├── cuda/           # NVIDIA CUDA
│   │   ├── vulkan/         # Vulkan compute
│   │   ├── metal/          # Apple Metal
│   │   └── rocm/           # AMD ROCm
│   │
│   ├── sampling/           # Token sampling strategies
│   ├── tokenizer/          # Tokenizer implementations
│   └── server/             # HTTP server
```

## Tensor System

```rust
pub struct Tensor {
    data: TensorStorage,
    shape: Vec<usize>,
    dtype: DType,
    backend: BackendType,
}

pub enum DType {
    F32, F16, BF16,
    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
    Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
    IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL,
}

#[repr(C)]
pub struct BlockQ4_0 {
    delta: f16,
    quants: [u8; 16],
}
```

## Backend Trait

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;

    // Memory management
    fn alloc(&self, size: usize) -> Result<BufferHandle>;
    fn free(&self, handle: BufferHandle);
    fn copy_to_device(&self, data: &[u8], handle: BufferHandle) -> Result<()>;
    fn copy_to_host(&self, handle: BufferHandle, dst: &mut [u8]) -> Result<()>;

    // Core operations
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()>;
    fn silu(&self, x: &mut Tensor) -> Result<()>;
    fn gelu(&self, x: &mut Tensor) -> Result<()>;
    fn softmax(&self, x: &mut Tensor, dim: i32) -> Result<()>;
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) -> Result<()>;
    fn rope(&self, q: &mut Tensor, k: &mut Tensor, pos: usize, config: &RopeConfig) -> Result<()>;
    fn quantize(&self, src: &Tensor, dtype: DType, dst: &mut Tensor) -> Result<()>;
    fn dequantize(&self, src: &Tensor, dst: &mut Tensor) -> Result<()>;
}
```

## Model Architecture

```rust
pub trait Model: Send + Sync {
    fn forward(&self, tokens: &[u32], pos: usize, ctx: &mut InferenceContext) -> Result<Tensor>;
    fn config(&self) -> &ModelConfig;
    fn architecture(&self) -> Architecture;
}

pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_config: RopeConfig,
    pub norm_eps: f32,
}

pub enum Architecture {
    Llama, Llama2, Llama3,
    Mistral, Mixtral,
    Qwen, Qwen2,
    Phi, Phi2, Phi3,
    Gemma, Gemma2,
    Falcon, StarCoder, MPT,
    // ... 50+ more
}
```

## KV Cache

```rust
pub struct KVCache {
    k_cache: Vec<Tensor>,
    v_cache: Vec<Tensor>,
    seq_len: usize,
    max_seq_len: usize,
}

pub struct InferenceContext {
    pub kv_cache: KVCache,
    pub backend: Arc<dyn Backend>,
    pub scratch: ScratchBuffer,
}
```

## Sampling

```rust
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub mirostat: Option<MirostatConfig>,
    pub seed: Option<u64>,
}
```

Sampling pipeline order:
1. Apply repeat/frequency/presence penalties
2. Temperature scaling
3. Top-K filtering
4. Top-P (nucleus) filtering
5. Min-P filtering
6. Typical sampling (if enabled)
7. Mirostat (if enabled)
8. Random selection

## Tokenizer

```rust
pub enum TokenizerType {
    BPE,
    SentencePiece,
    WordPiece,
    Unigram,
}

pub struct Tokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: Vec<(u32, u32)>,
    scores: Vec<f32>,
    tokenizer_type: TokenizerType,
    bos_token: u32,
    eos_token: u32,
    pad_token: Option<u32>,
    unk_token: Option<u32>,
}
```

## Server

OpenAI-compatible endpoints:
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `POST /v1/embeddings` - Embeddings
- `GET /v1/models` - List models

llama.cpp-specific endpoints:
- `POST /completion` - Native format
- `POST /tokenize` - Text to tokens
- `POST /detokenize` - Tokens to text
- `GET /health` - Health check
- `GET /slots` - Slot status

## CLI

```
llama-rs run <model.gguf> [options]
    -p, --prompt <text>
    -n, --n-predict <N>
    -c, --ctx-size <N>
    --temp <float>
    --top-k <N>
    --top-p <float>
    -i, --interactive
    -ngl, --n-gpu-layers <N>
    --backend <name>
    -t, --threads <N>

llama-rs serve <model.gguf> [options]
    --host <addr>
    --port <N>
    --parallel <N>

llama-rs info <model.gguf>
llama-rs quantize <in> <out> <type>
llama-rs convert <in> <out>
```

## Dependencies

```toml
[features]
default = ["cpu"]
cpu = []
cuda = ["dep:cudarc"]
vulkan = ["dep:ash", "dep:gpu-allocator"]
metal = ["dep:metal", "dep:objc"]
rocm = ["dep:hip-runtime"]
server = ["dep:axum", "dep:tokio", "dep:tower-http"]

[dependencies]
memmap2 = "0.9"
bytemuck = "1.14"
half = "2.3"
rayon = "1.8"
thiserror = "1.0"
tracing = "0.1"
wide = "0.7"
pulp = "0.18"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Optional
cudarc = { version = "0.11", optional = true }
ash = { version = "0.38", optional = true }
metal = { version = "0.28", optional = true }
axum = { version = "0.7", optional = true }
tokio = { version = "1.35", features = ["full"], optional = true }
```
