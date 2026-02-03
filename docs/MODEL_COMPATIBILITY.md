# Model Compatibility

This document tracks GGUF models that have been tested with llama-rs.

## Supported Architectures

llama-rs currently supports models that use LLaMA-style tensor naming conventions:

| Architecture | Status | Notes |
|--------------|--------|-------|
| LLaMA | ✅ Supported | Original LLaMA, LLaMA 2, LLaMA 3 |
| Mistral | ✅ Supported | Mistral 7B and derivatives |
| Qwen/Qwen2 | ✅ Supported | Includes attention biases |
| CodeLlama | ✅ Supported | Code-focused LLaMA variant |
| Yi | ✅ Supported | LLaMA-like architecture |
| DeepSeek | ✅ Supported | LLaMA-compatible models |
| InternLM | ✅ Supported | Uses LLaMA tensor names |
| Baichuan | ✅ Supported | Uses LLaMA tensor names |
| GPT-NeoX | ❌ Not Supported | Uses combined QKV tensors |
| GPT-2 | ❌ Not Supported | Different architecture |
| BLOOM | ❌ Not Supported | Different architecture |
| Falcon | ❌ Not Supported | Different attention structure |
| MPT | ❌ Not Supported | Different architecture |

## Tested Models

### Successfully Loading Models

| Model | HuggingFace Repo | Size | Quantization | Load | Inference |
|-------|------------------|------|--------------|------|-----------|
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | 469 MB | Q4_K_M | ✅ | ⚠️ |
| TinyLlama-1.1B-Chat | `TinyLlama/TinyLlama-1.1B-Chat-v0.2-GGUF` | 607 MB | Q4_0 | ✅ | ⚠️ |
| SmolLM-135M-Instruct | `MaziyarPanahi/SmolLM-135M-Instruct-GGUF` | 101 MB | Q4_K_M | ✅ | ⚠️ |
| DeepSeek-Coder-1.3B | `TheBloke/deepseek-coder-1.3b-instruct-GGUF` | 602 MB | Q2_K | ✅ | ⚠️ |

**Legend:**
- ✅ = Working
- ⚠️ = Loads and runs but output quality needs improvement
- ❌ = Not working

### Models That Fail to Load

| Model | HuggingFace Repo | Architecture | Error |
|-------|------------------|--------------|-------|
| Pythia-70M | `afrideva/pythia-70m-deduped-alpaca-cleaned-GGUF` | gptneox | Missing tensor: `blk.0.attn_q.weight` (uses combined QKV) |

## Quantization Format Support

| Format | Status | Notes |
|--------|--------|-------|
| F32 | ✅ | Full precision |
| F16 | ✅ | Half precision |
| Q8_0 | ✅ | 8-bit quantization |
| Q8_1 | ✅ | 8-bit with improved accuracy |
| Q6_K | ✅ | 6-bit K-quant |
| Q5_0 | ✅ | 5-bit quantization |
| Q5_1 | ✅ | 5-bit with improved accuracy |
| Q5_K | ✅ | 5-bit K-quant |
| Q4_0 | ✅ | 4-bit quantization |
| Q4_1 | ✅ | 4-bit with improved accuracy |
| Q4_K | ✅ | 4-bit K-quant |
| Q3_K | ✅ | 3-bit K-quant |
| Q2_K | ✅ | 2-bit K-quant |

## Downloading Models

Use the built-in download command to fetch models from HuggingFace:

```bash
# List available files in a repository
llama-rs download Qwen/Qwen2.5-0.5B-Instruct-GGUF

# Download a specific file
llama-rs download Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf

# Search for models
llama-rs models search "llama gguf"

# List cached models
llama-rs models list
```

## Model Features

### Weight Tying

Models that share embedding weights with output projection (weight tying) are automatically detected. If `output.weight` tensor is missing, the token embedding weights are used for the output projection.

**Models using weight tying:**
- SmolLM series
- Some smaller models

### Attention Biases

Some model architectures include biases in attention projections:

**Models with attention biases:**
- Qwen/Qwen2 series (Q, K, V biases)

### Grouped Query Attention (GQA)

Models with fewer KV heads than query heads are supported:

| Model | Query Heads | KV Heads | GQA Ratio |
|-------|-------------|----------|-----------|
| TinyLlama-1.1B | 32 | 4 | 8:1 |
| Qwen2.5-0.5B | 14 | 2 | 7:1 |
| SmolLM-135M | 9 | 3 | 3:1 |

## Known Issues

1. **Output Quality**: All models currently produce degraded output. This is under investigation and likely related to:
   - Grouped Query Attention implementation
   - RoPE positional encoding
   - Numerical precision in quantized operations

2. **GPT-NeoX Architecture**: Models using combined QKV projections (`attn_qkv.weight`) are not supported. This affects:
   - Pythia series
   - GPT-NeoX series
   - Some StableLM models

## Adding New Models

To test a new model:

1. Download using `llama-rs download <repo>`
2. Check model info with `llama-rs info <path>`
3. Verify architecture is LLaMA-compatible
4. Test inference with `llama-rs run <path> --prompt "test"`

Report compatibility results by opening an issue on GitHub.

## Test Environment

- **llama-rs version**: 0.1.0
- **Test date**: February 2026
- **Platform**: Linux x86_64
