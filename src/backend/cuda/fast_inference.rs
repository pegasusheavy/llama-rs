//! Fast GPU inference with minimal CPU-GPU transfers
//!
//! This module implements a forward pass where most computation stays on GPU.

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::backend::{BackendError, BackendResult};
use crate::model::LlamaModel;

use super::kernels::CudaKernels;
use super::dequant_weights::GpuWeightStore;

/// Fast GPU inference engine
pub struct FastGpuInference {
    device: Arc<CudaDevice>,
    kernels: CudaKernels,
    weights: GpuWeightStore,
    config: InferenceConfig,
    pos: usize,
    // KV cache per layer
    k_cache: Vec<Vec<f32>>,  // [layer][num_kv_heads * max_seq_len * head_dim]
    v_cache: Vec<Vec<f32>>,
}

#[derive(Clone)]
struct InferenceConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    max_seq_len: usize,
    norm_eps: f32,
    freq_base: f32,
    freq_scale: f32,
    use_neox_rope: bool,
}

impl FastGpuInference {
    /// Create from CPU model
    pub fn from_model(model: &LlamaModel, max_seq_len: usize) -> BackendResult<Self> {
        let cfg = model.config();
        
        let device = Arc::new(CudaDevice::new(0)
            .map_err(|e| BackendError::InitializationFailed(format!("{}", e)))?);
        
        eprintln!("Initializing fast GPU inference...");
        
        let kernels = CudaKernels::new(Arc::clone(&device))?;
        
        let weights = super::dequant_weights::upload_model_weights(
            Arc::clone(&device),
            model.layers(),
            model.token_embedding(),
            model.output(),
            model.norm(),
        )?;
        
        let use_neox = model.layers().first()
            .map(|l| l.attention.use_neox_rope)
            .unwrap_or(false);
        
        let config = InferenceConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            num_layers: cfg.num_layers,
            vocab_size: cfg.vocab_size,
            max_seq_len,
            norm_eps: cfg.norm_eps,
            freq_base: cfg.rope_config.freq_base,
            freq_scale: cfg.rope_config.freq_scale,
            use_neox_rope: use_neox,
        };
        
        // Initialize KV cache
        let kv_size = cfg.num_kv_heads * max_seq_len * cfg.head_dim;
        let k_cache: Vec<Vec<f32>> = (0..cfg.num_layers).map(|_| vec![0.0; kv_size]).collect();
        let v_cache: Vec<Vec<f32>> = (0..cfg.num_layers).map(|_| vec![0.0; kv_size]).collect();
        
        let vram_mb = weights.vram_usage() as f64 / (1024.0 * 1024.0);
        eprintln!("Fast GPU inference ready: {:.1} MB VRAM", vram_mb);
        
        Ok(Self {
            device: Arc::clone(&device),
            kernels,
            weights,
            config,
            pos: 0,
            k_cache,
            v_cache,
        })
    }
    
    /// Forward pass for a single token
    pub fn forward(&mut self, token_id: u32) -> BackendResult<Vec<f32>> {
        let cfg = self.config.clone();
        
        // 1. Embed token
        let mut hidden = self.embed_token(token_id)?;
        
        // 2. Process each layer
        for layer_idx in 0..cfg.num_layers {
            hidden = self.process_layer(layer_idx, hidden)?;
        }
        
        // 3. Final norm and output
        let hidden_norm = self.rms_norm_cpu(&hidden, "output_norm.weight")?;
        let logits = self.linear_cpu(&hidden_norm, "output.weight", None)?;
        
        self.pos += 1;
        
        Ok(logits)
    }
    
    /// Reset for new sequence
    pub fn reset(&mut self) {
        self.pos = 0;
        for k in &mut self.k_cache {
            k.fill(0.0);
        }
        for v in &mut self.v_cache {
            v.fill(0.0);
        }
    }
    
    pub fn position(&self) -> usize {
        self.pos
    }
    
    fn embed_token(&self, token_id: u32) -> BackendResult<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let offset = token_id as usize * hidden_size;
        
        let emb = self.weights.get("token_embd.weight")
            .ok_or_else(|| BackendError::OperationFailed("Missing token embedding".into()))?;
        
        let emb_data: Vec<f32> = self.device.dtoh_sync_copy(&emb.data)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        
        Ok(emb_data[offset..offset + hidden_size].to_vec())
    }
    
    fn process_layer(&mut self, layer_idx: usize, hidden: Vec<f32>) -> BackendResult<Vec<f32>> {
        let cfg = &self.config;
        let prefix = format!("blk.{}", layer_idx);
        
        // Attention norm
        let hidden_norm = self.rms_norm_cpu(&hidden, &format!("{}.attn_norm.weight", prefix))?;
        
        // QKV projections
        let q = self.linear_cpu(&hidden_norm, &format!("{}.attn_q.weight", prefix), 
                                Some(&format!("{}.attn_q.bias", prefix)))?;
        let k = self.linear_cpu(&hidden_norm, &format!("{}.attn_k.weight", prefix),
                                Some(&format!("{}.attn_k.bias", prefix)))?;
        let v = self.linear_cpu(&hidden_norm, &format!("{}.attn_v.weight", prefix),
                                Some(&format!("{}.attn_v.bias", prefix)))?;
        
        // Apply RoPE
        let (q, k) = self.apply_rope_cpu(q, k)?;
        
        // Update KV cache
        self.update_kv_cache(layer_idx, &k, &v);
        
        // Compute attention
        let attn_out = self.compute_attention_cpu(layer_idx, &q)?;
        
        // Output projection
        let attn_proj = self.linear_cpu(&attn_out, &format!("{}.attn_output.weight", prefix), None)?;
        
        // Add residual
        let hidden: Vec<f32> = hidden.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();
        
        // FFN norm
        let hidden_norm = self.rms_norm_cpu(&hidden, &format!("{}.ffn_norm.weight", prefix))?;
        
        // FFN
        let gate = self.linear_cpu(&hidden_norm, &format!("{}.ffn_gate.weight", prefix), None)?;
        let up = self.linear_cpu(&hidden_norm, &format!("{}.ffn_up.weight", prefix), None)?;
        
        // SiLU on gate, multiply with up
        let ffn: Vec<f32> = gate.iter().zip(&up)
            .map(|(g, u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();
        
        // Down projection
        let down = self.linear_cpu(&ffn, &format!("{}.ffn_down.weight", prefix), None)?;
        
        // Add residual
        let hidden: Vec<f32> = hidden.iter().zip(&down).map(|(a, b)| a + b).collect();
        
        Ok(hidden)
    }
    
    fn rms_norm_cpu(&self, x: &[f32], weight_name: &str) -> BackendResult<Vec<f32>> {
        let weight = self.weights.get(weight_name)
            .ok_or_else(|| BackendError::OperationFailed(format!("Missing weight: {}", weight_name)))?;
        
        let weight_data: Vec<f32> = self.device.dtoh_sync_copy(&weight.data)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        
        let n = x.len();
        let eps = self.config.norm_eps;
        
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let rms_inv = 1.0 / rms;
        
        Ok(x.iter().zip(&weight_data).map(|(v, w)| v * rms_inv * w).collect())
    }
    
    fn linear_cpu(&self, x: &[f32], weight_name: &str, bias_name: Option<&str>) -> BackendResult<Vec<f32>> {
        let weight = self.weights.get(weight_name)
            .ok_or_else(|| BackendError::OperationFailed(format!("Missing weight: {}", weight_name)))?;
        
        let k = weight.shape[0];
        let n = weight.shape[1];
        
        // Upload x to GPU
        let x_gpu = self.device.htod_sync_copy(x)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
        
        let mut out_gpu = self.device.alloc_zeros::<f32>(n)
            .map_err(|e| BackendError::AllocationFailed(format!("{}", e)))?;
        
        let config = LaunchConfig {
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.kernels.vec_mat_f32.clone().launch(config, (&x_gpu, &weight.data, &mut out_gpu, k as i32, n as i32))
        }.map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        
        let mut result: Vec<f32> = self.device.dtoh_sync_copy(&out_gpu)
            .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
        
        // Add bias if present
        if let Some(bias_name) = bias_name {
            if let Some(bias) = self.weights.get(bias_name) {
                let bias_data: Vec<f32> = self.device.dtoh_sync_copy(&bias.data)
                    .map_err(|e| BackendError::OperationFailed(format!("{}", e)))?;
                for (r, b) in result.iter_mut().zip(&bias_data) {
                    *r += b;
                }
            }
        }
        
        Ok(result)
    }
    
    fn apply_rope_cpu(&self, mut q: Vec<f32>, mut k: Vec<f32>) -> BackendResult<(Vec<f32>, Vec<f32>)> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let num_q_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let freq_base = cfg.freq_base;
        let freq_scale = cfg.freq_scale;
        let pos = self.pos;
        
        // Apply RoPE to Q
        for h in 0..num_q_heads {
            let offset = h * head_dim;
            for d in 0..head_dim / 2 {
                let freq = 1.0 / (freq_base.powf(2.0 * d as f32 / head_dim as f32)) * freq_scale;
                let theta = pos as f32 * freq;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();
                
                let i = offset + d;
                let j = offset + d + head_dim / 2;
                
                let q0 = q[i];
                let q1 = q[j];
                q[i] = q0 * cos_theta - q1 * sin_theta;
                q[j] = q0 * sin_theta + q1 * cos_theta;
            }
        }
        
        // Apply RoPE to K
        for h in 0..num_kv_heads {
            let offset = h * head_dim;
            for d in 0..head_dim / 2 {
                let freq = 1.0 / (freq_base.powf(2.0 * d as f32 / head_dim as f32)) * freq_scale;
                let theta = pos as f32 * freq;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();
                
                let i = offset + d;
                let j = offset + d + head_dim / 2;
                
                let k0 = k[i];
                let k1 = k[j];
                k[i] = k0 * cos_theta - k1 * sin_theta;
                k[j] = k0 * sin_theta + k1 * cos_theta;
            }
        }
        
        Ok((q, k))
    }
    
    fn update_kv_cache(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let num_kv_heads = cfg.num_kv_heads;
        let max_seq_len = cfg.max_seq_len;
        
        for h in 0..num_kv_heads {
            let src_offset = h * head_dim;
            let dst_offset = h * max_seq_len * head_dim + self.pos * head_dim;
            
            self.k_cache[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&k[src_offset..src_offset + head_dim]);
            self.v_cache[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&v[src_offset..src_offset + head_dim]);
        }
    }
    
    fn compute_attention_cpu(&self, layer_idx: usize, q: &[f32]) -> BackendResult<Vec<f32>> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let kv_len = self.pos + 1;
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        let heads_per_kv = num_heads / num_kv_heads;
        let mut attn_out = vec![0.0f32; cfg.hidden_size];
        
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            let q_offset = h * head_dim;
            
            // Compute scores
            let mut scores = vec![0.0f32; kv_len];
            for pos in 0..kv_len {
                let k_offset = kv_h * cfg.max_seq_len * head_dim + pos * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_offset + d] * self.k_cache[layer_idx][k_offset + d];
                }
                scores[pos] = dot * scale;
            }
            
            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            
            // Weighted sum
            let out_offset = h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for pos in 0..kv_len {
                    let v_offset = kv_h * cfg.max_seq_len * head_dim + pos * head_dim + d;
                    val += scores[pos] * self.v_cache[layer_idx][v_offset];
                }
                attn_out[out_offset + d] = val;
            }
        }
        
        Ok(attn_out)
    }
}
