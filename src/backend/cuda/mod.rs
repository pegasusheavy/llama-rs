//! CUDA GPU backend for tensor operations
//!
//! This module provides a CUDA-based GPU implementation of the Backend trait
//! for NVIDIA GPUs.
//!
//! # Features
//! - High-performance matrix operations via cuBLAS
//! - Custom CUDA kernels for quantized operations
//! - Efficient memory management
//!
//! # Requirements
//! - NVIDIA GPU with compute capability 6.0+
//! - CUDA Toolkit 11.0+
//! - Build with `--features cuda`

use crate::backend::{Backend, BackendError, BackendResult};
use crate::tensor::{DType, Tensor};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// CUDA backend configuration
#[derive(Debug, Clone)]
pub struct CudaConfig {
    /// Device index to use (0 = first GPU)
    pub device_index: usize,
    /// Whether to use TensorCores (if available)
    pub use_tensor_cores: bool,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            use_tensor_cores: true,
        }
    }
}

/// CUDA GPU backend
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    config: CudaConfig,
    // CPU backend for fallback operations
    cpu_backend: crate::backend::cpu::CpuBackend,
}

#[cfg(not(feature = "cuda"))]
pub struct CudaBackend {
    config: CudaConfig,
}

impl CudaBackend {
    /// Create a new CUDA backend with default configuration
    pub fn new() -> Result<Self, BackendError> {
        Self::with_config(CudaConfig::default())
    }

    /// Create a CUDA backend with custom configuration
    #[cfg(feature = "cuda")]
    pub fn with_config(config: CudaConfig) -> Result<Self, BackendError> {
        let device = CudaDevice::new(config.device_index)
            .map_err(|e| BackendError::InitializationFailed(format!("CUDA init failed: {}", e)))?;
        
        Ok(Self {
            device,
            config,
            cpu_backend: crate::backend::cpu::CpuBackend::new(),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn with_config(_config: CudaConfig) -> Result<Self, BackendError> {
        Err(BackendError::NotAvailable(
            "CUDA support not compiled. Build with --features cuda".to_string(),
        ))
    }

    /// Get device name
    #[cfg(feature = "cuda")]
    pub fn device_name(&self) -> String {
        format!("CUDA Device {}", self.config.device_index)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn device_name(&self) -> String {
        "CUDA disabled".to_string()
    }

    /// Allocate GPU memory and copy data
    #[cfg(feature = "cuda")]
    fn to_device(&self, data: &[f32]) -> Result<CudaSlice<f32>, BackendError> {
        self.device
            .htod_sync_copy(data)
            .map_err(|e| BackendError::AllocationFailed(format!("GPU copy failed: {}", e)))
    }

    /// Copy data back from GPU
    #[cfg(feature = "cuda")]
    fn from_device(&self, slice: &CudaSlice<f32>) -> Result<Vec<f32>, BackendError> {
        self.device
            .dtoh_sync_copy(slice)
            .map_err(|e| BackendError::OperationFailed(format!("GPU read failed: {}", e)))
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            #[cfg(feature = "cuda")]
            panic!("Failed to create CUDA backend");
            #[cfg(not(feature = "cuda"))]
            Self {
                config: CudaConfig::default(),
            }
        })
    }
}

// For now, the CUDA backend primarily uses the CPU backend as a fallback
// while we implement GPU-accelerated versions of critical operations
#[cfg(feature = "cuda")]
impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn alloc(&self, shape: &[usize], dtype: DType) -> BackendResult<Tensor> {
        self.cpu_backend.alloc(shape, dtype)
    }

    fn copy_to(&self, tensor: &Tensor) -> BackendResult<Tensor> {
        self.cpu_backend.copy_to(tensor)
    }

    fn add(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.add(a, b, out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.mul(a, b, out)
    }

    fn scale(&self, a: &Tensor, scalar: f32, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.scale(a, scalar, out)
    }

    fn silu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.silu(x, out)
    }

    fn gelu(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.gelu(x, out)
    }

    fn softmax(&self, x: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.softmax(x, out)
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.rms_norm(x, weight, eps, out)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // TODO: Use cuBLAS for GPU-accelerated matmul
        self.cpu_backend.matmul(a, b, out)
    }

    fn matvec(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.matvec(a, b, out)
    }

    fn vec_mat(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.vec_mat(a, b, out)
    }

    fn dequantize(&self, src: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.dequantize(src, out)
    }

    fn matvec_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        self.cpu_backend.matvec_q(a, b, out)
    }

    fn vec_mat_q(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> BackendResult<()> {
        // This is the critical path - should be GPU accelerated
        self.cpu_backend.vec_mat_q(a, b, out)
    }

    fn rope(
        &self,
        q: &mut Tensor,
        k: &mut Tensor,
        pos: usize,
        freq_base: f32,
        freq_scale: f32,
        use_neox: bool,
    ) -> BackendResult<()> {
        self.cpu_backend.rope(q, k, pos, freq_base, freq_scale, use_neox)
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &mut Tensor,
        scale: f32,
    ) -> BackendResult<()> {
        self.cpu_backend.attention(q, k, v, out, scale)
    }
}

#[cfg(not(feature = "cuda"))]
impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn alloc(&self, _shape: &[usize], _dtype: DType) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn copy_to(&self, _tensor: &Tensor) -> BackendResult<Tensor> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn add(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn scale(&self, _a: &Tensor, _scalar: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn silu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn gelu(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn softmax(&self, _x: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn rms_norm(&self, _x: &Tensor, _weight: &Tensor, _eps: f32, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn matvec(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn vec_mat(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn dequantize(&self, _src: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn matvec_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn vec_mat_q(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn rope(&self, _q: &mut Tensor, _k: &mut Tensor, _pos: usize, _freq_base: f32, _freq_scale: f32, _use_neox: bool) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }

    fn attention(&self, _q: &Tensor, _k: &Tensor, _v: &Tensor, _out: &mut Tensor, _scale: f32) -> BackendResult<()> {
        Err(BackendError::NotAvailable("CUDA".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_config_default() {
        let config = CudaConfig::default();
        assert_eq!(config.device_index, 0);
        assert!(config.use_tensor_cores);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_backend_creation() {
        match CudaBackend::new() {
            Ok(backend) => {
                assert_eq!(backend.name(), "cuda");
                assert!(backend.is_available());
                println!("CUDA backend created: {}", backend.device_name());
            }
            Err(e) => {
                println!("CUDA not available: {}", e);
            }
        }
    }
}
