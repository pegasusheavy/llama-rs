//! Integration tests for llama-rs
//!
//! These tests verify that the various components work together correctly.

use llama_rs::backend::default_backend;
use llama_rs::tensor::{DType, Tensor};

// =============================================================================
// Backend Basic Tests
// =============================================================================

#[test]
fn test_default_backend() {
    let backend = default_backend();
    assert_eq!(backend.name(), "cpu");
    assert!(backend.is_available());
}

#[test]
fn test_backend_alloc() {
    let backend = default_backend();
    let tensor = backend.alloc(&[4, 4], DType::F32).unwrap();
    assert_eq!(tensor.shape(), &[4, 4]);
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.numel(), 16);
}

#[test]
fn test_backend_copy() {
    let backend = default_backend();
    let original = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let copy = backend.copy_to(&original).unwrap();

    assert_eq!(copy.shape(), original.shape());
    assert_eq!(copy.as_f32().unwrap(), original.as_f32().unwrap());
}

// =============================================================================
// Element-wise Operation Tests
// =============================================================================

#[test]
fn test_add_operation() {
    let backend = default_backend();

    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let b = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
    let mut out = Tensor::zeros(vec![4], DType::F32);

    backend.add(&a, &b, &mut out).unwrap();

    let result = out.as_f32().unwrap();
    assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_mul_operation() {
    let backend = default_backend();

    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let b = Tensor::from_f32(&[2.0, 3.0, 4.0, 5.0], vec![4]).unwrap();
    let mut out = Tensor::zeros(vec![4], DType::F32);

    backend.mul(&a, &b, &mut out).unwrap();

    let result = out.as_f32().unwrap();
    assert_eq!(result, &[2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_scale_operation() {
    let backend = default_backend();

    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let mut out = Tensor::zeros(vec![4], DType::F32);

    backend.scale(&a, 2.5, &mut out).unwrap();

    let result = out.as_f32().unwrap();
    assert_eq!(result, &[2.5, 5.0, 7.5, 10.0]);
}

// =============================================================================
// Activation Function Tests
// =============================================================================

#[test]
fn test_silu_activation() {
    let backend = default_backend();

    let input = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
    let mut output = Tensor::zeros(vec![4], DType::F32);

    backend.silu(&input, &mut output).unwrap();

    let result = output.as_f32().unwrap();
    // SiLU(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // SiLU(1) ≈ 0.731
    assert!((result[1] - 0.731).abs() < 0.01);
    // SiLU(-1) ≈ -0.269
    assert!((result[2] - (-0.269)).abs() < 0.01);
    // SiLU(2) ≈ 1.762
    assert!((result[3] - 1.762).abs() < 0.01);
}

#[test]
fn test_gelu_activation() {
    let backend = default_backend();

    let input = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
    let mut output = Tensor::zeros(vec![4], DType::F32);

    backend.gelu(&input, &mut output).unwrap();

    let result = output.as_f32().unwrap();
    // GELU(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // GELU(1) ≈ 0.841
    assert!((result[1] - 0.841).abs() < 0.01);
    // GELU(-1) ≈ -0.159
    assert!((result[2] - (-0.159)).abs() < 0.01);
}

#[test]
fn test_softmax() {
    let backend = default_backend();

    let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let mut output = Tensor::zeros(vec![4], DType::F32);

    backend.softmax(&input, &mut output).unwrap();

    let result = output.as_f32().unwrap();

    // Sum should be 1.0
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Values should be monotonically increasing
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
    assert!(result[2] < result[3]);

    // All values should be positive
    for &v in result.iter() {
        assert!(v > 0.0);
    }
}

#[test]
fn test_softmax_2d() {
    let backend = default_backend();

    // 2x4 matrix - softmax along last dimension
    let input = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        vec![2, 4],
    )
    .unwrap();
    let mut output = Tensor::zeros(vec![2, 4], DType::F32);

    backend.softmax(&input, &mut output).unwrap();

    let result = output.as_f32().unwrap();

    // Each row should sum to 1.0
    let row1_sum: f32 = result[0..4].iter().sum();
    let row2_sum: f32 = result[4..8].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-6);
    assert!((row2_sum - 1.0).abs() < 1e-6);
}

// =============================================================================
// Normalization Tests
// =============================================================================

#[test]
fn test_rms_norm() {
    let backend = default_backend();

    let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let weight = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
    let mut output = Tensor::zeros(vec![4], DType::F32);

    backend.rms_norm(&input, &weight, 1e-5, &mut output).unwrap();

    let result = output.as_f32().unwrap();

    // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.7386
    // After normalization: [0.365, 0.730, 1.095, 1.460]
    assert!((result[0] - 0.365).abs() < 0.01);
    assert!((result[1] - 0.730).abs() < 0.01);
    assert!((result[2] - 1.095).abs() < 0.01);
    assert!((result[3] - 1.460).abs() < 0.01);
}

#[test]
fn test_rms_norm_with_weights() {
    let backend = default_backend();

    let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let weight = Tensor::from_f32(&[2.0, 2.0, 2.0, 2.0], vec![4]).unwrap();
    let mut output = Tensor::zeros(vec![4], DType::F32);

    backend.rms_norm(&input, &weight, 1e-5, &mut output).unwrap();

    let result = output.as_f32().unwrap();

    // With weight=2, values should be doubled
    assert!((result[0] - 0.730).abs() < 0.01);
    assert!((result[3] - 2.920).abs() < 0.01);
}

// =============================================================================
// Matrix Operation Tests
// =============================================================================

#[test]
fn test_matmul() {
    let backend = default_backend();

    // 2x3 @ 3x2 = 2x2
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let b = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    let mut out = Tensor::zeros(vec![2, 2], DType::F32);

    backend.matmul(&a, &b, &mut out).unwrap();

    let result = out.as_f32().unwrap();
    // [[1,2,3], [4,5,6]] @ [[1,2], [3,4], [5,6]]
    // = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
    // = [[22, 28], [49, 64]]
    assert_eq!(result, &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_matvec() {
    let backend = default_backend();

    // 3x4 @ 4 = 3
    let a = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
    )
    .unwrap();
    let b = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let mut out = Tensor::zeros(vec![3], DType::F32);

    backend.matvec(&a, &b, &mut out).unwrap();

    let result = out.as_f32().unwrap();
    // [1,2,3,4] · [1,2,3,4] = 30
    // [5,6,7,8] · [1,2,3,4] = 70
    // [9,10,11,12] · [1,2,3,4] = 110
    assert_eq!(result, &[30.0, 70.0, 110.0]);
}

// =============================================================================
// Tensor Tests
// =============================================================================

#[test]
fn test_tensor_from_f32() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_f32(&data, vec![2, 3]).unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.ndim(), 2);
    assert_eq!(tensor.numel(), 6);
    assert_eq!(tensor.dtype(), DType::F32);
}

#[test]
fn test_tensor_zeros() {
    let tensor = Tensor::zeros(vec![4, 4], DType::F32);
    let data = tensor.as_f32().unwrap();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_reshape() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_f32(&data, vec![2, 3]).unwrap();

    let reshaped = tensor.reshape(vec![3, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.numel(), 6);
}

#[test]
fn test_tensor_quantized_zeros() {
    let tensor = Tensor::zeros(vec![32], DType::Q4_0);
    assert_eq!(tensor.shape(), &[32]);
    assert_eq!(tensor.numel(), 32);
    assert_eq!(tensor.dtype(), DType::Q4_0);
    // Q4_0: 18 bytes per 32 elements
    assert_eq!(tensor.data().len(), 18);
}

// =============================================================================
// Quantization Tests
// =============================================================================

#[test]
fn test_dequantize_q4_0() {
    use llama_rs::tensor::quant::{dequantize_q4_0, quantize_q4_0};

    let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
    let block = quantize_q4_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q4_0(&block, &mut decoded);

    for (o, d) in original.iter().zip(decoded.iter()) {
        assert!(
            (o - d).abs() < 0.15,
            "Q4_0 roundtrip error too large: original={}, decoded={}",
            o,
            d
        );
    }
}

#[test]
fn test_dequantize_q8_0() {
    use llama_rs::tensor::quant::{dequantize_q8_0, quantize_q8_0};

    let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
    let block = quantize_q8_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q8_0(&block, &mut decoded);

    for (o, d) in original.iter().zip(decoded.iter()) {
        assert!(
            (o - d).abs() < 0.02,
            "Q8_0 roundtrip error too large: original={}, decoded={}",
            o,
            d
        );
    }
}
