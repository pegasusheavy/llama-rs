//! Tests for quantization/dequantization functions

use llama_gguf::tensor::quant::{
    dequantize_q4_0, dequantize_q4_1, dequantize_q5_0, dequantize_q5_1, dequantize_q8_0,
    quantize_q4_0, quantize_q4_1, quantize_q8_0,
};

#[test]
fn test_q4_0_roundtrip() {
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
fn test_q4_1_roundtrip() {
    // Q4_1 works better with positive ranges due to min value storage
    let original: [f32; 32] = std::array::from_fn(|i| (i as f32) * 0.1 + 1.0);

    let block = quantize_q4_1(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q4_1(&block, &mut decoded);

    for (o, d) in original.iter().zip(decoded.iter()) {
        assert!(
            (o - d).abs() < 0.15,
            "Q4_1 roundtrip error too large: original={}, decoded={}",
            o,
            d
        );
    }
}

#[test]
fn test_q8_0_roundtrip() {
    let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);

    let block = quantize_q8_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q8_0(&block, &mut decoded);

    // Q8 should be very close
    for (o, d) in original.iter().zip(decoded.iter()) {
        assert!(
            (o - d).abs() < 0.02,
            "Q8_0 roundtrip error too large: original={}, decoded={}",
            o,
            d
        );
    }
}

#[test]
fn test_q4_0_zeros() {
    let original = [0.0f32; 32];
    let block = quantize_q4_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q4_0(&block, &mut decoded);

    for d in decoded.iter() {
        assert_eq!(*d, 0.0, "Zero input should produce zero output");
    }
}

#[test]
fn test_q8_0_zeros() {
    let original = [0.0f32; 32];
    let block = quantize_q8_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q8_0(&block, &mut decoded);

    for d in decoded.iter() {
        assert_eq!(*d, 0.0, "Zero input should produce zero output");
    }
}

#[test]
fn test_q4_0_large_values() {
    // Test with larger values to check scaling
    // Note: Q4_0 only has 16 levels (-8 to 7 after bias), so with a range of 300
    // (-150 to +150), each quantization step is ~20 units
    let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 10.0);

    let block = quantize_q4_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q4_0(&block, &mut decoded);

    // Find the maximum value to understand the quantization step size
    let max_abs = original.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    // Q4_0 uses 15 levels (0-14, plus symmetric negative), so step ~= max_abs / 7
    let expected_step = max_abs / 7.0;

    for (o, d) in original.iter().zip(decoded.iter()) {
        // Error should be within one quantization step
        let error = (o - d).abs();
        assert!(
            error <= expected_step * 1.1, // Allow small margin
            "Q4_0 large value error: original={}, decoded={}, error={}, expected_step={}",
            o,
            d,
            error,
            expected_step
        );
    }
}

#[test]
fn test_q8_0_precision() {
    // Q8 should maintain good precision
    let original: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.01);

    let block = quantize_q8_0(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q8_0(&block, &mut decoded);

    // Calculate RMS error
    let rms_error: f32 = original
        .iter()
        .zip(decoded.iter())
        .map(|(o, d)| (o - d).powi(2))
        .sum::<f32>()
        / 32.0;
    let rms_error = rms_error.sqrt();

    assert!(rms_error < 0.005, "Q8_0 RMS error too high: {}", rms_error);
}

#[test]
fn test_q4_1_with_offset() {
    // Q4_1 should handle values with a non-zero minimum well
    let original: [f32; 32] = std::array::from_fn(|i| (i as f32) * 0.1 + 5.0);

    let block = quantize_q4_1(&original);
    let mut decoded = [0.0f32; 32];
    dequantize_q4_1(&block, &mut decoded);

    for (o, d) in original.iter().zip(decoded.iter()) {
        assert!(
            (o - d).abs() < 0.15,
            "Q4_1 offset test failed: original={}, decoded={}",
            o,
            d
        );
    }
}

#[test]
fn test_dequantize_q5_0() {
    use half::f16;
    use llama_gguf::tensor::quant::BlockQ5_0;

    // Create a simple Q5_0 block for testing
    // This tests that the 5-bit unpacking works correctly
    let block = BlockQ5_0 {
        d: f16::from_f32(0.1),
        qh: [0, 0, 0, 0], // All high bits are 0
        qs: [0x88; 16],   // Low nibble = 8, high nibble = 8 (after -16 shift: gives -8)
    };

    let mut decoded = [0.0f32; 32];
    dequantize_q5_0(&block, &mut decoded);

    // With d=0.1 and all 5-bit values = 8 (after shift: 8-16 = -8)
    // Expected: -8 * 0.1 = -0.8 for all values
    for (i, &d) in decoded.iter().enumerate() {
        assert!(
            (d - (-0.8)).abs() < 0.01,
            "Q5_0 dequant failed at {}: expected -0.8, got {}",
            i,
            d
        );
    }
}

#[test]
fn test_dequantize_q5_1() {
    use half::f16;
    use llama_gguf::tensor::quant::BlockQ5_1;

    // Create a simple Q5_1 block for testing
    let block = BlockQ5_1 {
        d: f16::from_f32(0.1),
        m: f16::from_f32(1.0),
        qh: [0, 0, 0, 0], // All high bits are 0
        qs: [0x88; 16],   // Low nibble = 8, high nibble = 8
    };

    let mut decoded = [0.0f32; 32];
    dequantize_q5_1(&block, &mut decoded);

    // With d=0.1, m=1.0 and all 5-bit values = 8
    // Expected: 8 * 0.1 + 1.0 = 1.8 for all values
    for (i, &d) in decoded.iter().enumerate() {
        assert!(
            (d - 1.8).abs() < 0.01,
            "Q5_1 dequant failed at {}: expected 1.8, got {}",
            i,
            d
        );
    }
}

#[test]
fn test_quantization_symmetry() {
    // Test that quantizing symmetric values produces symmetric results
    let positive: [f32; 32] = std::array::from_fn(|i| (i as f32) * 0.1);
    let negative: [f32; 32] = std::array::from_fn(|i| -(i as f32) * 0.1);

    let pos_block = quantize_q8_0(&positive);
    let neg_block = quantize_q8_0(&negative);

    let mut pos_decoded = [0.0f32; 32];
    let mut neg_decoded = [0.0f32; 32];

    dequantize_q8_0(&pos_block, &mut pos_decoded);
    dequantize_q8_0(&neg_block, &mut neg_decoded);

    // Check that the decoded values are approximately negatives of each other
    for (p, n) in pos_decoded.iter().zip(neg_decoded.iter()) {
        assert!(
            (p + n).abs() < 0.02,
            "Quantization not symmetric: {} + {} = {}",
            p,
            n,
            p + n
        );
    }
}
