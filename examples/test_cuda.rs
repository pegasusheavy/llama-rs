//! Test CUDA backend initialization and GPU kernels

#[cfg(feature = "cuda")]
fn main() {
    use llama_rs::backend::cuda::CudaBackend;
    use llama_rs::backend::Backend;
    use llama_rs::tensor::{Tensor, DType};
    
    println!("Testing CUDA backend...");
    println!();
    
    let backend = match CudaBackend::new() {
        Ok(b) => {
            println!("✓ CUDA backend initialized");
            println!("  Device: {}", b.device_name());
            b
        }
        Err(e) => {
            println!("✗ Failed to create CUDA backend: {}", e);
            std::process::exit(1);
        }
    };
    
    println!();
    println!("Testing GPU kernels...");
    println!();
    
    // Test add
    {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[0.5, 0.5, 0.5, 0.5], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);
        
        match backend.add(&a, &b, &mut out) {
            Ok(_) => {
                let result = out.as_f32().unwrap();
                let expected = [1.5, 2.5, 3.5, 4.5];
                let pass = result.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-5);
                println!("{} add: {:?}", if pass { "✓" } else { "✗" }, result);
            }
            Err(e) => println!("✗ add failed: {}", e),
        }
    }
    
    // Test mul
    {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_f32(&[2.0, 2.0, 2.0, 2.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);
        
        match backend.mul(&a, &b, &mut out) {
            Ok(_) => {
                let result = out.as_f32().unwrap();
                let expected = [2.0, 4.0, 6.0, 8.0];
                let pass = result.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-5);
                println!("{} mul: {:?}", if pass { "✓" } else { "✗" }, result);
            }
            Err(e) => println!("✗ mul failed: {}", e),
        }
    }
    
    // Test scale
    {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);
        
        match backend.scale(&a, 0.5, &mut out) {
            Ok(_) => {
                let result = out.as_f32().unwrap();
                let expected = [0.5, 1.0, 1.5, 2.0];
                let pass = result.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-5);
                println!("{} scale: {:?}", if pass { "✓" } else { "✗" }, result);
            }
            Err(e) => println!("✗ scale failed: {}", e),
        }
    }
    
    // Test silu
    {
        let x = Tensor::from_f32(&[0.0, 1.0, 2.0], vec![3]).unwrap();
        let mut out = Tensor::zeros(vec![3], DType::F32);
        
        match backend.silu(&x, &mut out) {
            Ok(_) => {
                let result = out.as_f32().unwrap();
                // SiLU(x) = x * sigmoid(x)
                // SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(2) ≈ 1.762
                let pass = result[0].abs() < 1e-5 && (result[1] - 0.731).abs() < 0.01;
                println!("{} silu: {:?}", if pass { "✓" } else { "✗" }, result);
            }
            Err(e) => println!("✗ silu failed: {}", e),
        }
    }
    
    // Test gelu
    {
        let x = Tensor::from_f32(&[0.0, 1.0, 2.0], vec![3]).unwrap();
        let mut out = Tensor::zeros(vec![3], DType::F32);
        
        match backend.gelu(&x, &mut out) {
            Ok(_) => {
                let result = out.as_f32().unwrap();
                // GELU(0) = 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.954
                let pass = result[0].abs() < 1e-5 && (result[1] - 0.841).abs() < 0.01;
                println!("{} gelu: {:?}", if pass { "✓" } else { "✗" }, result);
            }
            Err(e) => println!("✗ gelu failed: {}", e),
        }
    }
    
    // Test softmax
    {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);
        
        match backend.softmax(&x, &mut out) {
            Ok(_) => {
                let result = out.as_f32().unwrap();
                let sum: f32 = result.iter().sum();
                let pass = (sum - 1.0).abs() < 1e-5 && result[3] > result[2] && result[2] > result[1];
                println!("{} softmax: {:?} (sum={})", if pass { "✓" } else { "✗" }, result, sum);
            }
            Err(e) => println!("✗ softmax failed: {}", e),
        }
    }
    
    // Test rms_norm
    {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let w = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let mut out = Tensor::zeros(vec![4], DType::F32);
        
        match backend.rms_norm(&x, &w, 1e-5, &mut out) {
            Ok(_) => {
                let result = out.as_f32().unwrap();
                // Check that output has unit RMS (approximately)
                let rms: f32 = (result.iter().map(|x| x*x).sum::<f32>() / 4.0).sqrt();
                let pass = (rms - 1.0).abs() < 0.1;
                println!("{} rms_norm: {:?} (rms={})", if pass { "✓" } else { "✗" }, result, rms);
            }
            Err(e) => println!("✗ rms_norm failed: {}", e),
        }
    }
    
    // Test vec_mat
    {
        // vec = [1, 2], mat = [[1, 2, 3], [4, 5, 6]]
        // out = [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
        let vec = Tensor::from_f32(&[1.0, 2.0], vec![2]).unwrap();
        let mat = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let mut out = Tensor::zeros(vec![3], DType::F32);
        
        match backend.vec_mat(&vec, &mat, &mut out) {
            Ok(_) => {
                let result = out.as_f32().unwrap();
                let expected = [9.0, 12.0, 15.0];
                let pass = result.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-5);
                println!("{} vec_mat: {:?}", if pass { "✓" } else { "✗" }, result);
            }
            Err(e) => println!("✗ vec_mat failed: {}", e),
        }
    }
    
    println!();
    println!("GPU kernel tests completed!");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Build with: cargo build --features cuda");
    std::process::exit(1);
}
