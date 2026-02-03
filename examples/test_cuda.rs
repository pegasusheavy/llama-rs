//! Test CUDA backend initialization

#[cfg(feature = "cuda")]
fn main() {
    use llama_rs::backend::cuda::CudaBackend;
    use llama_rs::backend::Backend;
    
    println!("Testing CUDA backend...");
    println!();
    
    match CudaBackend::new() {
        Ok(backend) => {
            println!("✓ CUDA backend initialized successfully");
            println!("  Name: {}", backend.name());
            println!("  Available: {}", backend.is_available());
            println!("  Device: {}", backend.device_name());
            
            // Test basic tensor operations
            use llama_rs::tensor::{Tensor, DType};
            
            let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
            let b = Tensor::from_f32(&[0.5, 0.5, 0.5, 0.5], vec![4]).unwrap();
            let mut out = Tensor::zeros(vec![4], DType::F32);
            
            match backend.add(&a, &b, &mut out) {
                Ok(_) => {
                    let result = out.as_f32().unwrap();
                    println!();
                    println!("✓ Add operation succeeded");
                    println!("  [1,2,3,4] + [0.5,0.5,0.5,0.5] = {:?}", result);
                }
                Err(e) => {
                    println!("✗ Add operation failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ Failed to create CUDA backend: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Build with: cargo build --features cuda");
    std::process::exit(1);
}
