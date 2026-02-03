//! Test vec_mat kernel correctness

#[cfg(feature = "cuda")]
fn main() {
    use llama_cpp_rs::backend::{Backend, cpu::CpuBackend, cuda::CudaBackend};
    use llama_cpp_rs::tensor::{DType, Tensor};
    
    // Create a simple test: 
    // vec: [1, 2, 3, 4]
    // mat: [[1, 2], [3, 4], [5, 6], [7, 8]]  - 4x2 matrix
    // Expected: [50, 60]  (dot products)
    
    let vec_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mat_data: Vec<f32> = vec![
        1.0, 2.0,  // row 0
        3.0, 4.0,  // row 1
        5.0, 6.0,  // row 2
        7.0, 8.0,  // row 3
    ];
    
    // vec @ mat = [1*1 + 2*3 + 3*5 + 4*7, 1*2 + 2*4 + 3*6 + 4*8] = [50, 60]
    
    let vec_tensor = Tensor::from_f32(&vec_data, vec![4]).unwrap();
    let mat_tensor = Tensor::from_f32(&mat_data, vec![4, 2]).unwrap();
    let mut out_cpu = Tensor::zeros(vec![2], DType::F32);
    let mut out_gpu = Tensor::zeros(vec![2], DType::F32);
    
    // Test CPU
    let cpu = CpuBackend::new();
    cpu.vec_mat(&vec_tensor, &mat_tensor, &mut out_cpu).unwrap();
    let cpu_result = out_cpu.as_f32().unwrap();
    println!("CPU result: {:?}", cpu_result);
    
    // Test GPU
    let cuda = CudaBackend::new().unwrap();
    cuda.vec_mat(&vec_tensor, &mat_tensor, &mut out_gpu).unwrap();
    let gpu_result = out_gpu.as_f32().unwrap();
    println!("GPU result: {:?}", gpu_result);
    
    // Compare
    let diff: f32 = cpu_result.iter().zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("Total absolute difference: {}", diff);
    
    if diff < 0.01 {
        println!("PASS: GPU and CPU results match!");
    } else {
        println!("FAIL: Results differ significantly!");
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires CUDA. Build with: cargo build --features cuda");
}
