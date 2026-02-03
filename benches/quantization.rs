//! Performance benchmarks for llama-rs
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llama_gguf::tensor::{DType, Tensor};
use llama_gguf::backend::cpu::CpuBackend;
use llama_gguf::Backend;

/// Benchmark tensor creation and basic operations
fn tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    
    for size in [256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("zeros_f32", size), size, |b, &size| {
            b.iter(|| {
                black_box(Tensor::zeros(vec![size], DType::F32))
            });
        });
        
        group.bench_with_input(BenchmarkId::new("from_f32", size), size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            b.iter(|| {
                black_box(Tensor::from_f32(&data, vec![size]))
            });
        });
    }
    
    group.finish();
}

/// Benchmark matrix-vector multiplication
fn matvec_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("matvec");
    
    // Common LLM dimensions
    for (m, n) in [(1024, 1024), (2048, 2048), (4096, 4096)].iter() {
        let flops = (*m * *n * 2) as u64; // multiply-add = 2 ops
        group.throughput(Throughput::Elements(flops));
        
        group.bench_with_input(
            BenchmarkId::new("f32", format!("{}x{}", m, n)),
            &(*m, *n),
            |b, &(m, n)| {
                let matrix = Tensor::zeros(vec![m, n], DType::F32);
                let vector = Tensor::zeros(vec![n], DType::F32);
                let mut output = Tensor::zeros(vec![m], DType::F32);
                b.iter(|| {
                    backend.matvec(&matrix, &vector, &mut output).unwrap();
                    black_box(&output);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark matrix-matrix multiplication
fn matmul_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("matmul");
    
    for size in [128, 256, 512].iter() {
        let flops = (*size * *size * *size * 2) as u64;
        group.throughput(Throughput::Elements(flops));
        
        group.bench_with_input(BenchmarkId::new("f32", size), size, |b, &size| {
            let a = Tensor::zeros(vec![size, size], DType::F32);
            let b_mat = Tensor::zeros(vec![size, size], DType::F32);
            let mut c = Tensor::zeros(vec![size, size], DType::F32);
            b.iter(|| {
                backend.matmul(&a, &b_mat, &mut c).unwrap();
                black_box(&c);
            });
        });
    }
    
    group.finish();
}

/// Benchmark softmax operation
fn softmax_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("softmax");
    
    // Typical vocab sizes
    for size in [32000, 50257, 128256].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("vocab", size), size, |b, &size| {
            let tensor = Tensor::from_f32(
                &(0..size).map(|i| (i as f32) / 1000.0).collect::<Vec<_>>(),
                vec![size],
            ).unwrap();
            let mut output = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.softmax(&tensor, &mut output).unwrap();
                black_box(&output);
            });
        });
    }
    
    group.finish();
}

/// Benchmark RMS normalization
fn rms_norm_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("rms_norm");
    
    for size in [2048, 4096, 8192].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("hidden_dim", size), size, |b, &size| {
            let input = Tensor::from_f32(
                &(0..size).map(|i| (i as f32) / 1000.0).collect::<Vec<_>>(),
                vec![size],
            ).unwrap();
            let weights = Tensor::from_f32(&vec![1.0f32; size], vec![size]).unwrap();
            let mut output = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.rms_norm(&input, &weights, 1e-5, &mut output).unwrap();
                black_box(&output);
            });
        });
    }
    
    group.finish();
}

/// Benchmark SiLU activation
fn silu_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("silu");
    
    for size in [4096, 11008, 14336].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("size", size), size, |b, &size| {
            let tensor = Tensor::from_f32(
                &(0..size).map(|i| ((i as f32) - (size as f32 / 2.0)) / 1000.0).collect::<Vec<_>>(),
                vec![size],
            ).unwrap();
            let mut output = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.silu(&tensor, &mut output).unwrap();
                black_box(&output);
            });
        });
    }
    
    group.finish();
}

/// Benchmark dequantization
fn dequant_benchmark(c: &mut Criterion) {
    use llama_gguf::tensor::quant::{dequantize_q4_0, dequantize_q8_0, BlockQ4_0, BlockQ8_0};
    
    let mut group = c.benchmark_group("dequantize");
    
    // Number of blocks
    for n_blocks in [256, 1024, 4096].iter() {
        let n_elements = n_blocks * 32; // Q4_0 has 32 elements per block
        group.throughput(Throughput::Elements(n_elements as u64));
        
        // Create Q4_0 blocks
        let q4_0_blocks: Vec<BlockQ4_0> = (0..*n_blocks)
            .map(|i| BlockQ4_0 {
                d: half::f16::from_f32(0.1 * (i as f32 + 1.0)),
                qs: [((i * 7) % 256) as u8; 16],
            })
            .collect();
        
        group.bench_with_input(BenchmarkId::new("q4_0", n_blocks), &q4_0_blocks, |b, blocks| {
            b.iter(|| {
                let mut output = [0.0f32; 32];
                for block in blocks.iter() {
                    dequantize_q4_0(block, &mut output);
                    black_box(&output);
                }
            });
        });
        
        // Create Q8_0 blocks
        let q8_0_blocks: Vec<BlockQ8_0> = (0..*n_blocks)
            .map(|i| BlockQ8_0 {
                d: half::f16::from_f32(0.1 * (i as f32 + 1.0)),
                qs: std::array::from_fn(|j| ((i * 7 + j) % 256) as i8),
            })
            .collect();
        
        group.bench_with_input(BenchmarkId::new("q8_0", n_blocks), &q8_0_blocks, |b, blocks| {
            b.iter(|| {
                let mut output = [0.0f32; 32];
                for block in blocks.iter() {
                    dequantize_q8_0(block, &mut output);
                    black_box(&output);
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark dot product (SIMD critical path)
fn dot_product_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    
    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements((*size * 2) as u64)); // mul + add
        
        group.bench_with_input(BenchmarkId::new("f32", size), size, |b, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32 / 1000.0).collect();
            let b_vec: Vec<f32> = (0..size).map(|i| (size - i) as f32 / 1000.0).collect();
            b.iter(|| {
                let result: f32 = a.iter().zip(b_vec.iter()).map(|(x, y)| x * y).sum();
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Benchmark element-wise operations
fn elementwise_benchmark(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("elementwise");
    
    for size in [4096, 16384, 65536].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("add", size), size, |b, &size| {
            let a = Tensor::from_f32(&vec![1.0f32; size], vec![size]).unwrap();
            let b_tensor = Tensor::from_f32(&vec![2.0f32; size], vec![size]).unwrap();
            let mut out = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.add(&a, &b_tensor, &mut out).unwrap();
                black_box(&out);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("mul", size), size, |b, &size| {
            let a = Tensor::from_f32(&vec![1.5f32; size], vec![size]).unwrap();
            let b_tensor = Tensor::from_f32(&vec![2.5f32; size], vec![size]).unwrap();
            let mut out = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.mul(&a, &b_tensor, &mut out).unwrap();
                black_box(&out);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("scale", size), size, |b, &size| {
            let a = Tensor::from_f32(&vec![1.0f32; size], vec![size]).unwrap();
            let mut out = Tensor::zeros(vec![size], DType::F32);
            b.iter(|| {
                backend.scale(&a, 2.5, &mut out).unwrap();
                black_box(&out);
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    tensor_creation,
    matvec_benchmark,
    matmul_benchmark,
    softmax_benchmark,
    rms_norm_benchmark,
    silu_benchmark,
    dequant_benchmark,
    dot_product_benchmark,
    elementwise_benchmark,
);
criterion_main!(benches);
