//! Test GPU acceleration with statistics

#[cfg(feature = "cuda")]
fn main() {
    use llama_rs::gguf::GgufFile;
    use llama_rs::model::{InferenceContext, Model, ModelLoader};
    use llama_rs::tokenizer::Tokenizer;
    use llama_rs::sampling::{Sampler, SamplerConfig};
    use llama_rs::backend::cuda::CudaBackend;
    use std::sync::Arc;
    use std::io::{self, Write};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [prompt] [n_tokens]", args[0]);
        std::process::exit(1);
    }
    
    let model_path = &args[1];
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("Hello");
    let n_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    
    eprintln!("Loading model from: {}", model_path);
    
    // Load GGUF
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    eprintln!("Vocabulary size: {}", tokenizer.vocab_size);
    
    // Load model
    let loader = ModelLoader::load(model_path).expect("Failed to load model");
    let config = loader.config().clone();
    eprintln!("Model: {} layers, {} hidden dim", config.num_layers, config.hidden_size);
    
    let model = loader.build_model().expect("Failed to build model");
    
    // Create CUDA backend with GPU weights
    let mut cuda = CudaBackend::new().expect("Failed to init CUDA");
    eprintln!("Using CUDA backend: {}", cuda.device_name());
    
    eprintln!("\nUploading weights to GPU...");
    cuda.load_model_weights(&model).expect("Failed to load GPU weights");
    let vram_mb = cuda.gpu_weight_vram() as f64 / (1024.0 * 1024.0);
    eprintln!("VRAM used: {:.1} MB", vram_mb);
    
    let backend = Arc::new(cuda);
    let mut ctx = InferenceContext::new(&config, backend.clone());
    
    // Create sampler
    let sampler_config = SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    };
    let mut sampler = Sampler::new(sampler_config, config.vocab_size);
    
    // Encode prompt
    let add_bos = gguf.data.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(true);
    let tokens = tokenizer.encode(prompt, add_bos).expect("Failed to encode");
    
    eprintln!("\nGenerating {} tokens...", n_tokens);
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    
    let gen_start = Instant::now();
    let mut current_tokens = tokens.clone();
    
    for _ in 0..n_tokens {
        // Forward
        let logits = model.forward(&current_tokens, &mut ctx).expect("Forward failed");
        
        // Sample
        let next_token = sampler.sample(&logits, &[]);
        
        // Decode and print
        if let Ok(text) = tokenizer.decode(&[next_token]) {
            print!("{}", text);
            io::stdout().flush().unwrap();
        }
        
        current_tokens = vec![next_token];
        
        if next_token == tokenizer.special_tokens.eos_token_id {
            break;
        }
    }
    
    let gen_time = gen_start.elapsed();
    let tokens_per_sec = n_tokens as f32 / gen_time.as_secs_f32();
    
    println!();
    eprintln!();
    eprintln!("Generated {} tokens in {:.2}s ({:.2} tokens/sec)", 
              n_tokens, gen_time.as_secs_f32(), tokens_per_sec);
    
    // Print GPU stats (we can't easily access stats through Arc<dyn Backend>)
    eprintln!("(Note: Stats not accessible through trait object)");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires CUDA. Build with: cargo build --features cuda");
}
