//! Fast GPU inference example
//!
//! Tests the FastGpuInference engine that keeps weights on GPU.

#[cfg(feature = "cuda")]
fn main() {
    use llama_gguf::backend::cuda::fast_inference::FastGpuInference;
    use llama_gguf::gguf::GgufFile;
    use llama_gguf::model::ModelLoader;
    use llama_gguf::sampling::{Sampler, SamplerConfig};
    use llama_gguf::tokenizer::Tokenizer;
    use std::io::{self, Write};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [prompt] [n_tokens]", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("Hello");
    let n_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);

    eprintln!("Loading model from: {}", model_path);

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    let loader = ModelLoader::load(model_path).expect("Failed to load model");
    let config = loader.config().clone();
    let model = loader.build_model().expect("Failed to build model");

    eprintln!(
        "Model: {} layers, {} hidden dim",
        config.num_layers, config.hidden_size
    );

    // Create fast GPU inference engine
    eprintln!("\nCreating fast GPU inference engine...");
    let upload_start = Instant::now();
    let mut gpu =
        FastGpuInference::from_model(&model, 2048).expect("Failed to create GPU inference");
    eprintln!("GPU ready in {:.2}s", upload_start.elapsed().as_secs_f32());

    // Create sampler
    let sampler_config = SamplerConfig {
        temperature: 0.7,
        top_k: 40,
        ..Default::default()
    };
    let mut sampler = Sampler::new(sampler_config, config.vocab_size);

    // Encode prompt
    let add_bos = gguf
        .data
        .get_bool("tokenizer.ggml.add_bos_token")
        .unwrap_or(true);
    let tokens = tokenizer.encode(prompt, add_bos).expect("Failed to encode");

    eprintln!("\nGenerating {} tokens...", n_tokens);
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    let gen_start = Instant::now();

    // Process prompt tokens (all but last)
    for &token in &tokens[..tokens.len().saturating_sub(1)] {
        let _ = gpu.forward(token);
    }

    // Generate from last prompt token
    let mut current_token = *tokens.last().unwrap();
    let mut generated = 0;

    for _ in 0..n_tokens {
        let logits = gpu.forward(current_token).expect("Forward failed");

        // Sample
        let logits_tensor =
            llama_gguf::tensor::Tensor::from_f32(&logits, vec![logits.len()]).unwrap();
        let next_token = sampler.sample(&logits_tensor, &[]);

        // Decode and print
        if let Ok(text) = tokenizer.decode(&[next_token]) {
            print!("{}", text);
            io::stdout().flush().unwrap();
        }

        current_token = next_token;
        generated += 1;

        if next_token == tokenizer.special_tokens.eos_token_id {
            break;
        }
    }

    let gen_time = gen_start.elapsed();
    let tokens_per_sec = generated as f32 / gen_time.as_secs_f32();

    println!();
    eprintln!();
    eprintln!(
        "Generated {} tokens in {:.2}s ({:.2} tokens/sec)",
        generated,
        gen_time.as_secs_f32(),
        tokens_per_sec
    );
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires CUDA. Build with: cargo build --features cuda");
}
