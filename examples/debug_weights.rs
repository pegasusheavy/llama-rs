//! Debug tensor names for GPU weight matching

use llama_gguf::gguf::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    println!("Loading GGUF from: {}", model_path);
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");

    println!("\nTensor names in GGUF:");
    for tensor in &gguf.data.tensors {
        println!("  {} - {:?} {:?}", tensor.name, tensor.dims, tensor.dtype);
    }
}
