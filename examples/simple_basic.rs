//! # Simple Basic Example
//!
//! This is the simplest possible example of using the GPU Scatter-Gather library.
//! It generates a small wordlist and prints it to the console.
//!
//! ## What this example demonstrates:
//! - Creating a GPU context
//! - Defining character sets
//! - Creating a mask pattern
//! - Generating words
//! - Converting bytes to strings
//!
//! ## Expected output:
//! ```
//! Generated 9 words:
//! a1
//! a2
//! a3
//! b1
//! b2
//! b3
//! c1
//! c2
//! c3
//! ```

use anyhow::Result;
use gpu_scatter_gather::gpu::GpuContext;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🚀 GPU Scatter-Gather - Simple Basic Example\n");

    // Step 1: Create a GPU context
    // This initializes CUDA and loads the GPU kernels
    println!("📡 Initializing GPU context...");
    let gpu = GpuContext::new()?;
    println!("   ✅ GPU initialized successfully\n");

    // Step 2: Define character sets
    // Charset 0: lowercase letters (just a few for simplicity)
    // Charset 1: digits
    println!("📝 Defining character sets...");
    let mut charsets = HashMap::new();
    charsets.insert(0, b"abc".to_vec());       // Charset 0: a, b, c
    charsets.insert(1, b"123".to_vec());       // Charset 1: 1, 2, 3
    println!("   Charset 0: abc");
    println!("   Charset 1: 123\n");

    // Step 3: Create a mask pattern
    // Mask defines the structure: [0, 1] means "one char from charset 0, one from charset 1"
    // This will generate: a1, a2, a3, b1, b2, b3, c1, c2, c3
    println!("🎭 Creating mask pattern...");
    let mask = vec![0, 1];  // Pattern: ?0?1 (one from charset 0, one from charset 1)
    println!("   Pattern: ?0?1 (charset 0 + charset 1)\n");

    // Step 4: Calculate keyspace size
    // Keyspace = product of charset sizes
    // In this case: 3 letters × 3 digits = 9 combinations
    let keyspace = 3 * 3; // abc (3) × 123 (3) = 9 total combinations
    println!("📊 Keyspace size: {} combinations\n", keyspace);

    // Step 5: Generate the words
    // Format 2 = PACKED (no separators, just raw bytes)
    println!("⚡ Generating words on GPU...");
    let output = gpu.generate_batch(&charsets, &mask, 0, keyspace, 2)?;
    println!("   ✅ Generated {} bytes\n", output.len());

    // Step 6: Parse and display the results
    // Each word is 2 bytes (mask length), packed together
    let word_length = mask.len();
    let num_words = output.len() / word_length;

    println!("📤 Generated {} words:", num_words);
    println!("─────────────────────");

    for i in 0..num_words {
        let start = i * word_length;
        let end = start + word_length;
        let word_bytes = &output[start..end];
        let word = String::from_utf8_lossy(word_bytes);
        println!("{}", word);
    }

    println!("\n✨ Done!");

    Ok(())
}
