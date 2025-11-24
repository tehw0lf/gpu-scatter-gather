//! Validate GPU Output Against CPU Reference
//!
//! This example generates a small batch on both CPU and GPU, then compares
//! the output to ensure correctness before benchmarking performance.

use anyhow::Result;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::{Charset, WordlistGenerator};

fn main() -> Result<()> {
    println!("🔍 GPU Output Validation");
    println!("{}", "=".repeat(70));
    println!();

    // Initialize GPU
    println!("Initializing GPU...");
    let mut gpu = GpuContext::new()?;
    let device_name = gpu.device_name()?;
    let (major, minor) = gpu.compute_capability();
    println!("✅ GPU: {}", device_name);
    println!("   Compute Capability: {}.{}", major, minor);
    println!();

    // Test pattern: ?1?2 where ?1="abc", ?2="123"
    println!("Test pattern: ?1?2");
    println!("  Charset 1: abc");
    println!("  Charset 2: 123");
    println!("  Expected keyspace: 9 words");
    println!();

    let charset1 = Charset::new(b"abc".to_vec());
    let charset2 = Charset::new(b"123".to_vec());

    // Build CPU reference generator
    let gen = WordlistGenerator::builder()
        .charset(0, charset1.clone())
        .charset(1, charset2.clone())
        .mask(&[0, 1])
        .build()?;

    let keyspace = gen.keyspace_size();
    println!("Keyspace size: {}", keyspace);
    println!();

    // Generate all words on CPU
    println!("Generating reference wordlist on CPU...");
    let mut cpu_words = Vec::new();
    for i in 0..keyspace as u64 {
        let word = gen.index_to_word(i);
        cpu_words.push(word);
    }
    println!("✅ CPU generated {} words", cpu_words.len());
    println!();

    // Generate same words on GPU
    println!("Generating wordlist on GPU...");
    let mut charsets = std::collections::HashMap::new();
    charsets.insert(0, charset1.as_bytes().to_vec());
    charsets.insert(1, charset2.as_bytes().to_vec());

    let gpu_output = gpu.generate_batch(&charsets, &[0, 1], 0, keyspace as u64, 0)?;  // format=0 (newlines)
    println!("✅ GPU generated {} bytes", gpu_output.len());
    println!();

    // Parse GPU output into words
    let gpu_words: Vec<&[u8]> = gpu_output
        .split(|&b| b == b'\n')
        .filter(|w| !w.is_empty())
        .collect();

    println!("GPU parsed into {} words", gpu_words.len());
    println!();

    // Compare outputs
    println!("Comparing CPU vs GPU output...");
    println!("{}", "-".repeat(70));

    let mut matches = 0;
    let mut mismatches = 0;

    for (i, (cpu_word, gpu_word)) in cpu_words.iter().zip(gpu_words.iter()).enumerate() {
        let cpu_str = String::from_utf8_lossy(cpu_word);
        let gpu_str = String::from_utf8_lossy(gpu_word);

        if cpu_word == gpu_word {
            println!("[{}] ✅ {} == {}", i, cpu_str, gpu_str);
            matches += 1;
        } else {
            println!("[{}] ❌ {} != {} (MISMATCH!)", i, cpu_str, gpu_str);
            mismatches += 1;
        }
    }

    println!("{}", "-".repeat(70));
    println!();

    // Check for length mismatch
    if cpu_words.len() != gpu_words.len() {
        println!("❌ LENGTH MISMATCH!");
        println!("   CPU: {} words", cpu_words.len());
        println!("   GPU: {} words", gpu_words.len());
        anyhow::bail!("GPU generated different number of words than CPU");
    }

    // Report results
    println!("VALIDATION RESULTS:");
    println!("  Total words: {}", cpu_words.len());
    println!("  Matches: {} ({}%)", matches, (matches * 100) / cpu_words.len());
    println!("  Mismatches: {}", mismatches);
    println!();

    if mismatches == 0 {
        println!("🎉 SUCCESS! GPU output matches CPU reference perfectly!");
        println!();
        println!("The GPU kernel is CORRECT and ready for performance benchmarking.");
    } else {
        println!("❌ FAILURE! GPU output does not match CPU reference.");
        println!();
        println!("The GPU kernel needs debugging before proceeding.");
        anyhow::bail!("GPU validation failed");
    }

    Ok(())
}
