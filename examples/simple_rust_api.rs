//! # Rust API Example
//!
//! This example demonstrates the full Rust API including:
//! - Using predefined charsets
//! - Multiple output formats
//! - Multi-GPU context (automatically uses all GPUs)
//! - Batch generation with different start indices
//!
//! ## What this example demonstrates:
//! - GpuContext for single-GPU operation
//! - MultiGpuContext for automatic multi-GPU distribution
//! - Different output formats (NEWLINES, PACKED, FIXED_WIDTH)
//! - Generating subsets of the keyspace
//! - Performance measurement

use anyhow::Result;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 GPU Scatter-Gather - Rust API Example\n");
    println!("========================================\n");

    // Example 1: Single GPU with different output formats
    example_1_output_formats()?;

    println!("\n========================================\n");

    // Example 2: Multi-GPU context
    example_2_multi_gpu()?;

    println!("\n========================================\n");

    // Example 3: Partial keyspace generation
    example_3_partial_keyspace()?;

    println!("\n✨ All examples completed successfully!");

    Ok(())
}

/// Example 1: Demonstrating different output formats
fn example_1_output_formats() -> Result<()> {
    println!("📋 Example 1: Output Formats\n");

    let mut gpu = GpuContext::new()?;

    // Define a simple pattern: 2 letters + 2 digits
    let mut charsets = HashMap::new();
    charsets.insert(0, b"ab".to_vec()); // 2 letters
    charsets.insert(1, b"12".to_vec()); // 2 digits
    let mask = vec![0, 0, 1, 1]; // Pattern: ?0?0?1?1

    let keyspace = 2 * 2 * 2 * 2; // 16 combinations

    // Format 0: NEWLINES (each word on a new line)
    println!("Format 0: NEWLINES");
    println!("─────────────────────");
    let output = gpu.generate_batch(&charsets, &mask, 0, keyspace, 0)?;
    let words: Vec<&str> = std::str::from_utf8(&output)?.lines().collect();
    for (i, word) in words.iter().take(4).enumerate() {
        println!("  [{i}] {word}");
    }
    println!("  ... ({} total)\n", words.len());

    // Format 2: PACKED (no separators, most efficient)
    println!("Format 2: PACKED");
    println!("─────────────────────");
    let output = gpu.generate_batch(&charsets, &mask, 0, keyspace, 2)?;
    let word_length = mask.len();
    for i in 0..4 {
        let start = i * word_length;
        let end = start + word_length;
        let word = String::from_utf8_lossy(&output[start..end]);
        println!("  [{i}] {word}");
    }
    println!("  ... ({} total)\n", output.len() / word_length);

    Ok(())
}

/// Example 2: Multi-GPU context (automatically uses all available GPUs)
fn example_2_multi_gpu() -> Result<()> {
    println!("📋 Example 2: Multi-GPU Context\n");

    // Create multi-GPU context (automatically detects and uses all GPUs)
    let mut multi_gpu = MultiGpuContext::new()?;
    let num_gpus = multi_gpu.num_devices();

    println!("🔍 Detected {num_gpus} GPU(s)");

    if num_gpus == 1 {
        println!("   ℹ️  Single GPU detected - using optimized fast path");
    } else {
        println!("   🚀 Multiple GPUs detected - work will be distributed automatically");
    }

    // Generate a larger batch to demonstrate multi-GPU performance
    let mut charsets = HashMap::new();
    charsets.insert(0, b"abcdefghijklmnopqrstuvwxyz".to_vec()); // 26 letters
    charsets.insert(1, b"0123456789".to_vec()); // 10 digits
    let mask = vec![0, 0, 0, 0, 1, 1, 1, 1]; // 4 letters + 4 digits

    let batch_size = 10_000_000u64; // 10 million words

    println!("\n⚡ Generating 10M words...");
    let start = Instant::now();
    let output = multi_gpu.generate_batch(&charsets, &mask, 0, batch_size, 2)?;
    let duration = start.elapsed();

    let throughput = batch_size as f64 / duration.as_secs_f64() / 1_000_000.0;

    println!(
        "   ✅ Generated {} bytes in {:.4}s",
        output.len(),
        duration.as_secs_f64()
    );
    println!("   📊 Throughput: {throughput:.2} M words/s\n");

    // Show first few words
    let word_length = mask.len();
    println!("First 5 words:");
    for i in 0..5 {
        let start = i * word_length;
        let end = start + word_length;
        let word = String::from_utf8_lossy(&output[start..end]);
        println!("  [{i}] {word}");
    }

    Ok(())
}

/// Example 3: Generating a subset of the keyspace
fn example_3_partial_keyspace() -> Result<()> {
    println!("📋 Example 3: Partial Keyspace Generation\n");

    let mut gpu = GpuContext::new()?;

    let mut charsets = HashMap::new();
    charsets.insert(0, b"abc".to_vec());
    charsets.insert(1, b"123".to_vec());
    let mask = vec![0, 1]; // Pattern: ?0?1

    // Total keyspace: 3 × 3 = 9 words
    // Let's generate words 3-5 (indices 3, 4, 5)
    let start_idx = 3;
    let count = 3;

    println!("Total keyspace: 9 words (a1, a2, a3, b1, b2, b3, c1, c2, c3)");
    println!(
        "Generating subset: indices {} to {}\n",
        start_idx,
        start_idx + count - 1
    );

    let output = gpu.generate_batch(&charsets, &mask, start_idx, count, 2)?;

    let word_length = mask.len();
    println!("Generated words:");
    for i in 0..count {
        let start = i as usize * word_length;
        let end = start + word_length;
        let word = String::from_utf8_lossy(&output[start..end]);
        println!("  Index {}: {}", start_idx + i, word);
    }

    println!("\n💡 Use case: This is useful for distributed generation across multiple machines!");

    Ok(())
}
