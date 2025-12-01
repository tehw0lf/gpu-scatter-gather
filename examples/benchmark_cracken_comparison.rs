//! Cracken Comparison Benchmark
//!
//! Fair comparison with cracken using identical mask pattern:
//! - Mask: ?l?l?l?l?l?l?l?l?l?l?l?l?l?l?l?l (16 lowercase)
//! - Write to file in /data/claude
//! - Measure total throughput including disk I/O

use anyhow::Result;
use gpu_scatter_gather::Charset;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use gpu_scatter_gather::ffi::WG_FORMAT_PACKED;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🔬 GPU Scatter-Gather vs Cracken Comparison");
    println!("{}", "=".repeat(70));
    println!();

    // Create MultiGpuContext
    let mut ctx = MultiGpuContext::new()?;
    println!("✅ MultiGpuContext initialized");
    println!();

    // Charset: lowercase only (same as cracken's ?l)
    let lowercase = Charset::new(b"abcdefghijklmnopqrstuvwxyz".to_vec());

    let mut charsets = HashMap::new();
    charsets.insert(0, lowercase.as_bytes().to_vec());

    // Mask: 16 lowercase characters (same as cracken)
    let mask = vec![0; 16];  // All lowercase
    let batch_size = 50_000_000u64; // 50M words per batch
    let num_batches = 400; // 20 billion words total
    let word_length = 16;

    println!("Configuration:");
    println!("  Mask: ?l?l?l?l?l?l?l?l?l?l?l?l?l?l?l?l (16 lowercase)");
    println!("  Batch size: {} words", batch_size);
    println!("  Number of batches: {}", num_batches);
    println!("  Total words: {} billion", (batch_size * num_batches) / 1_000_000_000);
    println!("  Word length: {} bytes", word_length);
    println!("  Total data: {:.2} GB", (batch_size * num_batches * word_length as u64) as f64 / 1_000_000_000.0);
    println!();

    // Output file in /data/claude
    let output_file = "/data/claude/gpu_scatter_gather_16char.txt";
    println!("📝 Output file: {}", output_file);
    println!();

    // Warmup run
    println!("⏳ Warmup run...");
    {
        let warmup_file = File::create("/tmp/wordlist_warmup.txt")?;
        let mut warmup_writer = std::io::BufWriter::with_capacity(8 * 1024 * 1024, warmup_file);
        ctx.generate_batch_with(
            &charsets,
            &mask,
            0,
            1_000_000, // 1M words for warmup
            WG_FORMAT_PACKED,
            |data| warmup_writer.write_all(data)
        )?;
    }
    std::fs::remove_file("/tmp/wordlist_warmup.txt")?;
    println!("✅ Warmup complete");
    println!();

    // Main benchmark
    println!("🚀 Running benchmark...");
    println!();

    let file = File::create(output_file)?;
    let mut writer = std::io::BufWriter::with_capacity(16 * 1024 * 1024, file);

    let total_start = Instant::now();
    let mut total_words = 0u64;

    for batch in 0..num_batches {
        let batch_start = Instant::now();
        let start_index = batch * batch_size;

        ctx.generate_batch_with(
            &charsets,
            &mask,
            start_index,
            batch_size,
            WG_FORMAT_PACKED,
            |data| writer.write_all(data)
        )?;

        total_words += batch_size;
        let batch_elapsed = batch_start.elapsed();
        let batch_throughput = batch_size as f64 / batch_elapsed.as_secs_f64();

        // Print progress every 10 batches
        if (batch + 1) % 10 == 0 {
            let total_elapsed = total_start.elapsed();
            let overall_throughput = total_words as f64 / total_elapsed.as_secs_f64();
            println!("  Batch {}/{}: {:.2} M words/s (batch), {:.2} M words/s (overall), {} GB written",
                batch + 1,
                num_batches,
                batch_throughput / 1e6,
                overall_throughput / 1e6,
                (total_words * word_length as u64) / 1_000_000_000
            );
        }
    }

    writer.flush()?;
    drop(writer);

    let total_elapsed = total_start.elapsed();
    let overall_throughput = total_words as f64 / total_elapsed.as_secs_f64();
    let overall_bandwidth = (total_words * word_length as u64) as f64 / total_elapsed.as_secs_f64() / 1e9;

    println!();
    println!("📊 Final Results:");
    println!("{}", "=".repeat(70));
    println!("Total words generated: {:.2} billion", total_words as f64 / 1e9);
    println!("Total time: {:.2} seconds", total_elapsed.as_secs_f64());
    println!("Overall throughput: {:.2} M words/s", overall_throughput / 1e6);
    println!("Overall bandwidth: {:.2} GB/s", overall_bandwidth);
    println!();

    // Compare with cracken
    println!("🔍 Comparison with Cracken:");
    println!("{}", "=".repeat(70));
    println!("Cracken throughput: ~28-29 M words/s");
    println!("GPU Scatter-Gather: {:.2} M words/s", overall_throughput / 1e6);
    println!("Speedup: {:.2}x", overall_throughput / 28_500_000.0);
    println!();

    Ok(())
}
