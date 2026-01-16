//! Pure Generation Benchmark (No Disk I/O)
//!
//! Measures raw GPU candidate generation throughput without disk bottlenecks.
//! Fair comparison with hashcat's benchmark mode.

use anyhow::Result;
use gpu_scatter_gather::ffi::WG_FORMAT_PACKED;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use gpu_scatter_gather::Charset;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Pure GPU Generation Benchmark (No Disk I/O)");
    println!("{}", "=".repeat(70));
    println!();

    // Create MultiGpuContext
    let mut ctx = MultiGpuContext::new()?;
    println!("✅ MultiGpuContext initialized");
    println!();

    // Charset: lowercase only (same as cracken and hashcat ?l)
    let lowercase = Charset::new(b"abcdefghijklmnopqrstuvwxyz".to_vec());

    let mut charsets = HashMap::new();
    charsets.insert(0, lowercase.as_bytes().to_vec());

    // Mask: 16 lowercase characters
    let mask = vec![0; 16];
    let batch_size = 50_000_000u64; // 50M words per batch
    let num_batches = 100; // 5 billion words total
    let word_length = 16;

    println!("Configuration:");
    println!("  Mask: ?l?l?l?l?l?l?l?l?l?l?l?l?l?l?l?l (16 lowercase)");
    println!("  Batch size: {batch_size} words");
    println!("  Number of batches: {num_batches}");
    println!(
        "  Total words: {:.2} billion",
        (batch_size * num_batches) as f64 / 1e9
    );
    println!("  Word length: {word_length} bytes");
    println!();

    // Warmup run
    println!("⏳ Warmup run...");
    let total_bytes = Arc::new(AtomicU64::new(0));
    {
        let bytes_counter = total_bytes.clone();
        let _ = ctx.generate_batch_with(
            &charsets,
            &mask,
            0,
            1_000_000, // 1M words for warmup
            WG_FORMAT_PACKED,
            |data| -> std::io::Result<()> {
                bytes_counter.fetch_add(data.len() as u64, Ordering::Relaxed);
                Ok(())
            },
        )?;
    }
    println!(
        "✅ Warmup complete ({} bytes processed)",
        total_bytes.load(Ordering::Relaxed)
    );
    println!();

    // Main benchmark - just count bytes, don't write
    println!("🚀 Running pure generation benchmark...");
    println!();

    total_bytes.store(0, Ordering::Relaxed);
    let total_start = Instant::now();
    let mut total_words = 0u64;

    for batch in 0..num_batches {
        let batch_start = Instant::now();
        let start_index = batch * batch_size;

        let bytes_counter = total_bytes.clone();
        let _ = ctx.generate_batch_with(
            &charsets,
            &mask,
            start_index,
            batch_size,
            WG_FORMAT_PACKED,
            |data| -> std::io::Result<()> {
                // Just count bytes - no disk I/O!
                bytes_counter.fetch_add(data.len() as u64, Ordering::Relaxed);
                Ok(())
            },
        )?;

        total_words += batch_size;
        let batch_elapsed = batch_start.elapsed();
        let batch_throughput = batch_size as f64 / batch_elapsed.as_secs_f64();

        // Print progress every 10 batches
        if (batch + 1) % 10 == 0 {
            let total_elapsed = total_start.elapsed();
            let overall_throughput = total_words as f64 / total_elapsed.as_secs_f64();
            println!("  Batch {}/{}: {:.2} M words/s (batch), {:.2} M words/s (overall), {:.2} billion words",
                batch + 1,
                num_batches,
                batch_throughput / 1e6,
                overall_throughput / 1e6,
                total_words as f64 / 1e9
            );
        }
    }

    let total_elapsed = total_start.elapsed();
    let overall_throughput = total_words as f64 / total_elapsed.as_secs_f64();
    let overall_bandwidth =
        (total_words * word_length as u64) as f64 / total_elapsed.as_secs_f64() / 1e9;
    let total_bytes_processed = total_bytes.load(Ordering::Relaxed);

    println!();
    println!("📊 Final Results:");
    println!("{}", "=".repeat(70));
    println!(
        "Total words generated: {:.2} billion",
        total_words as f64 / 1e9
    );
    println!(
        "Total bytes processed: {:.2} GB",
        total_bytes_processed as f64 / 1e9
    );
    println!("Total time: {:.2} seconds", total_elapsed.as_secs_f64());
    println!(
        "Overall throughput: {:.2} M words/s",
        overall_throughput / 1e6
    );
    println!("Overall bandwidth: {overall_bandwidth:.2} GB/s");
    println!();

    // Compare with competitors
    println!("🔍 Comparison with Competitors:");
    println!("{}", "=".repeat(70));
    println!("Cracken (disk I/O):           ~28 M words/s");
    println!("Our tool (disk I/O):          ~17 M words/s");
    println!(
        "Our tool (pure generation):   {:.2} M words/s  ⬅️ THIS BENCHMARK",
        overall_throughput / 1e6
    );
    println!();
    println!("Hashcat MD5 benchmark:        72,621 MH/s (includes hashing)");
    println!("  → If we assume MD5 is ~10% overhead, generation ≈ 80,000 M words/s");
    println!("  → But hashcat generates on-GPU for direct hashing (different use case)");
    println!();
    println!(
        "Speedup vs Cracken:           {:.1}x",
        overall_throughput / 28_000_000.0
    );
    println!();

    Ok(())
}
