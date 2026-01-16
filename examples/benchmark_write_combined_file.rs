//! Write-Combined Memory Benchmark - File I/O Pattern
//!
//! This benchmark tests the hypothesis that write-combined (WC) memory improves
//! performance for write-only access patterns like direct file I/O.
//!
//! Test: Uses generate_batch_with() callback API to write directly to file,
//! which should benefit from WC memory's faster write speeds.
//!
//! Expected: WC memory may show 5-15% improvement since the callback writes
//! data to file without reading it back (write-only pattern).

use anyhow::Result;
use gpu_scatter_gather::ffi::WG_FORMAT_PACKED;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use gpu_scatter_gather::Charset;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🧪 Write-Combined Memory Experiment - File I/O Pattern");
    println!("{}", "=".repeat(70));
    println!();

    // Create MultiGpuContext (uses write-combined memory if enabled)
    let mut ctx = MultiGpuContext::new()?;
    println!("✅ MultiGpuContext initialized");
    println!();

    // Charsets
    let lowercase = Charset::new(b"abcdefghijklmnopqrstuvwxyz".to_vec());
    let digits = Charset::new(b"0123456789".to_vec());

    let mut charsets = HashMap::new();
    charsets.insert(0, lowercase.as_bytes().to_vec()); // ?l (index 0)
    charsets.insert(1, digits.as_bytes().to_vec()); // ?d (index 1)

    // Test configuration: 16-char password (mask as indices)
    let mask = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]; // Equivalent to ?l?l?l?l?l?l?l?l?l?l?l?l?d?d?d?d
    let batch_size = 50_000_000u64; // 50M words
    let word_length = 16;

    println!("Configuration:");
    println!("  Mask: ?l?l?l?l?l?l?l?l?l?l?l?l?d?d?d?d (12 lowercase + 4 digits)");
    println!("  Batch size: {batch_size} words");
    println!("  Word length: {word_length} bytes");
    println!(
        "  Total data: {:.2} MB",
        (batch_size * word_length as u64) as f64 / 1_000_000.0
    );
    println!();

    // Warmup run (not timed)
    println!("⏳ Warmup run...");
    {
        let warmup_file = File::create("/tmp/wordlist_warmup.txt")?;
        let mut warmup_writer = std::io::BufWriter::new(warmup_file);
        let _ = ctx.generate_batch_with(
            &charsets,
            &mask,
            0,
            1_000_000, // 1M words for warmup
            WG_FORMAT_PACKED,
            |data| warmup_writer.write_all(data),
        )?;
    }
    std::fs::remove_file("/tmp/wordlist_warmup.txt")?;
    println!("✅ Warmup complete");
    println!();

    // Benchmark: File I/O with callback API (write-only pattern)
    println!("🚀 Running benchmark: File I/O via callback API");
    println!();

    let mut timings = Vec::new();
    const NUM_ITERATIONS: usize = 5;

    for iteration in 1..=NUM_ITERATIONS {
        let output_file = format!("/tmp/wordlist_bench_{iteration}.txt");
        let file = File::create(&output_file)?;
        let mut writer = std::io::BufWriter::new(file);

        let start = Instant::now();

        let _ =
            ctx.generate_batch_with(&charsets, &mask, 0, batch_size, WG_FORMAT_PACKED, |data| {
                writer.write_all(data)
            })?;

        let elapsed = start.elapsed();
        timings.push(elapsed.as_secs_f64());

        let throughput = batch_size as f64 / elapsed.as_secs_f64();
        let bandwidth = (batch_size * word_length as u64) as f64 / elapsed.as_secs_f64() / 1e9;

        println!(
            "  Iteration {}: {:.2} M words/s ({:.2} GB/s) in {:.3}s",
            iteration,
            throughput / 1e6,
            bandwidth,
            elapsed.as_secs_f64()
        );

        // Cleanup
        std::fs::remove_file(output_file)?;
    }

    // Statistics
    println!();
    println!("📊 Results Summary:");
    println!("{}", "-".repeat(70));

    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
    let mean_throughput = batch_size as f64 / mean;
    let mean_bandwidth = (batch_size * word_length as u64) as f64 / mean / 1e9;

    let variance = timings.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / timings.len() as f64;
    let std_dev = variance.sqrt();
    let cv = (std_dev / mean) * 100.0;

    println!("Mean throughput: {:.2} M words/s", mean_throughput / 1e6);
    println!("Mean bandwidth: {mean_bandwidth:.2} GB/s");
    println!("Standard deviation: {std_dev:.3}s ({cv:.2}%)");
    println!();

    // Save results to JSON for comparison
    let results = serde_json::json!({
        "test": "write_combined_file_io",
        "mask": mask,
        "batch_size": batch_size,
        "word_length": word_length,
        "iterations": NUM_ITERATIONS,
        "mean_throughput_words_per_sec": mean_throughput,
        "mean_bandwidth_gbps": mean_bandwidth,
        "std_dev_seconds": std_dev,
        "coefficient_of_variation_percent": cv,
        "timings": timings,
    });

    let results_file = "benchmark_write_combined_file.json";
    std::fs::write(results_file, serde_json::to_string_pretty(&results)?)?;
    println!("✅ Results saved to {results_file}");

    Ok(())
}
