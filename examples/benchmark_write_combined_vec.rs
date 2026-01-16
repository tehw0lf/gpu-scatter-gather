//! Write-Combined Memory Benchmark - Vec Collection Pattern
//!
//! This benchmark tests the hypothesis that write-combined (WC) memory may
//! regress performance for patterns that read the data back (like to_vec()).
//!
//! Test: Uses generate_batch_with() callback API to collect data into Vec,
//! which requires the CPU to read back the data after GPU writes it.
//!
//! Expected: WC memory may show 0-5% regression since reading from WC memory
//! is uncached and slower than reading from regular pinned memory.

use anyhow::Result;
use gpu_scatter_gather::ffi::WG_FORMAT_PACKED;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use gpu_scatter_gather::Charset;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🧪 Write-Combined Memory Experiment - Vec Collection Pattern");
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
        let mut _warmup_vec = Vec::with_capacity(1_000_000 * word_length);
        let _ = ctx.generate_batch_with(
            &charsets,
            &mask,
            0,
            1_000_000, // 1M words for warmup
            WG_FORMAT_PACKED,
            |data| {
                _warmup_vec.extend_from_slice(data);
                Ok::<(), std::io::Error>(())
            },
        )?;
    }
    println!("✅ Warmup complete");
    println!();

    // Benchmark: Collect to Vec (read-after-write pattern)
    println!("🚀 Running benchmark: Vec collection via callback API");
    println!();

    let mut timings = Vec::new();
    const NUM_ITERATIONS: usize = 5;

    for iteration in 1..=NUM_ITERATIONS {
        let mut result_vec = Vec::with_capacity((batch_size * word_length as u64) as usize);

        let start = Instant::now();

        let _ =
            ctx.generate_batch_with(&charsets, &mask, 0, batch_size, WG_FORMAT_PACKED, |data| {
                result_vec.extend_from_slice(data);
                Ok::<(), std::io::Error>(())
            })?;

        let elapsed = start.elapsed();
        timings.push(elapsed.as_secs_f64());

        let throughput = batch_size as f64 / elapsed.as_secs_f64();
        let bandwidth = (batch_size * word_length as u64) as f64 / elapsed.as_secs_f64() / 1e9;

        println!(
            "  Iteration {}: {:.2} M words/s ({:.2} GB/s) in {:.3}s (collected {} bytes)",
            iteration,
            throughput / 1e6,
            bandwidth,
            elapsed.as_secs_f64(),
            result_vec.len()
        );

        // Verify we got the right amount of data
        assert_eq!(
            result_vec.len(),
            (batch_size * word_length as u64) as usize,
            "Vec size mismatch"
        );
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
        "test": "write_combined_vec_collection",
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

    let results_file = "benchmark_write_combined_vec.json";
    std::fs::write(results_file, serde_json::to_string_pretty(&results)?)?;
    println!("✅ Results saved to {results_file}");

    Ok(())
}
