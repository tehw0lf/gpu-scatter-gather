//! Multi-GPU Async Optimization Benchmark
//!
//! This benchmark compares:
//! - Multi-GPU synchronous (v1.1.0 baseline)
//! - Multi-GPU asynchronous with pinned memory and CUDA streams (v1.2.0 optimization)
//!
//! Expected improvement: 20-30% throughput gain with async optimizations

use anyhow::Result;
use cuda_driver_sys::*;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 GPU Scatter-Gather - MULTI-GPU ASYNC OPTIMIZATION Benchmark");
    println!("{}", "=".repeat(80));
    println!("Comparing: Sync (v1.1.0) vs Async with Pinned Memory (v1.2.0)");
    println!();

    // Get device count
    let device_count = unsafe {
        let mut count = 0i32;
        if cuInit(0) != CUresult::CUDA_SUCCESS {
            anyhow::bail!("Failed to initialize CUDA");
        }
        if cuDeviceGetCount(&mut count) != CUresult::CUDA_SUCCESS {
            anyhow::bail!("Failed to get device count");
        }
        count
    };

    if device_count < 1 {
        anyhow::bail!("No CUDA devices found");
    }

    println!("CUDA Devices Found: {}", device_count);
    println!();

    // Print device information
    for device_id in 0..device_count {
        let gpu = GpuContext::with_device(device_id)?;
        let device_name = gpu.device_name()?;
        let (major, minor) = gpu.compute_capability();
        println!("  Device {}: {} (sm_{}{})", device_id, device_name, major, minor);
    }
    println!();

    // Charsets
    let lowercase = b"abcdefghijklmnopqrstuvwxyz".to_vec(); // 26 chars
    let digits = b"0123456789".to_vec(); // 10 chars

    let mut charsets = HashMap::new();
    charsets.insert(0, lowercase); // ?l
    charsets.insert(1, digits);     // ?d

    // Test configuration: 10-char password (realistic use case)
    let mask = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1]; // ?l?l?l?l?l?l?d?d?d?d
    let word_length = 10;
    let pattern = "?l?l?l?l?l?l?d?d?d?d";

    // Batch sizes to test
    let batch_sizes = vec![
        10_000_000u64,   // 10M
        50_000_000u64,   // 50M
        100_000_000u64,  // 100M
    ];

    println!("{}", "=".repeat(80));
    println!("BENCHMARK CONFIGURATION");
    println!("{}", "=".repeat(80));
    println!("Pattern: {} ({}-char passwords)", pattern, word_length);
    println!("Output Format: PACKED (no separators - optimal bandwidth)");
    println!("Iterations: 3 (average of best 2 to reduce noise)");
    println!();

    for &batch_size in &batch_sizes {
        println!("{}", "=".repeat(80));
        println!("📊 Batch Size: {} words ({:.2e})", batch_size, batch_size as f64);
        println!("{}", "=".repeat(80));
        println!();

        // Warmup run (not timed)
        println!("  🔥 Warmup run...");
        let _ = run_sync_benchmark(&charsets, &mask, batch_size)?;
        println!("  ✅ Warmup complete");
        println!();

        // Test 1: Synchronous (v1.1.0 baseline)
        println!("  📏 Testing SYNC mode (v1.1.0 baseline)...");
        let mut sync_times = Vec::new();
        for i in 1..=3 {
            let (duration, words) = run_sync_benchmark(&charsets, &mask, batch_size)?;
            let throughput = words as f64 / duration.as_secs_f64() / 1_000_000.0;
            sync_times.push(duration);
            println!("    Run {}: {:.2} M words/s ({:.3}s)", i, throughput, duration.as_secs_f64());
        }

        // Average best 2
        sync_times.sort();
        let avg_sync_time = (sync_times[0] + sync_times[1]).as_secs_f64() / 2.0;
        let sync_throughput = batch_size as f64 / avg_sync_time / 1_000_000.0;
        println!();
        println!("    ⚡ SYNC Average: {:.2} M words/s", sync_throughput);
        println!();

        // Test 2: Asynchronous with pinned memory (v1.2.0)
        println!("  📏 Testing ASYNC mode (v1.2.0 - pinned memory + streams)...");
        let mut async_times = Vec::new();
        for i in 1..=3 {
            let (duration, words) = run_async_benchmark(&charsets, &mask, batch_size)?;
            let throughput = words as f64 / duration.as_secs_f64() / 1_000_000.0;
            async_times.push(duration);
            println!("    Run {}: {:.2} M words/s ({:.3}s)", i, throughput, duration.as_secs_f64());
        }

        // Average best 2
        async_times.sort();
        let avg_async_time = (async_times[0] + async_times[1]).as_secs_f64() / 2.0;
        let async_throughput = batch_size as f64 / avg_async_time / 1_000_000.0;
        println!();
        println!("    ⚡ ASYNC Average: {:.2} M words/s", async_throughput);
        println!();

        // Calculate improvement
        let improvement = ((async_throughput - sync_throughput) / sync_throughput) * 100.0;
        let speedup = async_throughput / sync_throughput;

        println!("  📊 RESULTS");
        println!("  {}", "-".repeat(76));
        println!("    SYNC (baseline):  {:.2} M words/s", sync_throughput);
        println!("    ASYNC (optimized): {:.2} M words/s", async_throughput);
        println!("  {}", "-".repeat(76));
        if improvement > 0.0 {
            println!("    ✅ Improvement: +{:.1}% ({:.2}× speedup)", improvement, speedup);
        } else {
            println!("    ⚠️  Regression: {:.1}%", improvement);
        }
        println!();
    }

    println!("{}", "=".repeat(80));
    println!("✅ Benchmark Complete");
    println!("{}", "=".repeat(80));

    Ok(())
}

fn run_sync_benchmark(charsets: &HashMap<usize, Vec<u8>>, mask: &[usize], batch_size: u64) -> Result<(std::time::Duration, u64)> {
    // Create sync multi-GPU context
    let ctx = MultiGpuContext::new()?;

    let start = Instant::now();
    let output = ctx.generate_batch(charsets, mask, 0, batch_size, 2)?; // Format 2 = PACKED
    let duration = start.elapsed();

    let word_length = mask.len();
    let words_generated = output.len() / word_length;

    Ok((duration, words_generated as u64))
}

fn run_async_benchmark(charsets: &HashMap<usize, Vec<u8>>, mask: &[usize], batch_size: u64) -> Result<(std::time::Duration, u64)> {
    // Create async multi-GPU context (with pinned memory + streams)
    let ctx = MultiGpuContext::new_async()?;

    let start = Instant::now();
    let output = ctx.generate_batch(charsets, mask, 0, batch_size, 2)?; // Format 2 = PACKED
    let duration = start.elapsed();

    let word_length = mask.len();
    let words_generated = output.len() / word_length;

    Ok((duration, words_generated as u64))
}
