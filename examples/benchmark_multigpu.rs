//! Multi-GPU Scaling Benchmark
//!
//! This benchmark tests multi-GPU scaling efficiency by comparing:
//! - Single GPU baseline performance
//! - Multi-GPU parallel performance
//! - Scaling efficiency calculation
//!
//! Test patterns use realistic password lengths (8-12 characters)
//! with large enough keyspaces to saturate multiple GPUs.

use anyhow::Result;
use cuda_driver_sys::*;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use std::collections::HashMap;
use std::ptr;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 GPU Scatter-Gather - MULTI-GPU SCALING Benchmark");
    println!("{}", "=".repeat(70));
    println!("Output Format: PACKED (no separators - optimal bandwidth)");
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

    // Test configurations: (word_length, num_lowercase, num_digits, description, keyspace)
    let test_configs = vec![
        (8,  4, 4, "?l?l?l?l?d?d?d?d", 26u64.pow(4) * 10u64.pow(4)),          // 456,976,000
        (10, 6, 4, "?l?l?l?l?l?l?d?d?d?d", 26u64.pow(6) * 10u64.pow(4)),      // 3,089,157,760,000
        (12, 8, 4, "?l?l?l?l?l?l?l?l?d?d?d?d", 26u64.pow(8) * 10u64.pow(4)), // 20,863,377,862,720,000
    ];

    // Batch size to test (large enough to saturate multiple GPUs)
    let batch_size = 100_000_000u64; // 100M words

    println!("{}", "=".repeat(70));
    println!("BENCHMARK CONFIGURATION");
    println!("{}", "=".repeat(70));
    println!("Batch Size: {} words ({:.2e})", batch_size, batch_size as f64);
    println!();

    for (word_length, num_lower, num_digit, pattern, keyspace) in test_configs {
        // Skip if batch size exceeds keyspace
        if batch_size > keyspace {
            println!("⏭  Skipping {}-char passwords: batch exceeds keyspace", word_length);
            println!();
            continue;
        }

        println!("{}", "=".repeat(70));
        println!("📊 Testing {}-character passwords: {}", word_length, pattern);
        println!("{}", "=".repeat(70));
        println!("   Lowercase: {}, Digits: {}", num_lower, num_digit);
        println!("   Total keyspace: {} combinations ({:.2e})", keyspace, keyspace as f64);
        println!();

        // Build mask pattern
        let mut mask = Vec::new();
        for _ in 0..num_lower {
            mask.push(0); // lowercase
        }
        for _ in 0..num_digit {
            mask.push(1); // digit
        }

        // Test 1: Single GPU baseline
        println!("🔷 Single GPU Baseline (Device 0)");
        println!("{}", "-".repeat(70));

        let single_gpu_throughput = benchmark_single_gpu(
            &charsets,
            &mask,
            batch_size,
            word_length,
        )?;

        println!();

        // Test 2: Multi-GPU (all available devices)
        if device_count > 1 {
            println!("🔶 Multi-GPU ({} devices)", device_count);
            println!("{}", "-".repeat(70));

            let multi_gpu_throughput = benchmark_multi_gpu(
                &charsets,
                &mask,
                batch_size,
                word_length,
                device_count as usize,
            )?;

            println!();

            // Calculate scaling efficiency
            let expected_throughput = single_gpu_throughput * device_count as f64;
            let scaling_efficiency = (multi_gpu_throughput / expected_throughput) * 100.0;

            println!("📈 SCALING ANALYSIS");
            println!("{}", "-".repeat(70));
            println!("   Single GPU:       {:>10.2} M words/s", single_gpu_throughput / 1e6);
            println!("   Multi-GPU ({}x):    {:>10.2} M words/s", device_count, multi_gpu_throughput / 1e6);
            println!("   Expected (ideal): {:>10.2} M words/s", expected_throughput / 1e6);
            println!("   Speedup:          {:>10.2}x", multi_gpu_throughput / single_gpu_throughput);
            println!("   Efficiency:       {:>10.2}%", scaling_efficiency);
            println!();

            if scaling_efficiency >= 90.0 {
                println!("   ✅ Excellent scaling efficiency (≥90%)");
            } else if scaling_efficiency >= 80.0 {
                println!("   ⚠️  Good scaling efficiency (80-90%)");
            } else {
                println!("   ❌ Poor scaling efficiency (<80%)");
                println!("   Overhead: {:.2}%", 100.0 - scaling_efficiency);
            }
            println!();
        } else {
            println!("⏭  Skipping multi-GPU test: only 1 device available");
            println!();
        }
    }

    println!("{}", "=".repeat(70));
    println!("KEY INSIGHTS");
    println!("{}", "=".repeat(70));
    println!();
    println!("Scaling Efficiency Targets:");
    println!("  • Excellent: ≥90% (minimal overhead)");
    println!("  • Good:      80-90% (acceptable overhead)");
    println!("  • Poor:      <80% (optimization needed)");
    println!();
    println!("Overhead Sources:");
    println!("  • Context switching:     ~1-2%");
    println!("  • Output aggregation:    ~1-3%");
    println!("  • Thread synchronization: ~1%");
    println!("  • Load imbalance:        ~2-5%");
    println!("  • Total expected:        5-11%");
    println!();
    println!("Performance optimization opportunities:");
    println!("  • Pinned memory for faster host↔device transfers");
    println!("  • Pre-allocated buffers to reduce allocation overhead");
    println!("  • Asynchronous kernel launches with CUDA streams");
    println!();

    Ok(())
}

fn benchmark_single_gpu(
    charsets: &HashMap<usize, Vec<u8>>,
    mask: &[usize],
    batch_size: u64,
    word_length: usize,
) -> Result<f64> {
    let gpu = GpuContext::with_device(0)?;

    unsafe {
        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        check_cuda(cuEventCreate(&mut start_event, 0))?;
        check_cuda(cuEventCreate(&mut end_event, 0))?;

        // Record start
        check_cuda(cuEventRecord(start_event, ptr::null_mut()))?;

        // Generate batch (includes kernel + memory I/O)
        let _output = gpu.generate_batch(charsets, mask, 0, batch_size, 2)?; // format=2 (PACKED)

        // Record end
        check_cuda(cuEventRecord(end_event, ptr::null_mut()))?;
        check_cuda(cuEventSynchronize(end_event))?;

        // Get elapsed time
        let mut elapsed_ms = 0.0f32;
        check_cuda(cuEventElapsedTime(&mut elapsed_ms, start_event, end_event))?;

        let elapsed_secs = elapsed_ms / 1000.0;
        let words_per_second = batch_size as f64 / elapsed_secs as f64;
        let mb_per_second = (batch_size as f64 * word_length as f64) / elapsed_secs as f64 / 1e6;

        println!(
            "   Batch: {:>12} words | Time: {:>7.4} s | {:>7.2} M words/s | {:>8.2} MB/s",
            batch_size,
            elapsed_secs,
            words_per_second / 1e6,
            mb_per_second
        );

        check_cuda(cuEventDestroy_v2(start_event))?;
        check_cuda(cuEventDestroy_v2(end_event))?;

        Ok(words_per_second)
    }
}

fn benchmark_multi_gpu(
    charsets: &HashMap<usize, Vec<u8>>,
    mask: &[usize],
    batch_size: u64,
    word_length: usize,
    num_devices: usize,
) -> Result<f64> {
    let multi_gpu = MultiGpuContext::new()?;

    // Use wall-clock time since multi-GPU spans multiple CUDA contexts
    let start = Instant::now();

    // Generate batch across all GPUs
    let _output = multi_gpu.generate_batch(charsets, mask, 0, batch_size, 2)?; // format=2 (PACKED)

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let words_per_second = batch_size as f64 / elapsed_secs;
    let mb_per_second = (batch_size as f64 * word_length as f64) / elapsed_secs / 1e6;

    println!(
        "   Batch: {:>12} words | Time: {:>7.4} s | {:>7.2} M words/s | {:>8.2} MB/s ({} GPUs)",
        batch_size,
        elapsed_secs,
        words_per_second / 1e6,
        mb_per_second,
        num_devices
    );

    Ok(words_per_second)
}

unsafe fn check_cuda(result: CUresult) -> Result<()> {
    if result != CUresult::CUDA_SUCCESS {
        let mut error_str = ptr::null();
        cuGetErrorString(result, &mut error_str);
        let error_msg = if !error_str.is_null() {
            std::ffi::CStr::from_ptr(error_str)
                .to_string_lossy()
                .into_owned()
        } else {
            format!("CUDA error code: {:?}", result)
        };
        anyhow::bail!("CUDA error: {}", error_msg);
    }
    Ok(())
}
