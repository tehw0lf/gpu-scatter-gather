//! Realistic Password Length Benchmark
//!
//! This benchmark tests realistic password lengths (8-16 characters) that are
//! the primary use case for GPU wordlist generation.
//!
//! Test patterns:
//! - 8 chars:  ?l?l?l?l?d?d?d?d  (lowercase + digits, e.g., "pass1234")
//! - 10 chars: ?l?l?l?l?l?l?d?d?d?d
//! - 12 chars: ?l?l?l?l?l?l?l?l?d?d?d?d
//! - 16 chars: ?l?l?l?l?l?l?l?l?l?l?l?l?d?d?d?d

use anyhow::Result;
use cuda_driver_sys::*;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::Charset;
use std::ptr;

fn main() -> Result<()> {
    println!("🚀 GPU Scatter-Gather - REALISTIC Password Length Benchmark");
    println!("{}", "=".repeat(70));
    println!("Output Format: PACKED (no separators - optimal bandwidth)");
    println!();

    // Initialize GPU
    let mut gpu = GpuContext::new()?;
    let device_name = gpu.device_name()?;
    let (major, minor) = gpu.compute_capability();

    println!("GPU: {device_name}");
    println!("Compute Capability: {major}.{minor}");
    println!();

    // Charsets
    let lowercase = Charset::new(b"abcdefghijklmnopqrstuvwxyz".to_vec()); // 26 chars
    let digits = Charset::new(b"0123456789".to_vec()); // 10 chars

    let mut charsets = std::collections::HashMap::new();
    charsets.insert(0, lowercase.as_bytes().to_vec()); // ?l
    charsets.insert(1, digits.as_bytes().to_vec()); // ?d

    // Test configurations: (word_length, num_lowercase, num_digits, description, keyspace)
    let test_configs = vec![
        (8, 4, 4, "?l?l?l?l?d?d?d?d", 26u64.pow(4) * 10u64.pow(4)), // 456,976,000
        (
            10,
            6,
            4,
            "?l?l?l?l?l?l?d?d?d?d",
            26u64.pow(6) * 10u64.pow(4),
        ), // 3,089,157,760,000
        (
            12,
            8,
            4,
            "?l?l?l?l?l?l?l?l?d?d?d?d",
            26u64.pow(8) * 10u64.pow(4),
        ), // 20,863,377,862,720,000
        (
            16,
            12,
            4,
            "?l?l?l?l?l?l?l?l?l?l?l?l?d?d?d?d",
            26u64.pow(12) * 10u64.pow(4),
        ), // ~9.5e16
    ];

    // Batch sizes to test (smaller for longer words due to larger keyspace)
    let batch_sizes = vec![
        10_000_000u64, // 10M
        50_000_000,    // 50M
        100_000_000,   // 100M
    ];

    for (word_length, num_lower, num_digit, pattern, keyspace) in test_configs {
        println!(
            "📊 Testing {word_length}-character passwords: {pattern}"
        );
        println!("   Lowercase: {num_lower}, Digits: {num_digit}");
        println!(
            "   Total keyspace: {} combinations ({:.2e})",
            keyspace, keyspace as f64
        );
        println!();

        // Build mask pattern
        let mut mask = Vec::new();
        for _ in 0..num_lower {
            mask.push(0); // lowercase
        }
        for _ in 0..num_digit {
            mask.push(1); // digit
        }

        // Create CUDA events for timing
        unsafe {
            let mut start_event = ptr::null_mut();
            let mut end_event = ptr::null_mut();
            check_cuda(cuEventCreate(&mut start_event, 0))?;
            check_cuda(cuEventCreate(&mut end_event, 0))?;

            for &batch_size in &batch_sizes {
                // Skip if batch size exceeds keyspace
                if batch_size > keyspace {
                    println!(
                        "   Batch: {batch_size:>12} words | SKIPPED (exceeds keyspace)"
                    );
                    continue;
                }

                // Record start
                check_cuda(cuEventRecord(start_event, ptr::null_mut()))?;

                // Generate batch (includes kernel + memory I/O)
                let _output = gpu.generate_batch(&charsets, &mask, 0, batch_size, 2)?; // format=2 (PACKED)

                // Record end
                check_cuda(cuEventRecord(end_event, ptr::null_mut()))?;
                check_cuda(cuEventSynchronize(end_event))?;

                // Get elapsed time
                let mut elapsed_ms = 0.0f32;
                check_cuda(cuEventElapsedTime(&mut elapsed_ms, start_event, end_event))?;

                let elapsed_secs = elapsed_ms / 1000.0;
                let words_per_second = batch_size as f64 / elapsed_secs as f64;
                let mb_per_second =
                    (batch_size as f64 * word_length as f64) / elapsed_secs as f64 / 1e6; // PACKED format has no separator

                println!(
                    "   Batch: {:>12} words | Time: {:>7.4} s | {:>7.2} M words/s | {:>8.2} MB/s",
                    batch_size,
                    elapsed_secs,
                    words_per_second / 1e6,
                    mb_per_second
                );
            }

            check_cuda(cuEventDestroy_v2(start_event))?;
            check_cuda(cuEventDestroy_v2(end_event))?;
        }

        println!();
    }

    println!("{}", "=".repeat(70));
    println!("KEY INSIGHTS:");
    println!("{}", "=".repeat(70));
    println!();
    println!("For password cracking / wordlist generation:");
    println!("  • 8-char passwords are common (old minimum standards)");
    println!("  • 12-char passwords are modern recommended minimum");
    println!("  • 16-char passwords are high-security targets");
    println!();
    println!("Performance metrics:");
    println!("  • Words/s: How many password candidates generated per second");
    println!("  • MB/s: Memory bandwidth utilization (higher = better GPU utilization)");
    println!();
    println!("Expected results:");
    println!("  • Longer words = more compute, more memory writes");
    println!("  • Should see HIGHER throughput with longer words (better GPU utilization)");
    println!("  • Memory bandwidth should be the bottleneck (not compute)");
    println!();

    Ok(())
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
            format!("CUDA error code: {result:?}")
        };
        anyhow::bail!("CUDA error: {error_msg}");
    }
    Ok(())
}
