//! Transposed Write Kernel Benchmark
//!
//! Compares the original uncoalesced kernel against the new transposed write kernel
//! to measure the impact of fully coalesced memory writes.
//!
//! Expected improvement: 5-13x speedup from 92% coalescing efficiency improvement

use anyhow::Result;
use cuda_driver_sys::*;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::Charset;
use std::ptr;

fn main() -> Result<()> {
    println!("🚀 GPU Scatter-Gather - TRANSPOSED WRITE Benchmark");
    println!("{}", "=".repeat(80));
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

    // Test 12-char passwords (most interesting case from analysis)
    let word_length = 12;
    let pattern = "?l?l?l?l?l?l?l?l?d?d?d?d";
    let keyspace = 26u64.pow(8) * 10u64.pow(4);

    let mut mask = Vec::new();
    mask.extend(std::iter::repeat_n(0, 8)); // lowercase
    mask.extend(std::iter::repeat_n(1, 4)); // digit

    println!("📊 Testing 12-character passwords: {pattern}");
    println!("   Lowercase: 8, Digits: 4");
    println!(
        "   Total keyspace: {} combinations ({:.2e})",
        keyspace, keyspace as f64
    );
    println!();

    // Test batch sizes
    let batch_sizes = vec![
        10_000_000u64, // 10M
        50_000_000,    // 50M
        100_000_000,   // 100M
    ];

    println!("{}", "=".repeat(80));
    println!("BASELINE: Original Uncoalesced Kernel");
    println!("{}", "=".repeat(80));
    println!();

    unsafe {
        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        check_cuda(cuEventCreate(&mut start_event, 0))?;
        check_cuda(cuEventCreate(&mut end_event, 0))?;

        for &batch_size in &batch_sizes {
            // Warmup
            let _ = gpu.generate_batch(&charsets, &mask, 0, 1000, 0)?; // format=0 (newlines)

            // Measure
            check_cuda(cuEventRecord(start_event, ptr::null_mut()))?;
            let _output = gpu.generate_batch(&charsets, &mask, 0, batch_size, 0)?; // format=0 (newlines)
            check_cuda(cuEventRecord(end_event, ptr::null_mut()))?;
            check_cuda(cuEventSynchronize(end_event))?;

            let mut elapsed_ms = 0.0f32;
            check_cuda(cuEventElapsedTime(&mut elapsed_ms, start_event, end_event))?;

            let elapsed_secs = elapsed_ms / 1000.0;
            let words_per_second = batch_size as f64 / elapsed_secs as f64;
            let mb_per_second =
                (batch_size as f64 * (word_length + 1) as f64) / elapsed_secs as f64 / 1e6;

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
    println!("{}", "=".repeat(80));
    println!("OPTIMIZED: Transposed Write Kernel (Fully Coalesced)");
    println!("{}", "=".repeat(80));
    println!();

    unsafe {
        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        check_cuda(cuEventCreate(&mut start_event, 0))?;
        check_cuda(cuEventCreate(&mut end_event, 0))?;

        for &batch_size in &batch_sizes {
            // Warmup
            let _ = gpu.generate_batch_transposed(&charsets, &mask, 0, 1000, 0)?; // format=0 (newlines)

            // Measure
            check_cuda(cuEventRecord(start_event, ptr::null_mut()))?;
            let _output = gpu.generate_batch_transposed(&charsets, &mask, 0, batch_size, 0)?; // format=0 (newlines)
            check_cuda(cuEventRecord(end_event, ptr::null_mut()))?;
            check_cuda(cuEventSynchronize(end_event))?;

            let mut elapsed_ms = 0.0f32;
            check_cuda(cuEventElapsedTime(&mut elapsed_ms, start_event, end_event))?;

            let elapsed_secs = elapsed_ms / 1000.0;
            let words_per_second = batch_size as f64 / elapsed_secs as f64;
            let mb_per_second =
                (batch_size as f64 * (word_length + 1) as f64) / elapsed_secs as f64 / 1e6;

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
    println!("{}", "=".repeat(80));
    println!("ANALYSIS:");
    println!("{}", "=".repeat(80));
    println!();
    println!("Expected improvements from transposed writes:");
    println!("  • Memory coalescing: 7.69% → 100% (13x efficiency gain)");
    println!("  • Memory transactions: 528M → ~40M per batch (13x reduction)");
    println!("  • L1 cache amplification: 13x → 1x (eliminate waste)");
    println!("  • Target throughput: 5-13x speedup");
    println!();
    println!("If results don't match expectations:");
    println!("  • Check shared memory bank conflicts");
    println!("  • Profile with Nsight Compute to verify coalescing");
    println!("  • Analyze warp divergence and occupancy");
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
