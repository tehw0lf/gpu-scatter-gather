//! Production GPU Benchmark with Realistic Memory I/O
//!
//! This benchmark measures REAL throughput including:
//! - GPU kernel execution
//! - Global memory writes
//! - PCIe transfer from GPU to CPU
//! - Memory allocation overhead
//!
//! Target: 500M-1B words/second (3-7x faster than maskprocessor's 142M/s)

use anyhow::Result;
use cuda_driver_sys::*;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::Charset;
use std::ptr;

fn main() -> Result<()> {
    println!("🚀 GPU Scatter-Gather - PRODUCTION Benchmark");
    println!("{}", "=".repeat(70));
    println!();

    // Initialize GPU
    let gpu = GpuContext::new()?;
    let device_name = gpu.device_name()?;
    let (major, minor) = gpu.compute_capability();

    println!("GPU: {}", device_name);
    println!("Compute Capability: {}.{}", major, minor);
    println!();

    // Test pattern: ?1?2 where ?1="abc", ?2="123"
    println!("Test pattern: ?1?2?1?2 (4-character words)");
    println!("  Charset 1: abc (3 chars)");
    println!("  Charset 2: 123 (3 chars)");
    println!("  Total keyspace: 3^4 = 81 combinations");
    println!();

    let charset1 = Charset::new(b"abc".to_vec());
    let charset2 = Charset::new(b"123".to_vec());

    let mut charsets = std::collections::HashMap::new();
    charsets.insert(0, charset1.as_bytes().to_vec());
    charsets.insert(1, charset2.as_bytes().to_vec());

    let mask = vec![0, 1, 0, 1]; // ?1?2?1?2

    // Batch sizes to test
    let batch_sizes = vec![
        10_000_000u64,      // 10M
        50_000_000,         // 50M
        100_000_000,        // 100M
        500_000_000,        // 500M
        1_000_000_000,      // 1B
    ];

    println!("Running benchmarks with realistic memory I/O...");
    println!();

    // Create CUDA events for timing
    unsafe {
        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        check_cuda(cuEventCreate(&mut start_event, 0))?;
        check_cuda(cuEventCreate(&mut end_event, 0))?;

        for &batch_size in &batch_sizes {
            // Record start
            check_cuda(cuEventRecord(start_event, ptr::null_mut()))?;

            // Generate batch (includes kernel + memory I/O)
            let _output = gpu.generate_batch(&charsets, &mask, 0, batch_size)?;

            // Record end
            check_cuda(cuEventRecord(end_event, ptr::null_mut()))?;
            check_cuda(cuEventSynchronize(end_event))?;

            // Get elapsed time
            let mut elapsed_ms = 0.0f32;
            check_cuda(cuEventElapsedTime(&mut elapsed_ms, start_event, end_event))?;

            let elapsed_secs = elapsed_ms / 1000.0;
            let words_per_second = batch_size as f64 / elapsed_secs as f64;
            let speedup_vs_maskprocessor = words_per_second / 142_000_000.0;

            println!(
                "Batch: {:>12} words | Time: {:>8.4} s | Throughput: {:>7.2} M words/s | Speedup: {:>5.2}x",
                batch_size,
                elapsed_secs,
                words_per_second / 1e6,
                speedup_vs_maskprocessor
            );
        }

        check_cuda(cuEventDestroy_v2(start_event))?;
        check_cuda(cuEventDestroy_v2(end_event))?;
    }

    println!();
    println!("{}", "=".repeat(70));
    println!("ANALYSIS:");
    println!("{}", "=".repeat(70));
    println!();
    println!("This benchmark includes:");
    println!("  ✅ GPU kernel execution (mixed-radix computation)");
    println!("  ✅ Global memory writes (GPU DRAM)");
    println!("  ✅ PCIe transfer (GPU → CPU)");
    println!("  ✅ Memory allocation overhead");
    println!();
    println!("Baseline comparison:");
    println!("  maskprocessor (CPU): 142 M words/s");
    println!("  crunch (CPU):        5 M words/s");
    println!();
    println!("If we hit 500M-1B words/s, we've achieved our target!");
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
            format!("CUDA error code: {:?}", result)
        };
        anyhow::bail!("CUDA error: {}", error_msg);
    }
    Ok(())
}
