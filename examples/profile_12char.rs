//! Profiling benchmark for 12-character passwords
//!
//! Simplified version for Nsight Compute profiling

use anyhow::Result;
use cuda_driver_sys::*;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::Charset;
use std::ptr;

fn main() -> Result<()> {
    println!("🔍 Profiling 12-character password generation");
    println!();

    // Initialize GPU
    let mut gpu = GpuContext::new()?;
    let device_name = gpu.device_name()?;
    let (major, minor) = gpu.compute_capability();

    println!("GPU: {}", device_name);
    println!("Compute Capability: {}.{}", major, minor);
    println!();

    // Charsets
    let lowercase = Charset::new(b"abcdefghijklmnopqrstuvwxyz".to_vec()); // 26 chars
    let digits = Charset::new(b"0123456789".to_vec()); // 10 chars

    let mut charsets = std::collections::HashMap::new();
    charsets.insert(0, lowercase.as_bytes().to_vec()); // ?l
    charsets.insert(1, digits.as_bytes().to_vec());     // ?d

    // Pattern: ?l?l?l?l?l?l?l?l?d?d?d?d (8 lowercase + 4 digits)
    let mask = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];

    println!("Test pattern: ?l?l?l?l?l?l?l?l?d?d?d?d (12 characters)");
    println!("  Lowercase: 8, Digits: 4");
    println!("  Keyspace: 26^8 * 10^4 = 2.09e15 combinations");
    println!();

    // Run a 100M word batch for profiling
    let batch_size = 100_000_000u64;

    println!("Generating {} words for profiling...", batch_size);

    unsafe {
        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        check_cuda(cuEventCreate(&mut start_event, 0))?;
        check_cuda(cuEventCreate(&mut end_event, 0))?;

        check_cuda(cuEventRecord(start_event, ptr::null_mut()))?;
        let _output = gpu.generate_batch(&charsets, &mask, 0, batch_size, 0)?;  // format=0 (newlines)
        check_cuda(cuEventRecord(end_event, ptr::null_mut()))?;
        check_cuda(cuEventSynchronize(end_event))?;

        let mut elapsed_ms = 0.0f32;
        check_cuda(cuEventElapsedTime(&mut elapsed_ms, start_event, end_event))?;

        let elapsed_secs = elapsed_ms / 1000.0;
        let words_per_second = batch_size as f64 / elapsed_secs as f64;

        println!();
        println!("Completed:");
        println!("  Time: {:.4} s", elapsed_secs);
        println!("  Throughput: {:.2} M words/s", words_per_second / 1e6);
        println!("  Memory bandwidth: {:.2} MB/s", (batch_size as f64 * 13.0) / elapsed_secs as f64 / 1e6);

        check_cuda(cuEventDestroy_v2(start_event))?;
        check_cuda(cuEventDestroy_v2(end_event))?;
    }

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
