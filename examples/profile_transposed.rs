//! Profile transposed write kernel with Nsight Compute
//!
//! Usage: ncu ./target/release/examples/profile_transposed

use anyhow::Result;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::Charset;

fn main() -> Result<()> {
    println!("🔍 Profiling transposed write kernel\n");

    let gpu = GpuContext::new()?;
    let device_name = gpu.device_name()?;
    let (major, minor) = gpu.compute_capability();

    println!("GPU: {}", device_name);
    println!("Compute Capability: {}.{}", major, minor);
    println!();

    // 12-char passwords
    let lowercase = Charset::new(b"abcdefghijklmnopqrstuvwxyz".to_vec());
    let digits = Charset::new(b"0123456789".to_vec());

    let mut charsets = std::collections::HashMap::new();
    charsets.insert(0, lowercase.as_bytes().to_vec());
    charsets.insert(1, digits.as_bytes().to_vec());

    let mut mask = vec![0, 0, 0, 0, 0, 0, 0, 0]; // 8x lowercase
    mask.extend(vec![1, 1, 1, 1]); // 4x digits

    println!("Test pattern: ?l?l?l?l?l?l?l?l?d?d?d?d (12 characters)");
    println!("  Lowercase: 8, Digits: 4");
    println!("  Keyspace: 26^8 * 10^4 = 2.09e15 combinations");
    println!();

    let batch_size = 100_000_000;
    println!("Generating {} words for profiling...", batch_size);

    let start = std::time::Instant::now();
    let _output = gpu.generate_batch_transposed(&charsets, &mask, 0, batch_size, 0)?;  // format=0 (newlines)
    let elapsed = start.elapsed();

    let throughput = batch_size as f64 / elapsed.as_secs_f64();
    let bandwidth = (batch_size as f64 * 13.0) / elapsed.as_secs_f64() / 1e6;

    println!();
    println!("Completed:");
    println!("  Time: {:.4} s", elapsed.as_secs_f64());
    println!("  Throughput: {:.2} M words/s", throughput / 1e6);
    println!("  Memory bandwidth: {:.2} MB/s", bandwidth);

    Ok(())
}
