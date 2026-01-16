use anyhow::Result;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🔍 Performance Comparison: Direct GPU vs Multi-GPU API");
    println!("{}", "=".repeat(80));

    let lowercase = b"abcdefghijklmnopqrstuvwxyz".to_vec();
    let digits = b"0123456789".to_vec();

    let mut charsets = HashMap::new();
    charsets.insert(0, lowercase);
    charsets.insert(1, digits);

    let mask = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1]; // 10 chars
    let batch_size = 100_000_000u64;

    println!("\n📊 Test: 100M words, 10-char pattern (?l?l?l?l?l?l?d?d?d?d)");
    println!("{}", "=".repeat(80));

    // Test 1: Direct GPU context
    println!("\n🔷 Test 1: Direct GpuContext API");
    let (duration1, throughput1, output_len1) = {
        let mut gpu = GpuContext::new()?;
        let start = Instant::now();
        let output1 = gpu.generate_batch(&charsets, &mask, 0, batch_size, 2)?;
        let duration1 = start.elapsed();
        let throughput1 = batch_size as f64 / duration1.as_secs_f64() / 1_000_000.0;
        (duration1, throughput1, output1.len())
    }; // gpu is dropped here
    println!("   Time: {:.4}s", duration1.as_secs_f64());
    println!("   Throughput: {throughput1:.2} M words/s");
    println!("   Output size: {output_len1} bytes");

    // Test 2: Multi-GPU sync (1 device)
    println!("\n🔷 Test 2: MultiGpuContext::new() (sync, 1 GPU)");
    let (duration2, throughput2, output_len2) = {
        let mut ctx_sync = MultiGpuContext::new()?;
        let start = Instant::now();
        let output2 = ctx_sync.generate_batch(&charsets, &mask, 0, batch_size, 2)?;
        let duration2 = start.elapsed();
        let throughput2 = batch_size as f64 / duration2.as_secs_f64() / 1_000_000.0;
        (duration2, throughput2, output2.len())
    }; // ctx_sync is dropped here
    println!("   Time: {:.4}s", duration2.as_secs_f64());
    println!("   Throughput: {throughput2:.2} M words/s");
    println!("   Output size: {output_len2} bytes");

    // Test 3: Multi-GPU async (1 device)
    println!("\n🔷 Test 3: MultiGpuContext::new_async() (async, 1 GPU)");
    let (duration3, throughput3, output_len3) = {
        let mut ctx_async = MultiGpuContext::new_async()?;
        let start = Instant::now();
        let output3 = ctx_async.generate_batch(&charsets, &mask, 0, batch_size, 2)?;
        let duration3 = start.elapsed();
        let throughput3 = batch_size as f64 / duration3.as_secs_f64() / 1_000_000.0;
        (duration3, throughput3, output3.len())
    }; // ctx_async is dropped here
    println!("   Time: {:.4}s", duration3.as_secs_f64());
    println!("   Throughput: {throughput3:.2} M words/s");
    println!("   Output size: {output_len3} bytes");

    println!("\n{}", "=".repeat(80));
    println!("📊 ANALYSIS");
    println!("{}", "=".repeat(80));

    let overhead_sync =
        ((duration2.as_secs_f64() - duration1.as_secs_f64()) / duration1.as_secs_f64()) * 100.0;
    let overhead_async =
        ((duration3.as_secs_f64() - duration1.as_secs_f64()) / duration1.as_secs_f64()) * 100.0;

    println!("\nDirect GPU:       {throughput1:.2} M words/s (baseline)");
    println!("Multi-GPU sync:   {throughput2:.2} M words/s ({overhead_sync:+.1}% overhead)");
    println!("Multi-GPU async:  {throughput3:.2} M words/s ({overhead_async:+.1}% overhead)");

    println!("\n❌ PROBLEM: Multi-GPU API should have <5% overhead, but has {overhead_sync:.1}%!");

    Ok(())
}
