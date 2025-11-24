//! Comprehensive benchmark comparing all three kernel approaches:
//! 1. Original uncoalesced kernel (baseline)
//! 2. Transposed kernel (attempted optimization)
//! 3. Hybrid column-major GPU + CPU transpose (expected winner)
//!
//! This benchmark will definitively show which approach achieves the best performance.

use anyhow::Result;
use gpu_scatter_gather::gpu::GpuContext;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== GPU Scatter-Gather Hybrid Architecture Benchmark ===\n");

    // Initialize GPU
    let mut gpu = GpuContext::new()?;
    println!("GPU Device: {}", gpu.device_name()?);
    println!("Compute Capability: {:?}\n", gpu.compute_capability());

    // Test configuration: realistic 12-char passwords
    let charset = b"abcdefghijklmnopqrstuvwxyz0123456789".to_vec();
    let mut charsets = HashMap::new();
    charsets.insert(1, charset);

    let mask: Vec<usize> = vec![1; 12]; // 12 character password
    let batch_size = 100_000_000u64; // 100M words

    println!("Configuration:");
    println!("  Word length: {} chars", mask.len());
    println!("  Charset size: 36 (lowercase + digits)");
    println!("  Batch size: {} words", batch_size / 1_000_000);
    println!("  Data transfer: {} MB\n", (batch_size * 13) / 1_000_000);

    // Warmup
    println!("Warming up GPU...");
    let _ = gpu.generate_batch(&charsets, &mask, 0, 1_000_000, 0)?;  // format=0 (newlines)
    println!("Warmup complete.\n");

    println!("{}", "=".repeat(80));
    println!("BENCHMARK 1: Original Uncoalesced Kernel");
    println!("{}", "=".repeat(80));
    println!("Memory pattern: Row-major writes (13 bytes apart)");
    println!("Expected coalescing: 7.69% (baseline)\n");

    let start = Instant::now();
    let output_original = gpu.generate_batch(&charsets, &mask, 0, batch_size, 0)?;  // format=0 (newlines)
    let duration_original = start.elapsed();

    let throughput_original = batch_size as f64 / duration_original.as_secs_f64();
    let bandwidth_original = (batch_size * 13) as f64 / duration_original.as_secs_f64() / 1e9;

    println!("Results:");
    println!("  Time: {:.2} ms", duration_original.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.2} M words/s", throughput_original / 1e6);
    println!("  Memory bandwidth: {:.2} GB/s", bandwidth_original);
    println!();

    // Verify sample output
    println!("Sample output (first 3 words):");
    for i in 0..3 {
        let word = &output_original[i * 13..(i + 1) * 13];
        println!("  {}: {}", i, String::from_utf8_lossy(word).trim());
    }
    println!();

    println!("{}", "=".repeat(80));
    println!("BENCHMARK 2: Transposed Kernel (Session 3 attempt)");
    println!("{}", "=".repeat(80));
    println!("Memory pattern: Shared memory staging -> row-major writes");
    println!("Expected coalescing: 7.69% (same as baseline - writes still uncoalesced)\n");

    let start = Instant::now();
    let output_transposed = gpu.generate_batch_transposed(&charsets, &mask, 0, batch_size, 0)?;  // format=0 (newlines)
    let duration_transposed = start.elapsed();

    let throughput_transposed = batch_size as f64 / duration_transposed.as_secs_f64();
    let bandwidth_transposed = (batch_size * 13) as f64 / duration_transposed.as_secs_f64() / 1e9;

    println!("Results:");
    println!("  Time: {:.2} ms", duration_transposed.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.2} M words/s", throughput_transposed / 1e6);
    println!("  Memory bandwidth: {:.2} GB/s", bandwidth_transposed);
    println!();

    // Verify correctness
    if output_transposed == output_original {
        println!("✓ Correctness: Output matches original kernel");
    } else {
        println!("✗ ERROR: Output differs from original kernel!");
    }
    println!();

    println!("{}", "=".repeat(80));
    println!("BENCHMARK 3: Hybrid Column-Major GPU + CPU Transpose (Session 4)");
    println!("{}", "=".repeat(80));
    println!("Memory pattern: Column-major writes (consecutive addresses!)");
    println!("Expected coalescing: 85-95% (11-12x improvement)");
    println!("CPU transpose: AVX2 SIMD\n");

    let start = Instant::now();
    let output_hybrid = gpu.generate_batch_hybrid(&charsets, &mask, 0, batch_size, 0)?;  // format=0 (newlines)
    let duration_hybrid = start.elapsed();

    let throughput_hybrid = batch_size as f64 / duration_hybrid.as_secs_f64();
    let bandwidth_hybrid = (batch_size * 13) as f64 / duration_hybrid.as_secs_f64() / 1e9;

    println!("Results:");
    println!("  Time: {:.2} ms", duration_hybrid.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.2} M words/s", throughput_hybrid / 1e6);
    println!("  Memory bandwidth: {:.2} GB/s", bandwidth_hybrid);
    println!();

    // Verify correctness
    if output_hybrid == output_original {
        println!("✓ Correctness: Output matches original kernel");
    } else {
        println!("✗ ERROR: Output differs from original kernel!");

        // Debug: show first mismatch
        for i in 0..output_hybrid.len().min(output_original.len()) {
            if output_hybrid[i] != output_original[i] {
                println!("  First mismatch at byte {}", i);
                println!("  Expected: {:?}", &output_original[i.saturating_sub(10)..i+10]);
                println!("  Got:      {:?}", &output_hybrid[i.saturating_sub(10)..i+10]);
                break;
            }
        }
    }
    println!();

    println!("{}", "=".repeat(80));
    println!("PERFORMANCE COMPARISON");
    println!("{}", "=".repeat(80));
    println!();

    println!("│ Kernel          │ Throughput (M/s) │ Bandwidth (GB/s) │ vs Original │ vs Transposed │");
    println!("├─────────────────┼──────────────────┼──────────────────┼─────────────┼───────────────┤");
    println!(
        "│ Original        │ {:>16.2} │ {:>16.2} │      1.00x  │        1.00x  │",
        throughput_original / 1e6,
        bandwidth_original
    );
    println!(
        "│ Transposed      │ {:>16.2} │ {:>16.2} │    {:>6.2}x  │        1.00x  │",
        throughput_transposed / 1e6,
        bandwidth_transposed,
        throughput_transposed / throughput_original
    );
    println!(
        "│ Hybrid          │ {:>16.2} │ {:>16.2} │    {:>6.2}x  │      {:>6.2}x  │",
        throughput_hybrid / 1e6,
        bandwidth_hybrid,
        throughput_hybrid / throughput_original,
        throughput_hybrid / throughput_transposed
    );
    println!();

    // Success criteria check
    println!("{}", "=".repeat(80));
    println!("SUCCESS CRITERIA VALIDATION");
    println!("{}", "=".repeat(80));
    println!();

    let speedup = throughput_hybrid / throughput_original;
    let target_speedup = 1.5;

    println!("Target: {:>5.1}x speedup vs original", target_speedup);
    println!("Actual: {:>5.2}x speedup", speedup);

    if speedup >= target_speedup {
        println!("✓ SUCCESS: Hybrid approach achieves target performance!");
        println!();
        println!("Estimated coalescing improvement:");
        println!("  Memory bandwidth increased by {:.2}x", bandwidth_hybrid / bandwidth_original);
        println!("  This suggests coalescing efficiency improved from 7.69% to ~{:.1}%",
            7.69 * (bandwidth_hybrid / bandwidth_original));
    } else {
        println!("✗ MISS: Hybrid approach did not achieve target speedup");
        println!();
        println!("Possible reasons:");
        println!("  - CPU transpose overhead too high");
        println!("  - GPU writes still not fully coalesced");
        println!("  - Need to profile with Nsight Compute for details");
    }
    println!();

    // Detailed breakdown (estimate GPU vs CPU time)
    println!("{}", "=".repeat(80));
    println!("ESTIMATED TIME BREAKDOWN (Hybrid Kernel)");
    println!("{}", "=".repeat(80));
    println!();

    // Estimate: assume original kernel is mostly GPU time
    let estimated_gpu_time = duration_original.as_secs_f64();
    let estimated_transpose_time = duration_hybrid.as_secs_f64() - estimated_gpu_time;
    let transpose_overhead_pct = (estimated_transpose_time / duration_hybrid.as_secs_f64()) * 100.0;

    println!("Total time:      {:.2} ms", duration_hybrid.as_secs_f64() * 1000.0);
    println!("Estimated GPU:   {:.2} ms ({:.1}%)",
        estimated_gpu_time * 1000.0,
        (estimated_gpu_time / duration_hybrid.as_secs_f64()) * 100.0);
    println!("Estimated CPU:   {:.2} ms ({:.1}%)",
        estimated_transpose_time * 1000.0,
        transpose_overhead_pct);
    println!();

    if transpose_overhead_pct < 20.0 {
        println!("✓ Transpose overhead < 20% (excellent!)");
    } else if transpose_overhead_pct < 30.0 {
        println!("⚠ Transpose overhead 20-30% (acceptable)");
    } else {
        println!("✗ Transpose overhead > 30% (CPU bottleneck!)");
    }
    println!();

    println!("{}", "=".repeat(80));
    println!("NEXT STEPS");
    println!("{}", "=".repeat(80));
    println!();

    if speedup >= target_speedup {
        println!("1. Profile with Nsight Compute to verify coalescing metrics:");
        println!("   ncu --metrics \\");
        println!("     smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,\\");
        println!("     l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \\");
        println!("     ./target/release/examples/profile_hybrid");
        println!();
        println!("2. Create Phase 3 Session 4 summary document");
        println!("3. Update README with final performance numbers");
        println!("4. Consider this a successful optimization!");
    } else {
        println!("1. Profile both kernels with Nsight Compute");
        println!("2. Compare memory coalescing metrics");
        println!("3. If coalescing improved: optimize CPU transpose");
        println!("4. If coalescing didn't improve: debug kernel address calculation");
    }
    println!();

    Ok(())
}
