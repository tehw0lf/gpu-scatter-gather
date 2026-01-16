//! Baseline Benchmark Suite for GPU Scatter-Gather
//!
//! This establishes scientific baseline metrics BEFORE Phase 3 optimizations.
//! Implements the methodology from docs/BASELINE_BENCHMARKING_PLAN.md

mod statistical_analysis;

use anyhow::Result;
use chrono::Local;
use cuda_driver_sys::*;
use gpu_scatter_gather::gpu::GpuContext;
use serde::{Deserialize, Serialize};
use statistical_analysis::{BenchmarkRun, StatisticalSummary};
use std::collections::HashMap;
use std::ptr;

/// Benchmark configuration for a test pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkPattern {
    pub name: String,
    pub mask_description: String,
    pub charsets: HashMap<usize, String>,
    pub mask: Vec<usize>,
    pub total_keyspace: u64,
    pub max_words_to_generate: Option<u64>,
}

impl BenchmarkPattern {
    /// Standard benchmark patterns for consistent testing
    pub fn standard_patterns() -> Vec<Self> {
        vec![
            // Pattern 1: Small (baseline)
            BenchmarkPattern {
                name: "small_4char_lowercase".to_string(),
                mask_description: "?l?l?l?l".to_string(),
                charsets: HashMap::from([(0, "abcdefghijklmnopqrstuvwxyz".to_string())]),
                mask: vec![0, 0, 0, 0],
                total_keyspace: 26_u64.pow(4), // 456,976
                max_words_to_generate: None,
            },
            // Pattern 2: Medium-Small
            BenchmarkPattern {
                name: "medium_6char_lowercase".to_string(),
                mask_description: "?l?l?l?l?l?l".to_string(),
                charsets: HashMap::from([(0, "abcdefghijklmnopqrstuvwxyz".to_string())]),
                mask: vec![0, 0, 0, 0, 0, 0],
                total_keyspace: 308_915_776, // 26^6
                max_words_to_generate: None,
            },
            // Pattern 3: Medium-Large (limited to 1B words)
            BenchmarkPattern {
                name: "large_8char_lowercase_limited".to_string(),
                mask_description: "?l?l?l?l?l?l?l?l".to_string(),
                charsets: HashMap::from([(0, "abcdefghijklmnopqrstuvwxyz".to_string())]),
                mask: vec![0, 0, 0, 0, 0, 0, 0, 0],
                total_keyspace: 208_827_064_576, // 26^8
                max_words_to_generate: Some(1_000_000_000), // Only generate 1B words
            },
            // Pattern 4: Mixed Charsets
            BenchmarkPattern {
                name: "mixed_upper_lower_digits".to_string(),
                mask_description: "?u?l?d?d?d?d?d?d".to_string(),
                charsets: HashMap::from([
                    (0, "ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_string()), // uppercase
                    (1, "abcdefghijklmnopqrstuvwxyz".to_string()), // lowercase
                    (2, "0123456789".to_string()),                  // digits
                ]),
                mask: vec![0, 1, 2, 2, 2, 2, 2, 2],
                total_keyspace: 676_000_000, // 26 * 26 * 10^6
                max_words_to_generate: None,
            },
            // Pattern 5: Special Characters
            BenchmarkPattern {
                name: "special_chars".to_string(),
                mask_description: "?l?l?s?s?s?s".to_string(),
                charsets: HashMap::from([
                    (0, "abcdefghijklmnopqrstuvwxyz".to_string()),     // lowercase
                    (1, " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".to_string()), // special (33 chars)
                ]),
                mask: vec![0, 0, 1, 1, 1, 1],
                total_keyspace: 778_377_624, // 26 * 26 * 33^4
                max_words_to_generate: None,
            },
        ]
    }
}

/// Run single benchmark iteration
unsafe fn run_single_benchmark(
    gpu: &mut GpuContext,
    pattern: &BenchmarkPattern,
) -> Result<BenchmarkRun> {
    // Prepare charsets as HashMap<usize, Vec<u8>>
    let charsets: HashMap<usize, Vec<u8>> = pattern
        .charsets
        .iter()
        .map(|(&id, s)| (id, s.as_bytes().to_vec()))
        .collect();

    // Determine how many words to generate
    let words_to_generate = pattern
        .max_words_to_generate
        .unwrap_or(pattern.total_keyspace);

    // Create CUDA events for accurate timing
    let mut start_event = ptr::null_mut();
    let mut end_event = ptr::null_mut();
    check_cuda(cuEventCreate(&mut start_event, 0))?;
    check_cuda(cuEventCreate(&mut end_event, 0))?;

    // Warm GPU (single small batch)
    let _ = gpu.generate_batch(&charsets, &pattern.mask, 0, 1000, 0)?;  // format=0 (newlines)

    // Record start time
    check_cuda(cuEventRecord(start_event, ptr::null_mut()))?;

    // Generate words in batches
    let batch_size = 10_000_000u64; // 10M words per batch
    let mut total_words = 0u64;
    let mut current_index = 0u64;

    while total_words < words_to_generate {
        let remaining = words_to_generate - total_words;
        let batch = batch_size.min(remaining);

        let _output = gpu.generate_batch(&charsets, &pattern.mask, current_index, batch, 0)?;  // format=0 (newlines)

        total_words += batch;
        current_index += batch;
    }

    // Record end time
    check_cuda(cuEventRecord(end_event, ptr::null_mut()))?;
    check_cuda(cuEventSynchronize(end_event))?;

    // Get elapsed time
    let mut elapsed_ms = 0.0f32;
    check_cuda(cuEventElapsedTime(&mut elapsed_ms, start_event, end_event))?;

    check_cuda(cuEventDestroy_v2(start_event))?;
    check_cuda(cuEventDestroy_v2(end_event))?;

    let throughput = (total_words as f64 / elapsed_ms as f64) * 1000.0;

    Ok(BenchmarkRun {
        pattern_name: pattern.name.clone(),
        kernel_time_ms: elapsed_ms as f64,
        total_time_ms: elapsed_ms as f64,
        words_generated: total_words,
        throughput_words_per_sec: throughput,
    })
}

/// Run complete benchmark suite with statistical analysis
pub fn run_baseline_suite() -> Result<HashMap<String, StatisticalSummary>> {
    // Initialize GPU
    let mut gpu = GpuContext::new()?;
    let device_name = gpu.device_name()?;
    let (major, minor) = gpu.compute_capability();

    println!("\n=== GPU Scatter-Gather Baseline Benchmarks ===");
    println!("GPU: {}", device_name);
    println!("Compute Capability: {}.{}", major, minor);
    println!();

    let patterns = BenchmarkPattern::standard_patterns();
    let mut results = HashMap::new();

    for pattern in &patterns {
        println!("\n=== Benchmarking: {} ===", pattern.name);
        println!("Mask: {}", pattern.mask_description);
        println!("Keyspace: {}", pattern.total_keyspace);
        if let Some(limit) = pattern.max_words_to_generate {
            println!("Generating: {} words (limited)", limit);
        }

        // Warm-up runs
        println!("\nWarm-up runs (3x)...");
        for i in 1..=3 {
            let run = unsafe { run_single_benchmark(&mut gpu, pattern)? };
            println!(
                "  Warm-up {}: {:.2}M words/s",
                i,
                run.throughput_words_per_sec / 1e6
            );
        }

        // Measurement runs
        println!("\nMeasurement runs (10x)...");
        let mut runs = Vec::new();
        for i in 1..=10 {
            let run = unsafe { run_single_benchmark(&mut gpu, pattern)? };
            println!(
                "  Run {}: {:.2}M words/s",
                i,
                run.throughput_words_per_sec / 1e6
            );
            runs.push(run);
        }

        // Compute statistics
        let summary = StatisticalSummary::from_runs(&runs);

        println!("\n--- Statistics ---");
        println!(
            "Mean:     {:.2}M words/s",
            summary.mean_throughput / 1e6
        );
        println!(
            "Median:   {:.2}M words/s",
            summary.median_throughput / 1e6
        );
        println!("Std Dev:  {:.2}M words/s", summary.std_dev / 1e6);
        println!("CV:       {:.2}%", summary.coefficient_of_variation * 100.0);
        println!(
            "95% CI:   [{:.2}M, {:.2}M] words/s",
            summary.confidence_interval_95.0 / 1e6,
            summary.confidence_interval_95.1 / 1e6
        );
        println!(
            "Range:    [{:.2}M, {:.2}M] words/s",
            summary.min_throughput / 1e6,
            summary.max_throughput / 1e6
        );

        if !summary.outliers.is_empty() {
            println!("Outliers: {} detected", summary.outliers.len());
        }

        if summary.is_stable() {
            println!("✅ Performance is STABLE (CV < 5%)");
        } else {
            println!("⚠️  Performance is UNSTABLE (CV >= 5%)");
        }

        results.insert(pattern.name.clone(), summary);
    }

    Ok(results)
}

/// Save results to JSON file
pub fn save_results(
    results: &HashMap<String, StatisticalSummary>,
    filename: &str,
) -> Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(filename, json)?;
    println!("\n✅ Results saved to: {}", filename);
    Ok(())
}

/// Generate markdown report
pub fn generate_report(
    results: &HashMap<String, StatisticalSummary>,
    filename: &str,
    gpu_info: &str,
) -> Result<()> {
    let mut report = String::new();

    report.push_str("# Baseline Benchmark Results\n\n");
    report.push_str(&format!(
        "**Date:** {}\n",
        Local::now().format("%Y-%m-%d %H:%M:%S")
    ));
    report.push_str(&format!("**GPU:** {}\n", gpu_info));
    report.push_str("\n---\n\n");

    report.push_str("## Summary Table\n\n");
    report.push_str("| Pattern | Mean Throughput | Median | Std Dev | CV | 95% CI |\n");
    report.push_str("|---------|-----------------|--------|---------|----|---------|\n");

    for (name, summary) in results {
        report.push_str(&format!(
            "| {} | {:.2}M words/s | {:.2}M | {:.2}M | {:.2}% | [{:.2}M, {:.2}M] |\n",
            name,
            summary.mean_throughput / 1e6,
            summary.median_throughput / 1e6,
            summary.std_dev / 1e6,
            summary.coefficient_of_variation * 100.0,
            summary.confidence_interval_95.0 / 1e6,
            summary.confidence_interval_95.1 / 1e6,
        ));
    }

    report.push_str("\n## Detailed Results\n\n");

    for (name, summary) in results {
        report.push_str(&format!("### {}\n\n", name));
        report.push_str(&format!(
            "- **Mean Throughput:** {:.2}M words/s\n",
            summary.mean_throughput / 1e6
        ));
        report.push_str(&format!(
            "- **Median Throughput:** {:.2}M words/s\n",
            summary.median_throughput / 1e6
        ));
        report.push_str(&format!(
            "- **Standard Deviation:** {:.2}M words/s\n",
            summary.std_dev / 1e6
        ));
        report.push_str(&format!(
            "- **Coefficient of Variation:** {:.2}%\n",
            summary.coefficient_of_variation * 100.0
        ));
        report.push_str(&format!(
            "- **95% Confidence Interval:** [{:.2}M, {:.2}M] words/s\n",
            summary.confidence_interval_95.0 / 1e6,
            summary.confidence_interval_95.1 / 1e6
        ));
        report.push_str(&format!(
            "- **Range:** [{:.2}M, {:.2}M] words/s\n",
            summary.min_throughput / 1e6,
            summary.max_throughput / 1e6
        ));
        report.push_str(&format!(
            "- **Stability:** {}\n",
            if summary.is_stable() {
                "✅ STABLE"
            } else {
                "⚠️ UNSTABLE"
            }
        ));

        if !summary.outliers.is_empty() {
            report.push_str(&format!(
                "- **Outliers:** {} detected\n",
                summary.outliers.len()
            ));
        }

        report.push_str("\n");
    }

    std::fs::write(filename, report)?;
    println!("✅ Report saved to: {}", filename);
    Ok(())
}

/// Helper function to check CUDA errors
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

fn main() -> Result<()> {
    println!("=== GPU Scatter-Gather Baseline Benchmarks ===\n");

    // Get GPU info first
    let gpu = GpuContext::new()?;
    let device_name = gpu.device_name()?;
    let (major, minor) = gpu.compute_capability();
    let gpu_info = format!("{} (Compute Capability {}.{})", device_name, major, minor);
    drop(gpu); // Drop before running suite

    // Run benchmark suite
    let results = run_baseline_suite()?;

    // Save results
    let timestamp = Local::now().format("%Y-%m-%d").to_string();
    let results_dir = "benches/scientific/results";
    std::fs::create_dir_all(results_dir)?;

    save_results(
        &results,
        &format!("{}/baseline_{}.json", results_dir, timestamp),
    )?;
    generate_report(
        &results,
        &format!("{}/baseline_report_{}.md", results_dir, timestamp),
        &gpu_info,
    )?;

    println!("\n✅ Baseline benchmarking complete!");
    Ok(())
}
