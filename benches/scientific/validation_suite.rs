//! Statistical Validation Suite
//!
//! Validates that the GPU scatter-gather algorithm produces statistically
//! unbiased output with uniform distribution and no unexpected patterns.

mod statistical_validation;

use anyhow::Result;
use chrono::Local;
use gpu_scatter_gather::gpu::GpuContext;
use serde::{Deserialize, Serialize};
use statistical_validation::{
    autocorrelation_test, chi_square_test, runs_test, ValidationResults,
};
use std::collections::HashMap;

/// Test pattern configuration
#[derive(Debug, Clone)]
pub struct ValidationPattern {
    pub name: String,
    pub mask_description: String,
    pub charsets: HashMap<usize, Vec<u8>>,
    pub charset_sizes: Vec<usize>,
    pub mask: Vec<usize>,
    pub sample_size: usize,
}

impl ValidationPattern {
    /// Standard validation patterns
    pub fn standard_patterns() -> Vec<Self> {
        vec![
            // Pattern 1: Simple binary (2 chars, 2 positions)
            ValidationPattern {
                name: "binary_2pos".to_string(),
                mask_description: "?1?1 (binary, 2 positions)".to_string(),
                charsets: HashMap::from([(0, b"01".to_vec())]),
                charset_sizes: vec![2, 2],
                mask: vec![0, 0],
                sample_size: 100_000,
            },
            // Pattern 2: Lowercase (26 chars, 3 positions)
            ValidationPattern {
                name: "lowercase_3pos".to_string(),
                mask_description: "?l?l?l (lowercase, 3 positions)".to_string(),
                charsets: HashMap::from([(0, b"abcdefghijklmnopqrstuvwxyz".to_vec())]),
                charset_sizes: vec![26, 26, 26],
                mask: vec![0, 0, 0],
                sample_size: 100_000,
            },
            // Pattern 3: Decimal (10 chars, 4 positions)
            ValidationPattern {
                name: "decimal_4pos".to_string(),
                mask_description: "?d?d?d?d (digits, 4 positions)".to_string(),
                charsets: HashMap::from([(0, b"0123456789".to_vec())]),
                charset_sizes: vec![10, 10, 10, 10],
                mask: vec![0, 0, 0, 0],
                sample_size: 100_000,
            },
            // Pattern 4: Mixed charsets
            ValidationPattern {
                name: "mixed_charsets".to_string(),
                mask_description: "?u?l?d (upper+lower+digit)".to_string(),
                charsets: HashMap::from([
                    (0, b"ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_vec()),
                    (1, b"abcdefghijklmnopqrstuvwxyz".to_vec()),
                    (2, b"0123456789".to_vec()),
                ]),
                charset_sizes: vec![26, 26, 10],
                mask: vec![0, 1, 2],
                sample_size: 100_000,
            },
            // Pattern 5: Hex (16 chars, 4 positions)
            ValidationPattern {
                name: "hex_4pos".to_string(),
                mask_description: "?h?h?h?h (hexadecimal, 4 positions)".to_string(),
                charsets: HashMap::from([(0, b"0123456789abcdef".to_vec())]),
                charset_sizes: vec![16, 16, 16, 16],
                mask: vec![0, 0, 0, 0],
                sample_size: 100_000,
            },
        ]
    }
}

/// Serializable validation results
#[derive(Debug, Serialize, Deserialize)]
pub struct SerializableResults {
    pub pattern_name: String,
    pub sample_size: usize,
    pub chi_square_statistic: f64,
    pub chi_square_df: usize,
    pub chi_square_critical: f64,
    pub chi_square_p_value: f64,
    pub chi_square_passed: bool,
    pub autocorr_max: f64,
    pub autocorr_significant_lags: Vec<usize>,
    pub autocorr_passed: bool,
    pub runs_count: usize,
    pub runs_expected: f64,
    pub runs_z_score: f64,
    pub runs_passed: bool,
    pub all_tests_passed: bool,
}

impl From<ValidationResults> for SerializableResults {
    fn from(results: ValidationResults) -> Self {
        let all_passed = results.all_passed();
        SerializableResults {
            pattern_name: results.pattern_name,
            sample_size: results.sample_size,
            chi_square_statistic: results.chi_square.chi_square_statistic,
            chi_square_df: results.chi_square.degrees_of_freedom,
            chi_square_critical: results.chi_square.critical_value_95,
            chi_square_p_value: results.chi_square.p_value_approx,
            chi_square_passed: results.chi_square.passed,
            autocorr_max: results.autocorrelation.max_autocorrelation,
            autocorr_significant_lags: results.autocorrelation.significant_lags,
            autocorr_passed: results.autocorrelation.passed,
            runs_count: results.runs_test.num_runs,
            runs_expected: results.runs_test.expected_runs,
            runs_z_score: results.runs_test.z_score,
            runs_passed: results.runs_test.passed,
            all_tests_passed: all_passed,
        }
    }
}

/// Run validation for a single pattern
fn validate_pattern(gpu: &GpuContext, pattern: &ValidationPattern) -> Result<ValidationResults> {
    println!("\n=== Validating: {} ===", pattern.name);
    println!("Mask: {}", pattern.mask_description);
    println!("Sample size: {}", pattern.sample_size);

    // Generate sample words
    println!("Generating sample...");
    let output = gpu.generate_batch(
        &pattern.charsets,
        &pattern.mask,
        0,
        pattern.sample_size as u64,
    )?;

    // Parse output into words
    let word_length = pattern.mask.len();
    let stride = word_length + 1; // +1 for newline
    let words: Vec<Vec<u8>> = output
        .chunks(stride)
        .take(pattern.sample_size)
        .map(|chunk| chunk[..word_length].to_vec())
        .collect();

    println!("Running statistical tests...");

    // Chi-square test
    println!("  Chi-square test for uniform distribution...");
    let chi_square = chi_square_test(&words, &pattern.charset_sizes);
    println!(
        "    χ² = {:.2}, df = {}, critical = {:.2}, p = {:.4}",
        chi_square.chi_square_statistic,
        chi_square.degrees_of_freedom,
        chi_square.critical_value_95,
        chi_square.p_value_approx
    );
    println!("    Result: {}", if chi_square.passed { "✅ PASS" } else { "❌ FAIL" });

    // Autocorrelation test
    println!("  Autocorrelation test for independence...");
    let autocorr = autocorrelation_test(&words, word_length.min(10));
    println!(
        "    Max autocorrelation = {:.4}",
        autocorr.max_autocorrelation
    );
    if !autocorr.significant_lags.is_empty() {
        println!(
            "    Significant lags: {:?}",
            autocorr.significant_lags
        );
    }
    println!("    Result: {}", if autocorr.passed { "✅ PASS" } else { "❌ FAIL" });

    // Runs test
    println!("  Runs test for randomness...");
    let runs = runs_test(&words);
    println!(
        "    Runs = {}, expected = {:.2}, z-score = {:.3}",
        runs.num_runs, runs.expected_runs, runs.z_score
    );
    println!("    Result: {}", if runs.passed { "✅ PASS" } else { "❌ FAIL" });

    Ok(ValidationResults {
        pattern_name: pattern.name.clone(),
        sample_size: pattern.sample_size,
        chi_square,
        autocorrelation: autocorr,
        runs_test: runs,
    })
}

/// Run complete validation suite
fn run_validation_suite() -> Result<Vec<ValidationResults>> {
    let gpu = GpuContext::new()?;
    let device_name = gpu.device_name()?;
    let (major, minor) = gpu.compute_capability();

    println!("=== GPU Scatter-Gather Statistical Validation Suite ===");
    println!("GPU: {}", device_name);
    println!("Compute Capability: {}.{}", major, minor);
    println!();

    let patterns = ValidationPattern::standard_patterns();
    let mut results = Vec::new();

    for pattern in &patterns {
        match validate_pattern(&gpu, pattern) {
            Ok(result) => {
                results.push(result);
            }
            Err(e) => {
                eprintln!("Error validating pattern {}: {}", pattern.name, e);
            }
        }
    }

    Ok(results)
}

/// Generate markdown report
fn generate_report(
    results: &[ValidationResults],
    filename: &str,
    gpu_info: &str,
) -> Result<()> {
    let mut report = String::new();

    report.push_str("# Statistical Validation Results\n\n");
    report.push_str(&format!(
        "**Date:** {}\n",
        Local::now().format("%Y-%m-%d %H:%M:%S")
    ));
    report.push_str(&format!("**GPU:** {}\n", gpu_info));
    report.push_str("\n---\n\n");

    report.push_str("## Executive Summary\n\n");

    let total_tests = results.len();
    let passed_tests = results.iter().filter(|r| r.all_passed()).count();

    report.push_str(&format!(
        "**Overall Result:** {}/{} patterns passed all tests\n\n",
        passed_tests, total_tests
    ));

    if passed_tests == total_tests {
        report.push_str("✅ **All patterns passed statistical validation!**\n\n");
        report.push_str("The GPU scatter-gather algorithm produces statistically unbiased output with:\n");
        report.push_str("- Uniform distribution of characters (Chi-square test)\n");
        report.push_str("- No significant autocorrelation (independence test)\n");
        report.push_str("- Random sequence ordering (runs test)\n\n");
    } else {
        report.push_str("⚠️ **Some patterns failed validation**\n\n");
    }

    report.push_str("## Summary Table\n\n");
    report.push_str("| Pattern | Sample Size | Chi-square | Autocorrelation | Runs Test | Overall |\n");
    report.push_str("|---------|-------------|------------|-----------------|-----------|----------|\n");

    for result in results {
        report.push_str(&format!(
            "| {} | {:>10} | {} | {} | {} | {} |\n",
            result.pattern_name,
            result.sample_size,
            if result.chi_square.passed { "✅" } else { "❌" },
            if result.autocorrelation.passed { "✅" } else { "❌" },
            if result.runs_test.passed { "✅" } else { "❌" },
            if result.all_passed() { "✅" } else { "❌" }
        ));
    }

    report.push_str("\n## Detailed Results\n\n");

    for result in results {
        report.push_str(&format!("### {}\n\n", result.pattern_name));
        report.push_str(&format!("**Sample Size:** {} words\n\n", result.sample_size));

        report.push_str("#### Chi-square Test (Uniform Distribution)\n\n");
        report.push_str(&format!(
            "- **χ² statistic:** {:.2}\n",
            result.chi_square.chi_square_statistic
        ));
        report.push_str(&format!(
            "- **Degrees of freedom:** {}\n",
            result.chi_square.degrees_of_freedom
        ));
        report.push_str(&format!(
            "- **Critical value (95%):** {:.2}\n",
            result.chi_square.critical_value_95
        ));
        report.push_str(&format!(
            "- **p-value (approx):** {:.4}\n",
            result.chi_square.p_value_approx
        ));
        report.push_str(&format!(
            "- **Result:** {}\n\n",
            if result.chi_square.passed { "✅ PASS" } else { "❌ FAIL" }
        ));

        report.push_str("#### Autocorrelation Test (Independence)\n\n");
        report.push_str(&format!(
            "- **Max lag tested:** {}\n",
            result.autocorrelation.max_lag_tested
        ));
        report.push_str(&format!(
            "- **Max autocorrelation:** {:.4}\n",
            result.autocorrelation.max_autocorrelation
        ));
        if !result.autocorrelation.significant_lags.is_empty() {
            report.push_str(&format!(
                "- **Significant lags:** {:?}\n",
                result.autocorrelation.significant_lags
            ));
        } else {
            report.push_str("- **Significant lags:** None\n");
        }
        report.push_str(&format!(
            "- **Result:** {}\n\n",
            if result.autocorrelation.passed { "✅ PASS" } else { "❌ FAIL" }
        ));

        report.push_str("#### Runs Test (Randomness)\n\n");
        report.push_str(&format!(
            "- **Number of runs:** {}\n",
            result.runs_test.num_runs
        ));
        report.push_str(&format!(
            "- **Expected runs:** {:.2}\n",
            result.runs_test.expected_runs
        ));
        report.push_str(&format!(
            "- **Standard deviation:** {:.2}\n",
            result.runs_test.std_dev
        ));
        report.push_str(&format!(
            "- **Z-score:** {:.3}\n",
            result.runs_test.z_score
        ));
        report.push_str(&format!(
            "- **Critical z (95%):** {:.2}\n",
            result.runs_test.critical_z_95
        ));
        report.push_str(&format!(
            "- **Result:** {}\n\n",
            if result.runs_test.passed { "✅ PASS" } else { "❌ FAIL" }
        ));
    }

    report.push_str("## Interpretation\n\n");
    report.push_str("### Chi-square Test\n");
    report.push_str("Tests if characters at each position follow a uniform distribution. ");
    report.push_str("A passing result means each character in the charset appears with equal probability, ");
    report.push_str("indicating no bias in the mixed-radix algorithm.\n\n");

    report.push_str("### Autocorrelation Test\n");
    report.push_str("Tests if there are correlations between character positions. ");
    report.push_str("A passing result means positions are independent, with no patterns ");
    report.push_str("like \"if position i is 'a', position i+1 tends to be 'b'\".\n\n");

    report.push_str("### Runs Test\n");
    report.push_str("Tests if the sequence of generated words is random. ");
    report.push_str("A passing result means words appear in a random order, not systematically ");
    report.push_str("(e.g., not always increasing or always decreasing).\n\n");

    std::fs::write(filename, report)?;
    println!("\n✅ Report saved to: {}", filename);
    Ok(())
}

fn main() -> Result<()> {
    println!("=== GPU Scatter-Gather Statistical Validation Suite ===\n");

    // Get GPU info
    let gpu = GpuContext::new()?;
    let device_name = gpu.device_name()?;
    let (major, minor) = gpu.compute_capability();
    let gpu_info = format!("{} (Compute Capability {}.{})", device_name, major, minor);
    drop(gpu);

    // Run validation suite
    let results = run_validation_suite()?;

    // Save results
    let timestamp = Local::now().format("%Y-%m-%d").to_string();
    let results_dir = "benches/scientific/results";
    std::fs::create_dir_all(results_dir)?;

    // Convert to serializable format
    let serializable: Vec<SerializableResults> = results
        .iter()
        .map(|r| SerializableResults::from(r.clone()))
        .collect();

    let json = serde_json::to_string_pretty(&serializable)?;
    std::fs::write(
        format!("{}/validation_{}.json", results_dir, timestamp),
        json,
    )?;

    // Generate markdown report
    generate_report(
        &results,
        &format!("docs/STATISTICAL_VALIDATION_{}.md", timestamp),
        &gpu_info,
    )?;

    // Print summary
    println!("\n=== Summary ===");
    let total = results.len();
    let passed = results.iter().filter(|r| r.all_passed()).count();
    println!("Total patterns tested: {}", total);
    println!("Patterns passed: {}", passed);
    println!("Patterns failed: {}", total - passed);

    if passed == total {
        println!("\n✅ All validation tests PASSED!");
    } else {
        println!("\n⚠️ Some validation tests FAILED");
    }

    Ok(())
}
