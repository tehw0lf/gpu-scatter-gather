//! Statistical analysis utilities for benchmark results
//!
//! This module provides statistical analysis functions for multiple benchmark runs,
//! including mean, median, standard deviation, coefficient of variation, and
//! confidence intervals.

use serde::{Deserialize, Serialize};

/// Results from a single benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub pattern_name: String,
    pub kernel_time_ms: f64,
    pub total_time_ms: f64,
    pub words_generated: u64,
    pub throughput_words_per_sec: f64,
}

/// Statistical summary of multiple runs
#[derive(Debug, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub pattern_name: String,
    pub num_runs: usize,
    pub mean_throughput: f64,
    pub median_throughput: f64,
    pub std_dev: f64,
    pub coefficient_of_variation: f64,
    pub confidence_interval_95: (f64, f64),
    pub min_throughput: f64,
    pub max_throughput: f64,
    pub outliers: Vec<f64>,
}

impl StatisticalSummary {
    /// Compute statistics from multiple benchmark runs
    pub fn from_runs(runs: &[BenchmarkRun]) -> Self {
        assert!(!runs.is_empty(), "Need at least one run");

        let pattern_name = runs[0].pattern_name.clone();
        let throughputs: Vec<f64> = runs.iter()
            .map(|r| r.throughput_words_per_sec)
            .collect();

        // Calculate mean
        let mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

        // Calculate median
        let mut sorted = throughputs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Calculate standard deviation
        let variance = throughputs.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / throughputs.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate coefficient of variation
        let cv = std_dev / mean;

        // Calculate 95% confidence interval using t-distribution
        // t-value for 95% CI with different degrees of freedom
        let t_value = match throughputs.len() {
            1 => 12.706,      // df=0 (not really valid, but prevents panic)
            2 => 4.303,       // df=1
            3 => 3.182,       // df=2
            4 => 2.776,       // df=3
            5 => 2.571,       // df=4
            6 => 2.447,       // df=5
            7 => 2.365,       // df=6
            8 => 2.306,       // df=7
            9 => 2.262,       // df=8
            10 => 2.228,      // df=9
            _ => 1.96,        // df>30, approximate with normal distribution
        };
        let margin = t_value * (std_dev / (throughputs.len() as f64).sqrt());
        let ci = (mean - margin, mean + margin);

        // Outlier detection using IQR method
        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        let outliers: Vec<f64> = throughputs.iter()
            .filter(|&&x| x < lower_bound || x > upper_bound)
            .copied()
            .collect();

        StatisticalSummary {
            pattern_name,
            num_runs: throughputs.len(),
            mean_throughput: mean,
            median_throughput: median,
            std_dev,
            coefficient_of_variation: cv,
            confidence_interval_95: ci,
            min_throughput: *sorted.first().unwrap(),
            max_throughput: *sorted.last().unwrap(),
            outliers,
        }
    }

    /// Check if performance is stable (CV < 5%)
    pub fn is_stable(&self) -> bool {
        self.coefficient_of_variation < 0.05
    }
}

/// Format throughput in human-readable form
#[allow(dead_code)]
pub fn format_throughput(words_per_sec: f64) -> String {
    if words_per_sec >= 1e9 {
        format!("{:.2}B words/s", words_per_sec / 1e9)
    } else if words_per_sec >= 1e6 {
        format!("{:.2}M words/s", words_per_sec / 1e6)
    } else if words_per_sec >= 1e3 {
        format!("{:.2}K words/s", words_per_sec / 1e3)
    } else {
        format!("{:.2} words/s", words_per_sec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_calculation() {
        let runs = vec![
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 100.0,
            },
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 200.0,
            },
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 300.0,
            },
        ];

        let summary = StatisticalSummary::from_runs(&runs);
        assert_eq!(summary.mean_throughput, 200.0);
        assert_eq!(summary.median_throughput, 200.0);
        assert_eq!(summary.min_throughput, 100.0);
        assert_eq!(summary.max_throughput, 300.0);
    }

    #[test]
    fn test_median_even_count() {
        let runs = vec![
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 100.0,
            },
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 200.0,
            },
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 300.0,
            },
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 400.0,
            },
        ];

        let summary = StatisticalSummary::from_runs(&runs);
        assert_eq!(summary.median_throughput, 250.0); // Average of 200 and 300
    }

    #[test]
    fn test_std_dev_calculation() {
        let runs = vec![
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 100.0,
            },
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 100.0,
            },
        ];

        let summary = StatisticalSummary::from_runs(&runs);
        assert_eq!(summary.std_dev, 0.0);
        assert_eq!(summary.coefficient_of_variation, 0.0);
    }

    #[test]
    fn test_stability_check() {
        let runs_stable = vec![
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 1000.0,
            },
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 1010.0,
            },
        ];

        let summary = StatisticalSummary::from_runs(&runs_stable);
        assert!(summary.is_stable()); // CV should be < 5%

        let runs_unstable = vec![
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 1000.0,
            },
            BenchmarkRun {
                pattern_name: "test".to_string(),
                kernel_time_ms: 100.0,
                total_time_ms: 110.0,
                words_generated: 1000,
                throughput_words_per_sec: 2000.0,
            },
        ];

        let summary = StatisticalSummary::from_runs(&runs_unstable);
        assert!(!summary.is_stable()); // CV should be >= 5%
    }

    #[test]
    fn test_format_throughput() {
        assert_eq!(format_throughput(1.5e9), "1.50B words/s");
        assert_eq!(format_throughput(500e6), "500.00M words/s");
        assert_eq!(format_throughput(1.2e6), "1.20M words/s");
        assert_eq!(format_throughput(5000.0), "5.00K words/s");
        assert_eq!(format_throughput(50.0), "50.00 words/s");
    }
}
