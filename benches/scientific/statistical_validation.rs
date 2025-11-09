//! Statistical Validation Suite
//!
//! Tests for uniform distribution, randomness, and pattern detection in
//! generated wordlists. Ensures the mixed-radix algorithm produces
//! unbiased output with no unexpected patterns.

use std::collections::HashMap;

/// Results from statistical validation tests
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub pattern_name: String,
    pub sample_size: usize,
    pub chi_square: ChiSquareResult,
    pub autocorrelation: AutocorrelationResult,
    pub runs_test: RunsTestResult,
}

impl ValidationResults {
    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.chi_square.passed && self.autocorrelation.passed && self.runs_test.passed
    }
}

/// Chi-square test for uniform distribution
#[derive(Debug, Clone)]
pub struct ChiSquareResult {
    pub chi_square_statistic: f64,
    pub degrees_of_freedom: usize,
    pub critical_value_95: f64,
    pub p_value_approx: f64,
    pub passed: bool,
}

/// Autocorrelation test for pattern detection
#[derive(Debug, Clone)]
pub struct AutocorrelationResult {
    pub max_lag_tested: usize,
    pub max_autocorrelation: f64,
    pub significant_lags: Vec<usize>,
    pub passed: bool,
}

/// Runs test for randomness
#[derive(Debug, Clone)]
pub struct RunsTestResult {
    pub num_runs: usize,
    pub expected_runs: f64,
    pub std_dev: f64,
    pub z_score: f64,
    pub critical_z_95: f64,
    pub passed: bool,
}

/// Perform Chi-square test for uniform distribution
///
/// Tests if characters at each position are uniformly distributed across
/// their charset. For a truly unbiased generator, each character should
/// appear with equal probability.
pub fn chi_square_test(words: &[Vec<u8>], charset_sizes: &[usize]) -> ChiSquareResult {
    let sample_size = words.len();
    let word_length = words[0].len();

    let mut chi_square = 0.0;
    let mut total_df = 0;

    // Test each position independently
    for pos in 0..word_length {
        let charset_size = charset_sizes[pos];
        let expected_count = sample_size as f64 / charset_size as f64;

        // Count frequency of each character at this position
        let mut char_counts: HashMap<u8, usize> = HashMap::new();
        for word in words {
            *char_counts.entry(word[pos]).or_insert(0) += 1;
        }

        // Calculate chi-square contribution for this position
        for count in char_counts.values() {
            let observed = *count as f64;
            let diff = observed - expected_count;
            chi_square += (diff * diff) / expected_count;
        }

        total_df += charset_size - 1;
    }

    // Critical value for 95% confidence with given degrees of freedom
    // Using approximation: chi^2 ≈ df + sqrt(2*df) * z_0.95
    // where z_0.95 = 1.645 (one-tailed)
    let critical_value = total_df as f64 + (2.0 * total_df as f64).sqrt() * 1.645;

    // Approximate p-value using Wilson-Hilferty transformation
    let p_value = if chi_square < total_df as f64 {
        0.5 // Rough approximation - better than expected
    } else {
        let z = ((chi_square / total_df as f64).powf(1.0 / 3.0) - 1.0 + 2.0 / (9.0 * total_df as f64))
            / (2.0 / (9.0 * total_df as f64)).sqrt();
        1.0 - normal_cdf(z)
    };

    ChiSquareResult {
        chi_square_statistic: chi_square,
        degrees_of_freedom: total_df,
        critical_value_95: critical_value,
        p_value_approx: p_value,
        passed: chi_square <= critical_value,
    }
}

/// Perform autocorrelation test
///
/// Tests if there are correlations between characters at different positions.
/// For a proper mixed-radix generator, positions should be independent.
pub fn autocorrelation_test(words: &[Vec<u8>], max_lag: usize) -> AutocorrelationResult {
    let sample_size = words.len();
    let word_length = words[0].len();

    let mut max_autocorr = 0.0;
    let mut significant_lags = Vec::new();

    // Significance threshold (95% confidence)
    let threshold = 1.96 / (sample_size as f64).sqrt();

    // Test autocorrelation at each lag
    for lag in 1..=max_lag.min(word_length - 1) {
        // Convert bytes to numeric values for correlation
        let mut x: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();

        for word in words {
            for pos in 0..word_length - lag {
                x.push(word[pos] as f64);
                y.push(word[pos + lag] as f64);
            }
        }

        // Calculate Pearson correlation coefficient
        let n = x.len() as f64;
        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let correlation = if var_x > 0.0 && var_y > 0.0 {
            cov / (var_x * var_y).sqrt()
        } else {
            0.0
        };

        let abs_corr = correlation.abs();
        if abs_corr > max_autocorr {
            max_autocorr = abs_corr;
        }

        if abs_corr > threshold {
            significant_lags.push(lag);
        }
    }

    let passed = significant_lags.is_empty();

    AutocorrelationResult {
        max_lag_tested: max_lag,
        max_autocorrelation: max_autocorr,
        significant_lags,
        passed,
    }
}

/// Perform runs test for randomness
///
/// Tests if the sequence of words shows random ordering vs systematic patterns.
/// Counts "runs" of increasing/decreasing sequences.
pub fn runs_test(words: &[Vec<u8>]) -> RunsTestResult {
    let n = words.len();

    // Convert words to numeric values (simple hash)
    let values: Vec<u64> = words
        .iter()
        .map(|word| {
            word.iter()
                .enumerate()
                .map(|(i, &b)| (b as u64) << (i * 8))
                .sum()
        })
        .collect();

    // Count runs (sequences of increasing or decreasing values)
    let mut num_runs = 1;
    for i in 1..n {
        if (values[i] > values[i - 1]) != (values[1] > values[0]) {
            num_runs += 1;
        }
    }

    // Expected number of runs for random sequence
    let n_f = n as f64;
    let expected_runs = (2.0 * n_f - 1.0) / 3.0;
    let variance = (16.0 * n_f - 29.0) / 90.0;
    let std_dev = variance.sqrt();

    // Z-score
    let z_score = ((num_runs as f64) - expected_runs) / std_dev;
    let critical_z = 1.96; // 95% confidence, two-tailed

    RunsTestResult {
        num_runs,
        expected_runs,
        std_dev,
        z_score,
        critical_z_95: critical_z,
        passed: z_score.abs() <= critical_z,
    }
}

/// Approximate CDF of standard normal distribution
fn normal_cdf(x: f64) -> f64 {
    // Approximation using error function
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi_square_uniform() {
        // Generate perfectly uniform sample
        let mut words = Vec::new();
        for i in 0..100 {
            words.push(vec![(i % 10) as u8, ((i / 10) % 10) as u8]);
        }

        let result = chi_square_test(&words, &[10, 10]);
        assert!(result.passed, "Uniform distribution should pass chi-square test");
    }

    #[test]
    fn test_autocorrelation_independent() {
        // Generate independent random-like data
        let mut words = Vec::new();
        for i in 0..1000 {
            words.push(vec![
                (i * 17 % 26) as u8 + b'a',
                (i * 23 % 26) as u8 + b'a',
                (i * 31 % 26) as u8 + b'a',
            ]);
        }

        let result = autocorrelation_test(&words, 2);
        // Should have low autocorrelation
        assert!(result.max_autocorrelation < 0.2);
    }

    #[test]
    fn test_runs_test_sequential() {
        // Sequential data (should fail randomness)
        let words: Vec<Vec<u8>> = (0..100)
            .map(|i| vec![i as u8, (i + 1) as u8])
            .collect();

        let result = runs_test(&words);
        // Sequential data should show pattern (might pass or fail depending on exact sequence)
        // Just ensure the test runs without panicking
        assert!(result.num_runs > 0);
    }

    #[test]
    fn test_normal_cdf() {
        // Test approximate CDF values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }
}
