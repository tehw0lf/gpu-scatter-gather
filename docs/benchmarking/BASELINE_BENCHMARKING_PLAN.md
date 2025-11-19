# Baseline Benchmarking Plan - Pre-Phase 3

**Date:** October 17, 2025
**Project:** GPU Scatter-Gather Wordlist Generator
**Purpose:** Establish scientific baseline metrics BEFORE Phase 3 optimization work
**Effort:** 1-3 days
**Status:** Ready to implement

## Executive Summary

**Goal:** Create minimal but scientifically rigorous baseline benchmarks to measure current performance before Phase 3 optimizations begin.

**Why Now:**
- Phase 3 is all about optimization (kernel improvements, memory transfers, multi-GPU)
- Without baseline metrics, we can't prove optimizations work
- Can't detect regressions or measure ROI of optimization efforts
- Industry best practice: "Measure first, optimize second"

**What We'll Build:**
- Automated benchmark script with statistical analysis
- Baseline measurements for standard test patterns
- JSON results storage for historical comparison
- Simple report generation
- Foundation for future full scientific framework (Enhancement 2)

---

## Current Performance Baseline (Informal)

From Phase 2 development:
- **Measured throughput:** 635M-1.2B words/s (RTX 4070 Ti SUPER)
- **vs maskprocessor:** 4.5-8.7x faster (maskprocessor ~142M words/s)
- **vs cracken:** 3.8-7.1x faster (cracken ~168M words/s)

**Problem:** These are single-run measurements, not statistically rigorous.

**Need:** Multiple runs with mean, std dev, confidence intervals.

---

## Benchmark Architecture

### Design Principles

1. **Minimal but Complete:** Don't build full Enhancement 2 yet, just what's needed
2. **Automated:** One command runs entire baseline suite
3. **Statistical:** Multiple runs with proper analysis
4. **Reproducible:** Document environment, store raw data
5. **Extensible:** Easy to expand to full Enhancement 2 later

### Components

```
benches/
├── scientific/
│   ├── baseline_benchmark.rs          # Main benchmark implementation
│   ├── statistical_analysis.rs        # Stats helper functions
│   ├── results/
│   │   ├── baseline_2025-10-17.json   # Raw benchmark data
│   │   └── baseline_report_2025-10-17.md  # Human-readable report
│   └── README.md                      # How to run benchmarks

scripts/
├── run_baseline_benchmark.sh          # Automated benchmark runner
└── compare_benchmarks.sh              # Compare before/after results
```

---

## Standard Test Patterns

Define **5 standard patterns** to benchmark consistently:

### Pattern 1: Small (Baseline)
- **Mask:** `?l?l?l?l` (lowercase, 4 chars)
- **Keyspace:** 456,976 words
- **Purpose:** Small keyspace, measures kernel launch overhead
- **Expected:** ~0.001s (very fast)

### Pattern 2: Medium-Small
- **Mask:** `?l?l?l?l?l?l` (lowercase, 6 chars)
- **Keyspace:** 308,915,776 words (~309M)
- **Purpose:** Medium keyspace, measures sustained throughput
- **Expected:** ~0.3-0.5s

### Pattern 3: Medium-Large
- **Mask:** `?l?l?l?l?l?l?l?l` (lowercase, 8 chars)
- **Keyspace:** 208,827,064,576 words (~209B)
- **Purpose:** Large keyspace, full GPU saturation
- **Expected:** ~200-300s (partial generation for testing)
- **Note:** Generate first 1B words only (avoid excessive runtime)

### Pattern 4: Mixed Charsets
- **Mask:** `?u?l?d?d?d?d?d?d` (upper + lower + 6 digits)
- **Charset sizes:** 26, 26, 10, 10, 10, 10, 10, 10
- **Keyspace:** 676,000,000 words (~676M)
- **Purpose:** Tests variable charset sizes
- **Expected:** ~0.6-1.0s

### Pattern 5: Special Characters
- **Mask:** `?l?l?s?s?s?s` (2 lowercase + 4 special)
- **Charset sizes:** 26, 26, 33, 33, 33, 33
- **Keyspace:** ~778M words
- **Purpose:** Tests complex charset (special chars)
- **Expected:** ~0.7-1.2s

---

## Statistical Analysis Requirements

For each pattern, measure:

### Execution Metrics
- **Kernel execution time** (GPU-only, CUDA events)
- **Total wall-clock time** (including transfers)
- **Throughput** (words/second)
- **Memory bandwidth utilization** (GB/s)

### Statistical Analysis (per pattern)
- **Warm-up runs:** 3 runs (discarded, eliminate cold-start)
- **Measurement runs:** 10 runs (for statistical analysis)
- **Metrics:**
  - Mean throughput (words/s)
  - Median throughput (words/s)
  - Standard deviation (σ)
  - Coefficient of variation (CV = σ/μ)
  - 95% confidence interval (CI)
  - Min/max throughput
  - Outlier detection (IQR method)

### Acceptance Criteria
- **CV < 5%** (low variance, stable performance)
- **Outliers documented** (explain any anomalies)
- **Reproducible** (can re-run and get similar results)

---

## Implementation Plan

### Step 1: Create Statistical Analysis Module

**File:** `benches/scientific/statistical_analysis.rs`

**Purpose:** Reusable stats functions

```rust
use std::collections::HashMap;

/// Results from a single benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkRun {
    pub pattern_name: String,
    pub kernel_time_ms: f64,
    pub total_time_ms: f64,
    pub words_generated: u64,
    pub throughput_words_per_sec: f64,
}

/// Statistical summary of multiple runs
#[derive(Debug)]
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

        let mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

        let mut sorted = throughputs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let variance = throughputs.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / throughputs.len() as f64;
        let std_dev = variance.sqrt();

        let cv = std_dev / mean;

        // 95% CI using t-distribution (approximation for n > 10)
        let t_value = 2.262; // t-value for 95% CI, df=9 (10 runs)
        let margin = t_value * (std_dev / (throughputs.len() as f64).sqrt());
        let ci = (mean - margin, mean + margin);

        // Outlier detection (IQR method)
        let q1 = sorted[sorted.len() / 4];
        let q3 = sorted[3 * sorted.len() / 4];
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
```

**Tasks:**
- [ ] Create `benches/scientific/statistical_analysis.rs`
- [ ] Implement `BenchmarkRun` struct
- [ ] Implement `StatisticalSummary` with all metrics
- [ ] Add helper functions (format_throughput, etc.)
- [ ] Write unit tests for statistical calculations

---

### Step 2: Create Baseline Benchmark Suite

**File:** `benches/scientific/baseline_benchmark.rs`

**Purpose:** Main benchmark implementation

```rust
use criterion::{black_box, Criterion};
use gpu_scatter_gather::{WordlistGenerator, Charset, Mask};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

mod statistical_analysis;
use statistical_analysis::{BenchmarkRun, StatisticalSummary};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkPattern {
    pub name: String,
    pub mask: String,
    pub charsets: HashMap<usize, String>,
    pub total_keyspace: u64,
    pub max_words_to_generate: Option<u64>, // Limit for large keyspaces
}

impl BenchmarkPattern {
    /// Standard benchmark patterns
    pub fn standard_patterns() -> Vec<Self> {
        vec![
            // Pattern 1: Small
            BenchmarkPattern {
                name: "small_4char_lowercase".to_string(),
                mask: "?l?l?l?l".to_string(),
                charsets: HashMap::from([(1, "abcdefghijklmnopqrstuvwxyz".to_string())]),
                total_keyspace: 456_976,
                max_words_to_generate: None,
            },

            // Pattern 2: Medium-Small
            BenchmarkPattern {
                name: "medium_6char_lowercase".to_string(),
                mask: "?l?l?l?l?l?l".to_string(),
                charsets: HashMap::from([(1, "abcdefghijklmnopqrstuvwxyz".to_string())]),
                total_keyspace: 308_915_776,
                max_words_to_generate: None,
            },

            // Pattern 3: Medium-Large (limited to 1B words)
            BenchmarkPattern {
                name: "large_8char_lowercase_limited".to_string(),
                mask: "?l?l?l?l?l?l?l?l".to_string(),
                charsets: HashMap::from([(1, "abcdefghijklmnopqrstuvwxyz".to_string())]),
                total_keyspace: 208_827_064_576,
                max_words_to_generate: Some(1_000_000_000), // Only generate 1B words
            },

            // Pattern 4: Mixed Charsets
            BenchmarkPattern {
                name: "mixed_upper_lower_digits".to_string(),
                mask: "?u?l?d?d?d?d?d?d".to_string(),
                charsets: HashMap::from([
                    (1, "ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_string()),
                    (2, "abcdefghijklmnopqrstuvwxyz".to_string()),
                    (3, "0123456789".to_string()),
                ]),
                total_keyspace: 676_000_000,
                max_words_to_generate: None,
            },

            // Pattern 5: Special Characters
            BenchmarkPattern {
                name: "special_chars".to_string(),
                mask: "?l?l?s?s?s?s".to_string(),
                charsets: HashMap::from([
                    (1, "abcdefghijklmnopqrstuvwxyz".to_string()),
                    (2, " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".to_string()),
                ]),
                total_keyspace: 778_377_000,
                max_words_to_generate: None,
            },
        ]
    }
}

/// Run single benchmark iteration
fn run_single_benchmark(pattern: &BenchmarkPattern) -> BenchmarkRun {
    // Build generator
    let mask = Mask::parse(&pattern.mask).expect("Invalid mask");
    let mut builder = WordlistGenerator::builder();
    for (id, charset_str) in &pattern.charsets {
        builder = builder.charset(*id, Charset::from(charset_str.as_str()));
    }
    let mut generator = builder.mask(mask.pattern()).build().expect("Failed to build generator");

    // Determine how many words to generate
    let words_to_generate = pattern.max_words_to_generate
        .unwrap_or(pattern.total_keyspace);

    // Warm GPU (single small batch)
    let _ = generator.generate_batch(0, 1000);

    // Benchmark: generate words to /dev/null (measure pure generation speed)
    let start = Instant::now();

    let batch_size = 10_000_000; // 10M words per batch
    let mut total_words = 0u64;
    let mut current_index = 0u64;

    while total_words < words_to_generate {
        let remaining = words_to_generate - total_words;
        let batch = batch_size.min(remaining);

        let output = generator.generate_batch(current_index, batch);
        black_box(&output); // Prevent optimization

        total_words += batch;
        current_index += batch;
    }

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput = total_words as f64 / elapsed.as_secs_f64();

    BenchmarkRun {
        pattern_name: pattern.name.clone(),
        kernel_time_ms: elapsed_ms, // Approximate (would need CUDA events for exact)
        total_time_ms: elapsed_ms,
        words_generated: total_words,
        throughput_words_per_sec: throughput,
    }
}

/// Run complete benchmark suite with statistical analysis
pub fn run_baseline_suite() -> HashMap<String, StatisticalSummary> {
    let patterns = BenchmarkPattern::standard_patterns();
    let mut results = HashMap::new();

    for pattern in &patterns {
        println!("\n=== Benchmarking: {} ===", pattern.name);
        println!("Mask: {}", pattern.mask);
        println!("Keyspace: {}", pattern.total_keyspace);
        if let Some(limit) = pattern.max_words_to_generate {
            println!("Generating: {} words (limited)", limit);
        }

        // Warm-up runs
        println!("\nWarm-up runs (3x)...");
        for i in 1..=3 {
            let run = run_single_benchmark(pattern);
            println!("  Warm-up {}: {:.2}M words/s",
                i, run.throughput_words_per_sec / 1e6);
        }

        // Measurement runs
        println!("\nMeasurement runs (10x)...");
        let mut runs = Vec::new();
        for i in 1..=10 {
            let run = run_single_benchmark(pattern);
            println!("  Run {}: {:.2}M words/s",
                i, run.throughput_words_per_sec / 1e6);
            runs.push(run);
        }

        // Compute statistics
        let summary = StatisticalSummary::from_runs(&runs);

        println!("\n--- Statistics ---");
        println!("Mean:     {:.2}M words/s", summary.mean_throughput / 1e6);
        println!("Median:   {:.2}M words/s", summary.median_throughput / 1e6);
        println!("Std Dev:  {:.2}M words/s", summary.std_dev / 1e6);
        println!("CV:       {:.2}%", summary.coefficient_of_variation * 100.0);
        println!("95% CI:   [{:.2}M, {:.2}M] words/s",
            summary.confidence_interval_95.0 / 1e6,
            summary.confidence_interval_95.1 / 1e6);
        println!("Range:    [{:.2}M, {:.2}M] words/s",
            summary.min_throughput / 1e6,
            summary.max_throughput / 1e6);

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

    results
}

/// Save results to JSON file
pub fn save_results(results: &HashMap<String, StatisticalSummary>, filename: &str) {
    let json = serde_json::to_string_pretty(results)
        .expect("Failed to serialize results");
    std::fs::write(filename, json).expect("Failed to write results");
    println!("\n✅ Results saved to: {}", filename);
}

/// Generate markdown report
pub fn generate_report(results: &HashMap<String, StatisticalSummary>, filename: &str) {
    let mut report = String::new();

    report.push_str("# Baseline Benchmark Results\n\n");
    report.push_str(&format!("**Date:** {}\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    report.push_str(&format!("**GPU:** RTX 4070 Ti SUPER\n"));
    report.push_str(&format!("**CUDA Version:** {}\n", env!("CUDA_VERSION", "Unknown")));
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
        report.push_str(&format!("- **Mean Throughput:** {:.2}M words/s\n", summary.mean_throughput / 1e6));
        report.push_str(&format!("- **Median Throughput:** {:.2}M words/s\n", summary.median_throughput / 1e6));
        report.push_str(&format!("- **Standard Deviation:** {:.2}M words/s\n", summary.std_dev / 1e6));
        report.push_str(&format!("- **Coefficient of Variation:** {:.2}%\n", summary.coefficient_of_variation * 100.0));
        report.push_str(&format!("- **95% Confidence Interval:** [{:.2}M, {:.2}M] words/s\n",
            summary.confidence_interval_95.0 / 1e6,
            summary.confidence_interval_95.1 / 1e6));
        report.push_str(&format!("- **Range:** [{:.2}M, {:.2}M] words/s\n",
            summary.min_throughput / 1e6,
            summary.max_throughput / 1e6));
        report.push_str(&format!("- **Stability:** {}\n",
            if summary.is_stable() { "✅ STABLE" } else { "⚠️ UNSTABLE" }));

        if !summary.outliers.is_empty() {
            report.push_str(&format!("- **Outliers:** {} detected\n", summary.outliers.len()));
        }

        report.push_str("\n");
    }

    std::fs::write(filename, report).expect("Failed to write report");
    println!("✅ Report saved to: {}", filename);
}

fn main() {
    println!("=== GPU Scatter-Gather Baseline Benchmarks ===\n");

    // Run benchmark suite
    let results = run_baseline_suite();

    // Save results
    let timestamp = chrono::Local::now().format("%Y-%m-%d").to_string();
    save_results(&results, &format!("benches/scientific/results/baseline_{}.json", timestamp));
    generate_report(&results, &format!("benches/scientific/results/baseline_report_{}.md", timestamp));

    println!("\n✅ Baseline benchmarking complete!");
}
```

**Tasks:**
- [ ] Create `benches/scientific/baseline_benchmark.rs`
- [ ] Implement `BenchmarkPattern` with standard patterns
- [ ] Implement `run_single_benchmark()` function
- [ ] Implement `run_baseline_suite()` with warm-up + measurement
- [ ] Implement JSON result saving
- [ ] Implement markdown report generation
- [ ] Add dependencies: `serde`, `serde_json`, `chrono`

---

### Step 3: Create Benchmark Runner Script

**File:** `scripts/run_baseline_benchmark.sh`

**Purpose:** Automate environment setup and benchmark execution

```bash
#!/bin/bash
set -euo pipefail

# GPU Scatter-Gather Baseline Benchmark Runner
# Purpose: Establish pre-optimization performance baseline

echo "=== GPU Scatter-Gather Baseline Benchmark Runner ==="
echo ""

# Check if running on correct hardware
if ! nvidia-smi &> /dev/null; then
    echo "❌ Error: NVIDIA GPU not detected"
    exit 1
fi

echo "📊 Detected GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Set CPU governor to performance mode (requires sudo)
echo "⚙️  Setting CPU governor to performance mode..."
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g performance || echo "⚠️  Failed to set CPU governor (continuing anyway)"
else
    echo "⚠️  cpupower not found, skipping CPU governor setup"
fi

# Lock GPU clocks to prevent boosting/throttling (optional, requires nvidia-smi)
echo "⚙️  Locking GPU clocks..."
sudo nvidia-smi -pm 1 || echo "⚠️  Failed to enable persistence mode"
sudo nvidia-smi -lgc 2610,2610 || echo "⚠️  Failed to lock GPU clocks (continuing anyway)"

# Create results directory
mkdir -p benches/scientific/results

# Kill background processes that might interfere
echo "🧹 Minimizing background processes..."
# (Optional: add commands to stop services)

# Build benchmarks in release mode
echo "🔨 Building benchmarks..."
cargo build --release --bin baseline_benchmark

# Run benchmarks
echo ""
echo "🚀 Running baseline benchmarks..."
echo "   This will take approximately 10-20 minutes..."
echo ""

./target/release/baseline_benchmark

# Restore CPU governor
echo ""
echo "⚙️  Restoring CPU governor to ondemand..."
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g ondemand || true
fi

# Unlock GPU clocks
echo "⚙️  Unlocking GPU clocks..."
sudo nvidia-smi -rgc || true
sudo nvidia-smi -pm 0 || true

echo ""
echo "✅ Baseline benchmarking complete!"
echo ""
echo "📁 Results saved to: benches/scientific/results/"
ls -lh benches/scientific/results/baseline_*
```

**Tasks:**
- [ ] Create `scripts/run_baseline_benchmark.sh`
- [ ] Make executable: `chmod +x scripts/run_baseline_benchmark.sh`
- [ ] Test environment setup (CPU governor, GPU clocks)
- [ ] Test full benchmark run
- [ ] Document any hardware-specific setup needed

---

### Step 4: Create Comparison Script

**File:** `scripts/compare_benchmarks.sh`

**Purpose:** Compare baseline with post-optimization results

```bash
#!/bin/bash
set -euo pipefail

# Compare two benchmark results
# Usage: ./scripts/compare_benchmarks.sh baseline_2025-10-17.json optimized_2025-10-24.json

if [ $# -ne 2 ]; then
    echo "Usage: $0 <baseline.json> <optimized.json>"
    exit 1
fi

BASELINE="$1"
OPTIMIZED="$2"

echo "=== Benchmark Comparison ==="
echo "Baseline:  $BASELINE"
echo "Optimized: $OPTIMIZED"
echo ""

# Use jq to compare results
# (Simplified version - could be expanded with Python script for better analysis)

echo "🔍 Throughput Comparison:"
echo ""

# Extract pattern names
PATTERNS=$(jq -r 'keys[]' "$BASELINE")

for pattern in $PATTERNS; do
    baseline_mean=$(jq -r ".\"$pattern\".mean_throughput" "$BASELINE")
    optimized_mean=$(jq -r ".\"$pattern\".mean_throughput" "$OPTIMIZED")

    # Calculate improvement percentage
    improvement=$(echo "scale=2; (($optimized_mean - $baseline_mean) / $baseline_mean) * 100" | bc)

    echo "Pattern: $pattern"
    echo "  Baseline:  $(echo "scale=2; $baseline_mean / 1000000" | bc)M words/s"
    echo "  Optimized: $(echo "scale=2; $optimized_mean / 1000000" | bc)M words/s"
    echo "  Change:    ${improvement}%"
    echo ""
done

echo "✅ Comparison complete!"
```

**Tasks:**
- [ ] Create `scripts/compare_benchmarks.sh`
- [ ] Make executable
- [ ] Test with dummy JSON files
- [ ] Add option for generating comparison graphs (future enhancement)

---

### Step 5: Documentation

**File:** `benches/scientific/README.md`

**Purpose:** Instructions for running benchmarks

```markdown
# Scientific Benchmarking

This directory contains scientific baseline benchmarks for the GPU Scatter-Gather wordlist generator.

## Purpose

Establish rigorous, reproducible performance baseline BEFORE Phase 3 optimizations:
- Multiple runs with statistical analysis
- Standard test patterns for consistent comparison
- Results stored for historical tracking

## Running Benchmarks

### Automated (Recommended)

```bash
# Run complete baseline benchmark suite
./scripts/run_baseline_benchmark.sh
```

This script will:
1. Set CPU governor to performance mode
2. Lock GPU clocks (prevent boost/throttle)
3. Build benchmarks in release mode
4. Run complete benchmark suite (warm-up + measurement)
5. Generate JSON results and markdown report
6. Restore system settings

**Duration:** 10-20 minutes

### Manual

```bash
# Build
cargo build --release --bin baseline_benchmark

# Run
./target/release/baseline_benchmark
```

## Results

Results are saved to `results/` directory:

- **JSON:** `baseline_YYYY-MM-DD.json` (raw data)
- **Report:** `baseline_report_YYYY-MM-DD.md` (human-readable)

## Standard Test Patterns

1. **small_4char_lowercase:** `?l?l?l?l` (456K words)
2. **medium_6char_lowercase:** `?l?l?l?l?l?l` (~309M words)
3. **large_8char_lowercase_limited:** `?l?l?l?l?l?l?l?l` (1B words, limited)
4. **mixed_upper_lower_digits:** `?u?l?d?d?d?d?d?d` (676M words)
5. **special_chars:** `?l?l?s?s?s?s` (~778M words)

## Interpreting Results

### Good Performance
- **CV < 5%:** Stable, consistent performance
- **95% CI narrow:** High confidence in measurements
- **No outliers:** Clean benchmark runs

### Concerning Performance
- **CV >= 5%:** Unstable, investigate thermal throttling or background processes
- **Wide CI:** High variance, need more runs
- **Many outliers:** System interference, check for background tasks

## Comparing Results

After optimizations, compare with baseline:

```bash
./scripts/compare_benchmarks.sh \
    results/baseline_2025-10-17.json \
    results/optimized_2025-10-24.json
```

This shows improvement percentage for each pattern.

## Hardware Requirements

- NVIDIA GPU (CUDA-capable)
- Linux with `nvidia-smi` and `cpupower` (optional)
- Sufficient GPU memory (8GB+ recommended)

## Troubleshooting

**High CV (>5%):**
- Check GPU temperature (thermal throttling?)
- Kill background processes (browsers, etc.)
- Ensure CPU governor is set to "performance"
- Lock GPU clocks to prevent boost variation

**Outliers detected:**
- Background processes interfering
- GPU throttling mid-benchmark
- Consider increasing warm-up runs

**Benchmark crashes:**
- Reduce `max_words_to_generate` for large patterns
- Check GPU memory usage with `nvidia-smi`
- Ensure CUDA drivers are up to date
```

**Tasks:**
- [ ] Create `benches/scientific/README.md`
- [ ] Document all standard patterns
- [ ] Add troubleshooting section
- [ ] Include example output

---

## Execution Checklist

### Day 1: Setup (2-4 hours)

- [ ] Create directory structure: `benches/scientific/`, `scripts/`
- [ ] Add dependencies to `Cargo.toml`:
  ```toml
  [dependencies]
  serde = { version = "1.0", features = ["derive"] }
  serde_json = "1.0"
  chrono = "0.4"

  [dev-dependencies]
  criterion = "0.5"
  ```
- [ ] Create `statistical_analysis.rs` module
- [ ] Write unit tests for statistical functions
- [ ] Test statistical calculations (mean, std dev, CI, etc.)

### Day 2: Implementation (4-6 hours)

- [ ] Create `baseline_benchmark.rs`
- [ ] Implement standard test patterns
- [ ] Implement benchmark runner (warm-up + measurement)
- [ ] Implement JSON result saving
- [ ] Implement markdown report generation
- [ ] Test with single pattern

### Day 3: Integration & Execution (2-4 hours)

- [ ] Create `run_baseline_benchmark.sh` script
- [ ] Test environment setup (CPU, GPU configuration)
- [ ] Run complete benchmark suite
- [ ] Verify results are stable (CV < 5%)
- [ ] Generate and review report
- [ ] Create `compare_benchmarks.sh` for future use
- [ ] Document results in `benches/scientific/README.md`
- [ ] Commit baseline results to repository

**Total Time:** 8-14 hours across 1-3 days

---

## Success Criteria

Baseline benchmarking is complete when:

- [x] All 5 standard patterns benchmarked
- [x] Each pattern has 10+ measurement runs
- [x] Statistical analysis computed (mean, median, std dev, CV, CI)
- [x] CV < 5% for all patterns (stable performance)
- [x] Results saved in JSON format
- [x] Markdown report generated
- [x] Baseline data committed to repository
- [x] Documentation complete

**Deliverables:**
1. `benches/scientific/results/baseline_2025-10-17.json`
2. `benches/scientific/results/baseline_report_2025-10-17.md`
3. `benches/scientific/README.md`
4. Executable scripts: `run_baseline_benchmark.sh`, `compare_benchmarks.sh`

---

## After Baseline is Established

### Proceed to Phase 3: Optimizations

With baseline data in hand, you can now:

1. **Profile GPU kernel** (Nsight Compute)
2. **Identify bottlenecks** (memory, compute, occupancy)
3. **Implement optimization** (e.g., Barrett reduction)
4. **Re-run benchmarks** (same patterns, same script)
5. **Compare results** (use `compare_benchmarks.sh`)
6. **Keep or discard** optimization based on data

### Example Workflow

```bash
# Before optimization: establish baseline
./scripts/run_baseline_benchmark.sh
# Results: baseline_2025-10-17.json

# Implement optimization (e.g., Barrett reduction)
# ... edit kernel code ...

# After optimization: measure improvement
./scripts/run_baseline_benchmark.sh
# Results: optimized_2025-10-24.json

# Compare
./scripts/compare_benchmarks.sh \
    benches/scientific/results/baseline_2025-10-17.json \
    benches/scientific/results/optimized_2025-10-24.json

# Output shows: "+15% improvement on mixed charsets"
```

### Track Performance Over Time

As you complete Phase 3-6, keep running benchmarks:

```
results/
├── baseline_2025-10-17.json          # Pre-optimization
├── barrett_reduction_2025-10-20.json # After arithmetic optimization
├── memory_coalesce_2025-10-25.json   # After memory optimization
├── multi_gpu_2025-11-01.json         # After multi-GPU support
└── final_v1.0_2025-11-15.json        # Production release
```

This creates a **performance timeline** showing project evolution.

---

## Future: Upgrade to Full Enhancement 2

This baseline framework is **minimal but extensible**.

When ready for full scientific benchmarking (Enhancement 2):

**Easy upgrades:**
- Add more test patterns (long masks, edge cases)
- Increase measurement runs (10 → 30 for tighter CI)
- Add GPU profiling (Nsight Compute integration)
- Add visualization (graphs with `plotters`)
- Add CI/CD integration (GitHub Actions)
- Add roofline model analysis (theoretical vs achieved)

**But for now:** Minimal baseline is sufficient for Phase 3!

---

## Dependencies

**Rust crates needed:**

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = "0.4"

[dev-dependencies]
criterion = "0.5"
```

**System tools needed:**
- `nvidia-smi` (CUDA toolkit)
- `cpupower` (Linux CPU frequency management, optional)
- `jq` (JSON processing for comparison script)
- `bc` (calculator for percentage calculations)

**Install on Ubuntu/Debian:**
```bash
sudo apt install nvidia-cuda-toolkit linux-tools-generic jq bc
```

---

## Appendix: Example Output

### Console Output (Expected)

```
=== GPU Scatter-Gather Baseline Benchmarks ===

=== Benchmarking: small_4char_lowercase ===
Mask: ?l?l?l?l
Keyspace: 456976

Warm-up runs (3x)...
  Warm-up 1: 1205.43M words/s
  Warm-up 2: 1198.76M words/s
  Warm-up 3: 1202.31M words/s

Measurement runs (10x)...
  Run 1: 1201.55M words/s
  Run 2: 1199.23M words/s
  Run 3: 1203.88M words/s
  Run 4: 1197.45M words/s
  Run 5: 1204.22M words/s
  Run 6: 1200.11M words/s
  Run 7: 1202.76M words/s
  Run 8: 1198.99M words/s
  Run 9: 1201.03M words/s
  Run 10: 1200.55M words/s

--- Statistics ---
Mean:     1200.98M words/s
Median:   1200.83M words/s
Std Dev:  2.13M words/s
CV:       0.18%
95% CI:   [1199.46M, 1202.50M] words/s
Range:    [1197.45M, 1204.22M] words/s
✅ Performance is STABLE (CV < 5%)

[... repeat for other patterns ...]

✅ Results saved to: benches/scientific/results/baseline_2025-10-17.json
✅ Report saved to: benches/scientific/results/baseline_report_2025-10-17.md

✅ Baseline benchmarking complete!
```

### Generated Report (Example)

See `benches/scientific/results/baseline_report_2025-10-17.md`:

```markdown
# Baseline Benchmark Results

**Date:** 2025-10-17 14:32:15
**GPU:** RTX 4070 Ti SUPER
**CUDA Version:** 12.3

---

## Summary Table

| Pattern | Mean Throughput | Median | Std Dev | CV | 95% CI |
|---------|-----------------|--------|---------|----|------------|
| small_4char_lowercase | 1200.98M words/s | 1200.83M | 2.13M | 0.18% | [1199.46M, 1202.50M] |
| medium_6char_lowercase | 1152.34M words/s | 1151.88M | 3.45M | 0.30% | [1149.88M, 1154.80M] |
| large_8char_lowercase_limited | 1198.77M words/s | 1199.01M | 2.88M | 0.24% | [1196.73M, 1200.81M] |
| mixed_upper_lower_digits | 1145.22M words/s | 1144.99M | 4.12M | 0.36% | [1142.27M, 1148.17M] |
| special_chars | 1088.45M words/s | 1088.12M | 5.67M | 0.52% | [1084.42M, 1092.48M] |

## Detailed Results

### small_4char_lowercase

- **Mean Throughput:** 1200.98M words/s
- **Median Throughput:** 1200.83M words/s
- **Standard Deviation:** 2.13M words/s
- **Coefficient of Variation:** 0.18%
- **95% Confidence Interval:** [1199.46M, 1202.50M] words/s
- **Range:** [1197.45M, 1204.22M] words/s
- **Stability:** ✅ STABLE

[... etc ...]
```

---

**Document Version:** 1.0
**Last Updated:** October 17, 2025
**Author:** tehw0lf + Claude Code
**Status:** Ready to implement
**Next Step:** Execute Day 1 tasks to establish baseline
