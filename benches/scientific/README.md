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
    benches/scientific/results/baseline_2025-11-09.json \
    benches/scientific/results/optimized_2025-11-10.json
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

## Statistical Analysis

Each pattern is measured with:
- **3 warm-up runs** (discarded, eliminate cold-start effects)
- **10 measurement runs** (for statistical analysis)

### Metrics Computed
- Mean throughput (words/s)
- Median throughput (words/s)
- Standard deviation (σ)
- Coefficient of variation (CV = σ/μ)
- 95% confidence interval (CI)
- Min/max throughput
- Outlier detection (IQR method)

### Acceptance Criteria
- **CV < 5%** (stable performance)
- **No unexplained outliers**
- **Reproducible** (re-running gives similar results)

## Example Output

```
=== GPU Scatter-Gather Baseline Benchmarks ===
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
Compute Capability: 8.9

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
  ...

--- Statistics ---
Mean:     1200.98M words/s
Median:   1200.83M words/s
Std Dev:  2.13M words/s
CV:       0.18%
95% CI:   [1199.46M, 1202.50M] words/s
Range:    [1197.45M, 1204.22M] words/s
✅ Performance is STABLE (CV < 5%)
```

## Implementation Details

### Benchmark Architecture

```
benches/scientific/
├── baseline_benchmark.rs          # Main benchmark binary
├── statistical_analysis.rs        # Statistical analysis module
├── results/
│   ├── baseline_2025-11-09.json   # Raw benchmark data
│   └── baseline_report_2025-11-09.md  # Human-readable report
└── README.md                      # This file
```

### Test Pattern Design

Patterns are chosen to test different aspects:
- **Small patterns:** Test kernel launch overhead
- **Medium patterns:** Test sustained throughput
- **Large patterns:** Test GPU saturation (limited to avoid excessive runtime)
- **Mixed charsets:** Test variable charset sizes
- **Special chars:** Test complex character sets

### Statistical Methods

- **Mean:** Average throughput across all runs
- **Median:** Middle value (robust to outliers)
- **Std Dev:** Measure of variability
- **CV:** Normalized variability (std dev / mean)
- **95% CI:** Range containing true mean with 95% probability
- **Outliers:** Values >1.5 IQR from Q1/Q3 quartiles

## References

- [BASELINE_BENCHMARKING_PLAN.md](../../docs/BASELINE_BENCHMARKING_PLAN.md) - Full implementation plan
- [FORMAL_SPECIFICATION.md](../../docs/FORMAL_SPECIFICATION.md) - Mathematical foundations
- [TODO.md](../../docs/TODO.md) - Project roadmap

## Contributing

If you notice performance regressions or have suggestions for additional test patterns, please open an issue or pull request.
