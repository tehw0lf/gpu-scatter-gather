# Baseline Benchmark Results

**Date:** 2025-11-20 20:49:04
**GPU:** NVIDIA GeForce RTX 4070 Ti SUPER (Compute Capability 8.9)

---

## Summary Table

| Pattern | Mean Throughput | Median | Std Dev | CV | 95% CI |
|---------|-----------------|--------|---------|----|---------|
| medium_6char_lowercase | 725.57M words/s | 732.35M | 19.99M | 2.75% | [711.48M, 739.65M] |
| large_8char_lowercase_limited | 553.24M words/s | 561.56M | 14.69M | 2.66% | [542.89M, 563.59M] |
| small_4char_lowercase | 560.73M words/s | 575.79M | 38.03M | 6.78% | [533.94M, 587.53M] |
| mixed_upper_lower_digits | 553.16M words/s | 551.83M | 12.48M | 2.26% | [544.36M, 561.95M] |
| special_chars | 720.43M words/s | 718.46M | 10.14M | 1.41% | [713.28M, 727.57M] |

## Detailed Results

### medium_6char_lowercase

- **Mean Throughput:** 725.57M words/s
- **Median Throughput:** 732.35M words/s
- **Standard Deviation:** 19.99M words/s
- **Coefficient of Variation:** 2.75%
- **95% Confidence Interval:** [711.48M, 739.65M] words/s
- **Range:** [682.61M, 753.78M] words/s
- **Stability:** ✅ STABLE
- **Outliers:** 1 detected

### large_8char_lowercase_limited

- **Mean Throughput:** 553.24M words/s
- **Median Throughput:** 561.56M words/s
- **Standard Deviation:** 14.69M words/s
- **Coefficient of Variation:** 2.66%
- **95% Confidence Interval:** [542.89M, 563.59M] words/s
- **Range:** [520.89M, 569.07M] words/s
- **Stability:** ✅ STABLE

### small_4char_lowercase

- **Mean Throughput:** 560.73M words/s
- **Median Throughput:** 575.79M words/s
- **Standard Deviation:** 38.03M words/s
- **Coefficient of Variation:** 6.78%
- **95% Confidence Interval:** [533.94M, 587.53M] words/s
- **Range:** [486.87M, 592.97M] words/s
- **Stability:** ⚠️ UNSTABLE
- **Outliers:** 2 detected

### mixed_upper_lower_digits

- **Mean Throughput:** 553.16M words/s
- **Median Throughput:** 551.83M words/s
- **Standard Deviation:** 12.48M words/s
- **Coefficient of Variation:** 2.26%
- **95% Confidence Interval:** [544.36M, 561.95M] words/s
- **Range:** [536.90M, 570.98M] words/s
- **Stability:** ✅ STABLE

### special_chars

- **Mean Throughput:** 720.43M words/s
- **Median Throughput:** 718.46M words/s
- **Standard Deviation:** 10.14M words/s
- **Coefficient of Variation:** 1.41%
- **95% Confidence Interval:** [713.28M, 727.57M] words/s
- **Range:** [698.45M, 733.93M] words/s
- **Stability:** ✅ STABLE

