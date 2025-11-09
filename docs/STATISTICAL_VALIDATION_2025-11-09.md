# Statistical Validation Results

**Date:** 2025-11-09 23:36:08
**GPU:** NVIDIA GeForce RTX 4070 Ti SUPER (Compute Capability 8.9)

---

## Executive Summary

**Overall Result:** 0/5 patterns passed all tests

⚠️ **Some patterns failed validation**

## Summary Table

| Pattern | Sample Size | Chi-square | Autocorrelation | Runs Test | Overall |
|---------|-------------|------------|-----------------|-----------|----------|
| binary_2pos |     100000 | ✅ | ✅ | ❌ | ❌ |
| lowercase_3pos |     100000 | ❌ | ✅ | ❌ | ❌ |
| decimal_4pos |     100000 | ✅ | ✅ | ❌ | ❌ |
| mixed_charsets |     100000 | ✅ | ❌ | ❌ | ❌ |
| hex_4pos |     100000 | ❌ | ✅ | ❌ | ❌ |

## Detailed Results

### binary_2pos

**Sample Size:** 100000 words

#### Chi-square Test (Uniform Distribution)

- **χ² statistic:** 0.00
- **Degrees of freedom:** 2
- **Critical value (95%):** 5.29
- **p-value (approx):** 0.5000
- **Result:** ✅ PASS

#### Autocorrelation Test (Independence)

- **Max lag tested:** 2
- **Max autocorrelation:** 0.0000
- **Significant lags:** None
- **Result:** ✅ PASS

#### Runs Test (Randomness)

- **Number of runs:** 50000
- **Expected runs:** 66666.33
- **Standard deviation:** 133.33
- **Z-score:** -124.999
- **Critical z (95%):** 1.96
- **Result:** ❌ FAIL

### lowercase_3pos

**Sample Size:** 100000 words

#### Chi-square Test (Uniform Distribution)

- **χ² statistic:** 653.71
- **Degrees of freedom:** 75
- **Critical value (95%):** 95.15
- **p-value (approx):** 0.0000
- **Result:** ❌ FAIL

#### Autocorrelation Test (Independence)

- **Max lag tested:** 3
- **Max autocorrelation:** 0.0003
- **Significant lags:** None
- **Result:** ✅ PASS

#### Runs Test (Randomness)

- **Number of runs:** 3847
- **Expected runs:** 66666.33
- **Standard deviation:** 133.33
- **Z-score:** -471.149
- **Critical z (95%):** 1.96
- **Result:** ❌ FAIL

### decimal_4pos

**Sample Size:** 100000 words

#### Chi-square Test (Uniform Distribution)

- **χ² statistic:** 0.00
- **Degrees of freedom:** 36
- **Critical value (95%):** 49.96
- **p-value (approx):** 0.5000
- **Result:** ✅ PASS

#### Autocorrelation Test (Independence)

- **Max lag tested:** 4
- **Max autocorrelation:** 0.0000
- **Significant lags:** None
- **Result:** ✅ PASS

#### Runs Test (Randomness)

- **Number of runs:** 10000
- **Expected runs:** 66666.33
- **Standard deviation:** 133.33
- **Z-score:** -425.001
- **Critical z (95%):** 1.96
- **Result:** ❌ FAIL

### mixed_charsets

**Sample Size:** 100000 words

#### Chi-square Test (Uniform Distribution)

- **χ² statistic:** 71.04
- **Degrees of freedom:** 59
- **Critical value (95%):** 76.87
- **p-value (approx):** 0.1353
- **Result:** ✅ PASS

#### Autocorrelation Test (Independence)

- **Max lag tested:** 3
- **Max autocorrelation:** 0.8893
- **Significant lags:** [1]
- **Result:** ❌ FAIL

#### Runs Test (Randomness)

- **Number of runs:** 10000
- **Expected runs:** 66666.33
- **Standard deviation:** 133.33
- **Z-score:** -425.001
- **Critical z (95%):** 1.96
- **Result:** ❌ FAIL

### hex_4pos

**Sample Size:** 100000 words

#### Chi-square Test (Uniform Distribution)

- **χ² statistic:** 10095.79
- **Degrees of freedom:** 60
- **Critical value (95%):** 78.02
- **p-value (approx):** 0.0000
- **Result:** ❌ FAIL

#### Autocorrelation Test (Independence)

- **Max lag tested:** 4
- **Max autocorrelation:** 0.0028
- **Significant lags:** None
- **Result:** ✅ PASS

#### Runs Test (Randomness)

- **Number of runs:** 6250
- **Expected runs:** 66666.33
- **Standard deviation:** 133.33
- **Z-score:** -453.127
- **Critical z (95%):** 1.96
- **Result:** ❌ FAIL

## Interpretation

### Chi-square Test
Tests if characters at each position follow a uniform distribution. A passing result means each character in the charset appears with equal probability, indicating no bias in the mixed-radix algorithm.

### Autocorrelation Test
Tests if there are correlations between character positions. A passing result means positions are independent, with no patterns like "if position i is 'a', position i+1 tends to be 'b'".

### Runs Test
Tests if the sequence of generated words is random. A passing result means words appear in a random order, not systematically (e.g., not always increasing or always decreasing).

