# Memory Coalescing Optimization - Final Decision

**Date**: December 1, 2025
**Status**: ❌ **PERMANENTLY ABANDONED** (Three Strikes Rule)
**Branch**: `experiment/memory-coalescing-v3` (archived, not merged)

---

## Executive Summary

After three comprehensive attempts to optimize memory coalescing in the CUDA kernel, we have **permanently decided to abandon this optimization path**. While Nsight Compute profiling shows 84.55% potential speedup from fixing uncoalesced memory accesses, **all experimental attempts resulted in 2-5× performance regressions**.

**Current performance (365-765 M words/s) is accepted as optimal** for this workload.

---

## The Three Strikes

### ⚾ Strike 1: CPU Transpose (Phase 3, November 2025)

**Approach**: Generate words in column-major format, transpose on CPU to row-major

**Result**: **5.3× SLOWER** (85 M words/s vs 440 M words/s baseline)

**Root Cause**: CPU RAM bandwidth bottleneck (3.9 GB/s << GPU 504 GB/s)

**Documentation**: `docs/archive/PHASE3_SESSION4_SUMMARY.md`

---

### ⚾ Strike 2: GPU Transpose (November 22, 2025)

**Approach**: Generate words in column-major format, transpose on GPU using shared memory tiling

**Result**: **2-4× SLOWER** (99-170 M words/s vs 401-440 M words/s baseline)

**Root Cause**: Extreme aspect ratio (50M × 13) causes poor tile utilization
- Only 13/32 = 40% of each tile used (23 rows wasted)
- 1.5M blocks launched, most 97% empty
- GPU scheduling overhead dominates

**Documentation**: `docs/archive/GPU_TRANSPOSE_ATTEMPT_2025-11-22.md`

---

### ⚾ Strike 3: Profiling Analysis (December 1, 2025)

**Approach**: Comprehensive Nsight Compute profiling to understand bottleneck

**Result**: **NO IMPLEMENTATION ATTEMPTED**

**Finding**: Profiling confirms 84.55% potential speedup, but:
1. All logical solutions have already been tried (strikes 1 & 2)
2. All failed with significant regressions
3. Profiler estimates don't account for workload-specific characteristics

**Profiling Data** (50M word batch, 8-char passwords):

| Metric | Value | Analysis |
|--------|-------|----------|
| **Memory Throughput** | 95.16% | Saturated |
| **Compute Throughput** | 33.78% | Underutilized |
| **Uncoalesced Sectors** | 17.5M excessive (85% waste) | Confirmed issue |
| **Global Store Efficiency** | 4.0 / 32 bytes (12.5%) | Very poor |
| **Global Load Efficiency** | 8.3 / 32 bytes (25.9%) | Poor |
| **L2 Hit Rate** | 100% | Perfect caching |
| **Achieved Occupancy** | 95.16% | Excellent |

**Nsight Compute Recommendations**:
- Est. speedup from fixing stores: 50.52%
- Est. speedup from fixing loads: 42.82%
- **Est. speedup from fixing uncoalesced accesses: 84.55%**

**Documentation**: `EXPERIMENT_NOTES.md` (on `experiment/memory-coalescing-v3` branch)

---

## Why All Attempts Failed

### The Paradox

- **Profiler says**: 84.55% speedup possible
- **Experimental evidence says**: All attempts resulted in 2-5× *slowdowns*

### Root Cause: Workload Characteristics

Wordlist generation has unique properties that break standard GPU optimization assumptions:

1. **Extreme aspect ratio**: 50,000,000 words × 13 chars = 3,846,154:1 ratio
   - Standard GPU algorithms assume roughly square matrices
   - Tile-based transpose requires both dimensions > tile size
   - Ours: 50M >> 32, but 13 << 32

2. **One dimension << cache line size**: 13 chars vs 32-byte sectors
   - Standard coalescing techniques work when both dimensions are large
   - Ours: One dimension is smaller than the optimization granularity

3. **Every fix makes something else worse**:
   - Coalesce writes → uncoalesce reads (charset data access)
   - Add transpose → dominate total time with overhead (2-5× slower)
   - Use shared memory → poor occupancy due to extreme aspect ratio

### Why Profiler Estimates Are Misleading

Nsight Compute's 84.55% speedup estimate assumes:
- ✅ Coalescing improves throughput (true in general)
- ❌ Transpose overhead is negligible (FALSE for our aspect ratio)
- ❌ Standard algorithms apply (FALSE - tile-based transpose fails)
- ❌ Improvements apply uniformly (FALSE - reads become uncoalesced)

**Reality**: Transpose overhead (2-5×) >> coalescing benefit (0.5-1×)

---

## Three Strikes Rule - Why Stop?

### Sufficient Evidence

1. ✅ **Comprehensive profiling**: Understand the problem
2. ✅ **Multiple solution attempts**: Tried all logical approaches
3. ✅ **Consistent negative results**: All attempts regressed 2-5×
4. ✅ **Diminishing returns**: Current performance already excellent

### Risk vs Reward

**Risks**:
- Destabilizing production-ready code
- Wasted development time on dead ends
- Opportunity cost (could work on other features)

**Rewards**:
- Uncertain gains (all evidence points to negative)
- Profiler estimates proven unreliable for this workload
- Current performance already 3.8-15.3× faster than CPU

**Verdict**: Risk >> Reward → Stop optimization attempts

---

## Final Decision

### ✅ ACCEPT CURRENT PERFORMANCE AS OPTIMAL

**Performance**: 365-765 M words/s (depending on password length)

**Competitive Position**: 3.8-15.3× faster than fastest CPU competitor (cracken)

**Conclusion**: The "uncoalesced" writes shown in profiling are not a fixable bottleneck - they're an inherent characteristic of this workload's extreme aspect ratio.

### 🚫 NO FURTHER MEMORY COALESCING ATTEMPTS

**Rule**: Three strikes and you're out.

**Status**: Permanently closed optimization path.

**Rationale**:
- All logical solutions attempted
- All failed with significant regressions
- Profiler recommendations don't apply to this workload
- Time better spent on features or other projects

---

## What We Learned

### 1. Profiler Estimates Aren't Universal Truth

- Nsight Compute provides excellent data
- But speedup estimates assume standard workloads
- Extreme aspect ratios break standard assumptions
- Must validate estimates with experiments

### 2. Some Bottlenecks Can't Be Fixed

- Not every bottleneck shown in profiling is addressable
- Some are fundamental to the algorithm/workload
- Attempting to "fix" them can make performance worse
- Know when to accept current performance

### 3. Three Strikes Rule Works

- First attempt: Fail fast, learn from failure
- Second attempt: Try different approach, fail again
- Third attempt: Analyze comprehensively, recognize pattern
- **Don't attempt a fourth** - diminishing returns

### 4. Production Performance Metrics

- 365-765 M words/s is excellent for this use case
- 3.8-15.3× faster than fastest CPU competitor
- "Good enough" is better than "perfect but unstable"
- Ship working code, move on to next project

---

## References

### Experiment Documentation

1. **Strike 1**: `docs/archive/PHASE3_SESSION4_SUMMARY.md`
2. **Strike 2**: `docs/archive/GPU_TRANSPOSE_ATTEMPT_2025-11-22.md`
3. **Strike 3**: `experiment/memory-coalescing-v3` branch → `EXPERIMENT_NOTES.md`

### Profiling Data

- **Nsight Compute Profile**: `profile_v3.ncu-rep` (on experiment branch)
- **Analysis**: `EXPERIMENT_NOTES.md` (experiment branch)

### Related Documentation

- `docs/design/FORMAL_SPECIFICATION.md` - Algorithm design
- `docs/benchmarking/COMPETITIVE_RESULTS.md` - Performance validation
- `docs/benchmarking/BASELINE_BENCHMARKING_PLAN.md` - Performance methodology

---

## Recommendations for Future Work

### ✅ Focus on These Instead

1. **Feature Development**
   - Hybrid masks (static + dynamic parts)
   - Rule-based generation
   - Python bindings for ease of use

2. **Integration Guides**
   - hashcat integration examples
   - John the Ripper integration examples
   - Best practices documentation

3. **Academic Publication**
   - Write paper on direct index-to-word mapping algorithm
   - Emphasize mathematical foundation
   - Discuss why standard GPU optimizations don't apply

### ❌ Don't Attempt

1. **Memory coalescing optimization** (three strikes)
2. **Write-combined memory** (failed catastrophically in separate experiment)
3. **Different transpose algorithms** (fundamental workload mismatch)

---

## Conclusion

After three comprehensive attempts spanning multiple weeks, we have definitively determined that **memory coalescing optimization is not beneficial for this workload** despite profiler recommendations.

**Current performance (365-765 M words/s, 3.8-15.3× faster than CPU) is accepted as optimal.**

This decision follows the "three strikes rule" and represents sound engineering judgment: know when to stop optimizing and ship production-ready code.

---

**Last Updated**: December 1, 2025
**Decision Status**: FINAL
**Archive Branch**: `experiment/memory-coalescing-v3`
