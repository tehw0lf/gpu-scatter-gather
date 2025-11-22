# Nsight Compute Profile - November 22, 2025

**Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER (Compute Capability 8.9)
**Library Version:** v1.0.0
**Test:** 50M 12-character passwords (?l?l?l?l?l?l?l?l?d?d?d?d)

---

## Executive Summary

**Performance:** 440 M words/s (5.3 GB/s)
**Primary Bottleneck:** Uncoalesced memory accesses (90% excessive sectors)
**Optimization Potential:** **~90% speedup possible** through memory coalescing improvements

---

## Key Metrics

### GPU Speed of Light
- **Memory Throughput:** 94.52% (L2 cache bound)
- **DRAM Throughput:** 14.58% (very low - data served from L2)
- **Compute (SM) Throughput:** 21.13% (under-utilized)
- **Duration:** 6.35 ms

**Finding:** Workload is memory-bound at L2 cache level, not compute-bound.

### Occupancy
- **Theoretical Occupancy:** 100%
- **Achieved Occupancy:** 96.28%
- **Active Warps Per SM:** 46.21 / 48
- **Registers Per Thread:** 38

**Finding:** Excellent occupancy - not a limiting factor.

### Memory Access Patterns

#### Critical Issue: Uncoalesced Memory Accesses 🔴

```
Excessive Sectors: 206,250,000 / 228,906,260 (90%)
Est. Speedup: 90.03%
```

**Global Loads:**
- Average utilization: **7.8 / 32 bytes per sector** (24% efficiency)
- Est. speedup: 43.36%

**Global Stores:**
- Average utilization: **2.7 / 32 bytes per sector** (8% efficiency)
- Est. speedup: 52.55%

**Root Cause:** Memory access stride between threads causing poor coalescing.

### Cache Performance
- **L1/TEX Hit Rate:** 90.69% (very good)
- **L2 Hit Rate:** 99.91% (excellent)
- **L2 Persisting Size:** 9.44 MB

**Finding:** Cache hierarchy performing well, but feeding inefficient memory patterns.

### Warp Efficiency
- **Warp Cycles Per Issued Instruction:** 56.10
- **Avg. Active Threads Per Warp:** 31.10 / 32 (96.9%)
- **Barrier Stalls:** 34.3% of total stall cycles (19.2 cycles/warp)

**Finding:** Warps are mostly full, but significant barrier stalls present.

---

## Optimization Opportunities

### Priority 1: Fix Memory Coalescing 🎯

**Problem:** Current kernel writes each character position sequentially per thread, causing strided access patterns.

**Current Pattern (uncoalesced):**
```
Thread 0: writes word[0], word[1], word[2], ..., word[11]
Thread 1: writes word[0], word[1], word[2], ..., word[11]
Thread 2: writes word[0], word[1], word[2], ..., word[11]
```

Each thread writes to a different memory location for each character, preventing coalescing.

**Solution: Transposed Memory Layout**

```
Phase 1 - Write character 0 of all words (coalesced):
  Thread 0: words[0][0]
  Thread 1: words[1][0]
  Thread 2: words[2][0]
  ...

Phase 2 - Write character 1 of all words (coalesced):
  Thread 0: words[0][1]
  Thread 1: words[1][1]
  Thread 2: words[2][1]
  ...
```

**Expected Gain:** 2-3× throughput (from 440 M/s to 900-1300 M/s)

**Implementation:**
- Modify kernel to write in column-major order
- May require CPU transpose step afterward (negligible cost)
- Alternative: Use shared memory for coalescing

### Priority 2: Reduce Barrier Stalls

**Problem:** 34.3% of stall cycles due to barrier synchronization (19.2 cycles/warp)

**Potential Solutions:**
1. Analyze if barriers are necessary (likely not for independent word generation)
2. If unavoidable, ensure uniform work distribution before barriers
3. Consider smaller block sizes (current: 256 threads) to reduce barrier wait time

**Expected Gain:** 5-10% improvement

### Priority 3: Increase Compute Utilization

**Current:** 21.13% SM throughput (under-utilized)

**Possible Approaches:**
1. Increase work per thread (reduce kernel launch overhead)
2. Optimize division operations (Barrett reduction)
3. Better instruction-level parallelism

**Expected Gain:** 10-20% improvement (limited by memory bottleneck)

---

## Detailed Analysis

### Launch Statistics
```
Grid Size: 195,313 blocks
Block Size: 256 threads
Total Threads: 50,000,128
Waves Per SM: 493.21
```

**Analysis:** Very large grid size leads to excellent GPU saturation. Not a bottleneck.

### Instruction Statistics
```
Executed Instructions: 806,640,806
Issued Instructions: 806,663,347
Branch Efficiency: 97.99%
Divergent Branches: 2,959 (0.004% of branches)
```

**Analysis:** Minimal branch divergence, good branch prediction. Not a bottleneck.

### Scheduler Statistics
```
Issued Warp Per Scheduler: 0.21 inst/cycle
One or More Eligible: 20.60%
No Eligible: 79.40%
```

**Analysis:** Schedulers are idle 79% of the time due to memory stalls. Confirms memory-bound behavior.

---

## Historical Context

### Previous Optimization Attempts (Phase 3)

See `docs/archive/PHASE3_OPTIMIZATION_RESULTS.md` for detailed history.

**Attempt 1: Transposed Writes (Row-to-Column)**
- Implementation: `generate_words_transposed_kernel`
- Result: **Regression** (slower than baseline)
- Reason: Uncoalesced reads from charset data

**Attempt 2: Column-Major + CPU Transpose**
- Implementation: `generate_words_columnmajor_kernel`
- Result: **2× speedup** (350M → 635M words/s on Phase 3 tests)
- Status: **Not yet integrated into v1.0.0 C API**

### Current v1.0.0 Status

The v1.0.0 release uses the **original row-major kernel** for stability and validation purposes. The column-major kernel showed promise but was not integrated due to:
1. Focus on formal validation and release
2. Need for comprehensive testing with FFI layer
3. Desire for additional profiling and tuning

**Opportunity:** Integrate and optimize the column-major kernel for v1.1.0.

---

## Recommendations for v1.1.0

### Short-Term (High Impact)

1. **Integrate Column-Major Kernel**
   - Port `generate_words_columnmajor_kernel` to FFI layer
   - Add transpose step to output formatting
   - Expected gain: **2× throughput** (440 → 900 M/s)

2. **Optimize Charset Loading**
   - Use shared memory for frequently accessed charset data
   - Reduce global memory pressure
   - Expected gain: **10-15%**

3. **Benchmark on Different GPUs**
   - Profile on Turing, Ampere, Hopper architectures
   - Identify architecture-specific optimizations
   - Build empirical performance database

### Medium-Term (Foundational)

4. **Multi-GPU Support**
   - Distribute keyspace across devices
   - Overlap compute with inter-GPU communication
   - Expected gain: **Linear scaling** with GPU count

5. **Barrett Reduction for Division**
   - Replace integer division with Barrett reduction
   - Particularly beneficial for non-power-of-2 charset sizes
   - Expected gain: **15-20%** on compute-bound cases

6. **Dynamic Kernel Selection**
   - Choose kernel variant based on:
     - Password length (short vs. long)
     - Charset sizes (power-of-2 vs. arbitrary)
     - Batch size (small vs. large)
   - Maximize performance across diverse workloads

### Long-Term (Research)

7. **Warp-Level Primitives**
   - Use warp shuffle operations for data exchange
   - Reduce shared memory usage
   - Potential for novel algorithmic approaches

8. **Tensor Core Utilization**
   - Investigate if wordlist generation can leverage Tensor Cores
   - Likely limited applicability but worth exploring

9. **Persistent Kernels**
   - Keep threads alive across batches
   - Reduce kernel launch overhead
   - Beneficial for streaming workloads

---

## Profiling Commands

### Quick Profile (Brief Metrics)
```bash
ncu --set brief --target-processes all ./target/release/examples/benchmark_realistic
```

### Full Profile (All Metrics)
```bash
ncu --set full --kernel-name generate_words_kernel --launch-skip 7 --launch-count 1 \
    ./target/release/examples/benchmark_realistic
```

### Memory-Focused Profile
```bash
ncu --section MemoryWorkloadAnalysis --section SourceCounters \
    --kernel-name generate_words_kernel --launch-count 1 \
    ./target/release/examples/benchmark_realistic
```

### Export for Analysis
```bash
ncu --set full --export profile.ncu-rep \
    ./target/release/examples/benchmark_realistic
```

Then open with `ncu-ui profile.ncu-rep` for GUI analysis.

---

## Conclusion

The v1.0.0 kernel achieves **440-700 M words/s** and is **memory-bound** rather than compute-bound. The primary bottleneck is **uncoalesced memory accesses** (90% excessive sectors).

**Key Takeaways:**
1. ✅ **Occupancy is excellent** (96%)
2. ✅ **Cache hit rates are very good** (L1: 91%, L2: 99.9%)
3. ✅ **Branch efficiency is high** (98%)
4. 🔴 **Memory coalescing is poor** (loads: 24%, stores: 8%)
5. 🔴 **Barrier stalls are significant** (34% of total stalls)

**Next Steps:**
- Integrate column-major kernel (proven 2× speedup)
- Optimize memory access patterns for coalescing
- Profile on diverse GPU architectures
- Implement multi-GPU support for linear scaling

With these optimizations, we can realistically target **900-1300 M words/s** on current hardware, achieving **6-9× speedup** over maskprocessor instead of the current 4-7×.

---

*Generated: November 22, 2025*
*Hardware: NVIDIA GeForce RTX 4070 Ti SUPER*
*Software: gpu-scatter-gather v1.0.0*
