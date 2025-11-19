# Phase 3 Session 2 Summary - November 17, 2025

## Session Overview

This session focused on profiling and optimizing for **realistic password lengths (8-16 characters)** rather than the 4-character test patterns used previously.

---

## Key Accomplishments

### 1. Committed Phase 3 Optimization 1
- **Shared memory charset caching**
- **Performance gain**: +70% (684M → 1,163M words/s for 4-char patterns)
- **Commit**: `d8fb18f`

### 2. Created Realistic Benchmarks
- New benchmark: `examples/benchmark_realistic.rs`
- Tests 8, 10, and 12-character passwords
- Pattern: `?l?l?l?l?l?l?l?l?d?d?d?d` (lowercase + digits)

### 3. Identified PCIe Bottleneck
- **Finding**: System runs PCIe Gen 3 x16 (~16 GB/s), not Gen 4
- GPU supports Gen 4, but host/motherboard limited to Gen 3
- **Impact**: 35-44% of time spent on PCIe transfer for large batches

### 4. Profiled Realistic Workloads
- Created `examples/profile_12char.rs` for focused profiling
- Identified 95% memory-bound bottleneck persists with longer words
- Compute utilization only 18% (82% idle waiting for memory)

### 5. Attempted 16-Byte Alignment Optimization
- Implemented int4 vectorized writes for perfect memory coalescing
- **Result**: Did NOT improve performance (365M vs 438M words/s = 17% slower)
- **Reason**: Increased data transfer (16 vs 13 bytes) outweighed GPU improvements
- **Learning**: Memory stays at 95% even with perfect alignment - deeper issue

---

## Performance Results

### Realistic Password Lengths (Unaligned - CURRENT BEST)

| Word Length | Pattern | Throughput | Memory BW |
|-------------|---------|------------|-----------|
| 8 chars | `?l?l?l?l?d?d?d?d` | 618-676 M words/s | ~5.6-6.1 GB/s |
| 10 chars | `?l?l?l?l?l?l?d?d?d?d` | 472-561 M words/s | ~5.2-6.2 GB/s |
| 12 chars | `?l?l?l?l?l?l?l?l?d?d?d?d` | 419-438 M words/s | ~5.4-5.7 GB/s |

**Key Observation**: Performance DECREASES with longer words, opposite of expectation.

### Profiling Analysis (12-Character Passwords)

#### Unaligned Version (Current - BEST)
- **Throughput**: 424-438 M words/s
- **Memory Throughput**: 95.01% ⚠️ MEMORY-BOUND
- **Compute Throughput**: 17.91% (only 18% utilized!)
- **Occupancy**: 96.34% (excellent)
- **Bottleneck**: Memory writes (uncoalesced)

#### Aligned Version (16-byte padding - SLOWER)
- **Throughput**: 365 M words/s (17% worse)
- **Memory Throughput**: 95.17% (still saturated!)
- **Compute Throughput**: 41.77% (2.3x improvement)
- **Trade-off**: Better compute but 23% more data transfer = net slower

---

## Key Findings

### 1. We're Using Only ~1.2% of GPU Memory Bandwidth!
- **Available**: 504 GB/s (RTX 4070 Ti SUPER peak bandwidth)
- **Actual**: ~5-6 GB/s total (kernel + PCIe)
- **Huge opportunity** for optimization remaining

### 2. PCIe Transfer is Significant But Not Primary Bottleneck
**For 100M × 12-char words (1.3 GB):**
- **Total time**: 228ms
- **PCIe transfer**: ~81-100ms (~35-44%)
- **Kernel execution**: ~128-147ms (~56-65%)
- **Conclusion**: Kernel is still the primary bottleneck

### 3. Memory Throughput Stays at 95% Regardless of Alignment
- 16-byte alignment should enable perfect coalescing
- Yet memory throughput unchanged (95.01% → 95.17%)
- **Hypothesis**: Fundamental memory architecture limitation or other bottleneck

### 4. Alignment Paradox
- ✅ **Compute improved**: 17.91% → 41.77% (2.3x better)
- ❌ **Memory unchanged**: Still 95% saturated
- ❌ **More data transfer**: 16 bytes vs 13 bytes (+23%)
- **Result**: Net 17% slower overall

---

## Technical Artifacts Created

### New Files
1. `examples/benchmark_realistic.rs` - Realistic password length benchmarks
2. `examples/profile_12char.rs` - Focused profiling for 12-char passwords
3. `profiling/results/profile_12char_20251117_205503.ncu-rep` - Unaligned profiling
4. `profiling/results/profile_12char_20251117_205503_summary.txt` - Analysis
5. `profiling/results/profile_12char_aligned.ncu-rep` - Aligned version profiling

### Modified Files
1. `kernels/wordlist_poc.cu` - Attempted alignment (reverted)
2. `src/gpu/mod.rs` - Padding support (reverted)

---

## Decision Point for Next Session

We have multiple paths forward, each with trade-offs:

### Option 1: Investigate Memory Coalescing Deeper
**Goal**: Understand why 95% memory throughput persists despite alignment

**Approach**:
- Use Nsight Compute memory transaction metrics
- Analyze L1/L2 cache hit rates and sector usage
- Check for bank conflicts or other memory issues

**Pros**: Could unlock the root cause of bottleneck
**Cons**: Time-intensive, may reveal hardware limitations
**Expected Gain**: Unknown - could be 0% or 5x depending on findings

### Option 2: Implement Barrett Reduction (Compute Optimization)
**Goal**: Optimize div/mod operations in mixed-radix decomposition

**Current State**:
- Compute only 18% utilized (unaligned) or 42% (aligned)
- Many div/mod operations per word (12 for 12-char passwords)

**Pros**: Well-understood optimization, should help longer words
**Cons**: Won't address 95% memory bottleneck
**Expected Gain**: 5-15% (small, since compute underutilized)

### Option 3: Tune Block Size
**Goal**: Test different block sizes (128, 512, 1024 threads/block)

**Current**: 256 threads/block
**Pros**: Quick to test, may improve occupancy/coalescing
**Cons**: Unlikely to solve 95% memory bottleneck
**Expected Gain**: 5-10%

### Option 4: Accept Current Performance & Move to Phase 4
**Goal**: Document current state as "good enough" and move forward

**Current Performance**:
- 8-char: 676 M words/s
- 12-char: 438 M words/s
- vs CPU baseline (maskprocessor): 142 M words/s
- **Speedup**: 3-4.7x over CPU

**Pros**: Already significant improvement, time to ship
**Cons**: Leaving ~50-100x potential performance on table
**Decision**: Is 3-5x speedup sufficient for v1.0?

### Option 5: Hybrid Approach - Keep Padding for Write, Strip on CPU
**Goal**: Use 16-byte GPU writes but optimize CPU stripping

**Approach**:
- Keep int4 vectorized kernel (better compute utilization)
- Implement SIMD-optimized CPU-side padding removal
- Or: Stream processing to overlap transfer + strip

**Pros**: Gets compute improvement without transfer penalty
**Cons**: Added complexity, CPU processing overhead
**Expected Gain**: Could match or beat unaligned version

---

## Recommended Next Step

**My Recommendation**: **Option 1 - Investigate Memory Coalescing Deeper**

**Rationale**:
1. We're at 95% memory bottleneck - this IS the problem
2. Only using 1.2% of available bandwidth - huge headroom
3. Alignment didn't help - need to understand WHY
4. Compute optimizations won't help if memory is saturated
5. Could unlock 5-50x improvement if we solve it

**Alternative**: If time is limited, **Option 4 - Ship current version**
- 3-5x speedup is respectable for v1.0
- Can optimize further in v2.0 based on user feedback

---

## Files to Review for Decision

1. `profiling/results/profile_12char_20251117_205503_summary.txt` - Detailed metrics
2. `docs/PHASE3_OPTIMIZATION_RESULTS.md` - Optimization 1 results
3. `benches/scientific/results/baseline_2025-11-17.json` - Performance baselines

---

## Session Metrics

- **Time Invested**: ~2.5 hours
- **Commits**: 1 (Phase 3 Opt 1)
- **Performance Improvement**: +70% (from Phase 3 Opt 1)
- **New Understanding**: Memory bottleneck persists despite alignment
- **Decision Required**: Path forward for remaining optimizations
