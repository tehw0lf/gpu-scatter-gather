# Next Optimization Decision - Phase 3 Continuation

**Date**: 2025-11-17
**Current State**: Shared memory caching implemented (+70% improvement)
**Current Performance**: 438 M words/s (12-char passwords)
**Bottleneck**: 95% memory-bound despite optimizations

---

## Quick Decision Matrix

| Option | Time | Difficulty | Expected Gain | Risk | Recommendation |
|--------|------|------------|---------------|------|----------------|
| **1. Deep Memory Analysis** | High (4-6 hrs) | Hard | 0-50x | High (may find nothing) | ⭐ If pursuing max performance |
| **2. Barrett Reduction** | Medium (2-3 hrs) | Medium | 5-15% | Low | If want safe improvement |
| **3. Block Size Tuning** | Low (1 hr) | Easy | 5-10% | Very Low | Quick win, try first |
| **4. Ship Current Version** | Low (1 hr) | Easy | 0% | None | ⭐ If 3-5x is good enough |
| **5. Hybrid CPU/GPU** | High (4-5 hrs) | Hard | 0-20% | Medium | Not recommended |

---

## Detailed Analysis

### Option 1: Deep Memory Coalescing Investigation

**What We Know**:
- Memory throughput stuck at 95% regardless of alignment
- Only using ~1.2% of GPU's 504 GB/s bandwidth
- Alignment improved compute but not memory

**What To Investigate**:
```bash
# Memory transaction efficiency
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ...

# Coalescing metrics
ncu --metrics gld_efficiency,gst_efficiency ...

# Bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ...
```

**Possible Findings**:
1. **Uncoalesced transactions**: Even with alignment, transactions not coalescing
2. **L1/L2 bottleneck**: Cache architecture limiting throughput
3. **Bank conflicts**: Shared memory conflicts slowing down
4. **Sector utilization**: Wasted sectors in memory transactions
5. **Fundamental limit**: GPU memory controller saturation (unlikely at 1.2%)

**Action Plan**:
1. Profile with detailed memory metrics
2. Analyze transaction efficiency
3. Identify specific bottleneck
4. Implement targeted fix
5. Re-profile to validate

**Time Estimate**: 4-6 hours
**Success Probability**: 50% (may hit hardware limit)

---

### Option 2: Barrett Reduction

**Problem**: Integer division/modulo operations are expensive

**Current Code**:
```cuda
int char_idx = remaining % cs_size;  // Slow modulo
remaining /= cs_size;                 // Slow division
```

**Optimized Code** (Barrett Reduction):
```cuda
// Precompute: uint64_t m = (1ULL << 64) / cs_size;
uint64_t q = (remaining * m) >> 64;   // Fast multiply-shift
int char_idx = remaining - q * cs_size; // Fast multiply-subtract
remaining = q;
```

**Why It Might Help**:
- 12-char password = 12 div/mod operations per word
- Current compute utilization: 18% (lots of headroom)
- But: Memory still at 95%, so limited impact

**Expected Improvement**:
- Best case: 15% (if compute becomes bottleneck after)
- Likely case: 5-10% (memory still dominates)
- Worst case: 0% (memory completely saturated)

**Time Estimate**: 2-3 hours
**Success Probability**: 80% (will help, but not dramatically)

---

### Option 3: Block Size Tuning

**Current**: 256 threads/block
**Alternatives**: 128, 512, 1024

**Why It Might Help**:
- Different block sizes affect:
  - Occupancy (warps per SM)
  - Shared memory usage
  - Memory coalescing patterns
  - Warp scheduler efficiency

**Quick Test**:
```cuda
// In generate_words_kernel launch
const int BLOCK_SIZE = 512;  // Try 128, 256, 512, 1024
```

**Expected Results**:
- 128 threads: Lower occupancy, might reduce contention
- 512 threads: Higher occupancy, better SM utilization
- 1024 threads: Max occupancy (if registers allow)

**Time Estimate**: 1 hour
**Success Probability**: 60% (might find 5-10% improvement)

---

### Option 4: Ship Current Version (v1.0)

**Current Performance**:
- **8-char passwords**: 676 M words/s
- **12-char passwords**: 438 M words/s
- **vs maskprocessor (CPU)**: 142 M words/s
- **Speedup**: 3-5x

**Is This Good Enough?**

**Arguments FOR shipping**:
1. Significant speedup already achieved
2. Diminishing returns on optimization time
3. Can gather user feedback and prioritize v2.0
4. Time better spent on other features (CLI, formats, etc.)

**Arguments AGAINST shipping**:
1. Only using 1.2% of GPU bandwidth - huge potential left
2. Competition may have better performance
3. Users expect >10x speedup from GPU tools
4. Memory bottleneck suggests fundamental issue

**Recommendation**:
- If this is a **learning project**: Keep optimizing (Option 1)
- If this is a **product**: Ship and iterate based on feedback

---

### Option 5: Hybrid CPU/GPU Approach

**Idea**: Keep 16-byte aligned GPU writes, optimize CPU-side stripping

**Why This Failed Before**:
```rust
// Naive implementation: 100M iterations, extend_from_slice each
for i in 0..batch_size {
    output.extend_from_slice(&chunk[0..unpadded_word_size]);
}
// This was slower than the entire GPU kernel!
```

**Optimized Approach**:
```rust
// SIMD-optimized stripping with AVX2/AVX-512
// Or: Stream processing to overlap transfer + CPU work
// Or: Just keep the padding (users probably don't care)
```

**Complexity**: High
**Gain**: Uncertain (depends on CPU optimization effectiveness)
**Recommendation**: Not worth it - stick with unaligned or ship as-is

---

## My Recommendation

### Short Term (Next 1-2 Hours):

**Try Option 3 first** (Block Size Tuning):
- Quick to test
- Low risk
- Might find easy 5-10% win
- Takes only 1 hour

```cuda
// Test these in order:
BLOCK_SIZE = 512  // Test first (likely best)
BLOCK_SIZE = 1024 // If 512 is good, try this
BLOCK_SIZE = 128  // If both above are worse
```

### Medium Term (If You Want to Continue):

**Then do Option 1** (Deep Memory Analysis):
- This is where the real bottleneck is
- Only way to unlock 5-50x remaining potential
- Learning opportunity even if it doesn't work
- Will definitively answer "can we fix this?"

### Long Term:

**Option 4** (Ship v1.0):
- If Options 1+3 don't yield >50% improvement
- 3-5x speedup is respectable
- Get user feedback
- Optimize v2.0 based on real-world usage

---

## Decision Checklist

Before next session, decide:

- [ ] What is the performance target? (e.g., "10x faster than CPU")
- [ ] How much time to invest in Phase 3? (e.g., "2 more sessions max")
- [ ] Is this a learning project or a product? (affects optimization depth)
- [ ] What matters more: max performance or shipping quickly?

**Based on answers, choose**:
- **Learning/Max Performance**: Option 3 → Option 1
- **Quick Win**: Option 3 only
- **Ship Product**: Option 4

---

## Resources for Next Session

### If doing Memory Analysis (Option 1):
- Review: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables
- Metrics to collect: `gld_efficiency`, `gst_efficiency`, `l1tex__*`
- Look for: sector utilization < 100%, uncoalesced transactions

### If doing Barrett Reduction (Option 2):
- Read: https://en.wikipedia.org/wiki/Barrett_reduction
- Implementation: https://www.nayuki.io/page/barrett-reduction-algorithm
- Test with: different charset sizes (2, 10, 26, 62)

### If doing Block Size Tuning (Option 3):
- Try: 128, 256, 512, 1024 threads/block
- Measure: occupancy, throughput, memory utilization
- Document: which size works best for which word lengths

---

## Command to Start Next Session

```bash
# Option 3 (Block Size Tuning) - START HERE
vim kernels/wordlist_poc.cu
# Change: const int block_size = 512;
cargo build --release && cargo run --release --example profile_12char

# Option 1 (Memory Analysis)
ncu --set full --section MemoryWorkloadAnalysis \
    -o profiling/results/memory_deep_dive \
    ./target/release/examples/profile_12char

# Option 4 (Ship)
git status  # Review what to commit
git add ...
git commit -m "docs: Add Phase 3 Session 2 findings and decision docs"
```
