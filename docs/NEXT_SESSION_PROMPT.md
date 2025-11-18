# Next Session: Phase 3 Session 4 - Hybrid Architecture

**Quick Start**: Read `docs/PHASE3_SESSION4_PROMPT.md` for full details.

---

## TL;DR

Implement **column-major GPU writes + AVX-512 CPU transpose** to achieve 2-3x additional speedup.

**Current State**: 440 M words/s (3.1x vs CPU), bottlenecked by uncoalesced writes (7.69% efficiency)

**Goal**: 900-1200 M words/s (6-8x vs CPU), using fully coalesced GPU writes (85-95% efficiency)

---

## The Plan

### 1. Column-Major GPU Kernel (2-3h)
```cuda
// CURRENT (uncoalesced):
output[tid * (word_length + 1) + pos]  // 13 bytes apart!

// NEW (coalesced):
output[pos * batch_size + tid]  // consecutive addresses!
```

### 2. AVX-512 CPU Transpose (1-2h)
```rust
// Input:  [char0_all_words][char1_all_words]...  (column-major)
// Output: [word0_all_chars][word1_all_chars]...  (row-major)
```

### 3. Benchmark & Profile (1h)
- Verify coalescing improvement (7.69% → 85-95%)
- Measure transpose overhead (target: <20%)
- Confirm end-to-end speedup (target: 1.5x minimum)

---

## Why This Will Work

**Problem**: Cannot coalesce writes with row-major output (threads write 13 bytes apart)

**Solution**:
1. GPU writes column-major → perfect coalescing ✅
2. CPU transposes fast with SIMD → minimal overhead ✅
3. Output is still row-major → compatibility maintained ✅

---

## Success Criteria

- ✅ Coalescing efficiency >80%
- ✅ Transpose overhead <30%
- ✅ End-to-end speedup ≥1.5x
- ✅ Correctness validated

**If fails**: Ship v1.0 with current 3-5x speedup (still good!)

---

## Key Files

**Read First**:
- `docs/PHASE3_SESSION4_PROMPT.md` - Detailed implementation guide (454 lines)
- `docs/PCIE_BOTTLENECK_ANALYSIS.md` - Why we need this approach
- `docs/PHASE3_SESSION3_SUMMARY.md` - What we tried and learned

**Implement**:
- `kernels/wordlist_poc.cu` - Add `generate_words_columnmajor_kernel`
- `src/transpose.rs` - New SIMD transpose module
- `examples/benchmark_hybrid.rs` - Comprehensive benchmark

---

## Quick Reminders

1. **Address calculation is critical**: `pos * batch_size + tid` for coalescing
2. **Process 32-64 words at a time** for cache efficiency
3. **Use runtime CPU feature detection** (AVX-512 / AVX2 / scalar)
4. **Profile with Nsight to verify coalescing** before optimizing further
5. **Test correctness** - hybrid output must match original exactly

---

## Expected Timeline

- Column-major kernel: 2-3 hours
- CPU transpose (SIMD): 1-2 hours
- Integration: 1 hour
- Benchmarking: 1 hour
- **Total**: 4-6 hours (one focused session)

---

## What Success Looks Like

```
Benchmark Results:
  Original kernel:  440 M words/s
  Hybrid kernel:    1000 M words/s  (+2.3x)
  vs CPU baseline:  7.0x faster

Nsight Compute:
  Coalescing efficiency: 7.69% → 92% (+12x)
  L1 amplification: 13x → 1.3x
  Memory transactions: 528M → 43M (-11x)

Conclusion: Reached near-theoretical maximum! 🚀
```

---

## Start Here

```bash
# 1. Review context
cat docs/PHASE3_SESSION4_PROMPT.md

# 2. Check CPU features
cat /proc/cpuinfo | grep avx

# 3. Start with column-major kernel
vim kernels/wordlist_poc.cu

# 4. Test and iterate
cargo build --release --example benchmark_hybrid
./target/release/examples/benchmark_hybrid
```

**Let's see how fast we can really go! 🔥**
