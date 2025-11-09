# Next Session Prompt - Phase 3: GPU Kernel Optimization

**Date Prepared:** November 9, 2025
**Current Status:** Phase 2.6 Complete - Ready for Phase 3
**Working Directory:** `/path/to/gpu-scatter-gather`

---

## Copy-Paste Prompt for Next Session

```
I'm working on gpu-scatter-gather (Rust + CUDA wordlist generator).

Current Status:
- Phase 2.6 COMPLETE: All validation done (mathematical proofs, cross-validation,
  baseline benchmarks, statistical validation)
- Baseline performance: 572-757M words/s on RTX 4070 Ti SUPER (all patterns CV < 5%)
- Publication-ready validation package complete
- Working directory: /path/to/gpu-scatter-gather

Ready to begin Phase 3: GPU Kernel Optimization

Please help me start Phase 3 optimizations according to docs/TODO.md.

Priority optimizations from TODO.md Phase 3:
1. GPU Kernel Optimization
   - Profile current kernel with Nsight Compute
   - Implement Barrett reduction for faster modulo operations
   - Optimize memory coalescing patterns
   - Tune occupancy (block size, registers per thread)

2. CPU-GPU Transfer Optimization
   - Implement pinned memory for faster transfers
   - Add asynchronous transfers with CUDA streams
   - Optimize batch size tuning

3. Multi-GPU Support (optional, can defer)
   - Implement keyspace partitioning
   - Add load balancing

Before starting any optimization:
- Run baseline benchmark to establish pre-optimization metrics
- Use scripts/compare_benchmarks.sh to measure improvement objectively

Key context files:
- docs/TODO.md - Phase 3 plan
- docs/BASELINE_BENCHMARKING_PLAN.md - Benchmarking methodology
- docs/DEVELOPMENT_LOG.md - Implementation history
- benches/scientific/results/baseline_2025-11-09.json - Baseline data

Let's start with profiling the current kernel to identify bottlenecks.
```

---

## Session Context Summary

### What's Complete (Phase 2.6)

**Validation:**
- ✅ Mathematical proofs (bijection, completeness, ordering)
- ✅ Cross-validation (100% match with maskprocessor)
- ✅ Baseline benchmarks (10 runs per pattern, statistical analysis)
- ✅ Statistical validation (chi-square, autocorrelation, runs test)
- ✅ Publication guide (ready for USENIX Security, ACM CCS, etc.)

**Performance Baseline (RTX 4070 Ti SUPER):**
```
Pattern                     Mean Throughput    CV      95% CI
small_4char_lowercase       749.47M words/s   1.39%   [742.12M, 756.82M]
medium_6char_lowercase      756.60M words/s   1.17%   [750.38M, 762.82M]
large_8char_1B_limited      572.22M words/s   1.51%   [566.14M, 578.30M]
mixed_upper_lower_digits    579.93M words/s   0.65%   [577.29M, 582.56M]
special_chars               756.29M words/s   0.85%   [751.79M, 760.80M]
```

**vs State-of-the-Art:**
- vs maskprocessor: 4.0-5.3× faster
- vs cracken: 3.4-4.5× faster

### What's Next (Phase 3)

**Primary Goals:**
1. **Profile current implementation** - Find bottlenecks
2. **Kernel optimization** - Arithmetic improvements (Barrett reduction)
3. **Memory optimization** - Coalesced access, pinned memory
4. **Transfer optimization** - Async transfers, CUDA streams
5. **Measure objectively** - Compare vs baseline with statistical rigor

**Success Criteria:**
- Measure improvement with `scripts/compare_benchmarks.sh`
- Maintain or improve stability (keep CV < 5%)
- Document optimizations in DEVELOPMENT_LOG.md
- Update baseline if significant improvement achieved

### Key Files to Reference

**Documentation:**
- `docs/TODO.md` - Phase 3 detailed plan (lines 646-685)
- `docs/DEVELOPMENT_LOG.md` - Implementation history
- `docs/BASELINE_BENCHMARKING_PLAN.md` - Benchmarking methodology
- `docs/PUBLICATION_GUIDE.md` - For documenting improvements

**Code:**
- `src/gpu/mod.rs` - Current GPU implementation
- `kernels/wordlist_poc.cu` - CUDA kernel (likely location)
- `benches/scientific/baseline_benchmark.rs` - Benchmark runner

**Scripts:**
- `scripts/run_baseline_benchmark.sh` - Run benchmarks
- `scripts/compare_benchmarks.sh` - Compare results

**Baseline Data:**
- `benches/scientific/results/baseline_2025-11-09.json` - Pre-optimization baseline

### Recommended Approach for Phase 3

**Step 1: Profiling (Day 1)**
1. Install/verify Nsight Compute is available
2. Profile current kernel with representative pattern
3. Identify bottlenecks:
   - Compute-bound? → Arithmetic optimization
   - Memory-bound? → Coalescing optimization
   - Launch-bound? → Occupancy tuning

**Step 2: First Optimization (Days 2-3)**
Based on profiling, implement highest-impact optimization:
- If compute-bound: Barrett reduction for modulo
- If memory-bound: Improve coalescing patterns
- If launch-bound: Tune block size/grid size

**Step 3: Measurement (Day 3)**
1. Run full baseline benchmark suite
2. Compare with `scripts/compare_benchmarks.sh`
3. Verify improvement is significant and stable
4. Document in DEVELOPMENT_LOG.md

**Step 4: Iterate (Days 4-N)**
Repeat profile → optimize → measure cycle for additional optimizations

### Important Notes

**Before Any Optimization:**
```bash
# Establish baseline (if not using existing 2025-11-09 baseline)
./scripts/run_baseline_benchmark.sh

# Results saved to: benches/scientific/results/baseline_YYYY-MM-DD.json
```

**After Each Optimization:**
```bash
# Re-run benchmarks
./scripts/run_baseline_benchmark.sh

# Compare results
./scripts/compare_benchmarks.sh \
    benches/scientific/results/baseline_2025-11-09.json \
    benches/scientific/results/optimized_YYYY-MM-DD.json
```

**Commit Strategy:**
- One commit per optimization (atomic changes)
- Include before/after performance data in commit message
- Update DEVELOPMENT_LOG.md with each optimization

### Expected Outcomes

**Realistic Goals:**
- **Barrett reduction:** +5-15% improvement on non-power-of-2 charsets
- **Memory coalescing:** +10-20% improvement if currently inefficient
- **Pinned memory:** +5-10% improvement in total time (reduces transfer overhead)
- **Async transfers:** +10-30% improvement with pipelining

**Stretch Goals:**
- **Combined optimizations:** 1.5-2× total speedup
- **Approach 1B words/s** on optimal patterns
- **Multi-GPU:** Linear scaling (2 GPUs = 2× throughput)

### Git Repository State

**Current Branch:** main

**Recent Commits:**
```
99a4a0b - docs: Add comprehensive publication guide
c74ff45 - docs: Mark Statistical Validation Suite complete
d9f05f5 - feat: Implement statistical validation suite
21f31a2 - docs: Update TODO and DEVELOPMENT_LOG
f3f58b0 - docs: Complete Phase 2.6
850bb66 - feat: Implement scientific baseline benchmarking
```

**Working Tree:** Clean (all Phase 2.6 work committed)

### Hardware Environment

**GPU:** NVIDIA GeForce RTX 4070 Ti SUPER
- Compute Capability: 8.9
- Memory: 16 GB
- CUDA Cores: 8,448
- Base Clock: ~2.31 GHz
- Boost Clock: ~2.61 GHz

**Baseline Established:** November 9, 2025

### Questions to Address in Phase 3

1. **What's the bottleneck?** Compute, memory, or launch?
2. **Which optimization gives best ROI?** Profile first, then decide
3. **Can we maintain stability?** Keep CV < 5% after optimizations
4. **Do we need multi-GPU?** Depends on use case and single-GPU results

### Success Metrics

**Quantitative:**
- X% improvement in throughput (measured objectively)
- CV remains < 5% (stability maintained)
- Performance comparison in commit messages

**Qualitative:**
- Understanding of GPU performance characteristics
- Documented optimization rationale
- Reproducible improvements

---

## Alternative: Conservative Approach

If you prefer to start conservatively:

```
Let's begin Phase 3 by profiling the current kernel implementation.

Please help me:
1. Check if Nsight Compute is available
2. Profile one baseline pattern (e.g., medium_6char_lowercase)
3. Identify the primary bottleneck (compute, memory, or launch)
4. Recommend the highest-impact optimization to implement first

Working directory: /path/to/gpu-scatter-gather

Current baseline: 756.60M words/s (medium_6char_lowercase)
Goal: Understand bottlenecks before optimizing
```

---

## Alternative: Aggressive Approach

If you want to move quickly:

```
Let's aggressively optimize the GPU kernel for Phase 3.

Please help me implement Barrett reduction for faster modulo operations.

Context:
- Current performance: 572-757M words/s depending on pattern
- Bottleneck hypothesis: Integer division/modulo in mixed-radix conversion
- Target improvement: +10-20% on non-power-of-2 charsets

Working directory: /path/to/gpu-scatter-gather
Kernel location: kernels/wordlist_poc.cu (or src/gpu/mod.rs if inline)

Steps:
1. Implement Barrett reduction for division/modulo
2. Benchmark with scripts/run_baseline_benchmark.sh
3. Compare improvement with scripts/compare_benchmarks.sh
4. Document in DEVELOPMENT_LOG.md

Let's start by examining the current kernel implementation.
```

---

**Document Version:** 1.0
**Prepared By:** Claude Code
**Status:** Ready for Phase 3 kickoff
