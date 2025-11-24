# Next Session: Baseline Validation & Future Optimization Planning

**Status**: ✅ **v1.5.0 RELEASED** (November 24, 2025)
**Repository**: https://github.com/tehw0lf/gpu-scatter-gather
**Current Version**: v1.5.0
**Last Release**: v1.5.0 - https://github.com/tehw0lf/gpu-scatter-gather/releases/tag/v1.5.0
**Next Steps**: Validate baseline performance with 16-char passwords, then explore research optimizations

---

## Immediate Priority: Baseline Validation

### Task: Benchmark Main Branch with 16-Char Passwords

**Goal**: Establish baseline performance metrics with realistic 16-character passwords on main branch (PORTABLE-only pinned memory).

**Why**:
- Write-combined memory experiment tested 8, 12, and 16-char passwords
- Need to validate main branch performance with same 16-char pattern
- Provides reference point for future optimizations

**Expected Results** (based on v1.4.0 numbers):
- 16-char passwords: ~450-500 M words/s
- Slightly lower than 8-char (771 M/s) due to larger data size
- Should be significantly faster than WC memory experiment baseline (184-224 M/s)

**Steps**:
```bash
# 1. Run existing benchmark with 16-char pattern
cargo build --release --example benchmark_realistic
./target/release/examples/benchmark_realistic

# Or create specific 16-char benchmark if needed
# 2. Document baseline in CHANGELOG or development logs
# 3. Use as reference for any future optimizations
```

---

## Research Opportunities (Post-Baseline)

### Priority 1: Memory Coalescing Research (High Risk/Reward)
**Goal**: 2-3× potential improvement through kernel optimization

**Current State**: Column-major kernel with CPU transpose (~500-750 M words/s)

**Approach**:
1. Profile with Nsight Compute to identify bottlenecks
2. Analyze memory coalescing efficiency
3. Experiment with different memory access patterns
4. Test cache-line aligned writes (128 bytes)

**Risk**: High - may require complete kernel rewrite with uncertain payoff

**Files to Analyze**:
- `kernels/wordlist_poc.cu` - Current kernel implementation
- `src/gpu/mod.rs` - Kernel launch configuration

**Profiling Command**:
```bash
ncu --set full --export profile.ncu-rep ./target/release/examples/benchmark_realistic
```

**Questions to Answer**:
- Is the bottleneck memory bandwidth or coalescing?
- Are we hitting PCIe limits or GPU compute limits?
- Can different block/grid configurations improve performance?

---

### Priority 2: Persistent GPU Buffers (Low Hanging Fruit)
**Goal**: Eliminate repeated device allocations (1-2% improvement)

**Current State**: Allocate/free device memory per batch

**Approach**:
- Add persistent buffers to `GpuContext`
- Reuse buffers across batches
- Amortize allocation cost

**Benefits**:
- Reduce `cuMemAlloc()`/`cuMemFree()` overhead
- More predictable performance
- Minimal code changes

**Expected Benefit**: 1-2% improvement, low risk

**Implementation Sketch**:
```rust
struct GpuContext {
    persistent_device_buffer: Option<CUdeviceptr>,
    buffer_capacity: usize,
    // ... existing fields
}

impl GpuContext {
    fn ensure_device_buffer(&mut self, required_size: usize) -> Result<CUdeviceptr> {
        if let Some(ptr) = self.persistent_device_buffer {
            if self.buffer_capacity >= required_size {
                return Ok(ptr);  // Reuse existing buffer
            }
            unsafe { cuMemFree_v2(ptr); }
        }

        let mut ptr: CUdeviceptr = 0;
        unsafe { cuMemAlloc_v2(&mut ptr, required_size)?; }

        self.persistent_device_buffer = Some(ptr);
        self.buffer_capacity = required_size;
        Ok(ptr)
    }
}
```

---

## Alternative Development Directions

### Feature Development
1. **Hybrid Masks**: Combine static prefixes/suffixes with dynamic parts
2. **Rule-Based Generation**: Integrate hashcat-style rules
3. **OpenCL Backend**: Support AMD/Intel GPUs
4. **Python Bindings**: PyPI package for broader adoption

### Integration Work
1. **Hashcat Plugin**: Native integration as custom wordlist provider
2. **John the Ripper Module**: Direct integration
3. **Web API**: Remote generation service

### Academic Work
1. **ArXiv Preprint**: Formal paper on algorithm
2. **Conference Submission**: USENIX Security, ACM CCS
3. **Performance Study**: Compare with other GPU password tools

---

## Documentation References

### Core Documentation
- `CHANGELOG.md` - Complete version history (up to v1.5.0)
- `WRITE_COMBINED_MEMORY_EXPERIMENT.md` - Comprehensive WC memory experiment (8, 12, 16-char results)
- `docs/design/FORMAL_SPECIFICATION.md` - Mathematical foundation
- `docs/design/PINNED_MEMORY_DESIGN.md` - Pinned memory technical spec

### Development Logs
- `docs/development/DEVELOPMENT_LOG.md` - Detailed session notes
- `docs/development/TODO.md` - Outstanding tasks

### Benchmarking
- `docs/benchmarking/BASELINE_BENCHMARKING_PLAN.md` - Performance methodology
- `examples/benchmark_realistic.rs` - Primary performance benchmark
- `examples/benchmark_write_combined_*.rs` - Experimental benchmarks (16-char configured)

### Key Implementation Files
- `src/multigpu.rs` - Multi-GPU context, pinned memory, load balancing
- `src/gpu/mod.rs` - Single GPU context and kernel interface
- `kernels/wordlist_poc.cu` - CUDA kernels (3 variants)

---

## Current Branch State

```bash
# Latest commits
git log --oneline -10

4a359e5 test(experiment): Add 16-char password testing - regression worsens with length
84dc164 fix(experiment): Re-test write-combined memory with realistic 12-char passwords
2b4c4c3 docs: Update NEXT_SESSION_PROMPT and CHANGELOG for v1.5.0 completion
ae8d1f3 docs(experiment): Document write-combined memory experiment results
8b9f71b chore: Remove obsolete release notes files
6a6ff18 chore: Bump version to v1.5.0
eac2a2e feat(perf): Add dynamic load balancing for heterogeneous GPUs
a04b63f chore: Bump version to v1.4.0
45c5858 docs: Update NEXT_SESSION_PROMPT for v1.4.0 release readiness
e2e592d feat(perf): Add zero-copy callback API - Phase 3 of 3 COMPLETE

# Tags
git tag -l
v1.0.0
v1.1.0
v1.2.0
v1.2.1
v1.3.0
v1.4.0
v1.5.0  # ← Latest release
```

---

## Performance Summary

### Current Baseline (v1.4.0/v1.5.0 - PORTABLE-only)
- **8-char passwords**: 771 M words/s
- **10-char passwords**: 554 M words/s
- **12-char passwords**: 497 M words/s
- **16-char passwords**: TBD - needs baseline validation

### Write-Combined Memory Experiment Results
Comprehensively tested and **rejected**:

| Length | File I/O Baseline | File I/O WC | Vec Baseline | Vec WC | Regression |
|--------|------------------:|------------:|-------------:|-------:|-----------:|
| 8-char | 345 M/s | 58 M/s | 290 M/s | 41 M/s | -83% to -86% |
| 12-char | 243 M/s | 38 M/s | 183 M/s | 27 M/s | -84% to -85% |
| 16-char | 184 M/s | 28 M/s | 224 M/s | 21 M/s | -85% to -91% |

**Key Finding**: Performance degrades with longer passwords (Vec: 86% → 91% regression).

---

## Quick Start for Next Session

### Option A: Baseline Validation (Recommended First Step)

**Goal**: Establish 16-char baseline on main branch

**Steps**:
```bash
# 1. Check current performance with realistic benchmark
cargo build --release --example benchmark_realistic
./target/release/examples/benchmark_realistic

# 2. Or create dedicated 16-char baseline benchmark
# 3. Document results in CHANGELOG or development log
# 4. Compare with v1.4.0 numbers (should be ~450-500 M/s for 16-char)
```

### Option B: Memory Coalescing Research (High Risk/Reward)

**Goal**: Profile and optimize CUDA kernel for 2-3× potential speedup

**Steps**:
```bash
# 1. Profile current kernel
ncu --set full --export profile.ncu-rep ./target/release/examples/benchmark_realistic

# 2. Analyze metrics (memory bandwidth, coalescing, cache hits)
# 3. Identify bottleneck (coalescing vs PCIe vs compute)
# 4. Experiment with kernel modifications
# 5. Benchmark improvements

# WARNING: May require extensive kernel rewrite with uncertain payoff
```

### Option C: Persistent GPU Buffers (Quick Win)

**Goal**: 1-2% improvement from eliminating repeated allocations

**Steps**:
```bash
git checkout -b feature/persistent-gpu-buffers

# 1. Add persistent buffer fields to GpuContext
# 2. Implement ensure_device_buffer() method
# 3. Update generate_batch_device_stream() to use persistent buffers
# 4. Add Drop implementation for cleanup
# 5. Test and benchmark
# 6. Commit and merge

git add src/gpu/mod.rs tests/
git commit -m "feat(perf): Add persistent GPU buffers to reduce allocation overhead"
```

### Option D: Feature Development

**Goal**: Expand capabilities beyond pure performance

Ideas:
- Hybrid masks (static + dynamic parts)
- Rule-based generation
- OpenCL backend for AMD GPUs
- Python bindings

---

## Recommendation

**Start with Option A (Baseline Validation)** to establish a clear performance reference point with 16-character passwords on the main branch. This will:
1. Provide accurate baseline for future optimizations
2. Validate current performance is as expected
3. Give context for any future optimization attempts

After baseline validation, the optimization phase is effectively complete. Future work should focus on:
- Research optimizations (memory coalescing - high risk/reward)
- Marginal improvements (persistent buffers - 1-2%)
- Feature development (hybrid masks, rules, OpenCL, Python bindings)

---

*Last Updated: November 24, 2025*
*Version: 17.0 (v1.5.0 Released, WC Memory Experiment Complete)*
*Current Branch: main*
*Status: Ready for baseline validation, then optional research/features*
*Next: Validate 16-char baseline performance on main branch*
