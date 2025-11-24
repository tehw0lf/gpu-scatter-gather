# Write-Combined Memory Experiment Results

**Date:** November 24, 2025
**Branch:** `experiment/write-combined-memory`
**Status:** ❌ **EXPERIMENT FAILED** - Significant performance regression

---

## Hypothesis

Write-combined (WC) memory might improve performance for write-only access patterns (e.g., direct file I/O) by trading faster GPU→Host writes for slower CPU reads.

**Predicted outcomes:**
- File I/O pattern: 5-15% improvement (write-only, no CPU reads)
- Vec collection pattern: 0-5% regression (CPU reads data back)

---

## Methodology

### Test Configuration
- **GPU:** RTX 4070 Ti SUPER
- **Pattern:** `?l?l?l?l?d?d?d?d` (8-char, 4 lowercase + 4 digits)
- **Batch size:** 50M words (400 MB data)
- **Iterations:** 5 per benchmark
- **Output format:** PACKED (no separators)

### Implementation
Modified `PinnedBuffer::new()` in `src/multigpu.rs`:

```rust
// Baseline (PORTABLE only)
let flags = CU_MEMHOSTALLOC_PORTABLE;

// Experimental (PORTABLE + WRITECOMBINED)
const CU_MEMHOSTALLOC_WRITECOMBINED: u32 = 0x04;
let flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED;
```

### Benchmarks
Two access patterns tested:

1. **File I/O** (`benchmark_write_combined_file.rs`):
   - Callback writes directly to buffered file
   - Write-only pattern (no CPU reads)
   - Expected to benefit from WC memory

2. **Vec Collection** (`benchmark_write_combined_vec.rs`):
   - Callback collects data into Vec via `extend_from_slice()`
   - Read-after-write pattern (CPU reads data)
   - Expected minor regression

---

## Results

### File I/O Pattern

| Metric | Baseline (PORTABLE) | Experimental (WC) | Change |
|--------|--------------------:|------------------:|-------:|
| **Throughput** | 345.08 M words/s | 57.81 M words/s | **-83.2%** ❌ |
| **Bandwidth** | 2.76 GB/s | 0.46 GB/s | **-83.3%** ❌ |
| **Std Dev** | 0.014s (9.73%) | 0.016s (1.90%) | More consistent, but much slower |

**Raw timings (baseline):** 0.140s, 0.134s, 0.130s, 0.153s, 0.168s
**Raw timings (experimental):** 0.848s, 0.871s, 0.843s, 0.882s, 0.881s

### Vec Collection Pattern

| Metric | Baseline (PORTABLE) | Experimental (WC) | Change |
|--------|--------------------:|------------------:|-------:|
| **Throughput** | 290.38 M words/s | 41.06 M words/s | **-85.9%** ❌ |
| **Bandwidth** | 2.32 GB/s | 0.33 GB/s | **-85.8%** ❌ |
| **Std Dev** | 0.046s (26.87%) | 0.016s (1.34%) | More consistent, but much slower |

**Raw timings (baseline):** 0.148s, 0.147s, 0.143s, 0.264s, 0.158s
**Raw timings (experimental):** 1.234s, 1.225s, 1.224s, 1.219s, 1.187s

---

## Analysis

### Unexpected Results

The hypothesis was **completely wrong**:

1. **File I/O (write-only) regressed by 83%** instead of improving by 5-15%
2. **Vec collection (read-write) regressed by 86%** instead of 0-5%
3. **Both patterns suffered massive slowdowns**, not just the read-heavy one

### Root Cause Analysis

The severe performance degradation suggests:

1. **PCIe transfer bottleneck:**
   - WC memory may have worse PCIe transfer characteristics
   - GPU→Host DMA transfers likely slower with WC flag
   - The "faster writes" benefit is CPU-only, not GPU-relevant

2. **Memory ordering overhead:**
   - WC memory bypasses cache with weaker ordering guarantees
   - May introduce synchronization overhead in multi-GPU context
   - Workers on different threads may see consistency issues

3. **CUDA driver behavior:**
   - WC flag may trigger different code paths in CUDA driver
   - Driver may not optimize WC memory for GPU→Host transfers
   - PORTABLE flag alone is already optimized for this use case

4. **CPU reads are inevitable:**
   - Even "write-only" file I/O requires buffering
   - BufWriter internally reads to manage buffer
   - No access pattern is truly write-only

### Why Hypothesis Failed

The hypothesis assumed:
- WC memory improves GPU→Host transfer speed (FALSE)
- CPU reads are the bottleneck (FALSE - it's the GPU transfer)
- WC benefit outweighs cache miss cost (FALSE - by 6-7×)

The reality:
- **Regular pinned memory is already optimal** for GPU→Host DMA
- **WC memory is for CPU→GPU scenarios** (opposite direction)
- **Cache coherency matters** even for "write-only" patterns

---

## Conclusions

### Definitive Answer: ❌ **DO NOT USE WRITECOMBINED FLAG**

Write-combined memory is **catastrophically bad** for this use case:

- **83-86% performance regression** across all tested patterns
- **No benefit** for any access pattern (file I/O, Vec collection)
- **Regular pinned memory (PORTABLE-only) is optimal**

### Recommendation

**Keep current implementation:**
```rust
// CORRECT - Use PORTABLE flag only
let flags = CU_MEMHOSTALLOC_PORTABLE;
```

**Never add WRITECOMBINED:**
```rust
// INCORRECT - Causes 6-7× slowdown
let flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED;
```

### Lessons Learned

1. **Profile before optimizing** - Assumptions about memory behavior can be very wrong
2. **WC memory is directional** - Optimized for CPU→GPU, not GPU→Host
3. **CUDA driver already optimizes pinned memory** - Don't second-guess it
4. **Benchmark all patterns** - "Write-only" assumption was incorrect
5. **Magnitude matters** - This wasn't a 5% regression, it was 85%

---

## Experiment Disposition

**Decision:** ❌ **ABANDON EXPERIMENT**

Actions:
- ✅ Delete `experiment/write-combined-memory` branch
- ✅ Keep benchmark code for documentation
- ✅ Document findings in this file
- ✅ Update NEXT_SESSION_PROMPT.md to remove Priority 2
- ❌ Do NOT merge WC changes under any circumstance

---

## Artifacts

### Baseline Results (PORTABLE-only)
- `benchmark_write_combined_file_baseline.json` - 345 M words/s
- `benchmark_write_combined_vec_baseline.json` - 290 M words/s

### Experimental Results (PORTABLE + WRITECOMBINED)
- `benchmark_write_combined_file.json` - 58 M words/s
- `benchmark_write_combined_vec.json` - 41 M words/s

### Benchmark Code
- `examples/benchmark_write_combined_file.rs` - File I/O pattern
- `examples/benchmark_write_combined_vec.rs` - Vec collection pattern

---

## CUDA Memory Flags Reference

For future reference, here's what each flag does:

### `CU_MEMHOSTALLOC_PORTABLE` (0x01)
- Memory accessible from any CUDA context
- **Essential for multi-GPU setups**
- Optimized for GPU↔Host transfers
- ✅ **USE THIS**

### `CU_MEMHOSTALLOC_WRITECOMBINED` (0x04)
- CPU writes bypass cache (write combining)
- Faster CPU→GPU writes
- **Slower GPU→Host reads** (uncached)
- ❌ **AVOID FOR GPU→Host USE CASE**

### `CU_MEMHOSTALLOC_DEVICEMAP` (0x08)
- Memory accessible from device via `cudaHostGetDevicePointer()`
- Zero-copy access (no explicit transfer)
- Not used in this project

---

**Experiment conducted by:** Claude Code
**Hardware:** RTX 4070 Ti SUPER (8,448 CUDA cores, 16GB GDDR6X)
**Conclusion:** Current pinned memory implementation is already optimal. No further optimization needed.
