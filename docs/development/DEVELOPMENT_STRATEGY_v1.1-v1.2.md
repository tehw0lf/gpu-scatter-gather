# Development Strategy: v1.1.0 & v1.2.0

**Created:** November 22, 2025
**Status:** Planning Phase

---

## Overview

After v1.0.0 release and comprehensive profiling, we have identified two distinct optimization paths. This document outlines the strategy for pursuing both in parallel.

---

## Strategy: Multi-GPU First, Single-GPU Research Second

### Why This Order?

**v1.1.0 - Multi-GPU Support (6 weeks)**
- ✅ **Proven approach** - Linear scaling is well-understood
- ✅ **Testable with 1 GPU** - No special hardware needed for development
- ✅ **Immediate user value** - Benefits anyone with 2+ GPUs
- ✅ **Design complete** - `docs/design/MULTI_GPU_DESIGN.md` already done
- ✅ **Predictable outcome** - 90-95% scaling efficiency expected

**v1.2.0 - Single-GPU Optimization (research-driven)**
- 🔬 **Complex research** - Multiple failed attempts already
- 🔬 **Uncertain gains** - May hit fundamental algorithm limits
- 🔬 **Benefits from time** - Community feedback can guide research
- 🔬 **No hardware blocker** - Can research while v1.1.0 is deployed

---

## v1.1.0: Multi-GPU Support

### Implementation Plan (6 Weeks)

**Week 1: Device Enumeration**
```c
int wg_get_device_count(void);
int wg_get_device_info(int device_id, char* name, int* compute_cap, ...);
```
- Enumerate available CUDA devices
- Query device properties
- Unit tests for single-GPU and mock multi-GPU

**Week 2: Multi-Context Management**
```rust
struct GpuWorker {
    device_id: i32,
    context: GpuContext,
    stream: CUstream,
}

struct MultiGpuContext {
    workers: Vec<GpuWorker>,
}
```
- Create per-device CUDA contexts
- Isolate contexts to prevent interference
- Handle context switching and cleanup

**Week 3: Keyspace Partitioning**
```rust
fn partition_keyspace(total: u64, num_gpus: usize) -> Vec<(u64, u64)> {
    // Divide keyspace evenly across GPUs
    // Handle remainder for perfect distribution
}
```
- Static partitioning algorithm
- Edge case handling (keyspace < num_gpus)
- Unit tests for various scenarios

**Week 4: Parallel Generation**
```rust
impl MultiGpuContext {
    pub fn generate_batch(...) -> Result<Vec<u8>> {
        // Launch kernels on all GPUs in parallel
        // Collect results
        // Merge in correct order
    }
}
```
- Thread pool for concurrent GPU execution
- Async kernel launches with streams
- Output aggregation in correct order

**Week 5: FFI Integration**
```c
WG_MultiGpuGenerator* wg_multigpu_create(...);
int wg_multigpu_generate(...);
void wg_multigpu_destroy(...);
```
- Expose multi-GPU API to C
- Statistics collection
- C integration tests

**Week 6: Optimization & Testing**
- Pinned memory for faster transfers
- Benchmark scaling efficiency (1, 2, 4 GPUs)
- Test on diverse GPU combinations
- Document performance characteristics

### Testing Strategy

**With Single GPU (Development):**
```bash
# Should behave identically to single-GPU generator
./test_multigpu_single_device

# Unit tests with mocked device count
cargo test test_partition_keyspace
cargo test test_multi_context_single_gpu
```

**With Multiple GPUs (Community Testing):**
```bash
# Validate linear scaling
./benchmark_multigpu --gpus 1,2,4

# Stress test
./stress_test_multigpu --duration 1h
```

### Success Criteria

- ✅ No regression with 1 GPU
- ✅ 90-95% scaling efficiency with 2+ GPUs
- ✅ 100% correctness (matches single-GPU output)
- ✅ Zero crashes in 24-hour stress test
- ✅ Complete FFI API with examples

---

## v1.2.0: Single-GPU Memory Optimization

### Current Situation

**Profiling Results (Nsight Compute):**
- 90% excessive sectors from uncoalesced memory
- Global loads: 24% efficiency (7.8/32 bytes used)
- Global stores: 8% efficiency (2.7/32 bytes used)
- Memory throughput: 94.52% (L2 bound)
- Compute throughput: 21.13% (under-utilized)

**Previous Failed Attempts:**
1. **Transposed kernel** (Phase 3 Session 3)
   - Result: Same speed as baseline
   - Issue: Different bottleneck emerged

2. **Column-major + CPU transpose** (Phase 3 Session 4)
   - Result: 5.3× SLOWER
   - Issue: CPU transpose bottleneck (81% overhead)
   - Bottleneck: DDR4-2400 RAM only 1.84 GB/s for transpose

### Research Directions

#### Option 1: Shared Memory Buffering

**Concept:** Stage writes through shared memory to coalesce before global write

```cuda
__shared__ char shared_buffer[BLOCK_SIZE * WORD_LENGTH];

// Phase 1: Each thread writes to shared memory (no coalescing needed)
shared_buffer[threadIdx.x * WORD_LENGTH + pos] = character;

__syncthreads();

// Phase 2: Coalesced write to global memory
// Thread 0 writes bytes [0, 32), thread 1 writes [32, 64), etc.
if (threadIdx.x < WORD_LENGTH) {
    for (int i = 0; i < BLOCK_SIZE; i += 32) {
        output[...] = shared_buffer[...];
    }
}
```

**Pros:**
- Shared memory is fast (~10 TB/s)
- Can reorganize data for coalesced global writes
- Well-understood CUDA pattern

**Cons:**
- Limited shared memory (48-100 KB per SM)
- Requires synchronization (may stall warps)
- Complex indexing logic
- May not eliminate all uncoalesced accesses

**Next Step:** Prototype and benchmark

#### Option 2: Warp-Level Primitives

**Concept:** Use warp shuffle operations to reorganize data

```cuda
// Each thread generates one character position for 32 consecutive words
// Use warp shuffle to exchange data between threads
// Write coalesced 32-byte chunks

char char_for_words[32];  // This thread's character for 32 words

// Generate characters
for (int i = 0; i < 32; i++) {
    char_for_words[i] = generate_char_for_word(warp_base_idx + i, pos);
}

// Shuffle to reorganize for coalesced writes
// Thread 0 gets word 0 complete, thread 1 gets word 1 complete, etc.
char my_word[WORD_LENGTH];
for (int pos = 0; pos < WORD_LENGTH; pos++) {
    my_word[pos] = __shfl_sync(0xFFFFFFFF, char_for_words[lane_id], pos);
}

// Coalesced write (thread 0 writes word 0, thread 1 writes word 1, etc.)
memcpy(output + threadIdx.x * WORD_LENGTH, my_word, WORD_LENGTH);
```

**Pros:**
- No shared memory usage
- Very fast shuffle operations
- No synchronization barriers

**Cons:**
- Complex algorithm restructuring
- Limited to warp size (32 threads)
- May increase register pressure
- Unclear if net benefit

**Next Step:** Analyze register usage and prototype

#### Option 3: GPU-Based Transpose

**Concept:** Column-major kernel + GPU transpose kernel (eliminate CPU bottleneck)

```cuda
// Kernel 1: Generate in column-major (coalesced writes)
__global__ void generate_columnmajor(...) {
    // Same as Phase 3 Session 4 kernel
}

// Kernel 2: Transpose on GPU (use shared memory tiling)
__global__ void transpose_kernel(
    const char* column_major,
    char* row_major,
    int num_words,
    int word_length
) {
    __shared__ char tile[TILE_SIZE][TILE_SIZE + 1];  // +1 avoids bank conflicts

    // Tiled transpose using shared memory
    // Well-known CUDA pattern, highly optimized
}
```

**Pros:**
- GPU transpose: ~500 GB/s (vs CPU: 1.84 GB/s)
- Eliminates CPU bottleneck from Phase 3 Session 4
- Column-major kernel already implemented
- Transpose is parallelizable and well-studied

**Cons:**
- Two kernel launches (overhead)
- Requires extra GPU memory (temporary buffer)
- Transpose overhead may still be significant

**Next Step:** Implement and benchmark

#### Option 4: Alternative Algorithm

**Concept:** Fundamentally different generation approach

**Ideas:**
- Generate in sorted order (different from index mapping)
- Use different mixed-radix representation
- Parallelize at different granularity (word-level vs character-level)

**Status:** Open research question

### Research Approach

**Phase 1: Lightweight Prototyping (1-2 weeks)**
- Implement minimal versions of Options 1-3
- Benchmark against baseline
- Profile with Nsight Compute
- Identify most promising direction

**Phase 2: Deep Dive (2-4 weeks)**
- Fully implement best approach
- Optimize based on profiling
- Comprehensive testing
- Document findings

**Phase 3: Decision Point**
- If >50% speedup achieved: Integrate into v1.2.0
- If <50% speedup: Document findings, accept 440 M/s as near-optimal
- Either outcome is valuable (proven optimization or proven limit)

### Community Involvement

**After v1.1.0 release:**
- Share profiling results and research directions
- Ask for community input on approaches
- Invite collaborators with optimization expertise
- May discover users with insights or different hardware profiles

---

## Timeline

```
Month 1-1.5:  v1.1.0 Multi-GPU Implementation
Month 1.5-2:  v1.1.0 Testing & Release
Month 2-3:    v1.2.0 Single-GPU Research (Phase 1: Prototyping)
Month 3-4:    v1.2.0 Implementation (if promising) or Documentation (if limit)
Month 4:      v1.2.0 Release OR v1.3.0 Planning (OpenCL)
```

**Flexible timeline** - v1.2.0 research benefits from not being rushed

---

## Success Metrics

### v1.1.0 Multi-GPU
- **Performance:** 90-95% scaling efficiency
- **Correctness:** 100% match with single-GPU
- **Stability:** Zero crashes in 24h stress test
- **Usability:** Clear documentation, working examples

### v1.2.0 Single-GPU
- **Best case:** >50% speedup (660+ M/s) with proven optimization
- **Good case:** 20-50% speedup (528-660 M/s) with documented approach
- **Acceptable case:** <20% speedup but proven that 440 M/s is near-optimal
- **All outcomes valuable** - Either faster code or proven limits

---

## Risk Mitigation

### Multi-GPU Risks
- **Can't test with multiple GPUs locally**
  - Mitigation: Comprehensive unit tests, community beta testing

- **API complexity**
  - Mitigation: Design review before implementation, clear examples

- **Performance overhead**
  - Mitigation: Benchmark each component, optimize critical paths

### Single-GPU Risks
- **Research may not yield improvements**
  - Mitigation: Accept that current performance may be optimal, document findings

- **Time investment without guaranteed outcome**
  - Mitigation: Set clear research milestones, be willing to conclude "limit reached"

- **Chasing diminishing returns**
  - Mitigation: Focus on breakthrough approaches, not incremental tweaks

---

## Key Principles

1. **Ship value early** - Multi-GPU helps users now
2. **Research needs time** - Don't rush single-GPU optimization
3. **Measure everything** - Profile before and after every change
4. **Community as resource** - Users may provide insights
5. **Document outcomes** - Both successes and proven limits are valuable

---

## References

- Multi-GPU Design: `docs/design/MULTI_GPU_DESIGN.md`
- Profiling Results: `docs/benchmarking/NSIGHT_COMPUTE_PROFILE_2025-11-22.md`
- Failed Attempts: `docs/archive/PHASE3_SESSION4_SUMMARY.md`
- CUDA Optimization Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

*Strategy approved: November 22, 2025*
*Author: AI-guided development (with human oversight)*
