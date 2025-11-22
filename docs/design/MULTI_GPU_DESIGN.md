# Multi-GPU Support Design Document

**Status:** Design Phase
**Target Version:** v1.1.0+
**Expected Performance:** Linear scaling with GPU count

---

## Table of Contents

- [Overview](#overview)
- [Current Architecture](#current-architecture)
- [Multi-GPU Architecture](#multi-gpu-architecture)
- [Keyspace Distribution Strategy](#keyspace-distribution-strategy)
- [API Design](#api-design)
- [Implementation Plan](#implementation-plan)
- [Performance Analysis](#performance-analysis)
- [Testing Strategy](#testing-strategy)
- [Alternative Approaches](#alternative-approaches)

---

## Overview

### Problem Statement

The current implementation (`v1.0.0`) uses a single GPU and achieves 440-700 M words/s. For very large keyspaces (e.g., 16+ character passwords), generation can still take considerable time. Multi-GPU support would provide:

1. **Linear scaling** - 2 GPUs → 880-1400 M words/s, 4 GPUs → 1760-2800 M words/s
2. **Reduced time-to-completion** - Critical for large keyspace scenarios
3. **Better hardware utilization** - Use all available GPUs in system
4. **Cost efficiency** - Faster results without upgrading to higher-end single GPU

### Goals

- ✅ **Linear scaling** - Each GPU contributes proportionally
- ✅ **Load balancing** - Equal work distribution across GPUs
- ✅ **Fault tolerance** - Graceful degradation if one GPU fails
- ✅ **Zero overhead** - No performance loss vs single-GPU when using one device
- ✅ **Simple API** - Minimal changes to existing code

### Non-Goals (v1.1.0)

- ❌ Multi-node distributed generation (future: v2.0.0+)
- ❌ Dynamic GPU addition/removal during generation
- ❌ GPU affinity/pinning to CPU cores
- ❌ NVLink-specific optimizations (use standard CUDA APIs)

---

## Current Architecture

### Single-GPU Flow

```
1. Initialize GpuContext (device 0)
2. Allocate device memory
3. Upload charset/mask data
4. For each batch:
   a. Launch kernel on device 0
   b. Wait for completion
   c. Copy results to host
5. Cleanup device memory
```

### Limitations

- **Single device utilization** - Other GPUs sit idle
- **Sequential processing** - No parallelism across devices
- **Hard-coded device 0** - No device selection mechanism

---

## Multi-GPU Architecture

### Design Principles

1. **Keyspace partitioning** - Divide total keyspace across GPUs
2. **Independent contexts** - Each GPU has own CUDA context
3. **Asynchronous generation** - Overlap compute on multiple GPUs
4. **Result aggregation** - Combine outputs in order

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│           MultiGpuWordlistGenerator                      │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ GPU 0      │  │ GPU 1      │  │ GPU N-1    │        │
│  │ Context    │  │ Context    │  │ Context    │        │
│  │ Thread     │  │ Thread     │  │ Thread     │        │
│  └────┬───────┘  └────┬───────┘  └────┬───────┘        │
│       │               │               │                 │
│       ├───────────────┴───────────────┤                 │
│       │     Keyspace Partitioner      │                 │
│       └───────────────┬───────────────┘                 │
│                       │                                 │
│            ┌──────────▼──────────┐                      │
│            │  Output Aggregator  │                      │
│            └─────────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### Component Diagram

```
MultiGpuContext
│
├── Vec<GpuWorker>
│   │
│   ├── GpuWorker(device_id: 0)
│   │   ├── CUcontext
│   │   ├── CUstream
│   │   └── GpuContext (existing)
│   │
│   ├── GpuWorker(device_id: 1)
│   │   ├── CUcontext
│   │   ├── CUstream
│   │   └── GpuContext (existing)
│   │
│   └── ...
│
├── KeyspacePartitioner
│   ├── total_keyspace: u64
│   ├── num_devices: usize
│   └── partition(device_id) -> (start_idx, count)
│
└── OutputAggregator
    ├── device_buffers: Vec<Vec<u8>>
    └── merge() -> Vec<u8>
```

---

## Keyspace Distribution Strategy

### Option 1: Static Partitioning (Recommended for v1.1.0)

**Strategy:** Divide keyspace evenly across GPUs at generation start.

**Algorithm:**
```rust
fn partition_keyspace(total: u64, num_gpus: usize) -> Vec<(u64, u64)> {
    let chunk_size = total / num_gpus as u64;
    let remainder = total % num_gpus as u64;

    let mut partitions = Vec::new();
    let mut start_idx = 0;

    for gpu_id in 0..num_gpus {
        // Give remainder to first GPU (load balancing)
        let count = if gpu_id == 0 {
            chunk_size + remainder
        } else {
            chunk_size
        };

        partitions.push((start_idx, count));
        start_idx += count;
    }

    partitions
}
```

**Example:**
```
Keyspace: 1,000,000 words
GPUs: 3

GPU 0: indices [0, 333,334)    (333,334 words)
GPU 1: indices [333,334, 666,667)  (333,333 words)
GPU 2: indices [666,667, 1,000,000) (333,333 words)
```

**Pros:**
- ✅ Simple implementation
- ✅ Predictable performance
- ✅ No synchronization overhead
- ✅ Works for all keyspace sizes

**Cons:**
- ❌ No dynamic load balancing (if GPUs have different speeds)
- ❌ Wasted work if generation stops early

### Option 2: Dynamic Work Queue (Future: v1.2.0+)

**Strategy:** GPUs pull batches from shared work queue.

**Pros:**
- ✅ Automatic load balancing (faster GPUs get more work)
- ✅ Handles heterogeneous GPU configurations
- ✅ Efficient for early termination (e.g., found target hash)

**Cons:**
- ❌ More complex synchronization
- ❌ Potential queue contention overhead
- ❌ Harder to guarantee output order

**Recommendation:** Start with **static partitioning** for simplicity, migrate to dynamic work queue if needed.

---

## API Design

### C FFI API

#### Device Enumeration

```c
/**
 * Get number of available CUDA devices
 *
 * Returns: Number of CUDA devices, or -1 on error
 */
int wg_get_device_count(void);

/**
 * Get device information
 *
 * @param device_id Device index (0 to wg_get_device_count() - 1)
 * @param name_out Buffer for device name (at least 256 bytes)
 * @param compute_cap_major_out Output for major compute capability
 * @param compute_cap_minor_out Output for minor compute capability
 * @param total_memory_out Output for total device memory in bytes
 *
 * Returns: WG_SUCCESS or error code
 */
int wg_get_device_info(
    int device_id,
    char* name_out,
    int* compute_cap_major_out,
    int* compute_cap_minor_out,
    uint64_t* total_memory_out
);
```

#### Multi-GPU Generator

```c
/**
 * Create multi-GPU generator using all available devices
 *
 * @param charsets Array of character sets
 * @param num_charsets Number of character sets
 * @param mask Mask pattern (e.g., "?1?2?3")
 *
 * Returns: Opaque handle or NULL on error
 */
WG_MultiGpuGenerator* wg_multigpu_create(
    const WG_Charset* charsets,
    int num_charsets,
    const char* mask
);

/**
 * Create multi-GPU generator using specific devices
 *
 * @param device_ids Array of device IDs to use
 * @param num_devices Number of devices
 * @param charsets Array of character sets
 * @param num_charsets Number of character sets
 * @param mask Mask pattern
 *
 * Returns: Opaque handle or NULL on error
 */
WG_MultiGpuGenerator* wg_multigpu_create_with_devices(
    const int* device_ids,
    int num_devices,
    const WG_Charset* charsets,
    int num_charsets,
    const char* mask
);

/**
 * Generate batch across all GPUs
 *
 * @param gen Multi-GPU generator handle
 * @param start_idx Starting index in keyspace
 * @param batch_size Total number of words to generate (distributed across GPUs)
 * @param format Output format (WG_FORMAT_NEWLINES, etc.)
 * @param output_ptr Output buffer pointer
 * @param output_size_out Actual output size in bytes
 *
 * Returns: WG_SUCCESS or error code
 */
int wg_multigpu_generate(
    WG_MultiGpuGenerator* gen,
    uint64_t start_idx,
    uint64_t batch_size,
    int format,
    uint8_t** output_ptr,
    size_t* output_size_out
);

/**
 * Get performance statistics
 *
 * @param gen Multi-GPU generator handle
 * @param stats_out Output statistics structure
 *
 * Returns: WG_SUCCESS or error code
 */
int wg_multigpu_get_stats(
    WG_MultiGpuGenerator* gen,
    WG_MultiGpuStats* stats_out
);

/**
 * Destroy multi-GPU generator
 *
 * @param gen Multi-GPU generator handle
 */
void wg_multigpu_destroy(WG_MultiGpuGenerator* gen);
```

#### Statistics Structure

```c
typedef struct {
    int num_devices;                // Number of GPUs used
    uint64_t total_generated;       // Total words generated
    uint64_t* per_device_generated; // Words generated per device
    double* per_device_throughput;  // M words/s per device
    double total_throughput;        // Aggregate M words/s
    double generation_time_ms;      // Total generation time
} WG_MultiGpuStats;
```

### Rust API

```rust
pub struct MultiGpuContext {
    workers: Vec<GpuWorker>,
    num_devices: usize,
}

impl MultiGpuContext {
    /// Create multi-GPU context with all available devices
    pub fn new() -> Result<Self>;

    /// Create multi-GPU context with specific devices
    pub fn with_devices(device_ids: &[i32]) -> Result<Self>;

    /// Generate batch across all GPUs
    pub fn generate_batch(
        &self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
    ) -> Result<Vec<u8>>;

    /// Get performance statistics
    pub fn stats(&self) -> MultiGpuStats;
}

struct GpuWorker {
    device_id: i32,
    context: GpuContext,
    stream: CUstream,
    thread_handle: Option<std::thread::JoinHandle<Result<Vec<u8>>>>,
}
```

---

## Implementation Plan

### Phase 1: Device Enumeration (Week 1)

**Goal:** Add device discovery and selection.

**Tasks:**
1. Implement `wg_get_device_count()`
2. Implement `wg_get_device_info()`
3. Add unit tests for device enumeration
4. Document API in C_API_SPECIFICATION.md

**Deliverables:**
- Device enumeration FFI functions
- Tests passing on single-GPU and multi-GPU systems

### Phase 2: Multi-Context Management (Week 2)

**Goal:** Create multiple CUDA contexts for different devices.

**Tasks:**
1. Modify `GpuContext::new()` to accept device ID
2. Create `MultiGpuContext` wrapper
3. Implement context switching and isolation
4. Add error handling for context creation failures

**Deliverables:**
- `MultiGpuContext` struct
- Per-device context creation
- Thread-safe context management

### Phase 3: Keyspace Partitioning (Week 3)

**Goal:** Implement static keyspace distribution.

**Tasks:**
1. Implement `partition_keyspace()` algorithm
2. Add tests for various keyspace sizes and GPU counts
3. Handle edge cases (keyspace < num_gpus, etc.)
4. Document partitioning strategy

**Deliverables:**
- Keyspace partitioner with unit tests
- Documentation of distribution algorithm

### Phase 4: Parallel Generation (Week 4)

**Goal:** Launch kernels on multiple GPUs concurrently.

**Tasks:**
1. Implement per-GPU thread pool
2. Add async kernel launch with streams
3. Implement output aggregation
4. Add synchronization for completion

**Deliverables:**
- Concurrent multi-GPU generation
- Output merging in correct order
- Thread-safe worker management

### Phase 5: FFI Integration (Week 5)

**Goal:** Expose multi-GPU API to C.

**Tasks:**
1. Implement `wg_multigpu_create()`
2. Implement `wg_multigpu_generate()`
3. Add statistics collection
4. Create C integration tests

**Deliverables:**
- Complete multi-GPU FFI API
- C test program demonstrating multi-GPU usage
- Updated C_API_SPECIFICATION.md

### Phase 6: Optimization & Testing (Week 6)

**Goal:** Optimize performance and validate correctness.

**Tasks:**
1. Profile multi-GPU overhead
2. Optimize memory transfers (use pinned memory)
3. Add comprehensive integration tests
4. Benchmark 1-GPU vs 2-GPU vs 4-GPU scaling
5. Test on different GPU combinations

**Deliverables:**
- Performance benchmarks showing linear scaling
- Test suite covering multi-GPU scenarios
- Optimization documentation

---

## Performance Analysis

### Expected Scaling

**Ideal (100% efficiency):**
```
1 GPU:  440-700 M words/s
2 GPUs: 880-1400 M words/s  (2.0× speedup)
4 GPUs: 1760-2800 M words/s (4.0× speedup)
```

**Realistic (90-95% efficiency):**
```
1 GPU:  440-700 M words/s
2 GPUs: 792-1330 M words/s  (1.8-1.9× speedup)
4 GPUs: 1584-2660 M words/s (3.6-3.8× speedup)
```

### Overhead Sources

1. **Context switching** - ~1-2% overhead
2. **Output aggregation** - ~1-3% overhead (memory copy)
3. **Thread synchronization** - ~1% overhead
4. **Load imbalance** - ~2-5% overhead (static partitioning)

**Total expected overhead:** 5-11%

### Bottleneck Analysis

**Not bottlenecks:**
- ✅ Kernel compute (each GPU independent)
- ✅ Memory bandwidth (per-device)
- ✅ PCIe bandwidth (parallel devices)

**Potential bottlenecks:**
- ⚠️ CPU aggregation of results (if copying from all GPUs simultaneously)
- ⚠️ Memory allocation (if allocating large buffers)

**Mitigation:**
- Use pinned memory for faster host<->device transfers
- Pre-allocate buffers once
- Pipeline generation (generate batch N+1 while copying batch N)

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_partition_keyspace_even() {
    let partitions = partition_keyspace(1000, 4);
    assert_eq!(partitions.len(), 4);
    assert_eq!(partitions[0], (0, 250));
    assert_eq!(partitions[1], (250, 250));
    assert_eq!(partitions[2], (500, 250));
    assert_eq!(partitions[3], (750, 250));
}

#[test]
fn test_partition_keyspace_uneven() {
    let partitions = partition_keyspace(1001, 4);
    assert_eq!(partitions[0], (0, 251)); // Extra 1 to first GPU
    assert_eq!(partitions[1], (251, 250));
    assert_eq!(partitions[2], (501, 250));
    assert_eq!(partitions[3], (751, 250));
}

#[test]
fn test_multi_gpu_correctness() {
    let multi_ctx = MultiGpuContext::new().unwrap();
    let single_ctx = GpuContext::new().unwrap();

    // Generate same keyspace with both
    let multi_output = multi_ctx.generate_batch(...);
    let single_output = single_ctx.generate_batch(...);

    // Should produce identical results
    assert_eq!(multi_output, single_output);
}
```

### Integration Tests

```c
// Test multi-GPU generation produces correct output
void test_multigpu_correctness() {
    WG_MultiGpuGenerator* gen = wg_multigpu_create(...);

    uint8_t* output;
    size_t output_size;

    int result = wg_multigpu_generate(gen, 0, 1000000, WG_FORMAT_NEWLINES, &output, &output_size);

    assert(result == WG_SUCCESS);
    assert(output_size > 0);

    // Cross-validate with single-GPU generator
    WG_Generator* single_gen = wg_create(...);
    uint8_t* single_output;
    size_t single_output_size;

    wg_generate(single_gen, 0, 1000000, WG_FORMAT_NEWLINES, &single_output, &single_output_size);

    // Should match
    assert(output_size == single_output_size);
    assert(memcmp(output, single_output, output_size) == 0);

    wg_free_output(output);
    wg_free_output(single_output);
    wg_destroy(single_gen);
    wg_multigpu_destroy(gen);
}
```

### Performance Tests

```bash
# Benchmark scaling efficiency
for num_gpus in 1 2 4; do
    echo "Testing with $num_gpus GPUs"
    ./benchmark_multigpu --gpus $num_gpus --batch 100000000 > results_${num_gpus}gpu.txt
done

# Analyze scaling
python scripts/analyze_scaling.py results_*gpu.txt
```

---

## Alternative Approaches

### Approach 1: Thread Pool (Rejected)

**Idea:** Use Rust thread pool (rayon) for parallel GPU execution.

**Why rejected:**
- CUDA contexts are thread-local (complicates API)
- Thread pool overhead not needed (fixed number of GPUs)
- Manual threading gives more control

### Approach 2: Unified Memory (Considered for v2.0+)

**Idea:** Use CUDA Unified Memory to avoid explicit memory management.

**Why deferred:**
- Performance overhead on current generation GPUs
- Requires compute capability 6.0+
- Less control over memory placement
- May revisit for future optimization

### Approach 3: Peer-to-Peer (P2P) Memory Access (Future)

**Idea:** Enable direct GPU-to-GPU memory transfers (NVLink).

**Why future work:**
- Not all systems have NVLink
- Requires additional complexity
- Current design doesn't need inter-GPU communication
- May be useful for distributed password cracking (v2.0+)

---

## Open Questions

1. **How to handle heterogeneous GPU configurations?**
   - Different compute capabilities
   - Different memory sizes
   - Different performance tiers

   **Answer:** Static partitioning works for v1.1.0. For v1.2.0, implement dynamic work queue.

2. **Should we support mixing different GPU architectures (e.g., Turing + Ampere)?**

   **Answer:** Yes, use PTX for each GPU's architecture. Already supported by current build.rs.

3. **What if user has GPUs but not all support required compute capability?**

   **Answer:** Skip unsupported GPUs, log warning, continue with supported ones.

4. **Memory management - who owns output buffers?**

   **Answer:** Same as current API - caller owns after wg_multigpu_generate(), must free.

5. **Error handling - what if one GPU fails mid-generation?**

   **Answer:** v1.1.0 - fail entire batch. v1.2.0+ - implement fault tolerance.

---

## Success Criteria

### v1.1.0 Release

- ✅ Multi-GPU API complete and documented
- ✅ Linear scaling (90%+ efficiency) on 2-4 GPUs
- ✅ 100% correctness (matches single-GPU output)
- ✅ Zero performance regression for single-GPU use case
- ✅ Integration tests passing on multi-GPU systems
- ✅ Comprehensive documentation and examples

### Performance Targets

| GPUs | Target Throughput | Efficiency |
|------|-------------------|------------|
| 1    | 440-700 M/s       | 100% (baseline) |
| 2    | 792-1330 M/s      | 90-95% |
| 4    | 1584-2660 M/s     | 90-95% |

### Quality Targets

- ⚡ **Performance:** >90% scaling efficiency
- ✅ **Correctness:** 100% match with single-GPU
- 🔒 **Stability:** No crashes or memory leaks in 24hr stress test
- 📖 **Usability:** Simple API, clear examples
- 🧪 **Testing:** >90% code coverage

---

## References

- [CUDA Multi-GPU Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system)
- [CUDA Streams and Concurrency](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- [NVIDIA Multi-GPU Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#multi-gpu)

---

*Last Updated: November 22, 2025*
*Status: Design Phase*
*Target: v1.1.0*
