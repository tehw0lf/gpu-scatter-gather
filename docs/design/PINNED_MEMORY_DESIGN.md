# Pinned Memory Optimization Design

**Date:** November 23, 2025
**Version:** v1.4.0-dev
**Status:** In Progress
**Priority:** 4 (10-15% expected improvement)

---

## Problem Statement

**Current bottleneck:** PCIe memory transfers using pageable memory

**Impact:**
- `cuMemcpyDtoH` with pageable memory: ~12-16 GB/s (PCIe 4.0 x16 theoretical)
- Extra overhead: DMA engine must copy from pageable → pinned → GPU
- Worker threads allocate `Vec<u8>` per batch (pageable memory)

**Goal:** Use pinned (page-locked) memory for faster PCIe transfers

---

## Solution: Pinned Memory Buffers with Persistent Workers

### Key Insight

With v1.3.0's persistent worker threads, we can now:
1. Allocate pinned buffers **once** during initialization
2. Reuse buffers across all batches
3. Use `CU_MEMHOSTALLOC_PORTABLE` for multi-context access

### CUDA Pinned Memory API

```c
// Allocate pinned memory
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);

// Flags:
#define CU_MEMHOSTALLOC_PORTABLE      0x01  // Multi-context access
#define CU_MEMHOSTALLOC_DEVICEMAP     0x02  // Map to device address space
#define CU_MEMHOSTALLOC_WRITECOMBINED 0x04  // Write-combined (faster writes)

// Free pinned memory
CUresult cuMemFreeHost(void *p);
```

**Key flag: `CU_MEMHOSTALLOC_PORTABLE`**
- Allows memory to be accessible from **any** CUDA context
- Critical for multi-GPU where each worker has its own context
- Without this flag: segfault when accessing from different context

---

## Architecture

### Current (v1.3.0)

```rust
struct MultiGpuContext {
    workers: Vec<GpuWorker>,
    worker_threads: Option<Vec<(Sender<WorkerMessage>, JoinHandle<()>)>>,
}

fn process_work_item(...) -> Result<Vec<u8>> {
    let mut host_buffer = vec![0u8; output_size];  // ❌ Pageable memory
    cuMemcpyDtoH(host_buffer.as_mut_ptr(), device_ptr, size);  // ❌ Slow
    Ok(host_buffer)
}
```

### Proposed (v1.4.0)

```rust
struct MultiGpuContext {
    workers: Vec<GpuWorker>,
    worker_threads: Option<Vec<(Sender<WorkerMessage>, JoinHandle<()>)>>,
    pinned_buffers: Vec<PinnedBuffer>,  // ✅ NEW: One per worker
}

struct PinnedBuffer {
    ptr: *mut u8,
    size: usize,
}

impl PinnedBuffer {
    fn new(size: usize) -> Result<Self> {
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let result = cuMemHostAlloc(
                &mut ptr,
                size,
                CU_MEMHOSTALLOC_PORTABLE,  // ✅ Multi-context access
            );
            if result != CUDA_SUCCESS {
                bail!("Failed to allocate pinned memory");
            }
            Ok(Self { ptr: ptr as *mut u8, size })
        }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe { cuMemFreeHost(self.ptr as *mut std::ffi::c_void); }
    }
}

fn process_work_item(pinned_ptr: *mut u8, ...) -> Result<usize> {
    // ✅ Write directly to pinned memory
    cuMemcpyDtoHAsync(pinned_ptr as *mut c_void, device_ptr, size, stream);
    cuStreamSynchronize(stream);
    Ok(size)  // Return size, data already in pinned buffer
}
```

---

## Implementation Plan

### Phase 1: Add PinnedBuffer Type ✅

**File:** `src/multigpu.rs`

```rust
/// Pinned (page-locked) memory buffer for fast PCIe transfers
struct PinnedBuffer {
    ptr: *mut u8,
    size: usize,
}

impl PinnedBuffer {
    /// Allocate pinned memory with PORTABLE flag for multi-context access
    fn new(size: usize) -> Result<Self>;

    /// Get raw pointer (for passing to CUDA)
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&mut self) -> *mut u8;

    /// Get slice view
    unsafe fn as_slice(&self, len: usize) -> &[u8];
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe { cuMemFreeHost(self.ptr as *mut c_void); }
    }
}

unsafe impl Send for PinnedBuffer {}
// NOT Sync - each buffer owned by single worker
```

### Phase 2: Integrate into MultiGpuContext ✅

**Changes:**

1. **Add pinned_buffers field**
```rust
pub struct MultiGpuContext {
    workers: Vec<GpuWorker>,
    num_devices: usize,
    async_mode: bool,
    worker_threads: Option<Vec<(Sender<WorkerMessage>, JoinHandle<()>)>>,
    pinned_buffers: Vec<PinnedBuffer>,  // NEW
    max_buffer_size: usize,              // NEW
}
```

2. **Allocate in constructor**
```rust
fn new_with_options(async_mode: bool) -> Result<Self> {
    // ... existing worker creation ...

    // Allocate pinned buffers (one per worker)
    let max_buffer_size = 1_000_000_000;  // 1 GB per buffer (configurable)
    let pinned_buffers: Vec<PinnedBuffer> = (0..num_devices)
        .map(|_| PinnedBuffer::new(max_buffer_size))
        .collect::<Result<Vec<_>>>()?;

    Ok(Self {
        workers,
        num_devices,
        async_mode,
        worker_threads,
        pinned_buffers,
        max_buffer_size,
    })
}
```

3. **Pass pinned pointers to workers**
```rust
struct WorkItem {
    charsets: HashMap<usize, Vec<u8>>,
    mask: Vec<usize>,
    partition: KeyspacePartition,
    output_format: i32,
    pinned_ptr: *mut u8,              // NEW: Pinned buffer pointer
    result_sender: Sender<Result<usize>>,  // CHANGED: Return size, not Vec
}
```

### Phase 3: Update Worker Threads ✅

**Changes to `process_work_item`:**

```rust
fn process_work_item(
    gpu_ctx: &GpuContext,
    partition: KeyspacePartition,
    charsets: &HashMap<usize, Vec<u8>>,
    mask: &[usize],
    output_format: i32,
    stream: CUstream,
    pinned_ptr: *mut u8,  // NEW: Write directly here
) -> Result<usize> {
    unsafe {
        // Calculate output size
        let word_length = mask.len();
        let bytes_per_word = match output_format {
            0 => word_length + 1,
            1 => word_length + 1,
            2 => word_length,
            _ => word_length + 1,
        };
        let output_size = partition.count as usize * bytes_per_word;

        // Generate to device memory
        let (device_ptr, size) = gpu_ctx.generate_batch_device_stream(
            charsets, mask, partition.start_idx, partition.count,
            stream, output_format,
        )?;

        // Copy directly to pinned memory (FAST!)
        let copy_result = if !stream.is_null() {
            cuMemcpyDtoHAsync_v2(
                pinned_ptr as *mut c_void,  // ✅ Pinned memory
                device_ptr,
                size,
                stream,
            )
        } else {
            cuMemcpyDtoH_v2(
                pinned_ptr as *mut c_void,  // ✅ Pinned memory
                device_ptr,
                size,
            )
        };

        if copy_result != CUresult::CUDA_SUCCESS {
            let _ = cuMemFree_v2(device_ptr);
            bail!("Failed to copy to pinned memory: {:?}", copy_result);
        }

        // Synchronize
        if !stream.is_null() {
            let _ = cuStreamSynchronize(stream);
        } else {
            let _ = cuCtxSynchronize();
        }

        // Free device memory
        let _ = cuMemFree_v2(device_ptr);

        Ok(size)  // Return size written
    }
}
```

### Phase 4: Update generate_batch Methods ✅

**Changes:**

1. **Send pinned pointers with work items**
2. **Receive size instead of Vec<u8>**
3. **Copy from pinned buffers to final output**

```rust
pub fn generate_batch_sync(...) -> Result<Vec<u8>> {
    // Fast path: single GPU
    if self.num_devices == 1 {
        let pinned_ptr = self.pinned_buffers[0].as_mut_ptr();
        let size = self.workers[0].context().generate_to_pinned(
            charsets, mask, start_idx, batch_size, output_format, pinned_ptr
        )?;

        // Copy from pinned to Vec (one final copy)
        unsafe {
            let mut result = vec![0u8; size];
            std::ptr::copy_nonoverlapping(pinned_ptr, result.as_mut_ptr(), size);
            Ok(result)
        }
    } else {
        // Multi-GPU path: send work to threads
        let partitions = self.partition(start_idx, batch_size);
        let mut result_receivers = Vec::new();

        for (worker_id, partition) in partitions.iter().enumerate() {
            let (result_sender, result_receiver) = channel();
            let work_item = WorkItem {
                charsets: charsets.clone(),
                mask: mask.to_vec(),
                partition: *partition,
                output_format,
                pinned_ptr: self.pinned_buffers[worker_id].as_mut_ptr(),  // ✅
                result_sender,
            };

            self.worker_threads.as_ref().unwrap()[worker_id]
                .0.send(WorkerMessage::Work(work_item))?;
            result_receivers.push(result_receiver);
        }

        // Collect results
        let mut total_size = 0;
        for receiver in &result_receivers {
            total_size += receiver.recv()??;
        }

        // Concatenate from pinned buffers
        unsafe {
            let mut result = vec![0u8; total_size];
            let mut offset = 0;
            for (worker_id, size) in result_receivers.iter().enumerate() {
                let size = size.recv()??;
                std::ptr::copy_nonoverlapping(
                    self.pinned_buffers[worker_id].as_ptr(),
                    result.as_mut_ptr().add(offset),
                    size,
                );
                offset += size;
            }
            Ok(result)
        }
    }
}
```

---

## Expected Performance Impact

### PCIe Transfer Speed

| Memory Type | Bandwidth | Notes |
|-------------|-----------|-------|
| Pageable | ~6-8 GB/s | DMA copies pageable → pinned → GPU |
| Pinned | ~12-16 GB/s | Direct DMA transfer |

**Speedup:** 1.5-2× on memory transfers

### Overall Throughput Impact

Current bottleneck breakdown:
- Kernel execution: 60-70% of time
- Memory transfer: 30-40% of time

**Expected improvement:**
- Memory transfer: 2× faster → saves 15-20% total time
- **Overall:** +10-15% throughput ✅

---

## Memory Requirements

**Per-worker buffer size:** Configurable, default 1 GB

**For 8-char lowercase words:**
- 1 GB ÷ 9 bytes/word = 111M words per buffer
- Covers most realistic batch sizes

**For 2 GPUs:**
- 2 × 1 GB = 2 GB pinned memory
- Reasonable for systems with 16+ GB RAM

**Considerations:**
- Pinned memory reduces available system RAM for paging
- Too much can hurt system performance
- Make buffer size configurable via API

---

## Safety Considerations

### Multi-Context Access

✅ **CRITICAL:** Use `CU_MEMHOSTALLOC_PORTABLE`
- Allows access from any CUDA context
- Each worker has its own context
- Without this: segfault when worker accesses buffer

### Thread Safety

✅ **Each worker owns its buffer**
- No sharing between threads
- No synchronization needed
- PinnedBuffer is `Send`, not `Sync`

### Memory Leaks

✅ **RAII with Drop**
- PinnedBuffer::drop() calls cuMemFreeHost
- Automatic cleanup on MultiGpuContext::drop()

---

## Testing Plan

1. **Correctness:** All existing 48 tests must pass
2. **Performance:** Benchmark before/after
3. **Memory:** Verify no leaks with valgrind/cuda-memcheck
4. **Edge cases:**
   - Very large batches (exceed buffer size)
   - Multiple batch runs (buffer reuse)
   - Error paths (allocation failures)

---

## Rollout Plan

1. ✅ Design document (this file)
2. ⏳ Implement PinnedBuffer type
3. ⏳ Integrate into MultiGpuContext
4. ⏳ Update worker threads
5. ⏳ Update generate_batch methods
6. ⏳ Add configuration for buffer size
7. ⏳ Benchmark and validate
8. ⏳ Document in v1.4.0 release notes

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pinned memory allocation fails | High | Fallback to pageable memory with warning |
| Buffer too small for batch | High | Check size, allocate temp buffer if needed |
| Multi-context access segfault | Critical | Use CU_MEMHOSTALLOC_PORTABLE flag |
| System RAM exhaustion | Medium | Make buffer size configurable, reasonable defaults |

---

## Future Optimizations (Post-v1.4.0)

1. **Write-combined memory** (`CU_MEMHOSTALLOC_WRITECOMBINED`)
   - Faster host writes (device → host)
   - May provide additional 5-10%

2. **Dynamic buffer sizing**
   - Allocate based on typical batch size
   - Reallocate if needed (rare)

3. **Memory pooling**
   - Share buffers across multiple MultiGpuContext instances
   - Advanced use case

---

**Status:** Design complete, ready for implementation
**Next:** Implement PinnedBuffer type in src/multigpu.rs
