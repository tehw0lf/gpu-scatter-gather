# Next Session: C API Phase 2 - Device Pointer Implementation

**Quick Start**: Implement zero-copy GPU operation for maximum performance

---

## TL;DR

Implement **device pointer API** to eliminate PCIe bottleneck and achieve 800-1200 M words/s.

**Current State**: Phase 1 complete - host memory API working (440 M words/s)

**Goal**: Zero-copy GPU operation (800-1200 M words/s, 2-3x improvement)

---

## What Phase 1 Delivered

✅ **Core C FFI Layer Complete**:
- 8 C API functions for host-side generation
- Automatic header generation with cbindgen
- Thread-local error handling
- Full input validation and safety guarantees
- All integration tests passing

**Files Created**:
- `src/ffi.rs` (350 lines) - Core FFI implementation
- `cbindgen.toml` - Header generation config
- `include/wordlist_generator.h` - Auto-generated C header
- `tests/ffi_basic_test.c` - C integration tests
- `docs/PHASE1_SUMMARY.md` - Complete documentation

**Current Performance**: ~440 M words/s (PCIe bottleneck)

---

## Phase 2 Objectives

### 1. Device Batch Structure (30 min)

Add to `src/ffi.rs`:

```rust
/// Device batch result (exposed to C)
#[repr(C)]
pub struct BatchDevice {
    /// Device pointer to candidates
    pub data: u64,  // CUdeviceptr
    /// Number of candidates
    pub count: u64,
    /// Length of each word (chars)
    pub word_length: usize,
    /// Bytes between word starts
    pub stride: usize,
    /// Total buffer size
    pub total_bytes: usize,
    /// Output format used
    pub format: i32,
}
```

### 2. Device Generation Functions (2-3h)

```rust
/// Generate batch in GPU memory (zero-copy)
#[no_mangle]
pub extern "C" fn wg_generate_batch_device(
    gen: *mut WordlistGenerator,
    start_idx: u64,
    count: u64,
    batch: *mut BatchDevice,
) -> i32 {
    // Validate inputs
    // Generate to GPU memory (no host transfer)
    // Fill BatchDevice struct
    // Store device pointer in internal state
    // Return WG_SUCCESS or error
}

/// Free device batch (optional early cleanup)
#[no_mangle]
pub extern "C" fn wg_free_batch_device(
    gen: *mut WordlistGenerator,
    batch: *mut BatchDevice,
) {
    // Free GPU memory
    // Set batch->data = 0
}
```

### 3. Internal State Updates (1h)

Update `GeneratorInternal`:

```rust
struct GeneratorInternal {
    gpu: GpuContext,
    charsets: HashMap<usize, Vec<u8>>,
    mask: Option<Vec<usize>>,
    current_batch: Option<CUdeviceptr>,  // Track active device memory
}

impl GeneratorInternal {
    fn free_current_batch(&mut self) {
        if let Some(ptr) = self.current_batch.take() {
            unsafe { cuMemFree_v2(ptr); }
        }
    }
}
```

### 4. External CUDA Context Support (1h)

Update `wg_create()` to accept user-provided CUDA context:

```rust
#[no_mangle]
pub extern "C" fn wg_create(
    ctx: CUcontext,  // User's context or NULL
    device_id: i32,
) -> *mut WordlistGenerator {
    // If ctx != NULL: use it
    // If ctx == NULL: create own context
    // Track ownership for cleanup
}
```

### 5. C Integration Tests (1h)

Add to `tests/ffi_basic_test.c`:

```c
void test_device_generation() {
    wg_handle_t gen = wg_create(NULL, 0);
    wg_set_charset(gen, 1, "abc", 3);
    int mask[] = {1, 1, 1};
    wg_set_mask(gen, mask, 3);

    wg_batch_device_t batch;
    int result = wg_generate_batch_device(gen, 0, 1000, &batch);
    assert(result == WG_SUCCESS);
    assert(batch.data != 0);  // Valid device pointer
    assert(batch.count == 1000);

    // Optional: Copy back and verify
    char* host_buffer = malloc(batch.total_bytes);
    cuMemcpyDtoH(host_buffer, batch.data, batch.total_bytes);
    // Verify contents...
    free(host_buffer);

    wg_free_batch_device(gen, &batch);
    wg_destroy(gen);
}
```

### 6. Documentation (30 min)

Update `docs/PHASE1_SUMMARY.md` → `docs/C_API_SUMMARY.md`:
- Document Phase 2 additions
- Zero-copy usage examples
- Performance comparison
- Device pointer lifetime rules

---

## Key Implementation Notes

### Device Pointer Lifetime

```c
// Pattern 1: Auto-free on next generation
wg_batch_device_t batch1, batch2;
wg_generate_batch_device(gen, 0, 1000, &batch1);
// batch1.data is valid

wg_generate_batch_device(gen, 1000, 1000, &batch2);
// batch1.data is now INVALID (freed automatically)
// batch2.data is valid

wg_destroy(gen);
// batch2.data is now INVALID
```

```c
// Pattern 2: Explicit early free
wg_batch_device_t batch;
wg_generate_batch_device(gen, 0, 1000, &batch);
// Use batch...
wg_free_batch_device(gen, &batch);
// batch.data is now INVALID (freed early)
```

### Integration with Hash Kernels

```c
// Example: hashcat-style zero-copy pipeline
wg_batch_device_t batch;
wg_generate_batch_device(gen, 0, 100000000, &batch);

// Use device pointer directly in hash kernel
md5_hash_kernel<<<grid, block>>>(
    (const char*)batch.data,
    batch.stride,
    batch.count,
    d_hashes_out
);

cuStreamSynchronize(stream);
```

---

## Success Criteria

- ✅ `wg_generate_batch_device()` generates to GPU memory
- ✅ No PCIe transfer (verified with nvprof)
- ✅ Device pointer valid until next generation or wg_destroy()
- ✅ All C tests pass
- ✅ Performance: 800-1200 M words/s (2-3x improvement)

---

## Testing Strategy

1. **Unit Tests**: Verify device pointer generation
2. **Memory Leak Tests**: Run with `cuda-memcheck`
3. **Performance Tests**: Benchmark vs Phase 1
4. **Integration Tests**: Simple hash kernel that consumes device pointer

---

## Expected Timeline

- Device batch structure: 30 min
- Device generation functions: 2-3 hours
- Internal state updates: 1 hour
- External context support: 1 hour
- C integration tests: 1 hour
- Documentation: 30 min
- **Total**: 5-7 hours (one focused session)

---

## What Success Looks Like

```
Phase 1 Baseline: 440 M words/s
Phase 2 Result:   1000 M words/s (+2.3x)

nvprof Analysis:
  No PCIe transfers detected (100% GPU-side)
  Memory bandwidth: 85-95% utilization
  Kernel time: Same as Phase 1
  End-to-end time: 2-3x faster (no host copy)

Conclusion: PCIe bottleneck eliminated! 🚀
```

---

## Files to Modify

**Core Implementation**:
- `src/ffi.rs` - Add device pointer functions
- `src/gpu/mod.rs` - Update to support device pointer return (if needed)

**Testing**:
- `tests/ffi_basic_test.c` - Add device pointer tests

**Documentation**:
- `docs/C_API_SUMMARY.md` - Update with Phase 2 features
- `docs/PHASE2_SUMMARY.md` - Create implementation summary

**Build**:
- No changes needed (cbindgen handles new structs automatically)

---

## Quick Reminders

1. **Memory Management**: Track device pointers, auto-free on next generation
2. **Safety**: Validate batch pointer before dereferencing
3. **Error Handling**: Return error codes, set thread-local messages
4. **Documentation**: Update header comments for cbindgen
5. **Testing**: Verify zero-copy with nvprof/Nsight

---

## Start Here

```bash
# 1. Review Phase 1 implementation
cat src/ffi.rs
cat docs/PHASE1_SUMMARY.md

# 2. Check current performance
./test_ffi

# 3. Add BatchDevice struct
vim src/ffi.rs

# 4. Implement wg_generate_batch_device()
# 5. Add tests
# 6. Benchmark

# 7. Profile with nvprof
nvprof --print-gpu-trace ./test_ffi_device

# Expected: No HtoD/DtoH transfers!
```

---

## After Phase 2

**Remaining Phases**:
- **Phase 3**: Output format modes (3-4 hours)
- **Phase 4**: Streaming API (2-3 hours)
- **Phase 5**: Utility functions (2-3 hours)

**Total Remaining**: 12-16 hours (2-3 more sessions)

---

**Let's eliminate that PCIe bottleneck! 🔥**
