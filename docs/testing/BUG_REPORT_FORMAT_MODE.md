# Bug Report: Output Format Mode Buffer Overrun

**Date Discovered**: November 20, 2025
**Severity**: CRITICAL
**Status**: IDENTIFIED - Fix documented in NEXT_SESSION_PROMPT.md
**Affects**: Phase 3 (Output Format Modes) - PACKED and FIXED_WIDTH formats

---

## Summary

Output format modes PACKED and FIXED_WIDTH cause memory corruption and crashes due to buffer overrun in the host API. The GPU kernel always generates data with newlines regardless of the format mode setting, causing a mismatch between calculated buffer size and actual data size.

---

## Impact

**Affected APIs:**
- ❌ `wg_generate_batch_host()` with PACKED format → **CRASHES**
- ❌ `wg_generate_batch_host()` with FIXED_WIDTH format → **LIKELY CRASHES** (untested)
- ✅ `wg_generate_batch_host()` with NEWLINES format → Works correctly
- ✅ `wg_generate_batch_device()` (all formats) → Works correctly (doesn't use host buffer)

**Production Impact:**
- **Blocks deployment** - Cannot use memory-saving PACKED format
- **Silent data corruption** - Buffer overrun may corrupt adjacent memory
- **Crashes** - "corrupted double-linked list" / "free(): invalid next size"

---

## Root Cause

The output format feature was added to the FFI layer (Phase 3) but **never implemented in the GPU kernel**:

1. **FFI Layer** correctly calculates buffer size based on format:
   ```c
   // For PACKED format, 8-char words, 100 count:
   buffer_size = 8 * 100 = 800 bytes
   ```

2. **GPU Kernel** always generates with newlines:
   ```cuda
   output[output_idx + word_length] = '\n';  // Always adds newline!
   // Actual output: 9 * 100 = 900 bytes
   ```

3. **Result**: 100-byte buffer overrun → memory corruption

---

## Reproduction

### Minimal Test Case

```c
#include "wordlist_generator.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);

    wg_set_charset(gen, 1, "xyz", 3);
    int mask[] = {1, 1, 1, 1, 1, 1, 1, 1};  // 8 chars
    wg_set_mask(gen, mask, 8);

    wg_set_format(gen, wg_WG_FORMAT_PACKED);  // Set PACKED format

    size_t buffer_size = wg_calculate_buffer_size(gen, 1000);
    // Returns: 8000 bytes (8 chars * 1000 words)

    uint8_t* buffer = malloc(buffer_size);
    ssize_t bytes = wg_generate_batch_host(gen, 0, 1000, buffer, buffer_size);
    // Actually writes: 9000 bytes!
    // Result: 1000-byte buffer overrun → CRASH

    free(buffer);
    wg_destroy(gen);
    return 0;
}
```

**Compile:**
```bash
gcc -o test_crash test_crash.c -I. -L./target/release \
    -lgpu_scatter_gather -lcuda -Wl,-rpath,./target/release
```

**Run:**
```bash
./test_crash
# Output: "free(): invalid next size (normal)" + crash
```

---

## Evidence

### Test Output (Debug Version)

```
=== Test 1: DEFAULT format ===
Buffer size for 100 words (DEFAULT): 900 bytes
Expected: 900 bytes (100 * 9)
Generated 900 bytes  ✅ CORRECT

=== Test 2: PACKED format ===
Setting format to PACKED...
Buffer size for 100 words (PACKED): 800 bytes
Expected: 800 bytes (100 * 8)
Calling wg_generate_batch_host...
Generated 900 bytes  ❌ WRONG! Should be 800!
ERROR: Buffer overrun detected at offset 800!
Expected 0xAA, got 0x0A (newline character)
```

### Stack Trace (GDB)

```
Thread 1 "test_simple_v2" received signal SIGABRT, Aborted.
corrupted double-linked list

#0  0x00007ffff1e9890c in ?? () from /usr/lib/libc.so.6
#1  0x00007ffff1e3e3a0 in raise () from /usr/lib/libc.so.6
#2  0x00007ffff1e2557a in abort () from /usr/lib/libc.so.6
#9  0x00007ffff7f620d7 in wg_generate_batch_host ()
    from ./target/release/libgpu_scatter_gather.so
#10 0x0000555555555705 in test_format_modes ()
#11 0x0000555555555c60 in main ()
```

---

## Technical Details

### Code Flow

1. **User sets PACKED format:**
   ```c
   wg_set_format(gen, wg_WG_FORMAT_PACKED);  // Sets internal.output_format = 2
   ```

2. **User calculates buffer size:**
   ```c
   size_t size = wg_calculate_buffer_size(gen, 100);
   // Correctly returns: 8 * 100 = 800 bytes
   ```

3. **User generates data:**
   ```c
   wg_generate_batch_host(gen, 0, 100, buffer, 800);
   ```

4. **Inside library (`src/ffi.rs`):**
   ```rust
   // ❌ BUG: Doesn't pass format to GPU!
   let result = internal.gpu.generate_batch(
       &charsets_map,
       mask,
       start_idx,
       count,
       // Missing: internal.output_format
   );
   ```

5. **Inside GPU code (`src/gpu/mod.rs`):**
   ```rust
   // ❌ BUG: Hardcodes newline
   let output_size = batch_size * (word_length + 1);  // Always +1!
   ```

6. **Inside CUDA kernel (`kernels/wordlist_poc.cu`):**
   ```cuda
   // ❌ BUG: Always writes newline
   output[output_idx + word_length] = '\n';
   ```

7. **Result:** Writes 900 bytes into 800-byte buffer → CRASH

---

## Fix Strategy

### Option A: Proper GPU Kernel Fix (Recommended)

**Pros:**
- Optimal performance (11% less bandwidth, 11% less memory)
- Clean implementation
- No CPU overhead

**Cons:**
- Requires changes to Rust + CUDA code
- Need to test all 3 kernel variants

**Implementation:** See `docs/NEXT_SESSION_PROMPT.md` for detailed steps

**Estimated Time:** 30-45 minutes

### Option B: CPU Post-Processing Workaround

**Pros:**
- Quick fix (5-10 minutes)
- Minimal code changes

**Cons:**
- Suboptimal performance (wastes PCIe bandwidth)
- Extra CPU work
- Doesn't fix root cause

**Not recommended for production**

---

## Affected Code Locations

### Rust (`src/ffi.rs`)
**Line ~449**: `wg_generate_batch_host()` doesn't pass format to GPU
```rust
let result = internal.gpu.generate_batch(
    &charsets_map,
    mask,
    start_idx,
    count,
    // MISSING: internal.output_format,
);
```

### Rust (`src/gpu/mod.rs`)
**Line ~15**: `generate_batch()` doesn't accept format parameter
**Line ~36**: `generate_batch_internal()` hardcodes newline in size calculation
```rust
let output_size = batch_size as usize * (word_length as usize + 1); // BUG!
```

### CUDA (`kernels/wordlist_poc.cu`)
**Line ~50** (3 places): All kernels hardcode newline write
```cuda
output[output_idx + word_length] = '\n';  // BUG!
```

---

## Workaround for Users

**Until fix is deployed, only use DEFAULT format:**

```c
// ✅ SAFE:
wg_set_format(gen, wg_WG_FORMAT_NEWLINES);  // or don't call wg_set_format at all

// ❌ UNSAFE:
wg_set_format(gen, wg_WG_FORMAT_PACKED);     // WILL CRASH!
wg_set_format(gen, wg_WG_FORMAT_FIXED_WIDTH); // LIKELY CRASHES!
```

**Or use device API (not affected):**

```c
// ✅ SAFE (device API doesn't have this bug):
struct wg_BatchDevice batch;
wg_set_format(gen, wg_WG_FORMAT_PACKED);  // OK with device API
wg_generate_batch_device(gen, 0, 1000, &batch);  // No crash
```

---

## Test Files

| File | Purpose | Status |
|------|---------|--------|
| `tests/ffi_test_host_packed.c` | Isolated reproduction | ❌ Crashes |
| `tests/ffi_test_host_packed_debug.c` | Buffer overrun detection | ❌ Shows overrun |
| `tests/ffi_integration_simple.c` | Full integration test | ❌ Crashes in test 3 |
| `tests/ffi_integration_minimal.c` | Working baseline | ✅ Passes (no format modes) |

---

## Verification After Fix

**Run these tests to verify fix:**

```bash
# 1. Isolated PACKED test
./test_host_packed
# Expected: PASS (no crash)

# 2. Debug test with buffer guards
./test_host_packed_debug
# Expected: No buffer overrun detected

# 3. Full integration tests
./test_ffi_integration_simple
# Expected: All 5 tests pass

# 4. Basic tests (should still pass)
./test_ffi
# Expected: 16/16 tests pass
```

---

## Performance Impact of Fix

### Current (Broken State with DEFAULT format)
- NEWLINES: 900 bytes → Works
- PACKED: Crashes (unusable)
- Effective bandwidth: 900 bytes per 100 words

### After Fix
- NEWLINES: 900 bytes
- PACKED: 800 bytes (11.1% reduction)
- FIXED_WIDTH: 900 bytes (with \0 padding)

**Expected improvement with PACKED format:**
- 11.1% less PCIe bandwidth
- 11.1% less GPU memory
- ~11% throughput increase (PCIe bottleneck)
- Estimated: 440 → 490 M words/s

---

## Related Issues

**None** - This is the first report of this bug

**Introduced in:** Phase 3 implementation (format modes added without kernel support)

**Discovered by:** Integration testing session (November 20, 2025)

---

## References

- **Fix Documentation:** `docs/NEXT_SESSION_PROMPT.md`
- **Integration Test Report:** `docs/testing/INTEGRATION_TEST_REPORT.md`
- **Phase 3 Specification:** `docs/api/PHASE3_SUMMARY.md`
- **Development Log:** `docs/development/DEVELOPMENT_LOG.md`

---

*Bug Report Created: November 20, 2025*
*Severity: CRITICAL*
*Priority: HIGH*
