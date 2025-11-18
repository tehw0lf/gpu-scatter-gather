# FFI Implementation Guide: Rust to C API Layer

**Version**: 2.0
**Date**: November 18, 2025
**Purpose**: Guide for implementing safe and correct C FFI bindings for Rust GPU library

---

## Overview

This guide explains how to implement the C Foreign Function Interface (FFI) layer that exposes our Rust GPU wordlist generator as a C library.

**Key challenges**:
- Type safety across language boundaries
- Memory ownership and lifetime management
- Error propagation (Rust `Result` → C error codes)
- Thread safety guarantees
- ABI stability

---

## Table of Contents

1. [Rust-C Type Mappings](#rust-c-type-mappings)
2. [FFI Safety Rules](#ffi-safety-rules)
3. [Opaque Handle Pattern](#opaque-handle-pattern)
4. [Error Handling Strategy](#error-handling-strategy)
5. [Memory Management](#memory-management)
6. [Thread Safety](#thread-safety)
7. [Testing FFI Code](#testing-ffi-code)

---

## Rust-C Type Mappings

### Primitive Types

| Rust Type | C Type | Notes |
|-----------|--------|-------|
| `u8` | `uint8_t` | ✅ Direct mapping |
| `u32` | `uint32_t` | ✅ Direct mapping |
| `u64` | `uint64_t` | ✅ Direct mapping |
| `i32` | `int32_t` | ✅ Direct mapping |
| `i64` | `int64_t` | ✅ Direct mapping |
| `usize` | `size_t` | ✅ Direct mapping |
| `isize` | `ssize_t` | ✅ Direct mapping |
| `bool` | `int` | ⚠️ Use `0`/`1`, not `true`/`false` |
| `()` | `void` | ✅ Return type only |

### Pointer Types

| Rust Type | C Type | Safety |
|-----------|--------|--------|
| `*const T` | `const T*` | ⚠️ Caller guarantees validity |
| `*mut T` | `T*` | ⚠️ Caller guarantees validity |
| `&T` | Not exposed | ❌ Use raw pointers in FFI |
| `&mut T` | Not exposed | ❌ Use raw pointers in FFI |
| `Box<T>` | `T*` | ⚠️ Ownership transfer |

### String Types

| Rust Type | C Type | Notes |
|-----------|--------|-------|
| `String` | Not exposed | ❌ Owned, don't cross FFI boundary |
| `&str` | Not exposed | ❌ Borrowed, lifetime issues |
| `CString` | `char*` | ✅ Null-terminated, ownership rules apply |
| `*const c_char` | `const char*` | ✅ Borrowed C string |
| `*mut c_char` | `char*` | ⚠️ Mutable C string |

### Array Types

| Rust Type | C Type | Notes |
|-----------|--------|-------|
| `Vec<T>` | Not exposed | ❌ Owned, don't cross FFI boundary |
| `&[T]` | `const T*, size_t` | ✅ Pass pointer + length |
| `&mut [T]` | `T*, size_t` | ✅ Pass pointer + length |

### CUDA Types

| Rust Type | C Type | Notes |
|-----------|--------|-------|
| `CUcontext` | `CUcontext` | ✅ Opaque pointer from cuda.h |
| `CUdevice` | `CUdevice` | ✅ Integer type from cuda.h |
| `CUdeviceptr` | `CUdeviceptr` | ✅ 64-bit pointer from cuda.h |
| `CUstream` | `CUstream` | ✅ Opaque pointer from cuda.h |

---

## FFI Safety Rules

### Rule 1: No Panics Across FFI Boundary

```rust
// BAD - panic can unwind into C code (undefined behavior!)
#[no_mangle]
pub extern "C" fn bad_function(ptr: *const u8) {
    let slice = unsafe { std::slice::from_raw_parts(ptr, 100) };
    slice[1000]; // Panic! 💥
}

// GOOD - catch panic and return error code
#[no_mangle]
pub extern "C" fn good_function(ptr: *const u8) -> i32 {
    let result = std::panic::catch_unwind(|| {
        let slice = unsafe { std::slice::from_raw_parts(ptr, 100) };
        // ... safe operations ...
        0
    });

    result.unwrap_or(-1) // Return error code if panic
}
```

### Rule 2: Validate All Pointers

```rust
// BAD - no null check
#[no_mangle]
pub extern "C" fn bad_function(ptr: *const u8) {
    unsafe {
        let value = *ptr; // May segfault if ptr is null!
    }
}

// GOOD - check for null
#[no_mangle]
pub extern "C" fn good_function(ptr: *const u8) -> i32 {
    if ptr.is_null() {
        return -1; // Error: null pointer
    }

    unsafe {
        let value = *ptr;
        // ... use value ...
        0
    }
}
```

### Rule 3: Use `#[repr(C)]` for Structs

```rust
// BAD - Rust struct layout is undefined
pub struct BatchDevice {
    data: u64,
    count: u64,
    word_length: usize,
}

// GOOD - C-compatible layout
#[repr(C)]
pub struct BatchDevice {
    data: u64,
    count: u64,
    word_length: usize,
}
```

### Rule 4: Use `extern "C"` Calling Convention

```rust
// BAD - default Rust ABI (unstable)
pub fn my_function(x: i32) -> i32 {
    x + 1
}

// GOOD - C calling convention (stable ABI)
#[no_mangle]
pub extern "C" fn my_function(x: i32) -> i32 {
    x + 1
}
```

### Rule 5: Handle Ownership Explicitly

```rust
// BAD - unclear ownership
#[no_mangle]
pub extern "C" fn create_string() -> *mut c_char {
    let s = CString::new("hello").unwrap();
    s.as_ptr() as *mut c_char // Dangling pointer! s is dropped!
}

// GOOD - transfer ownership to caller
#[no_mangle]
pub extern "C" fn create_string() -> *mut c_char {
    let s = CString::new("hello").unwrap();
    s.into_raw() // Transfer ownership, caller must free
}

#[no_mangle]
pub extern "C" fn free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr); // Take ownership back and drop
        }
    }
}
```

---

## Opaque Handle Pattern

### Why Opaque Handles?

Instead of exposing Rust structs directly to C, we use **opaque handles** (pointers to incomplete types).

**Benefits**:
- Hide implementation details
- Prevent C code from accessing internals
- Allow Rust to manage memory
- Enable future changes without breaking ABI

### Implementation

```rust
// Rust side (src/ffi.rs)

/// Opaque handle to WordlistGenerator
/// This is NOT a real struct - it's a marker type
#[repr(C)]
pub struct WordlistGenerator {
    _private: [u8; 0], // Zero-sized, prevents construction
}

// Internal representation (not exposed to C)
struct GeneratorInternal {
    gpu: gpu::GpuContext,
    charsets: HashMap<usize, Vec<u8>>,
    mask: Option<Vec<usize>>,
    last_error: Option<String>,
}

/// Create generator (returns opaque handle)
#[no_mangle]
pub extern "C" fn wg_create(
    ctx: CUcontext,
    device_id: i32,
) -> *mut WordlistGenerator {
    // Create internal state
    let internal = Box::new(GeneratorInternal {
        gpu: GpuContext::new().unwrap(),
        charsets: HashMap::new(),
        mask: None,
        last_error: None,
    });

    // Convert to opaque pointer
    Box::into_raw(internal) as *mut WordlistGenerator
}

/// Destroy generator
#[no_mangle]
pub extern "C" fn wg_destroy(gen: *mut WordlistGenerator) {
    if gen.is_null() {
        return;
    }

    unsafe {
        // Convert back to Box and drop
        let _ = Box::from_raw(gen as *mut GeneratorInternal);
    }
}

/// Helper: Convert handle to internal reference
unsafe fn handle_to_internal<'a>(
    gen: *mut WordlistGenerator
) -> Option<&'a mut GeneratorInternal> {
    if gen.is_null() {
        return None;
    }
    Some(&mut *(gen as *mut GeneratorInternal))
}
```

```c
// C side (include/wordlist_generator.h)

// Forward declaration - opaque type
typedef struct WordlistGenerator WordlistGenerator;

// Users can only manipulate pointers to it
WordlistGenerator* wg_create(CUcontext ctx, int device_id);
void wg_destroy(WordlistGenerator* gen);
```

---

## Error Handling Strategy

### Approach: Error Codes + Thread-Local Message

```rust
use std::cell::RefCell;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = RefCell::new(None);
}

/// Set thread-local error message
fn set_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(msg);
    });
}

/// Get thread-local error message
#[no_mangle]
pub extern "C" fn wg_get_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        match e.borrow().as_ref() {
            Some(err) => {
                // SAFETY: String is valid for 'static lifetime in thread-local
                err.as_ptr() as *const c_char
            }
            None => std::ptr::null(),
        }
    })
}

/// Example function with error handling
#[no_mangle]
pub extern "C" fn wg_set_charset(
    gen: *mut WordlistGenerator,
    charset_id: i32,
    chars: *const c_char,
    len: usize,
) -> i32 {
    // Validate handle
    let internal = unsafe {
        match handle_to_internal(gen) {
            Some(g) => g,
            None => {
                set_error("Invalid generator handle".to_string());
                return -1; // WG_ERROR_INVALID_HANDLE
            }
        }
    };

    // Validate charset_id
    if charset_id <= 0 || charset_id > 255 {
        set_error(format!("Invalid charset_id: {}", charset_id));
        return -2; // WG_ERROR_INVALID_PARAM
    }

    // Validate pointer
    if chars.is_null() {
        set_error("Null charset pointer".to_string());
        return -2; // WG_ERROR_INVALID_PARAM
    }

    // Convert C string to Rust
    let charset_bytes = unsafe {
        std::slice::from_raw_parts(chars as *const u8, len)
    }.to_vec();

    // Store charset
    internal.charsets.insert(charset_id as usize, charset_bytes);

    0 // WG_SUCCESS
}
```

### Error Code Constants

```rust
// In src/ffi.rs

pub const WG_SUCCESS: i32 = 0;
pub const WG_ERROR_INVALID_HANDLE: i32 = -1;
pub const WG_ERROR_INVALID_PARAM: i32 = -2;
pub const WG_ERROR_CUDA: i32 = -3;
pub const WG_ERROR_OUT_OF_MEMORY: i32 = -4;
pub const WG_ERROR_NOT_CONFIGURED: i32 = -5;
pub const WG_ERROR_BUFFER_TOO_SMALL: i32 = -6;
pub const WG_ERROR_KEYSPACE_OVERFLOW: i32 = -7;
```

---

## Memory Management

### Device Memory Lifetime

```rust
struct GeneratorInternal {
    // ... other fields ...
    current_batch: Option<CUdeviceptr>,
}

impl GeneratorInternal {
    /// Free current batch if it exists
    fn free_current_batch(&mut self) {
        if let Some(ptr) = self.current_batch.take() {
            unsafe {
                cuda_driver_sys::cuMemFree_v2(ptr);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn wg_generate_batch_device(
    gen: *mut WordlistGenerator,
    start_idx: u64,
    count: u64,
    batch: *mut BatchDevice,
) -> i32 {
    let internal = unsafe {
        match handle_to_internal(gen) {
            Some(g) => g,
            None => return WG_ERROR_INVALID_HANDLE,
        }
    };

    // Free previous batch (if any)
    internal.free_current_batch();

    // Generate new batch
    let result = internal.gpu.generate_batch(...);

    match result {
        Ok(data) => {
            // Store device pointer
            internal.current_batch = Some(data.device_ptr);

            // Fill output struct
            unsafe {
                (*batch).data = data.device_ptr;
                (*batch).count = count;
                (*batch).word_length = data.word_length;
                (*batch).stride = data.stride;
            }

            WG_SUCCESS
        }
        Err(e) => {
            set_error(format!("Generation failed: {}", e));
            WG_ERROR_CUDA
        }
    }
}
```

### Host Memory Management

```rust
#[no_mangle]
pub extern "C" fn wg_generate_batch_host(
    gen: *mut WordlistGenerator,
    start_idx: u64,
    count: u64,
    output_buffer: *mut u8,
    buffer_size: usize,
) -> isize {
    // Validate buffer
    if output_buffer.is_null() {
        set_error("Null output buffer".to_string());
        return WG_ERROR_INVALID_PARAM as isize;
    }

    let internal = unsafe {
        match handle_to_internal(gen) {
            Some(g) => g,
            None => return WG_ERROR_INVALID_HANDLE as isize,
        }
    };

    // Calculate required size
    let required = internal.calculate_buffer_size(count);
    if buffer_size < required {
        set_error(format!(
            "Buffer too small: need {} bytes, have {}",
            required, buffer_size
        ));
        return WG_ERROR_BUFFER_TOO_SMALL as isize;
    }

    // Generate to host
    match internal.gpu.generate_batch_host(...) {
        Ok(data) => {
            // Copy to caller's buffer
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    output_buffer,
                    data.len()
                );
            }
            data.len() as isize
        }
        Err(e) => {
            set_error(format!("Generation failed: {}", e));
            WG_ERROR_CUDA as isize
        }
    }
}
```

---

## Thread Safety

### Strategy: Thread-Local Error, No Shared Mutable State

```rust
// SAFE: Each handle owns its state
// Multiple threads can each create their own handle

#[no_mangle]
pub extern "C" fn wg_create(...) -> *mut WordlistGenerator {
    // Each call creates independent state
    // No shared mutable state
    // ✅ Thread-safe
}

// UNSAFE: Concurrent calls with same handle
// User must synchronize or use one handle per thread

// Documentation in header:
/**
 * Thread Safety:
 *   - wg_create(), wg_destroy(): Not thread-safe
 *   - wg_set_*(): Not thread-safe (configure before threading)
 *   - wg_generate_*(): Thread-safe with different handles
 *   - wg_get_error(): Thread-local (safe)
 */
```

---

## Testing FFI Code

### Unit Tests (Rust Side)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_destroy() {
        let gen = wg_create(std::ptr::null_mut(), 0);
        assert!(!gen.is_null());
        wg_destroy(gen);
    }

    #[test]
    fn test_set_charset() {
        let gen = wg_create(std::ptr::null_mut(), 0);
        let charset = b"abc\0";
        let result = wg_set_charset(gen, 1, charset.as_ptr() as *const i8, 3);
        assert_eq!(result, WG_SUCCESS);
        wg_destroy(gen);
    }

    #[test]
    fn test_null_handle() {
        let result = wg_set_charset(
            std::ptr::null_mut(),
            1,
            b"abc\0".as_ptr() as *const i8,
            3
        );
        assert_eq!(result, WG_ERROR_INVALID_HANDLE);
    }
}
```

### Integration Tests (C Side)

```c
// tests/ffi_test.c

#include <wordlist_generator.h>
#include <assert.h>
#include <stdio.h>

void test_basic_usage() {
    // Create
    wg_handle_t gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Configure
    int result = wg_set_charset(gen, 1, "abc", 3);
    assert(result == WG_SUCCESS);

    int mask[] = {1, 1};
    result = wg_set_mask(gen, mask, 2);
    assert(result == WG_SUCCESS);

    // Keyspace
    uint64_t keyspace = wg_keyspace_size(gen);
    assert(keyspace == 9); // 3 * 3

    // Destroy
    wg_destroy(gen);
    printf("✓ Basic usage test passed\n");
}

void test_error_handling() {
    wg_handle_t gen = wg_create(NULL, 0);

    // Invalid charset ID
    int result = wg_set_charset(gen, 0, "abc", 3);
    assert(result != WG_SUCCESS);

    const char* error = wg_get_error(gen);
    assert(error != NULL);
    printf("Error: %s\n", error);

    wg_destroy(gen);
    printf("✓ Error handling test passed\n");
}

int main() {
    test_basic_usage();
    test_error_handling();
    printf("All tests passed!\n");
    return 0;
}
```

---

## Common Pitfalls

### Pitfall 1: Returning References to Temporary Strings

```rust
// BAD
#[no_mangle]
pub extern "C" fn get_version() -> *const c_char {
    let version = format!("v{}.{}.{}", 1, 0, 0);
    let c_str = CString::new(version).unwrap();
    c_str.as_ptr() // Dangling pointer! c_str dropped at end of function
}

// GOOD - use static string
#[no_mangle]
pub extern "C" fn get_version() -> *const c_char {
    b"v1.0.0\0".as_ptr() as *const c_char
}

// OR - transfer ownership (caller must free)
#[no_mangle]
pub extern "C" fn get_version() -> *mut c_char {
    let version = format!("v{}.{}.{}", 1, 0, 0);
    CString::new(version).unwrap().into_raw()
}
```

### Pitfall 2: Not Checking for Null

```rust
// BAD
#[no_mangle]
pub extern "C" fn process_data(data: *const u8, len: usize) {
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    // ... process ... SEGFAULT if data is null!
}

// GOOD
#[no_mangle]
pub extern "C" fn process_data(data: *const u8, len: usize) -> i32 {
    if data.is_null() {
        return -1;
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    // ... process ...
    0
}
```

### Pitfall 3: Forgetting `#[no_mangle]`

```rust
// BAD - name will be mangled, C can't find it
pub extern "C" fn my_function() {}

// GOOD
#[no_mangle]
pub extern "C" fn my_function() {}
```

---

## Checklist for FFI Functions

Before marking an FFI function as complete, verify:

- [ ] `#[no_mangle]` attribute present
- [ ] `extern "C"` calling convention
- [ ] All pointer parameters checked for null
- [ ] Error handling in place (no panics)
- [ ] Memory ownership documented
- [ ] Return values documented
- [ ] Thread safety documented
- [ ] Unit test written
- [ ] Integration test written

---

**End of FFI Implementation Guide**
