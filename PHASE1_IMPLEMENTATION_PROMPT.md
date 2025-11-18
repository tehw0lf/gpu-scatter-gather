# Phase 1 Implementation Prompt: Core C API

**Goal**: Implement minimal viable C FFI layer for GPU wordlist generator library

**Estimated Time**: 8-10 hours

**Status**: Ready to implement

---

## Quick Start

**What you're building**: Transform the Rust GPU wordlist generator into a C library that can be embedded in password crackers like hashcat and John the Ripper.

**This phase delivers**: Core C API with create/destroy, configuration, and basic host-side generation. This is the foundation - device pointers and advanced features come in Phase 2+.

---

## Context: What We Have

### Current State (v1.0)
```
✅ Working GPU wordlist generator in Rust
✅ 440 M words/s throughput (3-5x faster than CPU)
✅ Clean internal API (GpuContext, generate_batch, etc.)
✅ Comprehensive documentation (2,300+ lines)
```

### Target State (Phase 1 Complete)
```
✅ C API for initialization and configuration
✅ Host-side batch generation (copy to host memory)
✅ Error handling with thread-local messages
✅ Auto-generated C header with cbindgen
✅ Basic C integration test
```

---

## Prerequisites: Read These First

Before starting implementation, **read these documents**:

1. **docs/FFI_IMPLEMENTATION_GUIDE.md** (Required)
   - Rust-C type mappings
   - FFI safety rules (no panics, null checks, etc.)
   - Opaque handle pattern
   - Error handling strategy
   - Memory management

2. **docs/C_API_SPECIFICATION.md** (Required)
   - Complete API specification
   - Function signatures
   - Expected behavior

3. **docs/CBINDGEN_SETUP.md** (Required)
   - How to configure cbindgen
   - Build process
   - Header generation

**Estimated reading time**: 45-60 minutes

---

## Implementation Checklist

### Step 1: Project Setup (30 minutes)

#### 1.1 Install cbindgen
```bash
cargo install cbindgen
```

#### 1.2 Create cbindgen configuration

**File**: `cbindgen.toml`

```toml
language = "C"
header = "/* libwordlist_generator - Auto-generated C API */"
include_guard = "WORDLIST_GENERATOR_H"
autogen_warning = "/* WARNING: Auto-generated from src/ffi.rs - DO NOT EDIT */"

sys_includes = ["stddef.h", "stdint.h", "cuda.h"]
tab_width = 4

documentation = true
documentation_style = "doxy"

[parse]
parse_deps = false
include = ["src/ffi.rs"]

[export]
prefix = "wg_"
```

#### 1.3 Create build script

**File**: `build.rs`

```rust
use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let config = cbindgen::Config::from_file("cbindgen.toml")
        .expect("Unable to find cbindgen.toml");

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("include/wordlist_generator.h");

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
```

#### 1.4 Update Cargo.toml

Add to `[lib]` section:
```toml
[lib]
name = "wordlist_generator"
crate-type = ["cdylib", "rlib"]  # cdylib for C library, rlib for Rust

[build-dependencies]
cbindgen = "0.26"
```

#### 1.5 Create include directory
```bash
mkdir -p include
```

### Step 2: FFI Module Structure (1 hour)

#### 2.1 Create FFI module

**File**: `src/ffi.rs`

```rust
//! C Foreign Function Interface (FFI) layer
//!
//! This module exposes the GPU wordlist generator as a C library.
//!
//! Safety: All functions validate inputs and never panic across FFI boundary.

use std::os::raw::c_char;
use std::ffi::CStr;
use std::collections::HashMap;
use std::cell::RefCell;
use crate::gpu::GpuContext;

// Error codes (matching C API specification)
pub const WG_SUCCESS: i32 = 0;
pub const WG_ERROR_INVALID_HANDLE: i32 = -1;
pub const WG_ERROR_INVALID_PARAM: i32 = -2;
pub const WG_ERROR_CUDA: i32 = -3;
pub const WG_ERROR_OUT_OF_MEMORY: i32 = -4;
pub const WG_ERROR_NOT_CONFIGURED: i32 = -5;
pub const WG_ERROR_BUFFER_TOO_SMALL: i32 = -6;
pub const WG_ERROR_KEYSPACE_OVERFLOW: i32 = -7;

/// Opaque handle to wordlist generator (exported to C)
#[repr(C)]
pub struct WordlistGenerator {
    _private: [u8; 0], // Zero-sized, prevents construction in C
}

/// Internal generator state (not exposed to C)
struct GeneratorInternal {
    gpu: GpuContext,
    charsets: HashMap<usize, Vec<u8>>,
    mask: Option<Vec<usize>>,
}

// Thread-local error storage
thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = RefCell::new(None);
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(msg);
    });
}

/// Helper: Convert opaque handle to internal reference
unsafe fn handle_to_internal<'a>(
    gen: *mut WordlistGenerator
) -> Option<&'a mut GeneratorInternal> {
    if gen.is_null() {
        return None;
    }
    Some(&mut *(gen as *mut GeneratorInternal))
}

// FFI functions will be added in next steps...
```

#### 2.2 Add FFI module to lib.rs

**File**: `src/lib.rs`

Add after other `pub mod` declarations:
```rust
pub mod ffi;
```

### Step 3: Implement Core Functions (2-3 hours)

#### 3.1 Initialization and teardown

Add to `src/ffi.rs`:

```rust
/// Create a new wordlist generator
///
/// # Arguments
/// * `ctx` - CUDA context (NULL to create new)
/// * `device_id` - CUDA device ID (0 for first GPU)
///
/// # Returns
/// Generator handle, or NULL on error
#[no_mangle]
pub extern "C" fn wg_create(
    _ctx: *mut std::ffi::c_void, // TODO: Phase 2 will use this
    device_id: i32,
) -> *mut WordlistGenerator {
    // Catch any panics and return NULL
    let result = std::panic::catch_unwind(|| {
        // Create GPU context
        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                set_error(format!("Failed to create GPU context: {}", e));
                return std::ptr::null_mut();
            }
        };

        // Create internal state
        let internal = Box::new(GeneratorInternal {
            gpu,
            charsets: HashMap::new(),
            mask: None,
        });

        // Convert to opaque pointer
        Box::into_raw(internal) as *mut WordlistGenerator
    });

    result.unwrap_or_else(|_| {
        set_error("Panic during initialization".to_string());
        std::ptr::null_mut()
    })
}

/// Destroy generator and free all resources
///
/// # Safety
/// Safe to call with NULL (no-op)
#[no_mangle]
pub extern "C" fn wg_destroy(gen: *mut WordlistGenerator) {
    if gen.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(gen as *mut GeneratorInternal);
        // Box drop automatically frees everything
    }
}
```

#### 3.2 Configuration functions

```rust
/// Define a charset for use in masks
///
/// # Arguments
/// * `gen` - Generator handle
/// * `charset_id` - Identifier (1-255)
/// * `chars` - Character array
/// * `len` - Length of character array
///
/// # Returns
/// WG_SUCCESS or error code
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
                return WG_ERROR_INVALID_HANDLE;
            }
        }
    };

    // Validate charset_id
    if charset_id <= 0 || charset_id > 255 {
        set_error(format!("Invalid charset_id: {}", charset_id));
        return WG_ERROR_INVALID_PARAM;
    }

    // Validate pointer
    if chars.is_null() {
        set_error("Null charset pointer".to_string());
        return WG_ERROR_INVALID_PARAM;
    }

    // Validate length
    if len == 0 || len > 512 {
        set_error(format!("Invalid charset length: {}", len));
        return WG_ERROR_INVALID_PARAM;
    }

    // Convert C string to Rust Vec<u8>
    let charset_bytes = unsafe {
        std::slice::from_raw_parts(chars as *const u8, len)
    }.to_vec();

    // Store charset
    internal.charsets.insert(charset_id as usize, charset_bytes);

    WG_SUCCESS
}

/// Set the mask pattern
///
/// # Arguments
/// * `gen` - Generator handle
/// * `mask` - Array of charset IDs
/// * `length` - Number of positions (word length)
///
/// # Returns
/// WG_SUCCESS or error code
#[no_mangle]
pub extern "C" fn wg_set_mask(
    gen: *mut WordlistGenerator,
    mask: *const i32,
    length: i32,
) -> i32 {
    let internal = unsafe {
        match handle_to_internal(gen) {
            Some(g) => g,
            None => {
                set_error("Invalid generator handle".to_string());
                return WG_ERROR_INVALID_HANDLE;
            }
        }
    };

    if mask.is_null() {
        set_error("Null mask pointer".to_string());
        return WG_ERROR_INVALID_PARAM;
    }

    if length <= 0 || length > 32 {
        set_error(format!("Invalid mask length: {}", length));
        return WG_ERROR_INVALID_PARAM;
    }

    // Convert C array to Rust Vec
    let mask_vec = unsafe {
        std::slice::from_raw_parts(mask, length as usize)
    }.iter().map(|&x| x as usize).collect::<Vec<_>>();

    // Validate all charset IDs exist
    for &charset_id in &mask_vec {
        if !internal.charsets.contains_key(&charset_id) {
            set_error(format!("Undefined charset ID in mask: {}", charset_id));
            return WG_ERROR_INVALID_PARAM;
        }
    }

    internal.mask = Some(mask_vec);

    WG_SUCCESS
}
```

#### 3.3 Keyspace information

```rust
/// Get total keyspace size
///
/// # Returns
/// Number of possible candidates, or 0 on error
#[no_mangle]
pub extern "C" fn wg_keyspace_size(gen: *mut WordlistGenerator) -> u64 {
    let internal = unsafe {
        match handle_to_internal(gen) {
            Some(g) => g,
            None => return 0,
        }
    };

    let mask = match &internal.mask {
        Some(m) => m,
        None => {
            set_error("Generator not configured (no mask set)".to_string());
            return 0;
        }
    };

    // Calculate keyspace: product of charset sizes
    let mut keyspace: u128 = 1;
    for &charset_id in mask {
        if let Some(charset) = internal.charsets.get(&charset_id) {
            keyspace = keyspace.saturating_mul(charset.len() as u128);
            if keyspace > u64::MAX as u128 {
                set_error("Keyspace overflow (>2^64)".to_string());
                return u64::MAX; // Saturate at max
            }
        }
    }

    keyspace as u64
}

/// Calculate required buffer size for host generation
///
/// # Returns
/// Required buffer size in bytes, or 0 on error
#[no_mangle]
pub extern "C" fn wg_calculate_buffer_size(
    gen: *mut WordlistGenerator,
    count: u64,
) -> usize {
    let internal = unsafe {
        match handle_to_internal(gen) {
            Some(g) => g,
            None => return 0,
        }
    };

    let mask = match &internal.mask {
        Some(m) => m,
        None => return 0,
    };

    let word_length = mask.len() + 1; // +1 for newline
    (count as usize).saturating_mul(word_length)
}
```

#### 3.4 Error handling

```rust
/// Get last error message for this thread
///
/// # Returns
/// Error message string, or NULL if no error
#[no_mangle]
pub extern "C" fn wg_get_error(_gen: *mut WordlistGenerator) -> *const c_char {
    LAST_ERROR.with(|e| {
        match e.borrow().as_ref() {
            Some(err) => err.as_ptr() as *const c_char,
            None => std::ptr::null(),
        }
    })
}
```

#### 3.5 Host-side generation

```rust
/// Generate batch and copy to host memory
///
/// # Returns
/// Number of bytes written, or negative error code
#[no_mangle]
pub extern "C" fn wg_generate_batch_host(
    gen: *mut WordlistGenerator,
    start_idx: u64,
    count: u64,
    output_buffer: *mut u8,
    buffer_size: usize,
) -> isize {
    let internal = unsafe {
        match handle_to_internal(gen) {
            Some(g) => g,
            None => return WG_ERROR_INVALID_HANDLE as isize,
        }
    };

    if output_buffer.is_null() {
        set_error("Null output buffer".to_string());
        return WG_ERROR_INVALID_PARAM as isize;
    }

    let mask = match &internal.mask {
        Some(m) => m,
        None => {
            set_error("Generator not configured".to_string());
            return WG_ERROR_NOT_CONFIGURED as isize;
        }
    };

    // Check buffer size
    let required = wg_calculate_buffer_size(gen, count);
    if buffer_size < required {
        set_error(format!(
            "Buffer too small: need {} bytes, have {}",
            required, buffer_size
        ));
        return WG_ERROR_BUFFER_TOO_SMALL as isize;
    }

    // Convert HashMap to format expected by generate_batch
    let charsets_map: HashMap<usize, Vec<u8>> = internal.charsets.clone();

    // Generate batch
    let result = internal.gpu.generate_batch(
        &charsets_map,
        mask,
        start_idx,
        count,
    );

    match result {
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

### Step 4: Build and Test (1 hour)

#### 4.1 Build library

```bash
cargo build --release

# Check that header was generated
ls -lh include/wordlist_generator.h

# Check that library was built
ls -lh target/release/libwordlist_generator.so
```

#### 4.2 Create C test program

**File**: `tests/ffi_basic_test.c`

```c
#include "../include/wordlist_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

void test_create_destroy() {
    printf("Test: create/destroy...\n");

    wg_handle_t gen = wg_create(NULL, 0);
    assert(gen != NULL && "Failed to create generator");

    wg_destroy(gen);

    printf("✓ create/destroy passed\n\n");
}

void test_configuration() {
    printf("Test: configuration...\n");

    wg_handle_t gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Set charset
    int result = wg_set_charset(gen, 1, "abc", 3);
    assert(result == 0 && "Failed to set charset");

    // Set mask
    int mask[] = {1, 1, 1, 1};
    result = wg_set_mask(gen, mask, 4);
    assert(result == 0 && "Failed to set mask");

    // Check keyspace
    uint64_t keyspace = wg_keyspace_size(gen);
    assert(keyspace == 81 && "Wrong keyspace (expected 3^4=81)");

    printf("  Keyspace: %llu\n", keyspace);

    wg_destroy(gen);

    printf("✓ configuration passed\n\n");
}

void test_generation() {
    printf("Test: host generation...\n");

    wg_handle_t gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Configure
    wg_set_charset(gen, 1, "ab", 2);
    int mask[] = {1, 1, 1};
    wg_set_mask(gen, mask, 3);

    // Allocate buffer
    uint64_t count = 8; // 2^3 = 8 total
    size_t buffer_size = wg_calculate_buffer_size(gen, count);
    printf("  Buffer size needed: %zu bytes\n", buffer_size);

    char* buffer = malloc(buffer_size);
    assert(buffer != NULL);

    // Generate
    ssize_t bytes = wg_generate_batch_host(gen, 0, count, buffer, buffer_size);
    assert(bytes > 0 && "Generation failed");
    printf("  Generated %zd bytes\n", bytes);

    // Verify first few words
    printf("  First 3 words:\n");
    char* ptr = buffer;
    for (int i = 0; i < 3; i++) {
        printf("    %d: %.*s", i, 3, ptr);
        ptr += 4; // 3 chars + newline
    }

    free(buffer);
    wg_destroy(gen);

    printf("✓ generation passed\n\n");
}

void test_error_handling() {
    printf("Test: error handling...\n");

    wg_handle_t gen = wg_create(NULL, 0);

    // Invalid charset ID
    int result = wg_set_charset(gen, 0, "abc", 3);
    assert(result != 0 && "Should fail with invalid charset_id");

    const char* error = wg_get_error(gen);
    assert(error != NULL && "Should have error message");
    printf("  Error message: %s\n", error);

    wg_destroy(gen);

    printf("✓ error handling passed\n\n");
}

int main() {
    printf("=== FFI Basic Tests ===\n\n");

    test_create_destroy();
    test_configuration();
    test_generation();
    test_error_handling();

    printf("=== All tests passed! ===\n");
    return 0;
}
```

#### 4.3 Compile and run test

```bash
# Compile test
gcc -o test_ffi tests/ffi_basic_test.c \
    -I. \
    -I/usr/local/cuda/include \
    -L./target/release \
    -lwordlist_generator \
    -Wl,-rpath,./target/release

# Run test
./test_ffi
```

Expected output:
```
=== FFI Basic Tests ===

Test: create/destroy...
✓ create/destroy passed

Test: configuration...
  Keyspace: 81
✓ configuration passed

Test: host generation...
  Buffer size needed: 32 bytes
  Generated 32 bytes
  First 3 words:
    0: aaa
    1: aab
    2: aba
✓ generation passed

Test: error handling...
  Error message: Invalid charset_id: 0
✓ error handling passed

=== All tests passed! ===
```

### Step 5: Documentation (1-2 hours)

#### 5.1 Add doc comments to FFI functions

Ensure all `#[no_mangle] pub extern "C"` functions have doc comments (they get included in generated header).

#### 5.2 Create Phase 1 summary

**File**: `docs/PHASE1_SUMMARY.md`

Document:
- What was implemented
- How to build
- How to use (examples)
- Known limitations (device pointers come in Phase 2)
- Next steps (Phase 2)

---

## Proof of Concept: Before Starting

To validate the approach, create a minimal PoC first:

**File**: `poc/ffi_poc.rs` (temporary, not committed)

```rust
use std::os::raw::c_char;

#[repr(C)]
pub struct Generator {
    _private: [u8; 0],
}

struct Internal {
    value: i32,
}

#[no_mangle]
pub extern "C" fn poc_create() -> *mut Generator {
    let internal = Box::new(Internal { value: 42 });
    Box::into_raw(internal) as *mut Generator
}

#[no_mangle]
pub extern "C" fn poc_get_value(gen: *mut Generator) -> i32 {
    if gen.is_null() {
        return -1;
    }
    unsafe {
        let internal = &*(gen as *const Internal);
        internal.value
    }
}

#[no_mangle]
pub extern "C" fn poc_destroy(gen: *mut Generator) {
    if gen.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(gen as *mut Internal);
    }
}
```

**Test PoC**:
```bash
# Build
cargo build --release

# Test in C
cat > poc_test.c <<EOF
#include <stdio.h>

struct Generator;
struct Generator* poc_create();
int poc_get_value(struct Generator* gen);
void poc_destroy(struct Generator* gen);

int main() {
    struct Generator* gen = poc_create();
    int value = poc_get_value(gen);
    printf("Value: %d\n", value);
    poc_destroy(gen);
    return 0;
}
EOF

gcc -o poc_test poc_test.c -L./target/release -lgpu_scatter_gather
./poc_test
# Should print: Value: 42
```

If PoC works, proceed with full implementation!

---

## Success Criteria

Phase 1 is complete when:

- [ ] FFI module (`src/ffi.rs`) compiles without warnings
- [ ] C header auto-generated successfully
- [ ] Library builds (`libwordlist_generator.so`)
- [ ] C test program compiles and links
- [ ] All C tests pass
- [ ] Error handling works (null checks, error messages)
- [ ] No memory leaks (verified with valgrind)
- [ ] Documentation complete

---

## Troubleshooting

### Issue: cbindgen not found
```bash
cargo install cbindgen
```

### Issue: Header not generated
```bash
# Check build.rs ran
cargo clean
cargo build --release -vv

# Manual generation
cbindgen --config cbindgen.toml --output include/wordlist_generator.h
```

### Issue: Linker can't find library
```bash
# Use rpath
gcc ... -Wl,-rpath,./target/release

# Or set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./target/release:$LD_LIBRARY_PATH
```

### Issue: Segfault in C test
```bash
# Run with gdb
gdb ./test_ffi
run

# Check for null pointer dereference
# Verify all FFI functions check for null
```

---

## Next Steps After Phase 1

Once Phase 1 is complete:

1. **Code review**: Review FFI code for safety
2. **Memory leak check**: Run valgrind on C test
3. **Commit**: Create commit for Phase 1
4. **Phase 2**: Implement device pointer support
5. **Phase 3**: Implement output format modes
6. **Phase 4**: Implement streaming API

---

## Time Budget

| Task | Estimated | Actual |
|------|-----------|--------|
| Project setup | 30 min | |
| FFI structure | 1 hour | |
| Core functions | 2-3 hours | |
| Build & test | 1 hour | |
| Documentation | 1-2 hours | |
| **Total** | **8-10 hours** | |

---

## Resources

- **FFI Guide**: `docs/FFI_IMPLEMENTATION_GUIDE.md`
- **API Spec**: `docs/C_API_SPECIFICATION.md`
- **cbindgen**: `docs/CBINDGEN_SETUP.md`
- **Rust FFI docs**: https://doc.rust-lang.org/nomicon/ffi.html

---

**Ready to start? Begin with the PoC, then follow the checklist!**

**Good luck! 🚀**
