//! C Foreign Function Interface (FFI) layer
//!
//! This module exposes the GPU wordlist generator as a C library.
//!
//! Safety: All functions validate inputs and never panic across FFI boundary.

use std::os::raw::c_char;
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
    _device_id: i32,
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
