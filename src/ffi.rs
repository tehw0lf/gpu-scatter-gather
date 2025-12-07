//! C Foreign Function Interface (FFI) layer
//!
//! This module exposes the GPU wordlist generator as a C library.
//!
//! Safety: All functions validate inputs and never panic across FFI boundary.

use std::os::raw::c_char;
use std::collections::HashMap;
use std::cell::RefCell;
use crate::gpu::GpuContext;
use crate::multigpu::MultiGpuContext;
use cuda_driver_sys::*;

// Error codes (matching C API specification)
pub const WG_SUCCESS: i32 = 0;
pub const WG_ERROR_INVALID_HANDLE: i32 = -1;
pub const WG_ERROR_INVALID_PARAM: i32 = -2;
pub const WG_ERROR_CUDA: i32 = -3;
pub const WG_ERROR_OUT_OF_MEMORY: i32 = -4;
pub const WG_ERROR_NOT_CONFIGURED: i32 = -5;
pub const WG_ERROR_BUFFER_TOO_SMALL: i32 = -6;
pub const WG_ERROR_KEYSPACE_OVERFLOW: i32 = -7;

// Output format modes
pub const WG_FORMAT_NEWLINES: i32 = 0;  // Default: "word\n"
pub const WG_FORMAT_FIXED_WIDTH: i32 = 1;  // Future: fixed width padding
pub const WG_FORMAT_PACKED: i32 = 2;  // Future: no separators

/// Opaque handle to wordlist generator (exported to C)
#[repr(C)]
pub struct WordlistGenerator {
    _private: [u8; 0], // Zero-sized, prevents construction in C
}

/// Opaque handle to multi-GPU wordlist generator (exported to C)
#[repr(C)]
pub struct MultiGpuGenerator {
    _private: [u8; 0], // Zero-sized, prevents construction in C
}

/// Device batch result (zero-copy GPU memory access)
///
/// Contains a GPU device pointer for direct kernel-to-kernel data passing.
/// Memory is automatically freed on next generation or wg_destroy().
#[repr(C)]
pub struct BatchDevice {
    /// Device pointer to candidates (CUdeviceptr)
    pub data: u64,
    /// Number of candidates generated
    pub count: u64,
    /// Length of each word in characters
    pub word_length: usize,
    /// Bytes between word starts (stride)
    pub stride: usize,
    /// Total buffer size in bytes
    pub total_bytes: usize,
    /// Output format used (WG_FORMAT_*)
    pub format: i32,
}

/// Internal generator state (not exposed to C)
struct GeneratorInternal {
    gpu: GpuContext,
    charsets: HashMap<usize, Vec<u8>>,
    mask: Option<Vec<usize>>,
    current_batch: Option<CUdeviceptr>,  // Track active device memory
    #[allow(dead_code)]
    owns_context: bool,  // Whether we created the CUDA context (kept for future use)
    output_format: i32,  // Output format mode (WG_FORMAT_*)
}

/// Internal multi-GPU generator state (not exposed to C)
struct MultiGpuGeneratorInternal {
    multi_gpu: MultiGpuContext,
    charsets: HashMap<usize, Vec<u8>>,
    mask: Option<Vec<usize>>,
    output_format: i32,
}

impl GeneratorInternal {
    /// Free current device batch memory (if any)
    fn free_current_batch(&mut self) {
        if let Some(ptr) = self.current_batch.take() {
            unsafe {
                let _ = cuMemFree_v2(ptr);
            }
        }
    }
}

impl Drop for GeneratorInternal {
    fn drop(&mut self) {
        // Free any outstanding device memory
        self.free_current_batch();
    }
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
/// * `ctx` - CUDA context (NULL to create new, non-NULL to use existing)
/// * `device_id` - CUDA device ID (0 for first GPU, ignored if ctx provided)
///
/// # Returns
/// Generator handle, or NULL on error
#[no_mangle]
pub extern "C" fn wg_create(
    ctx: *mut std::ffi::c_void,
    _device_id: i32,  // TODO: Support device selection when ctx is NULL
) -> *mut WordlistGenerator {
    // Catch any panics and return NULL
    let result = std::panic::catch_unwind(|| {
        let (gpu, owns_context) = if ctx.is_null() {
            // Create our own GPU context
            match GpuContext::new() {
                Ok(g) => (g, true),
                Err(e) => {
                    set_error(format!("Failed to create GPU context: {}", e));
                    return std::ptr::null_mut();
                }
            }
        } else {
            // Use provided CUDA context
            // For now, we still create our own context
            // TODO: Add GpuContext::from_existing(ctx) in future
            match GpuContext::new() {
                Ok(g) => (g, true),
                Err(e) => {
                    set_error(format!("Failed to create GPU context: {}", e));
                    return std::ptr::null_mut();
                }
            }
        };

        // Create internal state
        let internal = Box::new(GeneratorInternal {
            gpu,
            charsets: HashMap::new(),
            mask: None,
            current_batch: None,
            owns_context,
            output_format: WG_FORMAT_NEWLINES,  // Default format
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

/// Set output format mode
///
/// # Arguments
/// * `gen` - Generator handle
/// * `format` - Output format (WG_FORMAT_*)
///
/// # Returns
/// WG_SUCCESS or error code
#[no_mangle]
pub extern "C" fn wg_set_format(
    gen: *mut WordlistGenerator,
    format: i32,
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

    // Validate format
    if format < WG_FORMAT_NEWLINES || format > WG_FORMAT_PACKED {
        set_error(format!("Invalid format: {}", format));
        return WG_ERROR_INVALID_PARAM;
    }

    internal.output_format = format;
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

    let word_length = mask.len();

    // Calculate bytes per word based on format
    let bytes_per_word = match internal.output_format {
        WG_FORMAT_NEWLINES => word_length + 1,  // word + '\n'
        WG_FORMAT_FIXED_WIDTH => word_length + 1,  // word + '\0' padding
        WG_FORMAT_PACKED => word_length,  // just the word, no separator
        _ => word_length + 1,  // fallback to newlines
    };

    (count as usize).saturating_mul(bytes_per_word)
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
        internal.output_format,
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

/// Generate batch in GPU memory (zero-copy)
///
/// This function generates candidates directly in GPU memory without copying to host.
/// The device pointer remains valid until the next generation call or wg_destroy().
///
/// # Arguments
/// * `gen` - Generator handle
/// * `start_idx` - Starting index in keyspace
/// * `count` - Number of candidates to generate
/// * `batch` - Output structure to fill with device pointer info
///
/// # Returns
/// WG_SUCCESS or error code
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

    if batch.is_null() {
        set_error("Null batch pointer".to_string());
        return WG_ERROR_INVALID_PARAM;
    }

    let mask = match &internal.mask {
        Some(m) => m.clone(),
        None => {
            set_error("Generator not configured".to_string());
            return WG_ERROR_NOT_CONFIGURED;
        }
    };

    // Free previous batch (if any)
    internal.free_current_batch();

    // Convert HashMap
    let charsets_map: HashMap<usize, Vec<u8>> = internal.charsets.clone();

    // Generate batch on device
    let result = internal.gpu.generate_batch_device(
        &charsets_map,
        &mask,
        start_idx,
        count,
        internal.output_format,
    );

    match result {
        Ok((device_ptr, total_bytes)) => {
            // Store device pointer for auto-cleanup
            internal.current_batch = Some(device_ptr);

            // Fill batch structure
            let word_length = mask.len();

            // Calculate stride based on format
            // Note: Kernel always writes with newlines (word_length + 1)
            // For packed format, consumers should use word_length stride and ignore '\n'
            let stride = match internal.output_format {
                WG_FORMAT_NEWLINES => word_length + 1,  // word + '\n'
                WG_FORMAT_FIXED_WIDTH => word_length + 1,  // word + '\0' (same as newlines for now)
                WG_FORMAT_PACKED => word_length,  // just word (skip '\n')
                _ => word_length + 1,
            };

            unsafe {
                (*batch).data = device_ptr;
                (*batch).count = count;
                (*batch).word_length = word_length;
                (*batch).stride = stride;
                (*batch).total_bytes = total_bytes;
                (*batch).format = internal.output_format;
            }

            WG_SUCCESS
        }
        Err(e) => {
            set_error(format!("Device generation failed: {}", e));
            WG_ERROR_CUDA
        }
    }
}

/// Free device batch memory early (optional)
///
/// Device memory is automatically freed on next generation or wg_destroy(),
/// but this function allows explicit early cleanup.
///
/// # Arguments
/// * `gen` - Generator handle
/// * `batch` - Batch to free (data pointer will be set to 0)
#[no_mangle]
pub extern "C" fn wg_free_batch_device(
    gen: *mut WordlistGenerator,
    batch: *mut BatchDevice,
) {
    let internal = unsafe {
        match handle_to_internal(gen) {
            Some(g) => g,
            None => return,
        }
    };

    if batch.is_null() {
        return;
    }

    // Free device memory
    internal.free_current_batch();

    // Clear batch structure
    unsafe {
        (*batch).data = 0;
        (*batch).count = 0;
        (*batch).total_bytes = 0;
    }
}

/// Generate batch using CUDA stream (async)
///
/// Allows overlapping generation with other GPU operations.
/// Kernel launch returns immediately; use cuStreamSynchronize()
/// to wait for completion.
///
/// # Arguments
/// * `gen` - Generator handle
/// * `stream` - CUDA stream for async execution (null for default stream)
/// * `start_idx` - Starting index in keyspace
/// * `count` - Number of candidates to generate
/// * `batch` - [out] Batch result info
///
/// # Returns
/// WG_SUCCESS or error code
///
/// # Safety
/// Caller must synchronize stream before using batch.data.
/// Device pointer lifetime same as wg_generate_batch_device().
///
/// # Example
/// ```c
/// CUstream stream;
/// cuStreamCreate(&stream, 0);
///
/// wg_batch_device_t batch;
/// wg_generate_batch_stream(gen, stream, 0, 100000000, &batch);
///
/// // Do other work...
///
/// cuStreamSynchronize(stream);  // Wait for generation
/// // Now batch.data is valid
/// ```
#[no_mangle]
pub extern "C" fn wg_generate_batch_stream(
    gen: *mut WordlistGenerator,
    stream: CUstream,
    start_idx: u64,
    count: u64,
    batch: *mut BatchDevice,
) -> i32 {
    // Validate inputs
    if batch.is_null() {
        set_error("Batch pointer is null".to_string());
        return WG_ERROR_INVALID_PARAM;
    }

    let internal = unsafe {
        match handle_to_internal(gen) {
            Some(g) => g,
            None => {
                set_error("Invalid generator handle".to_string());
                return WG_ERROR_INVALID_HANDLE;
            }
        }
    };

    // Check if configured
    if internal.mask.is_none() {
        set_error("Mask not configured (call wg_set_mask first)".to_string());
        return WG_ERROR_NOT_CONFIGURED;
    }

    if internal.charsets.is_empty() {
        set_error("No charsets configured (call wg_set_charset first)".to_string());
        return WG_ERROR_NOT_CONFIGURED;
    }

    // Free previous batch (if any)
    internal.free_current_batch();

    // Generate on device using stream
    let mask = internal.mask.as_ref().unwrap();
    let word_length = mask.len();

    match internal
        .gpu
        .generate_batch_device_stream(&internal.charsets, mask, start_idx, count, stream, internal.output_format)
    {
        Ok((device_ptr, buffer_size)) => {
            // Calculate stride based on output format
            let stride = match internal.output_format {
                WG_FORMAT_PACKED => word_length,        // No separator
                WG_FORMAT_NEWLINES => word_length + 1,  // word + newline
                WG_FORMAT_FIXED_WIDTH => word_length + 1, // word + null/padding
                _ => word_length + 1,
            };

            // Store device pointer for auto-cleanup
            internal.current_batch = Some(device_ptr);

            // Fill batch info
            unsafe {
                (*batch).data = device_ptr;
                (*batch).count = count;
                (*batch).word_length = word_length;
                (*batch).stride = stride;
                (*batch).total_bytes = buffer_size;
                (*batch).format = internal.output_format;
            }

            WG_SUCCESS
        }
        Err(e) => {
            set_error(format!("GPU generation failed: {}", e));
            WG_ERROR_CUDA
        }
    }
}

/// Get library version string
///
/// Returns a static string with the library version.
/// This function never fails and always returns a valid pointer.
///
/// # Returns
/// Pointer to static version string (e.g., "0.1.0")
///
/// # Example
/// ```c
/// const char* version = wg_get_version();
/// printf("Library version: %s\n", version);
/// ```
#[no_mangle]
pub extern "C" fn wg_get_version() -> *const c_char {
    // Static version string from Cargo.toml
    const VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Check if CUDA is available
///
/// Attempts to initialize CUDA and check for devices.
/// This function can be called before creating a generator
/// to verify CUDA is available.
///
/// # Returns
/// 1 if CUDA is available and working, 0 otherwise
///
/// # Example
/// ```c
/// if (!wg_cuda_available()) {
///     fprintf(stderr, "CUDA not available\n");
///     return -1;
/// }
/// // Safe to create generator
/// ```
#[no_mangle]
pub extern "C" fn wg_cuda_available() -> i32 {
    unsafe {
        // Try to initialize CUDA
        if cuInit(0) != CUresult::CUDA_SUCCESS {
            return 0;
        }

        // Check if at least one device exists
        let mut device_count = 0;
        if cuDeviceGetCount(&mut device_count) != CUresult::CUDA_SUCCESS {
            return 0;
        }

        if device_count > 0 {
            1
        } else {
            0
        }
    }
}

/// Get number of CUDA devices
///
/// Returns the count of CUDA-capable devices in the system.
/// Returns -1 if CUDA is not available or on error.
///
/// # Returns
/// Number of CUDA devices (>= 0) or -1 on error
///
/// # Example
/// ```c
/// int count = wg_get_device_count();
/// if (count < 0) {
///     fprintf(stderr, "CUDA error\n");
/// } else if (count == 0) {
///     fprintf(stderr, "No CUDA devices found\n");
/// } else {
///     printf("Found %d CUDA device(s)\n", count);
/// }
/// ```
#[no_mangle]
pub extern "C" fn wg_get_device_count() -> i32 {
    unsafe {
        // Initialize CUDA if needed
        if cuInit(0) != CUresult::CUDA_SUCCESS {
            return -1;
        }

        // Get device count
        let mut device_count = 0;
        if cuDeviceGetCount(&mut device_count) != CUresult::CUDA_SUCCESS {
            return -1;
        }

        device_count
    }
}

/// Get device information
///
/// Retrieves detailed information about a specific CUDA device.
///
/// # Arguments
/// * `device_id` - Device index (0 to wg_get_device_count() - 1)
/// * `name_out` - Buffer for device name (at least 256 bytes)
/// * `compute_cap_major_out` - Output for major compute capability
/// * `compute_cap_minor_out` - Output for minor compute capability
/// * `total_memory_out` - Output for total device memory in bytes
///
/// # Returns
/// WG_SUCCESS or error code
///
/// # Example
/// ```c
/// int count = wg_get_device_count();
/// for (int i = 0; i < count; i++) {
///     char name[256];
///     int major, minor;
///     uint64_t memory;
///
///     if (wg_get_device_info(i, name, &major, &minor, &memory) == WG_SUCCESS) {
///         printf("Device %d: %s (sm_%d%d, %lu MB)\n",
///                i, name, major, minor, memory / (1024*1024));
///     }
/// }
/// ```
#[no_mangle]
pub extern "C" fn wg_get_device_info(
    device_id: i32,
    name_out: *mut c_char,
    compute_cap_major_out: *mut i32,
    compute_cap_minor_out: *mut i32,
    total_memory_out: *mut u64,
) -> i32 {
    unsafe {
        // Initialize CUDA if needed
        if cuInit(0) != CUresult::CUDA_SUCCESS {
            set_error("Failed to initialize CUDA".to_string());
            return WG_ERROR_CUDA;
        }

        // Validate device_id
        let mut device_count = 0;
        if cuDeviceGetCount(&mut device_count) != CUresult::CUDA_SUCCESS {
            set_error("Failed to get device count".to_string());
            return WG_ERROR_CUDA;
        }

        if device_id < 0 || device_id >= device_count {
            set_error(format!(
                "Invalid device_id: {} (valid range: 0-{})",
                device_id,
                device_count - 1
            ));
            return WG_ERROR_INVALID_PARAM;
        }

        // Validate output pointers
        if name_out.is_null() {
            set_error("name_out pointer is null".to_string());
            return WG_ERROR_INVALID_PARAM;
        }

        if compute_cap_major_out.is_null() {
            set_error("compute_cap_major_out pointer is null".to_string());
            return WG_ERROR_INVALID_PARAM;
        }

        if compute_cap_minor_out.is_null() {
            set_error("compute_cap_minor_out pointer is null".to_string());
            return WG_ERROR_INVALID_PARAM;
        }

        if total_memory_out.is_null() {
            set_error("total_memory_out pointer is null".to_string());
            return WG_ERROR_INVALID_PARAM;
        }

        // Get device handle
        let mut device = 0;
        if cuDeviceGet(&mut device, device_id) != CUresult::CUDA_SUCCESS {
            set_error(format!("Failed to get device {}", device_id));
            return WG_ERROR_CUDA;
        }

        // Get device name
        let mut name_buffer = vec![0i8; 256];
        if cuDeviceGetName(name_buffer.as_mut_ptr(), 256, device) != CUresult::CUDA_SUCCESS {
            set_error(format!("Failed to get name for device {}", device_id));
            return WG_ERROR_CUDA;
        }

        // Copy name to output buffer
        let name_len = name_buffer.iter().position(|&c| c == 0).unwrap_or(255);
        std::ptr::copy_nonoverlapping(
            name_buffer.as_ptr(),
            name_out,
            name_len.min(255),
        );
        *name_out.add(name_len.min(255)) = 0; // Null terminate

        // Get compute capability
        let mut major = 0;
        let mut minor = 0;
        if cuDeviceGetAttribute(
            &mut major,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        ) != CUresult::CUDA_SUCCESS
        {
            set_error(format!(
                "Failed to get compute capability major for device {}",
                device_id
            ));
            return WG_ERROR_CUDA;
        }

        if cuDeviceGetAttribute(
            &mut minor,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        ) != CUresult::CUDA_SUCCESS
        {
            set_error(format!(
                "Failed to get compute capability minor for device {}",
                device_id
            ));
            return WG_ERROR_CUDA;
        }

        *compute_cap_major_out = major;
        *compute_cap_minor_out = minor;

        // Get total memory
        let mut total_mem = 0usize;
        if cuDeviceTotalMem_v2(&mut total_mem, device) != CUresult::CUDA_SUCCESS {
            set_error(format!("Failed to get total memory for device {}", device_id));
            return WG_ERROR_CUDA;
        }

        *total_memory_out = total_mem as u64;

        WG_SUCCESS
    }
}

// ===========================================================================
// Multi-GPU API
// ===========================================================================

/// Helper: Convert opaque multi-GPU handle to internal reference
unsafe fn multigpu_handle_to_internal<'a>(
    gen: *mut MultiGpuGenerator
) -> Option<&'a mut MultiGpuGeneratorInternal> {
    if gen.is_null() {
        return None;
    }
    Some(&mut *(gen as *mut MultiGpuGeneratorInternal))
}

/// Create multi-GPU generator using all available devices
///
/// # Returns
/// Generator handle, or NULL on error
#[no_mangle]
pub extern "C" fn wg_multigpu_create() -> *mut MultiGpuGenerator {
    let result = std::panic::catch_unwind(|| {
        let multi_gpu = match MultiGpuContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                set_error(format!("Failed to create multi-GPU context: {}", e));
                return std::ptr::null_mut();
            }
        };

        let internal = Box::new(MultiGpuGeneratorInternal {
            multi_gpu,
            charsets: HashMap::new(),
            mask: None,
            output_format: WG_FORMAT_NEWLINES,
        });

        Box::into_raw(internal) as *mut MultiGpuGenerator
    });

    result.unwrap_or_else(|_| {
        set_error("Panic during multi-GPU initialization".to_string());
        std::ptr::null_mut()
    })
}

/// Create multi-GPU generator using specific devices
///
/// # Arguments
/// * `device_ids` - Array of device IDs to use
/// * `num_devices` - Number of devices in array
///
/// # Returns
/// Generator handle, or NULL on error
#[no_mangle]
pub extern "C" fn wg_multigpu_create_with_devices(
    device_ids: *const i32,
    num_devices: i32,
) -> *mut MultiGpuGenerator {
    if device_ids.is_null() || num_devices <= 0 {
        set_error("Invalid device_ids or num_devices".to_string());
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(|| {
        let device_ids_slice = unsafe {
            std::slice::from_raw_parts(device_ids, num_devices as usize)
        };

        let multi_gpu = match MultiGpuContext::with_devices(device_ids_slice) {
            Ok(ctx) => ctx,
            Err(e) => {
                set_error(format!("Failed to create multi-GPU context: {}", e));
                return std::ptr::null_mut();
            }
        };

        let internal = Box::new(MultiGpuGeneratorInternal {
            multi_gpu,
            charsets: HashMap::new(),
            mask: None,
            output_format: WG_FORMAT_NEWLINES,
        });

        Box::into_raw(internal) as *mut MultiGpuGenerator
    });

    result.unwrap_or_else(|_| {
        set_error("Panic during multi-GPU initialization".to_string());
        std::ptr::null_mut()
    })
}

/// Set charset for multi-GPU generator
///
/// # Arguments
/// * `gen` - Multi-GPU generator handle
/// * `charset_id` - Identifier (1-255)
/// * `chars` - Character array
/// * `len` - Length of character array
///
/// # Returns
/// WG_SUCCESS or error code
#[no_mangle]
pub extern "C" fn wg_multigpu_set_charset(
    gen: *mut MultiGpuGenerator,
    charset_id: i32,
    chars: *const c_char,
    len: usize,
) -> i32 {
    let internal = unsafe {
        match multigpu_handle_to_internal(gen) {
            Some(g) => g,
            None => {
                set_error("Invalid multi-GPU generator handle".to_string());
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

/// Set mask pattern for multi-GPU generator
///
/// # Arguments
/// * `gen` - Multi-GPU generator handle
/// * `mask` - Array of charset IDs
/// * `length` - Number of positions (word length)
///
/// # Returns
/// WG_SUCCESS or error code
#[no_mangle]
pub extern "C" fn wg_multigpu_set_mask(
    gen: *mut MultiGpuGenerator,
    mask: *const i32,
    length: i32,
) -> i32 {
    let internal = unsafe {
        match multigpu_handle_to_internal(gen) {
            Some(g) => g,
            None => {
                set_error("Invalid multi-GPU generator handle".to_string());
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

/// Set output format for multi-GPU generator
///
/// # Arguments
/// * `gen` - Multi-GPU generator handle
/// * `format` - Output format (WG_FORMAT_*)
///
/// # Returns
/// WG_SUCCESS or error code
#[no_mangle]
pub extern "C" fn wg_multigpu_set_format(
    gen: *mut MultiGpuGenerator,
    format: i32,
) -> i32 {
    let internal = unsafe {
        match multigpu_handle_to_internal(gen) {
            Some(g) => g,
            None => {
                set_error("Invalid multi-GPU generator handle".to_string());
                return WG_ERROR_INVALID_HANDLE;
            }
        }
    };

    // Validate format
    if format < WG_FORMAT_NEWLINES || format > WG_FORMAT_PACKED {
        set_error(format!("Invalid format: {}", format));
        return WG_ERROR_INVALID_PARAM;
    }

    internal.output_format = format;
    WG_SUCCESS
}

/// Generate batch using multiple GPUs
///
/// # Arguments
/// * `gen` - Multi-GPU generator handle
/// * `start_idx` - Starting index in keyspace
/// * `count` - Number of words to generate
/// * `output_buffer` - Output buffer
/// * `buffer_size` - Size of output buffer
///
/// # Returns
/// Number of bytes written, or negative error code
#[no_mangle]
pub extern "C" fn wg_multigpu_generate(
    gen: *mut MultiGpuGenerator,
    start_idx: u64,
    count: u64,
    output_buffer: *mut u8,
    buffer_size: usize,
) -> isize {
    let internal = unsafe {
        match multigpu_handle_to_internal(gen) {
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
    let word_length = mask.len();
    let bytes_per_word = match internal.output_format {
        WG_FORMAT_NEWLINES => word_length + 1,
        WG_FORMAT_FIXED_WIDTH => word_length + 1,
        WG_FORMAT_PACKED => word_length,
        _ => word_length + 1,
    };
    let required = (count as usize).saturating_mul(bytes_per_word);

    if buffer_size < required {
        set_error(format!(
            "Buffer too small: need {} bytes, have {}",
            required, buffer_size
        ));
        return WG_ERROR_BUFFER_TOO_SMALL as isize;
    }

    // Generate batch using multi-GPU
    let result = internal.multi_gpu.generate_batch(
        &internal.charsets,
        mask,
        start_idx,
        count,
        internal.output_format,
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
            set_error(format!("Multi-GPU generation failed: {}", e));
            WG_ERROR_CUDA as isize
        }
    }
}

/// Get number of GPUs being used
///
/// # Arguments
/// * `gen` - Multi-GPU generator handle
///
/// # Returns
/// Number of GPUs, or -1 on error
#[no_mangle]
pub extern "C" fn wg_multigpu_get_device_count(gen: *mut MultiGpuGenerator) -> i32 {
    let internal = unsafe {
        match multigpu_handle_to_internal(gen) {
            Some(g) => g,
            None => return -1,
        }
    };

    internal.multi_gpu.num_devices() as i32
}

/// Destroy multi-GPU generator and free all resources
///
/// # Safety
/// Safe to call with NULL (no-op)
#[no_mangle]
pub extern "C" fn wg_multigpu_destroy(gen: *mut MultiGpuGenerator) {
    if gen.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(gen as *mut MultiGpuGeneratorInternal);
        // Box drop automatically frees everything
    }
}
