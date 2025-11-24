//! GPU acceleration module using CUDA
//!
//! This module contains the GPU context management, kernel compilation,
//! and batch generation logic.

use anyhow::{Context, Result};
use cuda_driver_sys::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

/// GPU context for wordlist generation
pub struct GpuContext {
    context: CUcontext,
    module: CUmodule,
    kernel: CUfunction,
    kernel_transposed: CUfunction,
    kernel_columnmajor: CUfunction,
    device: CUdevice,
    device_id: i32,
    compute_capability: (i32, i32),
    // Persistent GPU buffers for reuse across batches
    persistent_output_buffer: Option<CUdeviceptr>,
    persistent_output_capacity: usize,
}

impl GpuContext {
    /// Initialize GPU context and load kernel (uses device 0)
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Initialize GPU context for specific device
    pub fn with_device(device_id: i32) -> Result<Self> {
        unsafe {
            // Initialize CUDA
            check_cuda(cuInit(0)).context("Failed to initialize CUDA")?;

            // Validate device_id
            let mut device_count = 0;
            check_cuda(cuDeviceGetCount(&mut device_count))
                .context("Failed to get device count")?;

            if device_id < 0 || device_id >= device_count {
                anyhow::bail!(
                    "Invalid device_id: {} (valid range: 0-{})",
                    device_id,
                    device_count - 1
                );
            }

            // Get device
            let mut device = 0;
            check_cuda(cuDeviceGet(&mut device, device_id))
                .with_context(|| format!("Failed to get CUDA device {}", device_id))?;

            // Get compute capability
            let mut compute_capability_major = 0;
            let mut compute_capability_minor = 0;
            check_cuda(cuDeviceGetAttribute(
                &mut compute_capability_major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            ))?;
            check_cuda(cuDeviceGetAttribute(
                &mut compute_capability_minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            ))?;

            // Create context
            let mut context = ptr::null_mut();
            check_cuda(cuCtxCreate_v2(&mut context, 0, device))
                .context("Failed to create CUDA context")?;

            // Load PTX module
            let ptx_path = format!(
                "{}/wordlist_poc_sm_{}{}.ptx",
                env!("CUDA_KERNELS_DIR"),
                compute_capability_major,
                compute_capability_minor
            );

            let ptx_data = std::fs::read(&ptx_path)
                .with_context(|| format!("Failed to read PTX file: {}", ptx_path))?;
            let ptx_cstring = CString::new(ptx_data)?;

            let mut module = ptr::null_mut();
            check_cuda(cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const _))
                .context("Failed to load CUDA module")?;

            // Get kernel functions
            let kernel_name = CString::new("generate_words_kernel")?;
            let mut kernel = ptr::null_mut();
            check_cuda(cuModuleGetFunction(&mut kernel, module, kernel_name.as_ptr()))
                .context("Failed to get kernel function")?;

            let kernel_transposed_name = CString::new("generate_words_transposed_kernel")?;
            let mut kernel_transposed = ptr::null_mut();
            check_cuda(cuModuleGetFunction(&mut kernel_transposed, module, kernel_transposed_name.as_ptr()))
                .context("Failed to get transposed kernel function")?;

            let kernel_columnmajor_name = CString::new("generate_words_columnmajor_kernel")?;
            let mut kernel_columnmajor = ptr::null_mut();
            check_cuda(cuModuleGetFunction(&mut kernel_columnmajor, module, kernel_columnmajor_name.as_ptr()))
                .context("Failed to get columnmajor kernel function")?;

            Ok(Self {
                context,
                module,
                kernel,
                kernel_transposed,
                kernel_columnmajor,
                device,
                device_id,
                compute_capability: (compute_capability_major, compute_capability_minor),
                persistent_output_buffer: None,
                persistent_output_capacity: 0,
            })
        }
    }

    /// Get device name
    pub fn device_name(&self) -> Result<String> {
        unsafe {
            let mut name = vec![0u8; 256];
            check_cuda(cuDeviceGetName(
                name.as_mut_ptr() as *mut i8,
                256,
                self.device,
            ))?;
            Ok(CString::from_vec_unchecked(name)
                .into_string()
                .unwrap_or_default()
                .trim_end_matches('\0')
                .to_string())
        }
    }

    /// Get compute capability
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Ensure persistent output buffer is allocated with at least the required size.
    /// Reuses existing buffer if large enough, otherwise reallocates.
    ///
    /// Returns the device pointer to use for output.
    unsafe fn ensure_output_buffer(&mut self, required_size: usize) -> Result<CUdeviceptr> {
        // Check if existing buffer is large enough
        if let Some(ptr) = self.persistent_output_buffer {
            if self.persistent_output_capacity >= required_size {
                // Reuse existing buffer
                return Ok(ptr);
            }

            // Free old buffer if too small
            check_cuda(cuMemFree_v2(ptr))?;
        }

        // Allocate new buffer
        let mut new_buffer: CUdeviceptr = 0;
        check_cuda(cuMemAlloc_v2(&mut new_buffer, required_size))?;

        self.persistent_output_buffer = Some(new_buffer);
        self.persistent_output_capacity = required_size;

        Ok(new_buffer)
    }

    /// Generate words using GPU (original uncoalesced kernel)
    pub fn generate_batch(
        &mut self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
    ) -> Result<Vec<u8>> {
        self.generate_batch_internal(charsets, mask, start_idx, batch_size, false, false, output_format)
    }

    /// Generate words using GPU with transposed writes (fully coalesced)
    pub fn generate_batch_transposed(
        &mut self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
    ) -> Result<Vec<u8>> {
        self.generate_batch_internal(charsets, mask, start_idx, batch_size, true, false, output_format)
    }

    /// Generate batch on device with optional stream (async)
    ///
    /// If stream is provided (non-null), kernel launches asynchronously.
    /// Caller must synchronize stream before using device pointer.
    ///
    /// # Arguments
    /// * `charsets` - Character set definitions
    /// * `mask` - Mask pattern
    /// * `start_idx` - Starting index in keyspace
    /// * `batch_size` - Number of words to generate
    /// * `stream` - Optional CUDA stream (null for default stream)
    /// * `output_format` - Output format (0=newlines, 1=fixed-width, 2=packed)
    ///
    /// # Returns
    /// Tuple of (device_pointer, output_size)
    pub fn generate_batch_device_stream(
        &self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        stream: CUstream,
        output_format: i32,
    ) -> Result<(CUdeviceptr, usize)> {
        unsafe {
            // Prepare charset data
            let mut charset_data = Vec::new();
            let mut charset_offsets = Vec::new();
            let mut charset_sizes = Vec::new();

            let max_charset_id = *mask.iter().max().unwrap_or(&0);
            for id in 0..=max_charset_id {
                if let Some(charset) = charsets.get(&id) {
                    charset_offsets.push(charset_data.len() as i32);
                    charset_sizes.push(charset.len() as i32);
                    charset_data.extend_from_slice(charset);
                } else {
                    charset_offsets.push(0);
                    charset_sizes.push(0);
                }
            }

            let mask_pattern: Vec<i32> = mask.iter().map(|&x| x as i32).collect();
            let word_length = mask.len() as i32;

            // Allocate GPU memory
            let mut d_charset_data = 0u64;
            let mut d_charset_offsets = 0u64;
            let mut d_charset_sizes = 0u64;
            let mut d_mask_pattern = 0u64;
            let mut d_output = 0u64;

            // Calculate output size based on format
            let bytes_per_word = match output_format {
                0 => word_length as usize + 1,  // WG_FORMAT_NEWLINES
                1 => word_length as usize + 1,  // WG_FORMAT_FIXED_WIDTH (pad with \0)
                2 => word_length as usize,      // WG_FORMAT_PACKED (no separator)
                _ => word_length as usize + 1,  // fallback
            };
            let output_size = batch_size as usize * bytes_per_word;

            check_cuda(cuMemAlloc_v2(&mut d_charset_data, charset_data.len()))?;
            check_cuda(cuMemAlloc_v2(
                &mut d_charset_offsets,
                charset_offsets.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemAlloc_v2(
                &mut d_charset_sizes,
                charset_sizes.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemAlloc_v2(
                &mut d_mask_pattern,
                mask_pattern.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemAlloc_v2(&mut d_output, output_size))?;

            // Copy data to GPU (async if stream provided)
            if !stream.is_null() {
                check_cuda(cuMemcpyHtoDAsync_v2(
                    d_charset_data,
                    charset_data.as_ptr() as *const _,
                    charset_data.len(),
                    stream,
                ))?;
                check_cuda(cuMemcpyHtoDAsync_v2(
                    d_charset_offsets,
                    charset_offsets.as_ptr() as *const _,
                    charset_offsets.len() * std::mem::size_of::<i32>(),
                    stream,
                ))?;
                check_cuda(cuMemcpyHtoDAsync_v2(
                    d_charset_sizes,
                    charset_sizes.as_ptr() as *const _,
                    charset_sizes.len() * std::mem::size_of::<i32>(),
                    stream,
                ))?;
                check_cuda(cuMemcpyHtoDAsync_v2(
                    d_mask_pattern,
                    mask_pattern.as_ptr() as *const _,
                    mask_pattern.len() * std::mem::size_of::<i32>(),
                    stream,
                ))?;
            } else {
                check_cuda(cuMemcpyHtoD_v2(
                    d_charset_data,
                    charset_data.as_ptr() as *const _,
                    charset_data.len(),
                ))?;
                check_cuda(cuMemcpyHtoD_v2(
                    d_charset_offsets,
                    charset_offsets.as_ptr() as *const _,
                    charset_offsets.len() * std::mem::size_of::<i32>(),
                ))?;
                check_cuda(cuMemcpyHtoD_v2(
                    d_charset_sizes,
                    charset_sizes.as_ptr() as *const _,
                    charset_sizes.len() * std::mem::size_of::<i32>(),
                ))?;
                check_cuda(cuMemcpyHtoD_v2(
                    d_mask_pattern,
                    mask_pattern.as_ptr() as *const _,
                    mask_pattern.len() * std::mem::size_of::<i32>(),
                ))?;
            }

            // Launch kernel
            let block_size: u32 = 256;
            let grid_size: u32 = ((batch_size + block_size as u64 - 1) / block_size as u64) as u32;

            let mut params = [
                &d_charset_data as *const _ as *mut _,
                &d_charset_offsets as *const _ as *mut _,
                &d_charset_sizes as *const _ as *mut _,
                &d_mask_pattern as *const _ as *mut _,
                &start_idx as *const _ as *mut _,
                &word_length as *const _ as *mut _,
                &d_output as *const _ as *mut _,
                &batch_size as *const _ as *mut _,
                &output_format as *const _ as *mut _,
            ];

            check_cuda(cuLaunchKernel(
                self.kernel,
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                stream,  // Use provided stream (null for default)
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;

            // Only synchronize if using default stream (null)
            if stream.is_null() {
                check_cuda(cuCtxSynchronize())?;
            }

            // Cleanup temporary GPU memory (keep d_output)
            // Note: For async operation, these frees happen immediately but
            // CUDA guarantees kernel completion before memory reuse
            check_cuda(cuMemFree_v2(d_charset_data))?;
            check_cuda(cuMemFree_v2(d_charset_offsets))?;
            check_cuda(cuMemFree_v2(d_charset_sizes))?;
            check_cuda(cuMemFree_v2(d_mask_pattern))?;

            // Return device pointer (caller must free)
            Ok((d_output, output_size))
        }
    }

    /// Generate words using hybrid column-major GPU + CPU transpose (FASTEST)
    ///
    /// This method achieves the highest performance by:
    /// 1. GPU writes words in column-major format (fully coalesced writes)
    /// 2. CPU transposes to row-major format using AVX2 SIMD
    ///
    /// Expected performance: 800-1200 M words/s for 12-char passwords
    /// (2-3x faster than standard kernel due to improved memory coalescing)
    pub fn generate_batch_hybrid(
        &mut self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
    ) -> Result<Vec<u8>> {
        // Generate column-major output on GPU
        let column_major = self.generate_batch_internal(
            charsets,
            mask,
            start_idx,
            batch_size,
            false, // don't use transposed kernel
            true,  // use columnmajor kernel
            output_format,
        )?;

        // Transpose to row-major on CPU using SIMD
        let word_length = mask.len() + 1; // +1 for newline
        crate::transpose::transpose_to_rowmajor(&column_major, batch_size as usize, word_length)
    }

    /// Generate batch on device and return device pointer (zero-copy)
    ///
    /// This method generates words directly in GPU memory without copying to host.
    /// The caller is responsible for freeing the returned device pointer.
    ///
    /// Returns: (device_pointer, total_bytes)
    pub fn generate_batch_device(
        &self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
    ) -> Result<(CUdeviceptr, usize)> {
        unsafe {
            // Prepare charset data
            let mut charset_data = Vec::new();
            let mut charset_offsets = Vec::new();
            let mut charset_sizes = Vec::new();

            let max_charset_id = *mask.iter().max().unwrap_or(&0);
            for id in 0..=max_charset_id {
                if let Some(charset) = charsets.get(&id) {
                    charset_offsets.push(charset_data.len() as i32);
                    charset_sizes.push(charset.len() as i32);
                    charset_data.extend_from_slice(charset);
                } else {
                    charset_offsets.push(0);
                    charset_sizes.push(0);
                }
            }

            let mask_pattern: Vec<i32> = mask.iter().map(|&x| x as i32).collect();
            let word_length = mask.len() as i32;

            // Allocate GPU memory
            let mut d_charset_data = 0u64;
            let mut d_charset_offsets = 0u64;
            let mut d_charset_sizes = 0u64;
            let mut d_mask_pattern = 0u64;
            let mut d_output = 0u64;

            // Calculate output size based on format
            let bytes_per_word = match output_format {
                0 => word_length as usize + 1,  // WG_FORMAT_NEWLINES
                1 => word_length as usize + 1,  // WG_FORMAT_FIXED_WIDTH (pad with \0)
                2 => word_length as usize,      // WG_FORMAT_PACKED (no separator)
                _ => word_length as usize + 1,  // fallback
            };
            let output_size = batch_size as usize * bytes_per_word;

            check_cuda(cuMemAlloc_v2(&mut d_charset_data, charset_data.len()))?;
            check_cuda(cuMemAlloc_v2(
                &mut d_charset_offsets,
                charset_offsets.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemAlloc_v2(
                &mut d_charset_sizes,
                charset_sizes.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemAlloc_v2(
                &mut d_mask_pattern,
                mask_pattern.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemAlloc_v2(&mut d_output, output_size))?;

            // Copy data to GPU
            check_cuda(cuMemcpyHtoD_v2(
                d_charset_data,
                charset_data.as_ptr() as *const _,
                charset_data.len(),
            ))?;
            check_cuda(cuMemcpyHtoD_v2(
                d_charset_offsets,
                charset_offsets.as_ptr() as *const _,
                charset_offsets.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemcpyHtoD_v2(
                d_charset_sizes,
                charset_sizes.as_ptr() as *const _,
                charset_sizes.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemcpyHtoD_v2(
                d_mask_pattern,
                mask_pattern.as_ptr() as *const _,
                mask_pattern.len() * std::mem::size_of::<i32>(),
            ))?;

            // Launch kernel (use standard kernel for now)
            let block_size: u32 = 256;
            let grid_size: u32 = ((batch_size + block_size as u64 - 1) / block_size as u64) as u32;

            let mut params = [
                &d_charset_data as *const _ as *mut _,
                &d_charset_offsets as *const _ as *mut _,
                &d_charset_sizes as *const _ as *mut _,
                &d_mask_pattern as *const _ as *mut _,
                &start_idx as *const _ as *mut _,
                &word_length as *const _ as *mut _,
                &d_output as *const _ as *mut _,
                &batch_size as *const _ as *mut _,
                &output_format as *const _ as *mut _,
            ];

            check_cuda(cuLaunchKernel(
                self.kernel,
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                ptr::null_mut(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;

            // Wait for completion
            check_cuda(cuCtxSynchronize())?;

            // Cleanup temporary GPU memory (keep d_output)
            check_cuda(cuMemFree_v2(d_charset_data))?;
            check_cuda(cuMemFree_v2(d_charset_offsets))?;
            check_cuda(cuMemFree_v2(d_charset_sizes))?;
            check_cuda(cuMemFree_v2(d_mask_pattern))?;

            // Return device pointer (caller must free)
            Ok((d_output, output_size))
        }
    }

    /// Internal implementation for word generation
    fn generate_batch_internal(
        &mut self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        use_transposed: bool,
        use_columnmajor: bool,
        output_format: i32,
    ) -> Result<Vec<u8>> {
        unsafe {
            // Prepare charset data
            let mut charset_data = Vec::new();
            let mut charset_offsets = Vec::new();
            let mut charset_sizes = Vec::new();

            let max_charset_id = *mask.iter().max().unwrap_or(&0);
            for id in 0..=max_charset_id {
                if let Some(charset) = charsets.get(&id) {
                    charset_offsets.push(charset_data.len() as i32);
                    charset_sizes.push(charset.len() as i32);
                    charset_data.extend_from_slice(charset);
                } else {
                    charset_offsets.push(0);
                    charset_sizes.push(0);
                }
            }

            let mask_pattern: Vec<i32> = mask.iter().map(|&x| x as i32).collect();
            let word_length = mask.len() as i32;

            // Calculate output size based on format
            let bytes_per_word = match output_format {
                0 => word_length as usize + 1,  // WG_FORMAT_NEWLINES
                1 => word_length as usize + 1,  // WG_FORMAT_FIXED_WIDTH (pad with \0)
                2 => word_length as usize,      // WG_FORMAT_PACKED (no separator)
                _ => word_length as usize + 1,  // fallback
            };
            let output_size = batch_size as usize * bytes_per_word;

            // Allocate temporary GPU memory for kernel inputs
            let mut d_charset_data = 0u64;
            let mut d_charset_offsets = 0u64;
            let mut d_charset_sizes = 0u64;
            let mut d_mask_pattern = 0u64;

            check_cuda(cuMemAlloc_v2(&mut d_charset_data, charset_data.len()))?;
            check_cuda(cuMemAlloc_v2(
                &mut d_charset_offsets,
                charset_offsets.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemAlloc_v2(
                &mut d_charset_sizes,
                charset_sizes.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemAlloc_v2(
                &mut d_mask_pattern,
                mask_pattern.len() * std::mem::size_of::<i32>(),
            ))?;

            // Use persistent output buffer (reuse or allocate)
            let d_output = self.ensure_output_buffer(output_size)?;

            // Copy data to GPU
            check_cuda(cuMemcpyHtoD_v2(
                d_charset_data,
                charset_data.as_ptr() as *const _,
                charset_data.len(),
            ))?;
            check_cuda(cuMemcpyHtoD_v2(
                d_charset_offsets,
                charset_offsets.as_ptr() as *const _,
                charset_offsets.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemcpyHtoD_v2(
                d_charset_sizes,
                charset_sizes.as_ptr() as *const _,
                charset_sizes.len() * std::mem::size_of::<i32>(),
            ))?;
            check_cuda(cuMemcpyHtoD_v2(
                d_mask_pattern,
                mask_pattern.as_ptr() as *const _,
                mask_pattern.len() * std::mem::size_of::<i32>(),
            ))?;

            // Launch kernel
            let block_size: u32 = 256;
            let grid_size: u32 = ((batch_size + block_size as u64 - 1) / block_size as u64) as u32;

            let mut params = [
                &d_charset_data as *const _ as *mut _,
                &d_charset_offsets as *const _ as *mut _,
                &d_charset_sizes as *const _ as *mut _,
                &d_mask_pattern as *const _ as *mut _,
                &start_idx as *const _ as *mut _,
                &word_length as *const _ as *mut _,
                &d_output as *const _ as *mut _,
                &batch_size as *const _ as *mut _,
                &output_format as *const _ as *mut _,
            ];

            let kernel_to_use = if use_columnmajor {
                self.kernel_columnmajor
            } else if use_transposed {
                self.kernel_transposed
            } else {
                self.kernel
            };

            check_cuda(cuLaunchKernel(
                kernel_to_use,
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                ptr::null_mut(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;

            // Wait for completion
            check_cuda(cuCtxSynchronize())?;

            // Copy results back
            let mut output = vec![0u8; output_size];
            check_cuda(cuMemcpyDtoH_v2(
                output.as_mut_ptr() as *mut _,
                d_output,
                output_size,
            ))?;

            // Cleanup temporary GPU memory (persistent buffer is reused)
            check_cuda(cuMemFree_v2(d_charset_data))?;
            check_cuda(cuMemFree_v2(d_charset_offsets))?;
            check_cuda(cuMemFree_v2(d_charset_sizes))?;
            check_cuda(cuMemFree_v2(d_mask_pattern))?;
            // Note: d_output is persistent and managed by self.persistent_output_buffer

            Ok(output)
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            // Free persistent output buffer if allocated
            if let Some(ptr) = self.persistent_output_buffer {
                let _ = cuMemFree_v2(ptr);
            }

            let _ = cuModuleUnload(self.module);
            let _ = cuCtxDestroy_v2(self.context);
        }
    }
}

/// Check CUDA result and convert to anyhow error
unsafe fn check_cuda(result: CUresult) -> Result<()> {
    if result != CUresult::CUDA_SUCCESS {
        let mut error_str = ptr::null();
        cuGetErrorString(result, &mut error_str);
        let error_msg = if !error_str.is_null() {
            std::ffi::CStr::from_ptr(error_str)
                .to_string_lossy()
                .into_owned()
        } else {
            format!("CUDA error code: {:?}", result)
        };
        anyhow::bail!("CUDA error: {}", error_msg);
    }
    Ok(())
}
