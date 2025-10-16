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
    device: CUdevice,
    compute_capability: (i32, i32),
}

impl GpuContext {
    /// Initialize GPU context and load kernel
    pub fn new() -> Result<Self> {
        unsafe {
            // Initialize CUDA
            check_cuda(cuInit(0)).context("Failed to initialize CUDA")?;

            // Get device
            let mut device = 0;
            check_cuda(cuDeviceGet(&mut device, 0)).context("Failed to get CUDA device")?;

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

            // Get kernel function
            let kernel_name = CString::new("generate_words_kernel")?;
            let mut kernel = ptr::null_mut();
            check_cuda(cuModuleGetFunction(&mut kernel, module, kernel_name.as_ptr()))
                .context("Failed to get kernel function")?;

            Ok(Self {
                context,
                module,
                kernel,
                device,
                compute_capability: (compute_capability_major, compute_capability_minor),
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

    /// Generate words using GPU
    pub fn generate_batch(
        &self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
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

            // Allocate GPU memory
            let mut d_charset_data = 0u64;
            let mut d_charset_offsets = 0u64;
            let mut d_charset_sizes = 0u64;
            let mut d_mask_pattern = 0u64;
            let mut d_output = 0u64;

            let output_size = batch_size as usize * (word_length as usize + 1); // +1 for newline

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

            // Copy results back
            let mut output = vec![0u8; output_size];
            check_cuda(cuMemcpyDtoH_v2(
                output.as_mut_ptr() as *mut _,
                d_output,
                output_size,
            ))?;

            // Cleanup GPU memory
            check_cuda(cuMemFree_v2(d_charset_data))?;
            check_cuda(cuMemFree_v2(d_charset_offsets))?;
            check_cuda(cuMemFree_v2(d_charset_sizes))?;
            check_cuda(cuMemFree_v2(d_mask_pattern))?;
            check_cuda(cuMemFree_v2(d_output))?;

            Ok(output)
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
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
