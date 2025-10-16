//! POC Benchmark: Prove 2.19 BILLION words/second!
//!
//! This example measures the PURE COMPUTE throughput of our CUDA kernel
//! without any I/O overhead. The kernel generates words but keeps them in registers.
//!
//! Target: RTX 4070 Ti SUPER with 8,448 CUDA cores @ 2.6 GHz
//! Theoretical max: 8,448 × 2.6 GHz / 10 cycles = 2.19 billion words/s

use anyhow::{Context, Result};
use cuda_driver_sys::*;
use std::ffi::CString;
use std::ptr;

const BATCH_SIZE: u64 = 1_000_000_000; // 1 billion words!

fn main() -> Result<()> {
    println!("🚀 GPU Scatter-Gather Wordlist Generator - POC Benchmark");
    println!("{}", "=".repeat(70));
    println!();

    unsafe {
        // Initialize CUDA
        check_cuda(cuInit(0)).context("Failed to initialize CUDA")?;

        // Get device
        let mut device = 0;
        check_cuda(cuDeviceGet(&mut device, 0)).context("Failed to get CUDA device")?;

        // Get device name and properties
        let mut name = vec![0u8; 256];
        check_cuda(cuDeviceGetName(name.as_mut_ptr() as *mut i8, 256, device))?;
        let name_str = CString::from_vec_unchecked(name)
            .into_string()
            .unwrap_or_default()
            .trim_end_matches('\0')
            .to_string();

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

        let mut multiprocessor_count = 0;
        check_cuda(cuDeviceGetAttribute(
            &mut multiprocessor_count,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device,
        ))?;

        println!("GPU: {}", name_str);
        println!("Compute Capability: {}.{}", compute_capability_major, compute_capability_minor);
        println!("Multiprocessors: {}", multiprocessor_count);
        println!();

        // Create context
        let mut context = ptr::null_mut();
        check_cuda(cuCtxCreate_v2(&mut context, 0, device)).context("Failed to create CUDA context")?;

        // Load PTX module
        let ptx_path = format!(
            "{}/wordlist_poc_sm_{}{}.ptx",
            env!("CUDA_KERNELS_DIR"),
            compute_capability_major,
            compute_capability_minor
        );

        println!("Loading kernel from: {}", ptx_path);

        let ptx_data = std::fs::read(&ptx_path)
            .with_context(|| format!("Failed to read PTX file: {}", ptx_path))?;
        let ptx_cstring = CString::new(ptx_data)?;

        let mut module = ptr::null_mut();
        check_cuda(cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const _))
            .context("Failed to load CUDA module")?;

        // Get kernel function
        let kernel_name = CString::new("poc_generate_words_compute_only")?;
        let mut kernel = ptr::null_mut();
        check_cuda(cuModuleGetFunction(&mut kernel, module, kernel_name.as_ptr()))
            .context("Failed to get kernel function")?;

        println!("✅ Kernel loaded successfully!");
        println!();

        // Prepare test data
        // Simple mask: "?1?2" where ?1="abc", ?2="123"
        let charset1 = b"abc";
        let charset2 = b"123";
        let mut charset_data = Vec::new();
        charset_data.extend_from_slice(charset1);
        charset_data.extend_from_slice(charset2);

        let charset_offsets = [0i32, 3];  // ?1 starts at 0, ?2 starts at 3
        let charset_sizes = [3i32, 3];     // Both have 3 characters
        let mask_pattern = [0i32, 1];      // Position 0 uses charset 0, position 1 uses charset 1
        let word_length = 2i32;

        // Allocate GPU memory
        let mut d_charset_data = 0u64;
        let mut d_charset_offsets = 0u64;
        let mut d_charset_sizes = 0u64;
        let mut d_mask_pattern = 0u64;
        let mut d_checksum = 0u64;

        check_cuda(cuMemAlloc_v2(&mut d_charset_data, charset_data.len()))?;
        check_cuda(cuMemAlloc_v2(&mut d_charset_offsets, 2 * std::mem::size_of::<i32>()))?;
        check_cuda(cuMemAlloc_v2(&mut d_charset_sizes, 2 * std::mem::size_of::<i32>()))?;
        check_cuda(cuMemAlloc_v2(&mut d_mask_pattern, 2 * std::mem::size_of::<i32>()))?;
        check_cuda(cuMemAlloc_v2(&mut d_checksum, std::mem::size_of::<u64>()))?;

        // Copy data to GPU
        check_cuda(cuMemcpyHtoD_v2(d_charset_data, charset_data.as_ptr() as *const _, charset_data.len()))?;
        check_cuda(cuMemcpyHtoD_v2(d_charset_offsets, charset_offsets.as_ptr() as *const _, 2 * std::mem::size_of::<i32>()))?;
        check_cuda(cuMemcpyHtoD_v2(d_charset_sizes, charset_sizes.as_ptr() as *const _, 2 * std::mem::size_of::<i32>()))?;
        check_cuda(cuMemcpyHtoD_v2(d_mask_pattern, mask_pattern.as_ptr() as *const _, 2 * std::mem::size_of::<i32>()))?;

        // Kernel launch configuration
        let block_size: u32 = 256;
        let grid_size: u32 = ((BATCH_SIZE + block_size as u64 - 1) / block_size as u64) as u32;

        println!("Launching kernel:");
        println!("  Batch size: {} words", BATCH_SIZE);
        println!("  Block size: {} threads", block_size);
        println!("  Grid size: {} blocks", grid_size);
        println!("  Total threads: {}", grid_size as u64 * block_size as u64);
        println!();

        // Prepare kernel parameters
        let start_idx = 0u64;
        let mut params = [
            &d_charset_data as *const _ as *mut _,
            &d_charset_offsets as *const _ as *mut _,
            &d_charset_sizes as *const _ as *mut _,
            &d_mask_pattern as *const _ as *mut _,
            &start_idx as *const _ as *mut _,
            &word_length as *const _ as *mut _,
            &BATCH_SIZE as *const _ as *mut _,
            &d_checksum as *const _ as *mut _,
        ];

        // Warm-up run
        println!("🔥 Warming up GPU...");
        check_cuda(cuLaunchKernel(
            kernel,
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
        check_cuda(cuCtxSynchronize())?;

        println!("✅ Warm-up complete!");
        println!();

        // THE BIG TEST!
        println!("🚀 RUNNING POC BENCHMARK...");
        println!("   Generating {} words...", BATCH_SIZE);
        println!();

        let start_time = std::time::Instant::now();

        check_cuda(cuLaunchKernel(
            kernel,
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

        check_cuda(cuCtxSynchronize())?;

        let elapsed = start_time.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        let words_per_second = BATCH_SIZE as f64 / elapsed_secs;

        println!("{}", "=".repeat(70));
        println!("🎉 RESULTS:");
        println!("{}", "=".repeat(70));
        println!();
        println!("  Generated {} words in {:.4} seconds", BATCH_SIZE, elapsed_secs);
        println!();
        println!("  ⚡ THROUGHPUT: {:.2} BILLION words/second", words_per_second / 1e9);
        println!();
        println!("  Speedup vs maskprocessor (142M/s): {:.2}x", words_per_second / 142_000_000.0);
        println!();

        // Theoretical maximum
        let theoretical_max = 2.19e9;  // 2.19 billion for RTX 4070 Ti SUPER
        let efficiency = (words_per_second / theoretical_max) * 100.0;
        println!("  Efficiency: {:.1}% of theoretical maximum ({:.2}B words/s)", efficiency, theoretical_max / 1e9);
        println!();

        if words_per_second > 1e9 {
            println!("🔥🔥🔥 WE HIT OVER 1 BILLION WORDS/SECOND! 🔥🔥🔥");
        }
        if words_per_second > 1.47e9 {
            println!("🚀🚀🚀 WE BEAT THE ORIGINAL TARGET (1.47B)! 🚀🚀🚀");
        }
        if words_per_second > 2e9 {
            println!("🎊🎊🎊 WE HIT 2 BILLION WORDS/SECOND! 🎊🎊🎊");
        }

        println!();
        println!("{}", "=".repeat(70));

        // Cleanup
        check_cuda(cuMemFree_v2(d_charset_data))?;
        check_cuda(cuMemFree_v2(d_charset_offsets))?;
        check_cuda(cuMemFree_v2(d_charset_sizes))?;
        check_cuda(cuMemFree_v2(d_mask_pattern))?;
        check_cuda(cuMemFree_v2(d_checksum))?;
        check_cuda(cuModuleUnload(module))?;
        check_cuda(cuCtxDestroy_v2(context))?;
    }

    Ok(())
}

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
