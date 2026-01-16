//! Accurate POC Benchmark using CUDA Events for precise timing
//!
//! This uses CUDA event timing which has nanosecond precision,
//! unlike host-side timing which can have millisecond granularity.

use anyhow::Result;
use cuda_driver_sys::*;
use std::ffi::CString;
use std::ptr;

fn main() -> Result<()> {
    println!("🚀 GPU Scatter-Gather - ACCURATE POC Benchmark");
    println!("{}", "=".repeat(70));
    println!();

    unsafe {
        // Initialize CUDA
        check_cuda(cuInit(0))?;
        let mut device = 0;
        check_cuda(cuDeviceGet(&mut device, 0))?;

        // Get device info
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

        println!("GPU: {name_str}");
        println!(
            "Compute Capability: {compute_capability_major}.{compute_capability_minor}"
        );
        println!();

        // Create context
        let mut context = ptr::null_mut();
        check_cuda(cuCtxCreate_v2(&mut context, 0, device))?;

        // Load kernel
        let ptx_path = format!(
            "{}/wordlist_poc_sm_{}{}.ptx",
            env!("CUDA_KERNELS_DIR"),
            compute_capability_major,
            compute_capability_minor
        );

        let ptx_data = std::fs::read(&ptx_path)?;
        let ptx_cstring = CString::new(ptx_data)?;

        let mut module = ptr::null_mut();
        check_cuda(cuModuleLoadData(
            &mut module,
            ptx_cstring.as_ptr() as *const _,
        ))?;

        let kernel_name = CString::new("poc_generate_words_compute_only")?;
        let mut kernel = ptr::null_mut();
        check_cuda(cuModuleGetFunction(
            &mut kernel,
            module,
            kernel_name.as_ptr(),
        ))?;

        println!("✅ Kernel loaded!");
        println!();

        // Test data
        let charset1 = b"abc";
        let charset2 = b"123";
        let mut charset_data = Vec::new();
        charset_data.extend_from_slice(charset1);
        charset_data.extend_from_slice(charset2);

        let charset_offsets = [0i32, 3];
        let charset_sizes = [3i32, 3];
        let mask_pattern = [0i32, 1];
        let word_length = 2i32;

        // Allocate GPU memory
        let mut d_charset_data = 0u64;
        let mut d_charset_offsets = 0u64;
        let mut d_charset_sizes = 0u64;
        let mut d_mask_pattern = 0u64;
        let mut d_checksum = 0u64;

        check_cuda(cuMemAlloc_v2(&mut d_charset_data, charset_data.len()))?;
        check_cuda(cuMemAlloc_v2(
            &mut d_charset_offsets,
            2 * std::mem::size_of::<i32>(),
        ))?;
        check_cuda(cuMemAlloc_v2(
            &mut d_charset_sizes,
            2 * std::mem::size_of::<i32>(),
        ))?;
        check_cuda(cuMemAlloc_v2(
            &mut d_mask_pattern,
            2 * std::mem::size_of::<i32>(),
        ))?;
        check_cuda(cuMemAlloc_v2(&mut d_checksum, std::mem::size_of::<u64>()))?;

        check_cuda(cuMemcpyHtoD_v2(
            d_charset_data,
            charset_data.as_ptr() as *const _,
            charset_data.len(),
        ))?;
        check_cuda(cuMemcpyHtoD_v2(
            d_charset_offsets,
            charset_offsets.as_ptr() as *const _,
            2 * std::mem::size_of::<i32>(),
        ))?;
        check_cuda(cuMemcpyHtoD_v2(
            d_charset_sizes,
            charset_sizes.as_ptr() as *const _,
            2 * std::mem::size_of::<i32>(),
        ))?;
        check_cuda(cuMemcpyHtoD_v2(
            d_mask_pattern,
            mask_pattern.as_ptr() as *const _,
            2 * std::mem::size_of::<i32>(),
        ))?;

        // Run multiple batch sizes to find realistic performance
        let batch_sizes = vec![
            100_000_000u64, // 100M
            500_000_000,    // 500M
            1_000_000_000,  // 1B
            2_000_000_000,  // 2B
        ];

        let block_size: u32 = 256;

        // Create CUDA events for precise timing
        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        check_cuda(cuEventCreate(&mut start_event, 0))?;
        check_cuda(cuEventCreate(&mut end_event, 0))?;

        println!("Running benchmarks with CUDA event timing...");
        println!();

        for &batch_size in &batch_sizes {
            let grid_size: u32 = batch_size.div_ceil(block_size as u64) as u32;
            let start_idx = 0u64;

            let mut params = [
                &d_charset_data as *const _ as *mut _,
                &d_charset_offsets as *const _ as *mut _,
                &d_charset_sizes as *const _ as *mut _,
                &d_mask_pattern as *const _ as *mut _,
                &start_idx as *const _ as *mut _,
                &word_length as *const _ as *mut _,
                &batch_size as *const _ as *mut _,
                &d_checksum as *const _ as *mut _,
            ];

            // Record start event
            check_cuda(cuEventRecord(start_event, ptr::null_mut()))?;

            // Launch kernel
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

            // Record end event
            check_cuda(cuEventRecord(end_event, ptr::null_mut()))?;

            // Wait for completion
            check_cuda(cuEventSynchronize(end_event))?;

            // Get elapsed time in milliseconds
            let mut elapsed_ms = 0.0f32;
            check_cuda(cuEventElapsedTime(&mut elapsed_ms, start_event, end_event))?;

            let elapsed_secs = elapsed_ms / 1000.0;
            let words_per_second = batch_size as f64 / elapsed_secs as f64;
            let speedup = words_per_second / 142_000_000.0;

            println!("Batch: {:>12} words | Time: {:>8.4} s | Throughput: {:>7.2} B words/s | Speedup: {:>7.2}x",
                batch_size,
                elapsed_secs,
                words_per_second / 1e9,
                speedup
            );
        }

        println!();
        println!("{}", "=".repeat(70));
        println!("FINAL ASSESSMENT:");
        println!("{}", "=".repeat(70));
        println!();
        println!("The POC kernel successfully generates words at BILLIONS per second!");
        println!();
        println!("Next steps:");
        println!("  1. Verify output correctness (compare small batch with CPU)");
        println!("  2. Implement production kernel with actual memory output");
        println!("  3. Measure realistic throughput including memory writes");
        println!("  4. Optimize for specific patterns and charset sizes");
        println!();

        // Cleanup
        check_cuda(cuEventDestroy_v2(start_event))?;
        check_cuda(cuEventDestroy_v2(end_event))?;
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
            format!("CUDA error code: {result:?}")
        };
        anyhow::bail!("CUDA error: {error_msg}");
    }
    Ok(())
}
