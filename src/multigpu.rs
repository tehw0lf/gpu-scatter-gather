//! Multi-GPU context management
//!
//! This module provides support for distributing wordlist generation across multiple GPUs.

use anyhow::{Context, Result};
use std::collections::HashMap;
use crate::gpu::GpuContext;

/// Worker managing a single GPU device
pub struct GpuWorker {
    /// Device ID (0-based)
    device_id: i32,
    /// GPU context for this device
    context: GpuContext,
}

impl GpuWorker {
    /// Create a new GPU worker for the specified device
    pub fn new(device_id: i32) -> Result<Self> {
        let context = GpuContext::with_device(device_id)
            .with_context(|| format!("Failed to create GPU context for device {}", device_id))?;

        Ok(Self {
            device_id,
            context,
        })
    }

    /// Get the device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get a reference to the GPU context
    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    /// Get a mutable reference to the GPU context
    pub fn context_mut(&mut self) -> &mut GpuContext {
        &mut self.context
    }
}

/// Multi-GPU context for parallel wordlist generation
pub struct MultiGpuContext {
    /// Workers for each GPU
    workers: Vec<GpuWorker>,
    /// Number of devices
    num_devices: usize,
}

impl MultiGpuContext {
    /// Create multi-GPU context with all available devices
    pub fn new() -> Result<Self> {
        // Get device count
        let device_count = unsafe {
            cuda_driver_sys::cuInit(0);
            let mut count = 0;
            if cuda_driver_sys::cuDeviceGetCount(&mut count) != cuda_driver_sys::CUresult::CUDA_SUCCESS {
                anyhow::bail!("Failed to get device count");
            }
            count
        };

        if device_count == 0 {
            anyhow::bail!("No CUDA devices found");
        }

        Self::with_devices(&(0..device_count).collect::<Vec<_>>())
    }

    /// Create multi-GPU context with specific devices
    pub fn with_devices(device_ids: &[i32]) -> Result<Self> {
        if device_ids.is_empty() {
            anyhow::bail!("Must specify at least one device");
        }

        // Create workers for each device
        let mut workers = Vec::with_capacity(device_ids.len());
        for &device_id in device_ids {
            match GpuWorker::new(device_id) {
                Ok(worker) => workers.push(worker),
                Err(e) => {
                    // Log warning but continue with remaining devices
                    eprintln!("Warning: Failed to initialize device {}: {}", device_id, e);
                    eprintln!("Continuing with remaining devices...");
                }
            }
        }

        if workers.is_empty() {
            anyhow::bail!("Failed to initialize any GPU devices");
        }

        let num_devices = workers.len();

        Ok(Self {
            workers,
            num_devices,
        })
    }

    /// Get number of active devices
    pub fn num_devices(&self) -> usize {
        self.num_devices
    }

    /// Get workers
    pub fn workers(&self) -> &[GpuWorker] {
        &self.workers
    }

    /// Get mutable workers
    pub fn workers_mut(&mut self) -> &mut [GpuWorker] {
        &mut self.workers
    }

    /// Get a specific worker by index
    pub fn worker(&self, index: usize) -> Option<&GpuWorker> {
        self.workers.get(index)
    }

    /// Get a specific worker by index (mutable)
    pub fn worker_mut(&mut self, index: usize) -> Option<&mut GpuWorker> {
        self.workers.get_mut(index)
    }

    /// Generate batch across all GPUs
    ///
    /// This is a placeholder for future implementation in Week 4
    pub fn generate_batch(
        &self,
        _charsets: &HashMap<usize, Vec<u8>>,
        _mask: &[usize],
        _start_idx: u64,
        _batch_size: u64,
        _output_format: i32,
    ) -> Result<Vec<u8>> {
        // TODO: Implement in Week 4 (parallel generation)
        anyhow::bail!("Multi-GPU generation not yet implemented (Week 4)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_context_creation() {
        // Try to create multi-GPU context
        // This will fail if no GPUs are available, which is fine
        match MultiGpuContext::new() {
            Ok(ctx) => {
                assert!(ctx.num_devices() >= 1);
                println!("Created multi-GPU context with {} device(s)", ctx.num_devices());
            }
            Err(e) => {
                println!("No GPUs available: {}", e);
            }
        }
    }

    #[test]
    fn test_multi_gpu_with_single_device() {
        // Test with device 0 only
        match MultiGpuContext::with_devices(&[0]) {
            Ok(ctx) => {
                assert_eq!(ctx.num_devices(), 1);
                assert_eq!(ctx.worker(0).unwrap().device_id(), 0);
            }
            Err(e) => {
                println!("Device 0 not available: {}", e);
            }
        }
    }

    #[test]
    fn test_gpu_worker_creation() {
        // Test creating individual worker
        match GpuWorker::new(0) {
            Ok(worker) => {
                assert_eq!(worker.device_id(), 0);
            }
            Err(e) => {
                println!("Failed to create worker for device 0: {}", e);
            }
        }
    }
}
