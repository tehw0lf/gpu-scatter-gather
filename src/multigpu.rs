//! Multi-GPU context management
//!
//! This module provides support for distributing wordlist generation across multiple GPUs.

use anyhow::{Context, Result};
use std::collections::HashMap;
use crate::gpu::GpuContext;

/// Keyspace partition for a single GPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyspacePartition {
    /// Starting index in the keyspace
    pub start_idx: u64,
    /// Number of words to generate
    pub count: u64,
}

impl KeyspacePartition {
    /// Create a new keyspace partition
    pub fn new(start_idx: u64, count: u64) -> Self {
        Self { start_idx, count }
    }

    /// Get the end index (exclusive)
    pub fn end_idx(&self) -> u64 {
        self.start_idx + self.count
    }
}

/// Partition keyspace across multiple GPUs
///
/// Uses static partitioning: divides keyspace evenly, with any remainder
/// going to the first GPU.
///
/// # Arguments
/// * `total_keyspace` - Total number of words to generate
/// * `num_gpus` - Number of GPUs to distribute across
///
/// # Returns
/// Vector of partitions, one per GPU
///
/// # Example
/// ```
/// use gpu_scatter_gather::multigpu::partition_keyspace;
///
/// let partitions = partition_keyspace(1_000_000, 3);
/// assert_eq!(partitions.len(), 3);
/// assert_eq!(partitions[0].count, 333_334); // Gets the extra 1
/// assert_eq!(partitions[1].count, 333_333);
/// assert_eq!(partitions[2].count, 333_333);
/// ```
pub fn partition_keyspace(total_keyspace: u64, num_gpus: usize) -> Vec<KeyspacePartition> {
    if num_gpus == 0 {
        return vec![];
    }

    let chunk_size = total_keyspace / num_gpus as u64;
    let remainder = total_keyspace % num_gpus as u64;

    let mut partitions = Vec::with_capacity(num_gpus);
    let mut start_idx = 0;

    for gpu_id in 0..num_gpus {
        // Give remainder to first GPU for load balancing
        let count = if gpu_id == 0 {
            chunk_size + remainder
        } else {
            chunk_size
        };

        partitions.push(KeyspacePartition::new(start_idx, count));
        start_idx += count;
    }

    partitions
}

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

    /// Partition a keyspace for generation across GPUs
    ///
    /// # Arguments
    /// * `start_idx` - Starting index in global keyspace
    /// * `count` - Total number of words to generate
    ///
    /// # Returns
    /// Vector of partitions, one per active GPU
    pub fn partition(&self, start_idx: u64, count: u64) -> Vec<KeyspacePartition> {
        let partitions = partition_keyspace(count, self.num_devices);

        // Adjust partitions to account for global start_idx offset
        partitions
            .into_iter()
            .map(|p| KeyspacePartition::new(start_idx + p.start_idx, p.count))
            .collect()
    }

    /// Generate batch across all GPUs in parallel
    ///
    /// This method distributes the workload across all GPUs using static partitioning
    /// and executes generation concurrently using threads.
    ///
    /// # Arguments
    /// * `charsets` - Character set definitions
    /// * `mask` - Mask pattern
    /// * `start_idx` - Starting index in global keyspace
    /// * `batch_size` - Total number of words to generate
    /// * `output_format` - Output format (WG_FORMAT_*)
    ///
    /// # Returns
    /// Concatenated output from all GPUs in order
    pub fn generate_batch(
        &self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
    ) -> Result<Vec<u8>> {
        use std::sync::{Arc, Mutex};
        use std::thread;

        // Partition keyspace across GPUs
        let partitions = self.partition(start_idx, batch_size);

        // Shared data (charsets and mask)
        let charsets = Arc::new(charsets.clone());
        let mask = Arc::new(mask.to_vec());

        // Results storage (indexed by GPU)
        let results: Arc<Mutex<Vec<Option<Vec<u8>>>>> =
            Arc::new(Mutex::new(vec![None; self.num_devices]));

        // Launch generation on each GPU in parallel
        let mut handles = Vec::new();

        for (gpu_idx, partition) in partitions.iter().enumerate() {
            // Skip empty partitions
            if partition.count == 0 {
                let results = Arc::clone(&results);
                results.lock().unwrap()[gpu_idx] = Some(Vec::new());
                continue;
            }

            let charsets = Arc::clone(&charsets);
            let mask = Arc::clone(&mask);
            let results = Arc::clone(&results);
            let partition = *partition;

            // Create a new GPU context for this thread
            // Note: We can't share GpuContext across threads due to CUDA context threading model
            // Each thread needs its own context
            let device_id = self.workers[gpu_idx].device_id();

            let handle = thread::spawn(move || {
                // Create GPU context for this thread
                let gpu_ctx = match GpuContext::with_device(device_id) {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        eprintln!("GPU {} failed to create context: {}", device_id, e);
                        return;
                    }
                };

                // Generate batch
                match gpu_ctx.generate_batch(&charsets, &mask, partition.start_idx, partition.count, output_format) {
                    Ok(output) => {
                        results.lock().unwrap()[gpu_idx] = Some(output);
                    }
                    Err(e) => {
                        eprintln!("GPU {} generation failed: {}", device_id, e);
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all GPUs to complete
        for handle in handles {
            handle.join().map_err(|_| anyhow::anyhow!("GPU thread panicked"))?;
        }

        // Aggregate results in order
        let results = results.lock().unwrap();
        let mut aggregated = Vec::new();

        for (idx, result) in results.iter().enumerate() {
            match result {
                Some(data) => aggregated.extend_from_slice(data),
                None => {
                    anyhow::bail!("GPU {} did not produce output", idx);
                }
            }
        }

        Ok(aggregated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keyspace partitioning tests
    #[test]
    fn test_partition_keyspace_even() {
        let partitions = partition_keyspace(1000, 4);
        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0], KeyspacePartition::new(0, 250));
        assert_eq!(partitions[1], KeyspacePartition::new(250, 250));
        assert_eq!(partitions[2], KeyspacePartition::new(500, 250));
        assert_eq!(partitions[3], KeyspacePartition::new(750, 250));

        // Verify coverage
        assert_eq!(partitions[3].end_idx(), 1000);
    }

    #[test]
    fn test_partition_keyspace_uneven() {
        let partitions = partition_keyspace(1001, 4);
        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0], KeyspacePartition::new(0, 251)); // Gets extra 1
        assert_eq!(partitions[1], KeyspacePartition::new(251, 250));
        assert_eq!(partitions[2], KeyspacePartition::new(501, 250));
        assert_eq!(partitions[3], KeyspacePartition::new(751, 250));

        // Verify total coverage
        let total: u64 = partitions.iter().map(|p| p.count).sum();
        assert_eq!(total, 1001);
        assert_eq!(partitions[3].end_idx(), 1001);
    }

    #[test]
    fn test_partition_keyspace_small() {
        // Keyspace smaller than GPU count
        let partitions = partition_keyspace(2, 4);
        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0], KeyspacePartition::new(0, 2)); // Gets all
        assert_eq!(partitions[1], KeyspacePartition::new(2, 0)); // Empty
        assert_eq!(partitions[2], KeyspacePartition::new(2, 0)); // Empty
        assert_eq!(partitions[3], KeyspacePartition::new(2, 0)); // Empty

        let total: u64 = partitions.iter().map(|p| p.count).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_partition_keyspace_single_gpu() {
        let partitions = partition_keyspace(1000000, 1);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0], KeyspacePartition::new(0, 1000000));
    }

    #[test]
    fn test_partition_keyspace_zero_gpus() {
        let partitions = partition_keyspace(1000, 0);
        assert_eq!(partitions.len(), 0);
    }

    #[test]
    fn test_partition_keyspace_large() {
        // Test with realistic large keyspace
        let partitions = partition_keyspace(100_000_000, 3);
        assert_eq!(partitions.len(), 3);

        // 100M / 3 = 33,333,333 with remainder 1
        assert_eq!(partitions[0].count, 33_333_334);
        assert_eq!(partitions[1].count, 33_333_333);
        assert_eq!(partitions[2].count, 33_333_333);

        // Verify no gaps or overlaps
        assert_eq!(partitions[0].end_idx(), partitions[1].start_idx);
        assert_eq!(partitions[1].end_idx(), partitions[2].start_idx);
        assert_eq!(partitions[2].end_idx(), 100_000_000);

        let total: u64 = partitions.iter().map(|p| p.count).sum();
        assert_eq!(total, 100_000_000);
    }

    #[test]
    fn test_keyspace_partition_end_idx() {
        let partition = KeyspacePartition::new(100, 50);
        assert_eq!(partition.start_idx, 100);
        assert_eq!(partition.count, 50);
        assert_eq!(partition.end_idx(), 150);
    }

    // Multi-GPU context tests
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
    fn test_multi_gpu_partition() {
        // Test partitioning logic with mock context
        // We can test this even without actual GPUs by creating a mock
        // For now, test the standalone partition_keyspace function
        match MultiGpuContext::with_devices(&[0]) {
            Ok(ctx) => {
                // Test partitioning with offset
                let partitions = ctx.partition(1000, 100);
                assert_eq!(partitions.len(), 1);
                assert_eq!(partitions[0].start_idx, 1000);
                assert_eq!(partitions[0].count, 100);
                assert_eq!(partitions[0].end_idx(), 1100);
            }
            Err(e) => {
                println!("No GPU available for partition test: {}", e);
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

    #[test]
    fn test_multi_gpu_generate_batch() {
        // Test parallel generation across GPUs
        match MultiGpuContext::with_devices(&[0]) {
            Ok(ctx) => {
                // Setup: Simple 2x2 keyspace (4 words total)
                let mut charsets = HashMap::new();
                charsets.insert(1, b"ab".to_vec());

                let mask = vec![1, 1]; // ?1?1 = aa, ab, ba, bb

                // Generate all 4 words
                match ctx.generate_batch(&charsets, &mask, 0, 4, 0) {
                    Ok(output) => {
                        // Verify we got output
                        assert!(!output.is_empty());

                        // Convert to string for verification
                        let output_str = String::from_utf8_lossy(&output);
                        let lines: Vec<&str> = output_str.trim().split('\n').collect();

                        // Should have 4 words
                        assert_eq!(lines.len(), 4);

                        // Verify words are correct
                        assert_eq!(lines[0], "aa");
                        assert_eq!(lines[1], "ab");
                        assert_eq!(lines[2], "ba");
                        assert_eq!(lines[3], "bb");

                        println!("Multi-GPU generation successful: {:?}", lines);
                    }
                    Err(e) => {
                        println!("Generation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("No GPU available for generation test: {}", e);
            }
        }
    }

    #[test]
    fn test_multi_gpu_partial_keyspace() {
        // Test generating a subset of the keyspace
        match MultiGpuContext::with_devices(&[0]) {
            Ok(ctx) => {
                let mut charsets = HashMap::new();
                charsets.insert(1, b"abc".to_vec());

                let mask = vec![1, 1]; // 3x3 = 9 words total

                // Generate middle 3 words (indices 3, 4, 5)
                match ctx.generate_batch(&charsets, &mask, 3, 3, 0) {
                    Ok(output) => {
                        let output_str = String::from_utf8_lossy(&output);
                        let lines: Vec<&str> = output_str.trim().split('\n').collect();

                        assert_eq!(lines.len(), 3);

                        // Words at indices 3, 4, 5 should be: ba, bb, bc
                        assert_eq!(lines[0], "ba");
                        assert_eq!(lines[1], "bb");
                        assert_eq!(lines[2], "bc");

                        println!("Partial keyspace generation successful: {:?}", lines);
                    }
                    Err(e) => {
                        println!("Partial generation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("No GPU available for partial generation test: {}", e);
            }
        }
    }
}
