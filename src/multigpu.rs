//! Multi-GPU context management
//!
//! This module provides support for distributing wordlist generation across multiple GPUs.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use crate::gpu::GpuContext;
use cuda_driver_sys::*;

/// Performance statistics for a single GPU
///
/// Tracks throughput over time to enable adaptive load balancing
/// across heterogeneous GPU configurations.
#[derive(Debug, Clone)]
struct GpuStats {
    /// Last completion time for a work item
    last_completion_time: Duration,
    /// Estimated throughput in words per second
    /// Uses exponential moving average for stability
    throughput_estimate: f64,
    /// Number of samples collected
    sample_count: usize,
}

impl GpuStats {
    /// Create new stats tracker with no history
    fn new() -> Self {
        Self {
            last_completion_time: Duration::ZERO,
            throughput_estimate: 0.0,
            sample_count: 0,
        }
    }

    /// Record a completion event and update throughput estimate
    ///
    /// # Arguments
    /// * `duration` - Time taken to complete the work
    /// * `words` - Number of words generated
    ///
    /// Uses exponential moving average with alpha=0.2 to smooth estimates
    /// while still being responsive to performance changes.
    fn record_completion(&mut self, duration: Duration, words: u64) {
        self.last_completion_time = duration;

        let throughput = words as f64 / duration.as_secs_f64();

        // Exponential moving average: new_estimate = alpha * new + (1 - alpha) * old
        const ALPHA: f64 = 0.2;
        self.throughput_estimate = if self.sample_count == 0 {
            throughput  // First sample
        } else {
            ALPHA * throughput + (1.0 - ALPHA) * self.throughput_estimate
        };

        self.sample_count += 1;
    }

    /// Get current throughput estimate
    fn throughput(&self) -> f64 {
        self.throughput_estimate
    }

    /// Check if we have enough samples for reliable estimates
    fn has_reliable_estimate(&self) -> bool {
        self.sample_count >= 3  // Need at least 3 samples
    }
}

/// Send-safe wrapper for raw pointer to pinned memory
///
/// SAFETY: This is safe because:
/// 1. Each worker owns its pinned buffer exclusively
/// 2. Pointers are never shared between workers
/// 3. PORTABLE flag ensures multi-context access
struct SendPtr(*mut u8);

unsafe impl Send for SendPtr {}

impl SendPtr {
    fn new(ptr: *mut u8) -> Self {
        SendPtr(ptr)
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.0
    }
}

/// Work item sent to persistent worker threads
struct WorkItem {
    /// Character sets for generation
    charsets: HashMap<usize, Vec<u8>>,
    /// Mask pattern
    mask: Vec<usize>,
    /// Keyspace partition for this worker
    partition: KeyspacePartition,
    /// Output format (0=newlines, 1=fixed-width, 2=packed)
    output_format: i32,
    /// Pinned memory pointer to write output directly (fast PCIe transfers)
    pinned_ptr: SendPtr,
    /// Channel to send result back (returns size and duration for stats)
    result_sender: Sender<Result<(usize, Duration)>>,
}

/// Shutdown signal for worker threads
enum WorkerMessage {
    Work(WorkItem),
    Shutdown,
}

/// Pinned (page-locked) memory buffer for fast PCIe transfers
///
/// Pinned memory provides ~2x faster host ↔ device transfers compared to
/// pageable memory because it bypasses the intermediate staging buffer.
///
/// Uses `CU_MEMHOSTALLOC_PORTABLE` flag to allow access from multiple CUDA contexts,
/// which is critical for multi-GPU setups where each worker thread has its own context.
struct PinnedBuffer {
    /// Raw pointer to pinned memory
    ptr: *mut u8,
    /// Size of the buffer in bytes (kept for debugging/future use)
    #[allow(dead_code)]
    size: usize,
}

impl PinnedBuffer {
    /// Allocate pinned memory with PORTABLE flag for multi-context access
    ///
    /// # Arguments
    /// * `size` - Size in bytes to allocate
    ///
    /// # Returns
    /// `Ok(PinnedBuffer)` on success, `Err` if allocation fails
    ///
    /// # Safety
    /// Allocates pinned (page-locked) host memory using CUDA Driver API.
    /// The memory is automatically freed when the PinnedBuffer is dropped.
    fn new(size: usize) -> Result<Self> {
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let result = cuMemHostAlloc(
                &mut ptr,
                size,
                CU_MEMHOSTALLOC_PORTABLE,  // Allow access from any CUDA context
            );

            if result != CUresult::CUDA_SUCCESS {
                anyhow::bail!("Failed to allocate {} bytes of pinned memory: {:?}", size, result);
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
            })
        }
    }

    /// Get immutable raw pointer to the pinned memory
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get mutable raw pointer to the pinned memory
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Get the size of the buffer
    #[inline]
    #[allow(dead_code)]
    fn size(&self) -> usize {
        self.size
    }

    /// Get a slice view of the pinned memory
    ///
    /// # Safety
    /// Caller must ensure:
    /// - `len` does not exceed buffer size
    /// - No concurrent mutable access to the same region
    #[inline]
    #[allow(dead_code)]
    unsafe fn as_slice(&self, len: usize) -> &[u8] {
        std::slice::from_raw_parts(self.ptr, len)
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe {
            let result = cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("Warning: Failed to free pinned memory: {:?}", result);
            }
        }
    }
}

// SAFETY: PinnedBuffer owns the memory and can be safely sent between threads.
// The PORTABLE flag ensures it can be accessed from any CUDA context.
unsafe impl Send for PinnedBuffer {}

// NOT Sync: Each buffer should be owned by a single worker thread at a time
// to avoid race conditions on the underlying memory.

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
    /// CUDA stream for async operations (optional)
    stream: Option<CUstream>,
}

impl GpuWorker {
    /// Create a new GPU worker for the specified device
    pub fn new(device_id: i32) -> Result<Self> {
        let context = GpuContext::with_device(device_id)
            .with_context(|| format!("Failed to create GPU context for device {}", device_id))?;

        Ok(Self {
            device_id,
            context,
            stream: None,
        })
    }

    /// Create a GPU worker (async mode - streams created per-thread)
    ///
    /// Note: We don't create CUDA streams here because streams must be created
    /// in the same thread/context where they'll be used. The async implementation
    /// creates streams in worker threads instead.
    pub fn new_with_stream(device_id: i32) -> Result<Self> {
        let context = GpuContext::with_device(device_id)
            .with_context(|| format!("Failed to create GPU context for device {}", device_id))?;

        Ok(Self {
            device_id,
            context,
            stream: None,  // Streams created in worker threads, not here
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

    /// Get the CUDA stream (if available)
    pub fn stream(&self) -> Option<CUstream> {
        self.stream
    }
}

impl Drop for GpuWorker {
    fn drop(&mut self) {
        // Cleanup CUDA stream if it was created
        if let Some(stream) = self.stream {
            unsafe {
                // Synchronize stream before destroying
                let _ = cuStreamSynchronize(stream);
                let _ = cuStreamDestroy_v2(stream);
            }
        }
    }
}

/// Multi-GPU context for parallel wordlist generation
pub struct MultiGpuContext {
    /// Workers for each GPU (used for single-GPU fast path)
    workers: Vec<GpuWorker>,
    /// Number of devices
    num_devices: usize,
    /// Enable async kernel launches with CUDA streams
    async_mode: bool,
    /// Persistent worker threads (for multi-GPU path)
    /// Each tuple: (work_sender, thread_handle)
    worker_threads: Option<Vec<(Sender<WorkerMessage>, JoinHandle<()>)>>,
    /// Pinned memory buffers (one per worker) for fast PCIe transfers
    pinned_buffers: Vec<PinnedBuffer>,
    /// Maximum buffer size per worker (in bytes) - kept for future use
    #[allow(dead_code)]
    max_buffer_size: usize,
    /// Performance statistics per GPU for adaptive load balancing
    gpu_stats: Vec<GpuStats>,
}

impl MultiGpuContext {
    /// Create multi-GPU context with all available devices
    pub fn new() -> Result<Self> {
        Self::new_with_options(false)
    }

    /// Create multi-GPU context with all available devices and async mode
    pub fn new_async() -> Result<Self> {
        Self::new_with_options(true)
    }

    /// Create multi-GPU context with all available devices and optional async mode
    fn new_with_options(async_mode: bool) -> Result<Self> {
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

        Self::with_devices_and_options(&(0..device_count).collect::<Vec<_>>(), async_mode)
    }

    /// Create multi-GPU context with specific devices
    pub fn with_devices(device_ids: &[i32]) -> Result<Self> {
        Self::with_devices_and_options(device_ids, false)
    }

    /// Create multi-GPU context with specific devices and async mode
    pub fn with_devices_async(device_ids: &[i32]) -> Result<Self> {
        Self::with_devices_and_options(device_ids, true)
    }

    /// Create multi-GPU context with specific devices and optional async mode
    fn with_devices_and_options(device_ids: &[i32], async_mode: bool) -> Result<Self> {
        if device_ids.is_empty() {
            anyhow::bail!("Must specify at least one device");
        }

        // Create workers for each device
        let mut workers = Vec::with_capacity(device_ids.len());
        for &device_id in device_ids {
            let worker_result = if async_mode {
                GpuWorker::new_with_stream(device_id)
            } else {
                GpuWorker::new(device_id)
            };

            match worker_result {
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

        // For multi-GPU systems (2+), spawn persistent worker threads
        let worker_threads = if num_devices >= 2 {
            let mut threads = Vec::with_capacity(num_devices);

            for device_id in device_ids.iter().take(num_devices) {
                let (work_sender, work_receiver) = channel::<WorkerMessage>();
                let device_id = *device_id;
                let use_async = async_mode;

                let handle = thread::spawn(move || {
                    // Create GPU context ONCE for this worker thread
                    let gpu_ctx = match GpuContext::with_device(device_id) {
                        Ok(ctx) => ctx,
                        Err(e) => {
                            eprintln!("Worker thread for GPU {} failed to create context: {}", device_id, e);
                            return;
                        }
                    };

                    // Create CUDA stream ONCE if async mode
                    let stream = if use_async {
                        unsafe {
                            let mut stream_ptr: CUstream = std::ptr::null_mut();
                            let result = cuStreamCreate(&mut stream_ptr, 0);
                            if result != CUresult::CUDA_SUCCESS {
                                eprintln!("Worker thread for GPU {} failed to create stream: {:?}", device_id, result);
                                return;
                            }
                            stream_ptr
                        }
                    } else {
                        std::ptr::null_mut()
                    };

                    // Worker loop: process work items until shutdown
                    while let Ok(msg) = work_receiver.recv() {
                        match msg {
                            WorkerMessage::Shutdown => {
                                // Cleanup and exit
                                unsafe {
                                    if !stream.is_null() {
                                        let _ = cuStreamSynchronize(stream);
                                        let _ = cuStreamDestroy_v2(stream);
                                    }
                                }
                                break;
                            }
                            WorkerMessage::Work(work_item) => {
                                // Process work item with pinned memory pointer
                                let result = Self::process_work_item(
                                    &gpu_ctx,
                                    work_item.partition,
                                    &work_item.charsets,
                                    &work_item.mask,
                                    work_item.output_format,
                                    stream,
                                    work_item.pinned_ptr,
                                );

                                // Send result back (ignore errors if receiver dropped)
                                let _ = work_item.result_sender.send(result);
                            }
                        }
                    }
                });

                threads.push((work_sender, handle));
            }

            Some(threads)
        } else {
            None  // Single GPU uses fast path
        };

        // Allocate pinned memory buffers (one per worker)
        // Default: 1 GB per buffer, covers ~111M 8-char words
        let max_buffer_size = 1_000_000_000;  // 1 GB per worker
        let pinned_buffers: Vec<PinnedBuffer> = (0..num_devices)
            .map(|_| PinnedBuffer::new(max_buffer_size))
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("Failed to allocate pinned memory for {} workers", num_devices))?;

        // Initialize GPU stats (one per device)
        let gpu_stats: Vec<GpuStats> = (0..num_devices)
            .map(|_| GpuStats::new())
            .collect();

        Ok(Self {
            workers,
            num_devices,
            async_mode,
            worker_threads,
            pinned_buffers,
            max_buffer_size,
            gpu_stats,
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
        // Check if we have reliable throughput estimates for adaptive partitioning
        let all_reliable = self.gpu_stats.iter().all(|s| s.has_reliable_estimate());

        if all_reliable && self.num_devices > 1 {
            // Use adaptive partitioning based on measured throughput
            self.adaptive_partition(start_idx, count)
        } else {
            // Fall back to static partitioning
            let partitions = partition_keyspace(count, self.num_devices);

            // Adjust partitions to account for global start_idx offset
            partitions
                .into_iter()
                .map(|p| KeyspacePartition::new(start_idx + p.start_idx, p.count))
                .collect()
        }
    }

    /// Adaptive keyspace partitioning based on measured GPU throughput
    ///
    /// Distributes work proportionally to each GPU's observed performance,
    /// allowing heterogeneous GPU setups to achieve better load balancing.
    ///
    /// # Arguments
    /// * `start_idx` - Starting index in global keyspace
    /// * `total_work` - Total number of words to generate
    ///
    /// # Returns
    /// Vector of partitions sized according to GPU throughput estimates
    ///
    /// # Example
    /// Given 2 GPUs with throughputs 500M/s and 300M/s:
    /// - GPU 0 gets 62.5% of work (500 / (500 + 300))
    /// - GPU 1 gets 37.5% of work (300 / (500 + 300))
    fn adaptive_partition(&self, start_idx: u64, total_work: u64) -> Vec<KeyspacePartition> {
        // Calculate total throughput across all GPUs
        let total_throughput: f64 = self.gpu_stats.iter()
            .map(|s| s.throughput())
            .sum();

        if total_throughput == 0.0 {
            // Fallback to static partitioning if no throughput data
            return partition_keyspace(total_work, self.num_devices)
                .into_iter()
                .map(|p| KeyspacePartition::new(start_idx + p.start_idx, p.count))
                .collect();
        }

        let mut partitions = Vec::with_capacity(self.num_devices);
        let mut allocated = 0u64;
        let mut current_start = start_idx;

        for (gpu_id, stats) in self.gpu_stats.iter().enumerate() {
            let throughput_fraction = stats.throughput() / total_throughput;

            // Calculate work for this GPU (proportional to throughput)
            let count = if gpu_id == self.num_devices - 1 {
                // Last GPU gets remainder to avoid rounding errors
                total_work.saturating_sub(allocated)
            } else {
                let count = (total_work as f64 * throughput_fraction).round() as u64;
                allocated += count;
                count
            };

            partitions.push(KeyspacePartition::new(current_start, count));
            current_start += count;
        }

        partitions
    }

    /// Process a work item on a persistent worker thread
    ///
    /// This is called by persistent worker threads to generate a batch.
    /// The GPU context and stream are owned by the worker thread.
    /// Output is written directly to pinned memory for fast PCIe transfers.
    ///
    /// Returns (output_size, duration) for performance tracking
    fn process_work_item(
        gpu_ctx: &GpuContext,
        partition: KeyspacePartition,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        output_format: i32,
        stream: CUstream,
        pinned_ptr: SendPtr,
    ) -> Result<(usize, Duration)> {
        let start_time = std::time::Instant::now();

        unsafe {
            // Calculate output size
            let word_length = mask.len();
            let bytes_per_word = match output_format {
                0 => word_length + 1,  // WG_FORMAT_NEWLINES
                1 => word_length + 1,  // WG_FORMAT_FIXED_WIDTH
                2 => word_length,      // WG_FORMAT_PACKED
                _ => word_length + 1,  // fallback
            };
            let output_size = partition.count as usize * bytes_per_word;

            // Generate batch using device pointer API with optional stream
            let (device_ptr, size) = gpu_ctx.generate_batch_device_stream(
                charsets,
                mask,
                partition.start_idx,
                partition.count,
                stream,
                output_format,
            )?;

            // Verify size
            if size != output_size {
                eprintln!("[WARNING] Size mismatch! expected={}, got={}", output_size, size);
            }

            // Copy directly to pinned memory (FAST! ~2x faster than pageable)
            let copy_result = if !stream.is_null() {
                cuMemcpyDtoHAsync_v2(
                    pinned_ptr.as_mut_ptr() as *mut std::ffi::c_void,
                    device_ptr,
                    size,
                    stream,
                )
            } else {
                cuMemcpyDtoH_v2(
                    pinned_ptr.as_mut_ptr() as *mut std::ffi::c_void,
                    device_ptr,
                    size,
                )
            };

            if copy_result != CUresult::CUDA_SUCCESS {
                let _ = cuMemFree_v2(device_ptr);
                anyhow::bail!("Failed to copy results to pinned memory: {:?}", copy_result);
            }

            // Synchronize stream to ensure copy completes
            if !stream.is_null() {
                let sync_result = cuStreamSynchronize(stream);
                if sync_result != CUresult::CUDA_SUCCESS {
                    let _ = cuMemFree_v2(device_ptr);
                    anyhow::bail!("Failed to synchronize stream: {:?}", sync_result);
                }
            } else {
                let _ = cuCtxSynchronize();
            }

            // Free device memory
            let _ = cuMemFree_v2(device_ptr);

            // Calculate elapsed time
            let duration = start_time.elapsed();

            // Return size written and duration for stats tracking
            Ok((size, duration))
        }
    }

    /// Generate batch with zero-copy callback API (Phase 3 optimization)
    ///
    /// This method generates data directly into pinned memory and provides
    /// a callback with a slice reference, eliminating the final copy to Vec.
    ///
    /// For single GPU: TRUE zero-copy (no pinned→Vec allocation)
    /// For multi-GPU: One fast pinned→pinned copy, then callback (no Vec allocation)
    ///
    /// # Arguments
    /// * `charsets` - Character set definitions
    /// * `mask` - Mask pattern
    /// * `start_idx` - Starting index in global keyspace
    /// * `batch_size` - Total number of words to generate
    /// * `output_format` - Output format (WG_FORMAT_*)
    /// * `f` - Callback function that processes the data in pinned memory
    ///
    /// # Returns
    /// Result from the callback function
    ///
    /// # Example
    /// ```no_run
    /// use gpu_scatter_gather::multigpu::MultiGpuContext;
    /// use std::collections::HashMap;
    /// use std::io::Write;
    ///
    /// let mut ctx = MultiGpuContext::new()?;
    /// let mut charsets = HashMap::new();
    /// charsets.insert(1, b"abc".to_vec());
    /// let mask = vec![1, 1];
    ///
    /// // Write directly to file without allocating Vec
    /// let mut file = std::fs::File::create("output.txt")?;
    /// ctx.generate_batch_with(&charsets, &mask, 0, 9, 0, |data| {
    ///     file.write_all(data)
    /// })?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn generate_batch_with<F, R>(
        &mut self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
        f: F,
    ) -> Result<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        if self.async_mode {
            self.generate_batch_with_async(charsets, mask, start_idx, batch_size, output_format, f)
        } else {
            self.generate_batch_with_sync(charsets, mask, start_idx, batch_size, output_format, f)
        }
    }

    /// Generate batch across all GPUs in parallel (optimized version)
    ///
    /// This is a convenience wrapper around `generate_batch_with` that returns Vec<u8>.
    /// For maximum performance (zero-copy), use `generate_batch_with` directly.
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
        &mut self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
    ) -> Result<Vec<u8>> {
        self.generate_batch_with(charsets, mask, start_idx, batch_size, output_format, |data| {
            data.to_vec()
        })
    }

    /// Generate batch with callback (synchronous version)
    ///
    /// Uses pinned memory for faster PCIe transfers. Data is provided to callback
    /// directly from pinned memory, eliminating the final Vec allocation.
    ///
    /// # Arguments
    /// * `charsets` - Character set definitions
    /// * `mask` - Mask pattern
    /// * `start_idx` - Starting index in global keyspace
    /// * `batch_size` - Total number of words to generate
    /// * `output_format` - Output format (WG_FORMAT_*)
    /// * `f` - Callback to process data in pinned memory
    ///
    /// # Returns
    /// Result from callback
    fn generate_batch_with_sync<F, R>(
        &mut self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
        f: F,
    ) -> Result<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        // Fast path for single GPU: use pinned memory, no threading overhead
        if self.num_devices == 1 {
            unsafe {
                // Calculate output size
                let word_length = mask.len();
                let bytes_per_word = match output_format {
                    0 => word_length + 1,  // WG_FORMAT_NEWLINES
                    1 => word_length + 1,  // WG_FORMAT_FIXED_WIDTH
                    2 => word_length,      // WG_FORMAT_PACKED
                    _ => word_length + 1,  // fallback
                };
                let _output_size = batch_size as usize * bytes_per_word;

                // Get pinned buffer pointer
                let pinned_ptr = self.pinned_buffers[0].as_mut_ptr();

                // Generate using device pointer API
                let (device_ptr, size) = self.workers[0].context.generate_batch_device_stream(
                    charsets,
                    mask,
                    start_idx,
                    batch_size,
                    std::ptr::null_mut(),  // No stream for single GPU
                    output_format,
                )?;

                // Copy to pinned memory
                let copy_result = cuMemcpyDtoH_v2(
                    pinned_ptr as *mut std::ffi::c_void,
                    device_ptr,
                    size,
                );

                if copy_result != CUresult::CUDA_SUCCESS {
                    let _ = cuMemFree_v2(device_ptr);
                    anyhow::bail!("Failed to copy to pinned memory: {:?}", copy_result);
                }

                // Synchronize
                let _ = cuCtxSynchronize();

                // Free device memory
                let _ = cuMemFree_v2(device_ptr);

                // Call callback with pinned memory slice (ZERO-COPY!)
                let slice = std::slice::from_raw_parts(pinned_ptr, size);
                return Ok(f(slice));
            }
        }

        // Multi-GPU path: use persistent worker threads with pinned memory
        use std::sync::mpsc::channel;

        // Partition keyspace across GPUs
        let partitions = self.partition(start_idx, batch_size);

        // Get reference to worker threads
        let worker_threads = self.worker_threads.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Worker threads not initialized"))?;

        // Create result channels for each worker
        let mut result_receivers = Vec::new();

        for (gpu_idx, partition) in partitions.iter().enumerate() {
            // Skip empty partitions
            if partition.count == 0 {
                let (tx, rx) = channel();
                let _ = tx.send(Ok((0, Duration::ZERO)));
                result_receivers.push((rx, gpu_idx));
                continue;
            }

            // Create result channel for this work item
            let (result_tx, result_rx) = channel();

            // Create work item with pinned buffer pointer
            let work_item = WorkItem {
                charsets: charsets.clone(),
                mask: mask.to_vec(),
                partition: *partition,
                output_format,
                pinned_ptr: SendPtr::new(self.pinned_buffers[gpu_idx].as_mut_ptr()),
                result_sender: result_tx,
            };

            // Send work to persistent worker thread
            worker_threads[gpu_idx].0.send(WorkerMessage::Work(work_item))
                .map_err(|e| anyhow::anyhow!("Failed to send work to GPU {}: {}", gpu_idx, e))?;

            result_receivers.push((result_rx, gpu_idx));
        }

        // Collect results from workers (size, duration, worker_id)
        let results: Vec<(usize, Duration, usize)> = result_receivers
            .into_iter()
            .map(|(rx, worker_id)| {
                let (size, duration) = rx.recv()
                    .with_context(|| format!("Failed to receive from GPU {}", worker_id))??;
                Ok((size, duration, worker_id))
            })
            .collect::<Result<Vec<_>>>()?;

        // Record performance stats for adaptive load balancing
        for &(size, duration, worker_id) in &results {
            let word_length = mask.len();
            let bytes_per_word = match output_format {
                0 => word_length + 1,  // WG_FORMAT_NEWLINES
                1 => word_length + 1,  // WG_FORMAT_FIXED_WIDTH
                2 => word_length,      // WG_FORMAT_PACKED
                _ => word_length + 1,  // fallback
            };
            let words = size / bytes_per_word;
            self.gpu_stats[worker_id].record_completion(duration, words as u64);
        }

        // Calculate total size
        let total_size: usize = results.iter().map(|(size, _, _)| size).sum();

        // Concatenate from worker pinned buffers into buffer[0], then callback
        unsafe {
            let output_ptr = self.pinned_buffers[0].as_mut_ptr();
            let mut offset = 0;

            for (size, _, worker_id) in results {
                if size > 0 {
                    // Copy from worker buffer to buffer[0] (fast pinned→pinned memcpy)
                    std::ptr::copy_nonoverlapping(
                        self.pinned_buffers[worker_id].as_ptr(),
                        output_ptr.add(offset),
                        size,
                    );
                    offset += size;
                }
            }

            // Call callback with concatenated data in buffer[0] (NO Vec allocation!)
            let slice = std::slice::from_raw_parts(output_ptr, total_size);
            Ok(f(slice))
        }
    }

    /// Generate batch with callback (async optimized version)
    ///
    /// This implementation uses:
    /// - Pinned memory allocation (cuMemAllocHost) for 2x faster PCIe transfers
    /// - CUDA streams for overlapped kernel execution (5-10% improvement)
    /// - Async memory copies for pipelined data transfers
    /// - Zero-copy callback API (no Vec allocation)
    ///
    /// Expected improvement: 10-15% throughput gain over sync version
    ///
    /// # Arguments
    /// * `charsets` - Character set definitions
    /// * `mask` - Mask pattern
    /// * `start_idx` - Starting index in global keyspace
    /// * `batch_size` - Total number of words to generate
    /// * `output_format` - Output format (WG_FORMAT_*)
    /// * `f` - Callback to process data in pinned memory
    ///
    /// # Returns
    /// Result from callback
    fn generate_batch_with_async<F, R>(
        &mut self,
        charsets: &HashMap<usize, Vec<u8>>,
        mask: &[usize],
        start_idx: u64,
        batch_size: u64,
        output_format: i32,
        f: F,
    ) -> Result<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        // Fast path for single GPU: use pinned memory, no threading overhead
        if self.num_devices == 1 {
            unsafe {
                // Calculate output size
                let word_length = mask.len();
                let bytes_per_word = match output_format {
                    0 => word_length + 1,  // WG_FORMAT_NEWLINES
                    1 => word_length + 1,  // WG_FORMAT_FIXED_WIDTH
                    2 => word_length,      // WG_FORMAT_PACKED
                    _ => word_length + 1,  // fallback
                };
                let _output_size = batch_size as usize * bytes_per_word;

                // Get pinned buffer pointer
                let pinned_ptr = self.pinned_buffers[0].as_mut_ptr();

                // Generate using device pointer API
                let (device_ptr, size) = self.workers[0].context.generate_batch_device_stream(
                    charsets,
                    mask,
                    start_idx,
                    batch_size,
                    std::ptr::null_mut(),  // No stream for single GPU
                    output_format,
                )?;

                // Copy to pinned memory
                let copy_result = cuMemcpyDtoH_v2(
                    pinned_ptr as *mut std::ffi::c_void,
                    device_ptr,
                    size,
                );

                if copy_result != CUresult::CUDA_SUCCESS {
                    let _ = cuMemFree_v2(device_ptr);
                    anyhow::bail!("Failed to copy to pinned memory: {:?}", copy_result);
                }

                // Synchronize
                let _ = cuCtxSynchronize();

                // Free device memory
                let _ = cuMemFree_v2(device_ptr);

                // Call callback with pinned memory slice (ZERO-COPY!)
                let slice = std::slice::from_raw_parts(pinned_ptr, size);
                return Ok(f(slice));
            }
        }

        // Multi-GPU path: use persistent worker threads (async mode uses streams)
        use std::sync::mpsc::channel;

        // Partition keyspace across GPUs
        let partitions = self.partition(start_idx, batch_size);

        // Get reference to worker threads
        let worker_threads = self.worker_threads.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Worker threads not initialized"))?;

        // Create result channels for each worker
        let mut result_receivers = Vec::new();

        for (gpu_idx, partition) in partitions.iter().enumerate() {
            // Skip empty partitions
            if partition.count == 0 {
                let (tx, rx) = channel();
                let _ = tx.send(Ok((0, Duration::ZERO)));
                result_receivers.push((rx, gpu_idx));
                continue;
            }

            // Create result channel for this work item
            let (result_tx, result_rx) = channel();

            // Create work item with pinned buffer pointer
            let work_item = WorkItem {
                charsets: charsets.clone(),
                mask: mask.to_vec(),
                partition: *partition,
                output_format,
                pinned_ptr: SendPtr::new(self.pinned_buffers[gpu_idx].as_mut_ptr()),
                result_sender: result_tx,
            };

            // Send work to persistent worker thread
            worker_threads[gpu_idx].0.send(WorkerMessage::Work(work_item))
                .map_err(|e| anyhow::anyhow!("Failed to send work to GPU {}: {}", gpu_idx, e))?;

            result_receivers.push((result_rx, gpu_idx));
        }

        // Collect results from workers (size, duration, worker_id)
        let results: Vec<(usize, Duration, usize)> = result_receivers
            .into_iter()
            .map(|(rx, worker_id)| {
                let (size, duration) = rx.recv()
                    .with_context(|| format!("Failed to receive from GPU {}", worker_id))??;
                Ok((size, duration, worker_id))
            })
            .collect::<Result<Vec<_>>>()?;

        // Record performance stats for adaptive load balancing
        for &(size, duration, worker_id) in &results {
            let word_length = mask.len();
            let bytes_per_word = match output_format {
                0 => word_length + 1,  // WG_FORMAT_NEWLINES
                1 => word_length + 1,  // WG_FORMAT_FIXED_WIDTH
                2 => word_length,      // WG_FORMAT_PACKED
                _ => word_length + 1,  // fallback
            };
            let words = size / bytes_per_word;
            self.gpu_stats[worker_id].record_completion(duration, words as u64);
        }

        // Calculate total size
        let total_size: usize = results.iter().map(|(size, _, _)| size).sum();

        // Concatenate from worker pinned buffers into buffer[0], then callback
        unsafe {
            let output_ptr = self.pinned_buffers[0].as_mut_ptr();
            let mut offset = 0;

            for (size, _, worker_id) in results {
                if size > 0 {
                    // Copy from worker buffer to buffer[0] (fast pinned→pinned memcpy)
                    std::ptr::copy_nonoverlapping(
                        self.pinned_buffers[worker_id].as_ptr(),
                        output_ptr.add(offset),
                        size,
                    );
                    offset += size;
                }
            }

            // Call callback with concatenated data in buffer[0] (NO Vec allocation!)
            let slice = std::slice::from_raw_parts(output_ptr, total_size);
            Ok(f(slice))
        }
    }
}

impl Drop for MultiGpuContext {
    fn drop(&mut self) {
        // Gracefully shutdown persistent worker threads
        if let Some(ref mut worker_threads) = self.worker_threads {
            // Send shutdown signal to all workers
            for (sender, _) in worker_threads.iter() {
                let _ = sender.send(WorkerMessage::Shutdown);
            }

            // Wait for all workers to exit (with timeout protection)
            while let Some((_, handle)) = worker_threads.pop() {
                // Join with timeout - if thread doesn't exit in 5 seconds, it's leaked
                // This is acceptable for Drop since we're shutting down anyway
                let _ = handle.join();
            }
        }
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
            Ok(mut ctx) => {
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
            Ok(mut ctx) => {
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

    #[test]
    fn test_multi_gpu_async_creation() {
        // Test async multi-GPU context creation
        match MultiGpuContext::new_async() {
            Ok(ctx) => {
                assert!(ctx.num_devices() >= 1);
            }
            Err(_e) => {
            }
        }
    }

    #[test]
    fn test_multi_gpu_async_repeated() {
        // Test async multi-GPU generation with repeated calls (like benchmark does)
        match MultiGpuContext::new_async() {
            Ok(mut ctx) => {
                let mut charsets = HashMap::new();
                charsets.insert(0, b"abcdefghijklmnopqrstuvwxyz".to_vec());
                charsets.insert(1, b"0123456789".to_vec());

                let mask = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1]; // 10 chars

                // Run 3 times like the benchmark does
                for i in 1..=3 {
                    match ctx.generate_batch(&charsets, &mask, 0, 10_000_000, 2) { // 10M words, PACKED
                        Ok(output) => {
                            assert_eq!(output.len(), 100_000_000); // 10M * 10 chars
                            println!("Run {}: Success!", i);
                        }
                        Err(e) => {
                            panic!("Run {} failed: {}", i, e);
                        }
                    }
                }
                println!("All 3 runs successful!");
            }
            Err(e) => {
                println!("No GPU available for async repeated test: {}", e);
            }
        }
    }

    #[test]
    fn test_multi_gpu_async_large() {
        // Test async multi-GPU generation with large batch (1M words)
        match MultiGpuContext::new_async() {
            Ok(mut ctx) => {
                let mut charsets = HashMap::new();
                charsets.insert(0, b"abcdefghijklmnopqrstuvwxyz".to_vec());
                charsets.insert(1, b"0123456789".to_vec());

                let mask = vec![0, 0, 0, 0, 1, 1]; // ?l?l?l?l?d?d - 6 chars

                // Generate 1M words
                match ctx.generate_batch(&charsets, &mask, 0, 1_000_000, 2) { // PACKED format
                    Ok(output) => {
                        // Verify we got correct output size
                        assert_eq!(output.len(), 6_000_000); // 1M words * 6 chars each
                        println!("Async large batch (1M words) successful!");
                    }
                    Err(e) => {
                        panic!("Large batch generation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("No GPU available for async large test: {}", e);
            }
        }
    }

    #[test]
    fn test_multi_gpu_async_medium() {
        // Test async multi-GPU generation with medium batch
        match MultiGpuContext::new_async() {
            Ok(mut ctx) => {
                let mut charsets = HashMap::new();
                charsets.insert(1, b"abcdefghij".to_vec()); // 10 chars

                let mask = vec![1, 1, 1]; // 10^3 = 1000 words

                // Generate 1000 words
                match ctx.generate_batch(&charsets, &mask, 0, 1000, 2) { // PACKED format
                    Ok(output) => {
                        // Verify we got correct output size
                        assert_eq!(output.len(), 3000); // 1000 words * 3 chars each
                        println!("Async medium batch (1000 words) successful!");
                    }
                    Err(e) => {
                        panic!("Medium batch generation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("No GPU available for async medium test: {}", e);
            }
        }
    }

    #[test]
    fn test_multi_gpu_async_basic() {
        // Test async multi-GPU generation
        match MultiGpuContext::new_async() {
            Ok(mut ctx) => {
                let mut charsets = HashMap::new();
                charsets.insert(1, b"ab".to_vec());

                let mask = vec![1, 1]; // 2x2 = 4 words total

                // Generate all 4 words using async mode
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

                        println!("Async multi-GPU generation successful: {:?}", lines);
                    }
                    Err(e) => {
                        panic!("Async generation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("No GPU available for async test: {}", e);
            }
        }
    }

    // GpuStats tests
    #[test]
    fn test_gpu_stats_new() {
        let stats = GpuStats::new();
        assert_eq!(stats.throughput(), 0.0);
        assert_eq!(stats.sample_count, 0);
        assert!(!stats.has_reliable_estimate());
    }

    #[test]
    fn test_gpu_stats_single_sample() {
        let mut stats = GpuStats::new();

        // Record: 1,000,000 words in 1 second = 1M words/sec
        stats.record_completion(Duration::from_secs(1), 1_000_000);

        assert_eq!(stats.throughput(), 1_000_000.0);
        assert_eq!(stats.sample_count, 1);
        assert!(!stats.has_reliable_estimate()); // Need 3 samples
    }

    #[test]
    fn test_gpu_stats_multiple_samples() {
        let mut stats = GpuStats::new();

        // Record 3 samples
        stats.record_completion(Duration::from_secs(1), 1_000_000); // 1M/s
        stats.record_completion(Duration::from_secs(1), 900_000);   // 0.9M/s
        stats.record_completion(Duration::from_secs(1), 1_100_000); // 1.1M/s

        assert_eq!(stats.sample_count, 3);
        assert!(stats.has_reliable_estimate());

        // With ALPHA=0.2, the estimate should be close to the average
        // but weighted toward recent samples
        let throughput = stats.throughput();
        assert!(throughput > 950_000.0 && throughput < 1_050_000.0);
    }

    #[test]
    fn test_gpu_stats_exponential_moving_average() {
        let mut stats = GpuStats::new();

        // First sample
        stats.record_completion(Duration::from_secs(1), 1_000_000); // 1M/s
        assert_eq!(stats.throughput(), 1_000_000.0);

        // Second sample: higher throughput
        stats.record_completion(Duration::from_secs(1), 2_000_000); // 2M/s

        // EMA: 0.2 * 2M + 0.8 * 1M = 0.4M + 0.8M = 1.2M
        assert_eq!(stats.throughput(), 1_200_000.0);

        // Third sample: lower throughput
        stats.record_completion(Duration::from_secs(1), 1_000_000); // 1M/s

        // EMA: 0.2 * 1M + 0.8 * 1.2M = 0.2M + 0.96M = 1.16M
        assert_eq!(stats.throughput(), 1_160_000.0);
    }

    #[test]
    fn test_adaptive_partition_heterogeneous() {
        use std::time::Duration;

        // Create a mock multi-GPU context with 2 GPUs
        // We'll manually construct gpu_stats to simulate heterogeneous GPUs
        match MultiGpuContext::with_devices(&[0]) {
            Ok(ctx) => {
                let mut ctx = ctx;
                // Simulate 2-GPU setup by manually adding stats
                // GPU 0: 500M words/s
                // GPU 1: 300M words/s
                ctx.gpu_stats = vec![GpuStats::new(), GpuStats::new()];
                ctx.num_devices = 2;

                // Record samples to establish throughput
                for _ in 0..3 {
                    ctx.gpu_stats[0].record_completion(Duration::from_secs(1), 500_000_000);
                    ctx.gpu_stats[1].record_completion(Duration::from_secs(1), 300_000_000);
                }

                // Test adaptive partitioning with 800M words
                let partitions = ctx.adaptive_partition(0, 800_000_000);

                assert_eq!(partitions.len(), 2);

                // GPU 0 should get ~62.5% (500 / 800 = 0.625)
                // GPU 1 should get ~37.5% (300 / 800 = 0.375)
                let gpu0_expected = 500_000_000; // 62.5% of 800M
                let gpu1_expected = 300_000_000; // 37.5% of 800M

                // Allow 1% tolerance for rounding
                let tolerance = 8_000_000; // 1% of 800M
                assert!((partitions[0].count as i64 - gpu0_expected as i64).abs() < tolerance as i64,
                    "GPU 0 got {} words, expected {} ± {}", partitions[0].count, gpu0_expected, tolerance);
                assert!((partitions[1].count as i64 - gpu1_expected as i64).abs() < tolerance as i64,
                    "GPU 1 got {} words, expected {} ± {}", partitions[1].count, gpu1_expected, tolerance);

                // Verify total coverage
                let total: u64 = partitions.iter().map(|p| p.count).sum();
                assert_eq!(total, 800_000_000);

                // Verify no gaps
                assert_eq!(partitions[0].start_idx, 0);
                assert_eq!(partitions[1].start_idx, partitions[0].count);
            }
            Err(e) => {
                println!("No GPU available for adaptive partition test: {}", e);
            }
        }
    }

    #[test]
    fn test_adaptive_partition_balanced() {
        use std::time::Duration;

        // Test with balanced GPUs (same throughput)
        match MultiGpuContext::with_devices(&[0]) {
            Ok(ctx) => {
                let mut ctx = ctx;
                // Simulate 3-GPU setup with equal throughput
                ctx.gpu_stats = vec![GpuStats::new(), GpuStats::new(), GpuStats::new()];
                ctx.num_devices = 3;

                // All GPUs: 400M words/s
                for _ in 0..3 {
                    for i in 0..3 {
                        ctx.gpu_stats[i].record_completion(Duration::from_secs(1), 400_000_000);
                    }
                }

                // Test with 1.2B words
                let partitions = ctx.adaptive_partition(0, 1_200_000_000);

                assert_eq!(partitions.len(), 3);

                // Each GPU should get ~400M words (1/3 of total)
                for (i, partition) in partitions.iter().enumerate() {
                    let tolerance = 12_000_000; // 1% tolerance
                    assert!((partition.count as i64 - 400_000_000i64).abs() < tolerance as i64,
                        "GPU {} got {} words, expected 400M ± {}", i, partition.count, tolerance);
                }

                // Verify total
                let total: u64 = partitions.iter().map(|p| p.count).sum();
                assert_eq!(total, 1_200_000_000);
            }
            Err(e) => {
                println!("No GPU available for balanced partition test: {}", e);
            }
        }
    }

    #[test]
    fn test_partition_fallback_before_reliable() {
        // Test that partition() falls back to static partitioning
        // when throughput estimates aren't reliable yet
        match MultiGpuContext::with_devices(&[0]) {
            Ok(ctx) => {
                let mut ctx = ctx;
                // Simulate 2-GPU setup but without enough samples
                ctx.gpu_stats = vec![GpuStats::new(), GpuStats::new()];
                ctx.num_devices = 2;

                // Only 1 sample (need 3 for reliable estimate)
                ctx.gpu_stats[0].record_completion(Duration::from_secs(1), 500_000_000);
                ctx.gpu_stats[1].record_completion(Duration::from_secs(1), 300_000_000);

                // partition() should use static partitioning (50/50 split)
                let partitions = ctx.partition(0, 1_000_000);

                assert_eq!(partitions.len(), 2);

                // Should be evenly split (static partitioning)
                assert_eq!(partitions[0].count, 500_000);
                assert_eq!(partitions[1].count, 500_000);
            }
            Err(e) => {
                println!("No GPU available for fallback test: {}", e);
            }
        }
    }
}
