# GPU Scatter-Gather Wordlist Generator

## Project Vision
Build the **world's fastest wordlist generator** using GPU acceleration and novel algorithms, achieving **500M-1B+ words/second** - orders of magnitude faster than any existing tool.

**Core Innovation:** Direct index-to-word mapping using mixed-radix arithmetic on GPU, enabling massive parallelism without sequential dependencies.

**Novel Experiment:** This project is also a showcase of **AI-assisted high-performance computing development**. We're using AI (Claude Code) to design, implement, and optimize CUDA kernels and systems programming. This represents a new frontier in collaborative human-AI software development for performance-critical applications.

## Implementation Decisions (2025-10-16)

### Technology Stack (FINALIZED)
- **CUDA Bindings**: `nvidia-cuda-runtime-sys` + manual kernel compilation
  - Rationale: Maximum control, minimal abstraction overhead, modern approach

- **Kernel Compilation Strategy**: Hybrid approach
  - Ship pre-compiled PTX for common compute capabilities (7.0, 7.5, 8.0, 8.6, 8.9, 9.0)
  - Fallback to runtime compilation (nvrtc) for unsupported architectures
  - Benefits: Fast startup for 95% of users + universal compatibility

### Optimization Strategy (PROGRESSIVE)
- **Phase 1 (POC)**: Simple division/modulo to prove concept
- **Phase 2 (v0.2)**: Barrett reduction for arbitrary charsets
- **Phase 3 (v0.3)**: Special-case power-of-2 charsets with bitwise operations
- **Rationale**: Validate correctness first, optimize later

### Memory Layout (PER-BINDING)
- **Stdout binding**: Include newlines (required for pipe compatibility with hashcat)
- **In-memory binding**: Skip newlines, return raw bytes for maximum performance
- **File binding**: Include newlines (standard wordlist format)
- **Rationale**: Each binding optimized for its specific use case

### Ethical Use Statement
This tool is designed for **defensive security research only**, including:
- Password security testing and auditing
- Security research and vulnerability assessment
- Educational purposes for understanding password strength
- Integration with legitimate security testing frameworks (hashcat, John the Ripper)

**Not intended for**: Unauthorized access, credential theft, or malicious activities.

## Project Goals

### Primary Objectives
1. **Maximum Speed**: Achieve 500M-1B+ words/second on consumer GPUs (RTX 4070 target)
2. **Universal Bindings**: Multiple consumption patterns (stdout, in-memory streaming, file output, network streaming)
3. **Random Access**: Jump to any keyspace position instantly for distributed workloads
4. **Zero Dependencies on External Tools**: Standalone wordlist generator, not tied to hashcat or any specific use case

### Success Metrics
- **Performance**: 3-7x faster than maskprocessor (~142M/s baseline)
- **Scalability**: Linear performance scaling with GPU cores
- **Flexibility**: Support multiple output bindings without performance degradation
- **Usability**: Simple CLI and programmatic API for all major languages

## Architecture Overview

### Core Algorithm: Index-to-Word Direct Mapping

Instead of sequential odometer iteration, use mathematical index conversion:

```
Index → Mixed-Radix Decomposition → Word
```

**Key Advantages:**
- **Massively parallel** - Every thread generates independently
- **No sequential dependencies** - No carry propagation
- **Random access** - Jump to any position O(1)
- **Perfect GPU utilization** - Coalesced memory access, minimal divergence

### Algorithm Pseudocode

```cuda
__global__ void generate_words_kernel(
    const char* __restrict__ charset_data,      // Flat array of all charset chars
    const int* __restrict__ charset_offsets,    // Start index for each charset
    const int* __restrict__ charset_sizes,      // Size of each charset
    const int* __restrict__ mask_pattern,       // Which charset for each position
    uint64_t start_idx,                         // Starting combination index
    int word_length,                            // Number of positions
    char* __restrict__ output_buffer,           // Output buffer for words
    uint64_t batch_size                         // Number of words to generate
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    uint64_t idx = start_idx + tid;
    char* word = output_buffer + (tid * (word_length + 1)); // +1 for newline

    // Convert global index to word using mixed-radix arithmetic
    uint64_t remaining = idx;
    for (int pos = word_length - 1; pos >= 0; pos--) {
        int charset_id = mask_pattern[pos];
        int cs_size = charset_sizes[charset_id];
        int char_idx = remaining % cs_size;
        word[pos] = charset_data[charset_offsets[charset_id] + char_idx];
        remaining /= cs_size;
    }
    word[word_length] = '\n';
}
```

### Performance Projection

**Hardware Target: RTX 4070**
- CUDA Cores: 5,888
- Clock Speed: ~2.5 GHz
- Memory Bandwidth: 504 GB/s
- Compute Capability: 8.9

**Theoretical Performance:**
- Assume 10 cycles per word (charset lookups + division + modulo)
- 5,888 cores × 2.5 GHz / 10 = **1.47 billion words/second**

**Realistic Performance (with overhead):**
- Kernel launch overhead: ~negligible when amortized over millions of words
- Memory transfers: 8 bytes/word × 1B words/s = 8 GB/s (well within 504 GB/s bandwidth)
- **Estimated: 500M-1B words/second** (3-7x faster than maskprocessor)

## Multiple Binding Architecture

### Binding 1: Stdout Stream (CLI Compatibility)
- **Use Case**: Drop-in replacement for maskprocessor, pipe to hashcat
- **Implementation**: GPU generates batches → CPU buffers → stdout stream
- **Buffering Strategy**: Double-buffering (GPU generates batch N+1 while CPU writes batch N)
- **Performance**: Limited by stdout/pipe bandwidth (~100-200M words/s realistic)

### Binding 2: In-Memory Stream (Zero-Copy API)
- **Use Case**: Programmatic access from Rust/Python/C++, no serialization overhead
- **Implementation**: GPU generates directly to pinned memory, return iterator/stream
- **API**: `WordlistStream::new(mask) -> impl Iterator<Item = &[u8]>`
- **Performance**: Full GPU speed (500M-1B words/s)
- **Memory**: Configurable batch size (e.g., 10M words × 8 bytes = 80MB per batch)

### Binding 3: Memory-Mapped File Output
- **Use Case**: Generate massive wordlists to disk with maximum throughput
- **Implementation**: GPU → pinned memory → async write to mmap'd file
- **Optimization**: Batched writes, no per-word syscalls
- **Performance**: Limited by SSD speed (NVMe: ~7 GB/s = ~875M words/s for 8-byte words)

### Binding 4: Network Stream (Distributed Generation)
- **Use Case**: Remote wordlist generation, distributed cracking
- **Implementation**: GPU → compression (optional) → TCP/UDP stream
- **Protocol**: Custom binary protocol or HTTP chunked transfer
- **Compression**: Optional LZ4/Zstd for bandwidth reduction
- **Performance**: Limited by network bandwidth (10 Gbps = ~1.25 GB/s = ~156M words/s uncompressed)

### Binding 5: Batch API (High-Level Language Bindings)
- **Use Case**: Python/Node.js/Go/Rust libraries
- **Implementation**: Generate batches on demand, return as native arrays/lists
- **API Example (Python)**:
  ```python
  gen = WordlistGenerator(mask='?1?1?2?2', charsets={1: 'abc', 2: '123'})
  batch = gen.next_batch(count=1_000_000)  # Returns list[bytes]
  ```
- **Performance**: Full GPU speed, bounded by language overhead

### Binding 6: Callback/Handler Interface
- **Use Case**: Custom processing per word without serialization
- **Implementation**: GPU generates → CPU callback per word or per batch
- **API Example (Rust)**:
  ```rust
  generator.generate_with_handler(|word: &[u8]| {
      // Custom processing here
  });
  ```
- **Performance**: Depends on callback complexity

## Implementation Plan

### Phase 1: Foundation & Proof of Concept

#### 1.1 Project Setup
- [ ] **Create new repository: gpu-scatter-gather-wordlist**
  - Location: New standalone repo (independent of wlgen-rs)
  - Initialize: `cargo init --name gpu-scatter-gather`
  - License: MIT or Apache-2.0 (permissive for maximum adoption)

- [x] **Technology Stack Selection** ✅ (FINALIZED 2025-10-16)
  - **Language**: Rust (for safety, performance, excellent bindings ecosystem)
  - **CUDA Bindings**: `nvidia-cuda-runtime-sys` + manual kernel compilation (NVRTC)
  - **CLI**: `clap` v4 for argument parsing
  - **Async I/O**: `tokio` for async file/network operations
  - **Benchmarking**:
    - `criterion` for CPU benchmarks
    - Custom GPU timing with CUDA events
    - `plotters` for generating performance graphs
    - Track performance metrics in JSON for historical comparison

- [ ] **Benchmarking Infrastructure (Day 1 Priority!)**
  - [ ] **Automated benchmark suite**
    - Run on every commit via CI
    - Store results in `benchmarks/results/` with timestamps
    - Generate comparison graphs automatically

  - [ ] **Performance tracking dashboard**
    - Plot throughput (words/s) over time
    - Plot kernel execution time vs batch size
    - Plot memory bandwidth utilization
    - Compare against maskprocessor baseline (142M words/s)
    - Export to GitHub Pages or similar

  - [ ] **Benchmark scenarios**
    - Small batches (1K, 10K, 100K words)
    - Medium batches (1M, 10M words)
    - Large batches (100M, 1B words)
    - Various mask patterns (short/long, uniform/mixed charsets)
    - Multi-GPU scaling tests

  - [ ] **Regression detection**
    - Alert if performance drops >5% between commits
    - Automatically bisect to find regression commit
    - Block merges that regress performance

- [ ] **Project Structure**
  ```
  gpu-scatter-gather/
  ├── Cargo.toml
  ├── README.md
  ├── LICENSE
  ├── TODO.md (this file)
  ├── src/
  │   ├── lib.rs              # Core library
  │   ├── main.rs             # CLI entry point
  │   ├── gpu/
  │   │   ├── mod.rs          # GPU module
  │   │   ├── kernel.cu       # CUDA kernel implementation
  │   │   └── context.rs      # GPU context management
  │   ├── bindings/
  │   │   ├── mod.rs          # Bindings module
  │   │   ├── stdout.rs       # Stdout streaming
  │   │   ├── memory.rs       # In-memory stream
  │   │   ├── file.rs         # File output
  │   │   ├── network.rs      # Network streaming
  │   │   └── callback.rs     # Callback interface
  │   ├── charset.rs          # Charset management
  │   ├── mask.rs             # Mask parsing
  │   └── keyspace.rs         # Keyspace calculations
  ├── benches/
  │   ├── gpu_bench.rs        # GPU performance benchmarks
  │   ├── binding_bench.rs    # Benchmark each binding
  │   ├── results/            # Historical benchmark results
  │   │   ├── 2025-10-15_baseline.json
  │   │   ├── 2025-10-16_poc.json
  │   │   └── ...
  │   └── graphs/             # Generated performance graphs
  │       ├── throughput_over_time.png
  │       ├── kernel_execution_time.png
  │       └── comparison_vs_maskprocessor.png
  ├── tests/
  │   ├── integration.rs      # Integration tests
  │   └── correctness.rs      # Output correctness tests
  ├── examples/
  │   ├── basic_usage.rs
  │   ├── streaming_api.rs
  │   └── distributed.rs
  └── bindings/
      ├── python/             # PyO3 Python bindings
      ├── node/               # Neon Node.js bindings
      └── c/                  # C FFI for maximum compatibility
  ```

#### 1.2 Core Algorithm Implementation

- [ ] **Implement keyspace calculator**
  ```rust
  pub fn calculate_keyspace(
      mask: &[usize],           // Mask pattern (charset indices)
      charset_sizes: &[usize]   // Size of each charset
  ) -> u128 {
      mask.iter()
          .map(|&charset_id| charset_sizes[charset_id] as u128)
          .product()
  }
  ```

- [ ] **Implement index-to-word conversion (CPU reference)**
  ```rust
  pub fn index_to_word(
      index: u64,
      mask: &[usize],
      charsets: &[Vec<u8>],
      output: &mut [u8]
  ) {
      let mut remaining = index;
      for (pos, &charset_id) in mask.iter().enumerate().rev() {
          let charset = &charsets[charset_id];
          let char_idx = (remaining % charset.len() as u64) as usize;
          output[pos] = charset[char_idx];
          remaining /= charset.len() as u64;
      }
  }
  ```

- [ ] **Validate CPU reference implementation**
  - Test against maskprocessor output for correctness
  - Ensure index-to-word mapping is bijective
  - Test edge cases (single charset, single position, overflow)

#### 1.3 CUDA Kernel Development

- [ ] **Set up CUDA development environment**
  - Install CUDA toolkit (12.x recommended)
  - Configure `cudarc` or `cust` in Cargo.toml
  - Test basic GPU context initialization
  - Verify GPU capabilities (compute capability, CUDA cores, memory)

- [ ] **🎯 POC: Prove 1.47 Billion Words/Second (THE DREAM!)**

  **Goal:** Demonstrate raw GPU generation speed WITHOUT any I/O overhead. Just prove the kernel can generate 1.47B words/s purely in GPU memory!

  - [ ] **Implement minimal kernel for POC**
    ```cuda
    // Stripped-down kernel: generate words but DON'T write to global memory
    // Just compute and discard (measures pure compute throughput)
    __global__ void poc_generate_words_compute_only(
        const char* charset_data,
        const int* charset_offsets,
        const int* charset_sizes,
        const int* mask_pattern,
        uint64_t start_idx,
        int word_length,
        uint64_t batch_size
    ) {
        uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= batch_size) return;

        uint64_t idx = start_idx + tid;
        char word[32]; // Local array, stays in registers

        // Convert index to word
        uint64_t remaining = idx;
        for (int pos = word_length - 1; pos >= 0; pos--) {
            int charset_id = mask_pattern[pos];
            int cs_size = charset_sizes[charset_id];
            int char_idx = remaining % cs_size;
            word[pos] = charset_data[charset_offsets[charset_id] + char_idx];
            remaining /= cs_size;
        }

        // Force computation (prevent compiler optimization)
        if (word[0] == 255) {
            // This will never happen, but prevents dead code elimination
            atomicAdd((unsigned long long*)&charset_data[0], (unsigned long long)word[0]);
        }
    }
    ```

  - [ ] **Benchmark POC kernel with CUDA events**
    ```rust
    // Time kernel execution with maximum batch size
    let batch_size = 1_000_000_000; // 1 billion words

    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;

    start.record(stream)?;
    launch_poc_kernel(batch_size)?;
    end.record(stream)?;
    end.synchronize()?;

    let elapsed_ms = start.elapsed_time(&end)?;
    let words_per_second = (batch_size as f64 / elapsed_ms as f64) * 1000.0;

    println!("🚀 POC: Generated {} words in {:.3}s", batch_size, elapsed_ms / 1000.0);
    println!("🔥 RAW THROUGHPUT: {:.2} BILLION words/second", words_per_second / 1e9);
    ```

  - [ ] **Test with different block/grid configurations**
    - Find optimal block size (128, 256, 512, 1024 threads)
    - Maximize occupancy for RTX 4070 (5,888 CUDA cores)
    - Profile with Nsight Compute to verify saturation

  - [ ] **Validate against theoretical limit**
    - RTX 4070: 5,888 cores × 2.5 GHz / 10 cycles = 1.47B words/s
    - Goal: Achieve >80% of theoretical (>1.17B words/s)
    - Stretch: Achieve >90% of theoretical (>1.32B words/s)
    - **ULTIMATE GOAL: Hit or exceed 1.47B words/s** 💪

  - [ ] **Document POC results**
    - Record throughput for various batch sizes
    - Record kernel execution time vs batch size
    - Generate performance graph
    - Compare against theoretical maximum
    - **CELEBRATE IF WE HIT 1.47B WORDS/S!** 🎉🎊🍾

- [ ] **Implement production CUDA kernel (kernel.cu)**
  ```cuda
  // Production kernel: actually write to output buffer
  __global__ void generate_words_kernel(
      const char* charset_data,
      const int* charset_offsets,
      const int* charset_sizes,
      const int* mask_pattern,
      uint64_t start_idx,
      int word_length,
      char* output_buffer,
      uint64_t batch_size
  ) {
      uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= batch_size) return;

      uint64_t idx = start_idx + tid;
      char* word = output_buffer + (tid * (word_length + 1)); // +1 for newline

      // Convert index to word
      uint64_t remaining = idx;
      for (int pos = word_length - 1; pos >= 0; pos--) {
          int charset_id = mask_pattern[pos];
          int cs_size = charset_sizes[charset_id];
          int char_idx = remaining % cs_size;
          word[pos] = charset_data[charset_offsets[charset_id] + char_idx];
          remaining /= cs_size;
      }
      word[word_length] = '\n';
  }
  ```

- [ ] **Implement GPU context manager (context.rs)**
  - GPU initialization and cleanup
  - Memory allocation (device memory for charsets, output buffer)
  - Kernel compilation and caching
  - Error handling for GPU operations
  - Performance monitoring (throughput, bandwidth)

- [ ] **Implement batch generation function**
  ```rust
  pub struct GpuGenerator {
      context: CudaContext,
      charsets: Vec<Vec<u8>>,
      mask: Vec<usize>,
      batch_size: usize,
      // Performance metrics
      total_words_generated: u64,
      total_time_ms: f64,
  }

  impl GpuGenerator {
      pub fn generate_batch(&mut self, start_idx: u64) -> Vec<u8> {
          // Launch kernel, copy results from GPU
          // Track performance metrics
      }

      pub fn throughput(&self) -> f64 {
          self.total_words_generated as f64 / (self.total_time_ms / 1000.0)
      }
  }
  ```

#### 1.4 Correctness Validation

- [ ] **Implement test suite**
  - Compare GPU output with CPU reference implementation
  - Test various mask patterns and charset sizes
  - Verify index-to-word bijection
  - Test boundary conditions (first word, last word, wraparound)

- [ ] **Cross-validation with maskprocessor**
  - Generate same patterns with both tools
  - Compare output byte-for-byte
  - Ensure ordering is identical

### Phase 2: Binding Implementation

#### 2.1 Binding 1: Stdout Stream

- [ ] **Implement stdout streaming (bindings/stdout.rs)**
  ```rust
  pub struct StdoutStream {
      generator: GpuGenerator,
      current_idx: u64,
      total_keyspace: u128,
  }

  impl StdoutStream {
      pub fn stream(&mut self) {
          let stdout = std::io::stdout();
          let mut writer = BufWriter::new(stdout.lock());

          while self.current_idx < self.total_keyspace as u64 {
              let batch = self.generator.generate_batch(self.current_idx);
              writer.write_all(&batch)?;
              self.current_idx += batch.len() as u64 / (word_length + 1);
          }
          writer.flush()?;
      }
  }
  ```

- [ ] **Implement double-buffering for stdout**
  - GPU generates batch N+1 while CPU writes batch N
  - Use async tasks or threads for parallel GPU/CPU work
  - Minimize idle time on both GPU and CPU

- [ ] **CLI implementation for stdout mode**
  ```bash
  gpu-scatter-gather -1 'abc' -2 '123' '?1?2' --output stdout
  ```

#### 2.2 Binding 2: In-Memory Stream

- [ ] **Implement zero-copy iterator (bindings/memory.rs)**
  ```rust
  pub struct MemoryStream {
      generator: GpuGenerator,
      current_batch: Vec<u8>,
      current_idx: u64,
      batch_offset: usize,
      word_length: usize,
  }

  impl Iterator for MemoryStream {
      type Item = &[u8];

      fn next(&mut self) -> Option<Self::Item> {
          if self.batch_offset >= self.current_batch.len() {
              self.current_batch = self.generator.generate_batch(self.current_idx);
              self.batch_offset = 0;
              if self.current_batch.is_empty() {
                  return None;
              }
          }
          let word = &self.current_batch[self.batch_offset..self.batch_offset + self.word_length];
          self.batch_offset += self.word_length + 1; // +1 for newline
          Some(word)
      }
  }
  ```

- [ ] **API design for library usage**
  ```rust
  let stream = WordlistGenerator::new()
      .mask("?1?2")
      .charset(1, "abc")
      .charset(2, "123")
      .build()?
      .into_memory_stream();

  for word in stream {
      // Process word
  }
  ```

#### 2.3 Binding 3: Memory-Mapped File Output

- [ ] **Implement mmap file writer (bindings/file.rs)**
  - Use `memmap2` crate for memory-mapped files
  - Pre-allocate file based on keyspace size
  - Write GPU batches directly to mmap'd region
  - Implement progress reporting

- [ ] **Async file I/O optimization**
  - Use `tokio::fs` for async writes
  - Batch writes to minimize syscalls
  - Configurable buffer sizes

- [ ] **CLI for file output**
  ```bash
  gpu-scatter-gather -1 'abc' -2 '123' '?1?2' --output file -o wordlist.txt
  ```

#### 2.4 Binding 4: Network Stream

- [ ] **Implement TCP server (bindings/network.rs)**
  ```rust
  pub struct NetworkStream {
      generator: GpuGenerator,
      listener: TcpListener,
      compression: Option<CompressionType>,
  }
  ```

- [ ] **Protocol design**
  - Custom binary protocol for efficiency
  - Header: keyspace size, word length, charset info
  - Body: chunked word batches
  - Optional compression (LZ4/Zstd)

- [ ] **CLI for network mode**
  ```bash
  gpu-scatter-gather -1 'abc' -2 '123' '?1?2' --output network --port 8080
  ```

- [ ] **Client library**
  - Rust client for consuming network stream
  - Example: distributed cracking coordinator

#### 2.5 Binding 5: Language Bindings

- [ ] **Python bindings (PyO3)**
  ```python
  from gpu_scatter_gather import WordlistGenerator

  gen = WordlistGenerator(mask='?1?2', charsets={1: 'abc', 2: '123'})
  for word in gen:
      print(word.decode('utf-8'))

  # Or batch mode
  batch = gen.next_batch(1_000_000)
  ```

- [ ] **Node.js bindings (Neon)**
  ```javascript
  const { WordlistGenerator } = require('gpu-scatter-gather');

  const gen = new WordlistGenerator({
      mask: '?1?2',
      charsets: { 1: 'abc', 2: '123' }
  });

  for await (const word of gen) {
      console.log(word.toString());
  }
  ```

- [ ] **C FFI for maximum compatibility**
  ```c
  #include "gpu_scatter_gather.h"

  gpu_sg_generator_t* gen = gpu_sg_create("?1?2", ...);
  char* batch = gpu_sg_next_batch(gen, 1000000);
  gpu_sg_free(gen);
  ```

#### 2.6 Binding 6: Callback Interface

- [ ] **Implement callback handler (bindings/callback.rs)**
  ```rust
  pub trait WordHandler {
      fn handle_word(&mut self, word: &[u8]);
  }

  impl GpuGenerator {
      pub fn generate_with_handler<H: WordHandler>(&mut self, handler: &mut H) {
          for batch_idx in 0..num_batches {
              let batch = self.generate_batch(batch_idx * batch_size);
              for word in batch.chunks(word_length + 1) {
                  handler.handle_word(&word[..word_length]);
              }
          }
      }
  }
  ```

### Phase 3: Optimization & Performance Tuning

#### 3.1 GPU Kernel Optimization

- [ ] **Profile GPU kernel with Nsight Compute**
  - Identify bottlenecks (memory, compute, occupancy)
  - Optimize block size and grid dimensions
  - Analyze warp divergence and stalls

- [ ] **Optimize memory access patterns**
  - Ensure coalesced global memory access
  - Use shared memory for frequently accessed data (charsets)
  - Minimize bank conflicts

- [ ] **Arithmetic optimization**
  - Replace division/modulo with faster alternatives if possible
  - Use `__ldg()` for read-only data (texture memory)
  - Consider warp-level primitives for coordination

- [ ] **Occupancy optimization**
  - Tune registers per thread
  - Adjust block size for maximum occupancy
  - Profile with `nvprof` or Nsight Compute

#### 3.2 CPU-GPU Transfer Optimization

- [ ] **Pinned memory allocation**
  - Use CUDA pinned memory for faster host-device transfers
  - Pre-allocate buffers to avoid allocation overhead

- [ ] **Asynchronous transfers**
  - Use CUDA streams for overlapping compute + transfer
  - Pipeline: GPU computes batch N, transfers batch N-1 simultaneously

- [ ] **Batch size tuning**
  - Experiment with different batch sizes (1M, 10M, 100M words)
  - Find optimal balance between transfer overhead and latency
  - Consider GPU memory constraints

#### 3.3 Multi-GPU Support

- [ ] **Implement multi-GPU coordinator**
  ```rust
  pub struct MultiGpuGenerator {
      generators: Vec<GpuGenerator>,
      keyspace_per_gpu: u64,
  }
  ```

- [ ] **Keyspace partitioning**
  - Divide keyspace equally among available GPUs
  - Each GPU generates non-overlapping ranges
  - Merge output streams in correct order

- [ ] **Load balancing**
  - Monitor GPU utilization
  - Dynamically adjust keyspace assignment
  - Handle GPU failures gracefully

#### 3.4 Compression & Deduplication

- [ ] **Optional compression for network/file output**
  - Integrate LZ4 for fast compression
  - Integrate Zstd for high compression ratio
  - Configurable compression level

- [ ] **Streaming compression**
  - Compress GPU batches before transfer to CPU
  - Decompress on-demand for consumers

### Phase 4: Testing & Validation

#### 4.1 Correctness Testing

- [ ] **Comprehensive test suite**
  - Unit tests for index-to-word conversion
  - Integration tests for each binding
  - Property-based testing (e.g., bijection tests)

- [ ] **Fuzz testing**
  - Generate random masks and charsets
  - Verify all outputs are valid
  - Check for crashes, panics, GPU errors

- [ ] **Cross-platform testing**
  - Linux (Ubuntu, Arch, RHEL)
  - Windows (10, 11)
  - Test on different GPUs (NVIDIA 20xx, 30xx, 40xx series)

#### 4.2 Performance Benchmarking

- [ ] **Benchmark against existing tools**
  - maskprocessor: ~142M words/s baseline
  - crunch: ~5M words/s (very slow)
  - hashcat built-in: varies by mode

- [ ] **Benchmark each binding**
  - Stdout: measure throughput to /dev/null
  - Memory: measure iterator throughput
  - File: measure write throughput to NVMe SSD
  - Network: measure throughput over loopback

- [ ] **GPU performance analysis**
  - Measure kernel execution time
  - Measure memory transfer time
  - Calculate achieved bandwidth vs. theoretical
  - Analyze occupancy and efficiency

- [ ] **Create performance comparison table**
  ```
  | Tool                          | Speed (words/s) | Speedup vs maskprocessor |
  |-------------------------------|-----------------|--------------------------|
  | maskprocessor (CPU)           | 142M            | 1.0x (baseline)          |
  | crunch (CPU)                  | 5M              | 0.035x                   |
  | GPU Scatter-Gather (stdout)   | 200M            | 1.4x                     |
  | GPU Scatter-Gather (memory)   | 800M            | 5.6x                     |
  | GPU Scatter-Gather (file)     | 600M            | 4.2x                     |
  ```

#### 4.3 Stress Testing

- [ ] **Large keyspace testing**
  - Test with keyspaces > 2^64 (use u128)
  - Verify no overflow/wraparound issues
  - Test resume functionality

- [ ] **Long-running stability tests**
  - Generate for hours continuously
  - Monitor memory leaks (GPU and CPU)
  - Check for GPU throttling/overheating

- [ ] **Resource exhaustion testing**
  - Test with very large charsets (10K+ characters)
  - Test with very long masks (100+ positions)
  - Handle out-of-memory gracefully

### Phase 5: Documentation & Polish

#### 5.1 User Documentation

- [ ] **README.md**
  - Project overview and goals
  - Installation instructions (pre-built binaries + source)
  - Quick start guide
  - Performance comparison table
  - Use cases and examples

- [ ] **API Documentation (rustdoc)**
  - Document all public APIs
  - Include usage examples in doc comments
  - Document performance characteristics

- [ ] **User Guide**
  - Detailed explanation of each binding
  - Configuration options
  - Performance tuning guide
  - Troubleshooting common issues

- [ ] **Tutorial: Distributed Cracking**
  - Example: coordinating multiple GPUs across machines
  - Keyspace partitioning strategies
  - Network protocol details

#### 5.2 Developer Documentation

- [ ] **Architecture documentation**
  - Explain index-to-word algorithm
  - CUDA kernel design rationale
  - Binding architecture

- [ ] **Contributing guide**
  - How to build from source
  - How to run tests and benchmarks
  - Code style guidelines

- [ ] **Algorithm explanation**
  - Mathematical derivation of mixed-radix conversion
  - Complexity analysis
  - Comparison with odometer algorithm

#### 5.3 Examples & Demos

- [ ] **Example: Basic CLI usage**
  ```bash
  # Simple wordlist generation
  gpu-scatter-gather -1 'abc' -2 '123' '?1?2' > wordlist.txt

  # Pipe to hashcat
  gpu-scatter-gather -1 'ABC' -2 '0123456789' '?1?1?2?2?2?2' | hashcat -m 2500 capture.hccapx
  ```

- [ ] **Example: Library usage (Rust)**
  ```rust
  use gpu_scatter_gather::WordlistGenerator;

  let mut gen = WordlistGenerator::new()
      .mask("?1?2?3")
      .charset(1, "abc")
      .charset(2, "123")
      .charset(3, "xyz")
      .build()?;

  for word in gen.into_memory_stream() {
      println!("{}", String::from_utf8_lossy(word));
  }
  ```

- [ ] **Example: Python API**
  ```python
  from gpu_scatter_gather import WordlistGenerator

  gen = WordlistGenerator(
      mask='?1?2?3',
      charsets={1: 'abc', 2: '123', 3: 'xyz'}
  )

  # Iterator interface
  for word in gen:
      print(word.decode('utf-8'))

  # Batch interface
  batch = gen.next_batch(1_000_000)
  ```

- [ ] **Example: Distributed generation**
  ```rust
  // Server
  let server = NetworkStream::bind("0.0.0.0:8080", mask, charsets)?;
  server.serve().await?;

  // Client
  let stream = NetworkClient::connect("server:8080")?;
  for word in stream {
      // Process word
  }
  ```

### Phase 6: Distribution & Release

#### 6.1 Build System

- [ ] **Cross-compilation setup**
  - Linux x86_64 (CUDA 11.8, 12.x)
  - Windows x86_64 (CUDA 11.8, 12.x)
  - Support for multiple CUDA compute capabilities (7.0+, 8.0+, 9.0+)

- [ ] **Pre-built binary releases**
  - Automated builds with GitHub Actions
  - Release on GitHub Releases
  - Include CUDA runtime dependencies

- [ ] **Package distribution**
  - Cargo crate for Rust users
  - PyPI package for Python bindings
  - npm package for Node.js bindings
  - AUR package for Arch Linux
  - Homebrew formula for macOS (if Metal/OpenCL port added)

#### 6.2 CI/CD Pipeline

- [ ] **GitHub Actions workflows**
  - Build and test on push
  - Run benchmarks on release
  - Automated release on version tags
  - Cross-platform builds

- [ ] **Benchmark tracking**
  - Track performance over time
  - Detect performance regressions
  - Publish results to GitHub Pages

#### 6.3 Release Checklist

- [ ] **v0.1.0: Proof of Concept**
  - Basic GPU kernel working
  - Stdout binding functional
  - Correctness validated against maskprocessor
  - Performance: >200M words/s

- [ ] **v0.2.0: Multi-Binding Support**
  - All 6 bindings implemented
  - Performance: >500M words/s for memory binding
  - Python bindings available

- [ ] **v0.3.0: Optimization Pass**
  - GPU kernel optimized (profile-guided)
  - Multi-GPU support
  - Performance: >800M words/s for memory binding

- [ ] **v1.0.0: Production Ready**
  - All features complete and tested
  - Comprehensive documentation
  - Stable API
  - Performance: approaching 1B words/s

## Technical Challenges & Solutions

### Challenge 1: Mixed-Radix Arithmetic Precision
**Problem:** Division/modulo operations on GPU can be slow, especially for large indices (u64).

**Solutions:**
- Use fast integer division libraries (e.g., libdivide)
- Precompute division magic numbers
- Consider using Barrett reduction or Montgomery multiplication
- For specific charset sizes (powers of 2), use bitwise operations

### Challenge 2: Output Ordering
**Problem:** Multi-GPU or multi-stream generation produces out-of-order results.

**Solutions:**
- Assign non-overlapping keyspace ranges to each GPU
- Merge streams in correct order using priority queue
- For memory binding, allow out-of-order if user doesn't care
- For stdout, enforce ordering with buffering

### Challenge 3: Large Keyspace (> 2^64)
**Problem:** Keyspaces can exceed u64 range, requiring u128 support.

**Solutions:**
- Use u128 for keyspace calculations
- Implement 128-bit arithmetic in CUDA kernel (PTX asm if needed)
- Partition keyspace into u64 ranges for GPU processing
- Document keyspace limits clearly

### Challenge 4: GPU Memory Constraints
**Problem:** Limited GPU memory for large batches or many charsets.

**Solutions:**
- Dynamically adjust batch size based on available GPU memory
- Stream charsets to GPU if they don't fit in memory
- Use texture memory for read-only charset data
- Implement memory pooling to reuse buffers

### Challenge 5: CPU-GPU Transfer Bottleneck
**Problem:** PCIe bandwidth can limit throughput for small word sizes.

**Solutions:**
- Use pinned memory for faster transfers
- Increase batch size to amortize transfer overhead
- Pipeline transfers with computation using CUDA streams
- For stdout binding, accept lower throughput as inherent limitation

### Challenge 6: Platform Compatibility
**Problem:** CUDA is NVIDIA-only, limiting hardware support.

**Solutions:**
- Primary target: NVIDIA GPUs with CUDA (widest adoption)
- Future: Add OpenCL backend for AMD/Intel GPUs
- Future: Add Metal backend for Apple Silicon
- Fall back to CPU implementation if no GPU available

## Comparison: GPU vs CPU Algorithms

| Aspect | CPU Odometer (maskprocessor) | GPU Scatter-Gather (this project) |
|--------|------------------------------|-----------------------------------|
| **Algorithm** | Sequential carry propagation | Direct index-to-word mapping |
| **Parallelism** | Single-threaded (or limited threads) | Massively parallel (1000s of threads) |
| **Memory Access** | Sequential, cache-friendly | Parallel, coalesced access |
| **Random Access** | Expensive (must iterate from start) | Free (O(1) jump to any position) |
| **Performance** | ~142M words/s (1 core) | 500M-1B words/s (GPU) |
| **Scalability** | Limited by single-core speed | Scales with GPU cores |
| **Use Cases** | General-purpose, pipe to tools | High-throughput, programmatic access |
| **Distributed** | Difficult (resume from position hard) | Easy (assign keyspace ranges) |

## Success Criteria

### Performance Targets
- [ ] **v0.1.0**: >200M words/s (1.4x faster than maskprocessor)
- [ ] **v0.2.0**: >500M words/s (3.5x faster than maskprocessor)
- [ ] **v1.0.0**: >800M words/s (5.6x faster than maskprocessor)
- [ ] **Stretch Goal**: >1B words/s (7x faster than maskprocessor)

### Usability Targets
- [ ] Drop-in CLI replacement for maskprocessor
- [ ] Simple API for Rust, Python, Node.js
- [ ] Clear documentation and examples
- [ ] Pre-built binaries for Linux and Windows

### Reliability Targets
- [ ] 100% output correctness (validated against maskprocessor)
- [ ] No crashes or GPU errors under normal use
- [ ] Graceful handling of edge cases and errors
- [ ] Stable performance over extended runtime

## Phase 2.6: Formal Validation & Scientific Rigor (October 16, 2025)

**Status:** ✅ **FORMAL SPECIFICATION COMPLETE**

### Mathematical Foundations

- [x] **Formal Algorithm Specification** (COMPLETE)
  - Mathematical notation and definitions
  - Formal specification of index-to-word function
  - Complete documentation in `docs/FORMAL_SPECIFICATION.md`

- [x] **Bijection Proof** (COMPLETE)
  - Proved injectivity (one-to-one): No two indices map to same word
  - Proved surjectivity (onto): Every word has corresponding index
  - Constructive proof with explicit inverse function

- [x] **Completeness Proof** (COMPLETE)
  - Proved algorithm generates entire keyspace exactly once
  - No gaps, no duplicates (mathematical certainty)

- [x] **Ordering Correctness Proof** (COMPLETE)
  - Proved output follows canonical mixed-radix ordering
  - Matches maskprocessor ordering (validated empirically + proven mathematically)

- [x] **Complexity Analysis** (COMPLETE)
  - Time: O(n · log m) per word (formal analysis)
  - Space: O(n + Σ|C_i|) total (constant per word)
  - GPU parallelism analysis with theoretical throughput derivation

### Remaining Formal Validation Tasks

- [ ] **Statistical Validation Suite**
  - [ ] Implement Chi-square test for uniform distribution
  - [ ] Implement autocorrelation tests (no unexpected patterns)
  - [ ] Implement runs test for sequence randomness
  - [ ] Document results in `docs/STATISTICAL_VALIDATION.md`

- [ ] **Scientific Benchmarking Framework**
  - [ ] Standardized methodology with multiple runs
  - [ ] Statistical analysis: mean, median, std dev, confidence intervals
  - [ ] Outlier detection and handling
  - [ ] Warm-up runs to exclude cold-start effects
  - [ ] Publish raw data and scripts for reproducibility

- [ ] **Formal Verification (Optional - Ambitious)**
  - [ ] Formalize proofs in Coq or Lean 4 (machine-verified)
  - [ ] Extract verified executable code
  - [ ] CUDA kernel verification with GPUVerify or Harmony

- [ ] **Academic Publication** (Optional)
  - [ ] Write whitepaper with formal proofs
  - [ ] Literature review of related work
  - [ ] Target: USENIX Security, ACM CCS, or arXiv
  - [ ] Include reproducibility package (code, data, proofs)

### Documentation

- [x] `docs/FORMAL_SPECIFICATION.md` - Complete mathematical specification with proofs
- [x] `docs/FORMAL_VALIDATION_PLAN.md` - Comprehensive plan for academic rigor
- [ ] `docs/STATISTICAL_VALIDATION.md` - Statistical test results
- [ ] `docs/SCIENTIFIC_BENCHMARKS.md` - Rigorous benchmarking methodology
- [ ] `paper/` - Optional academic paper directory

### Key Achievements

✅ **Mathematical Proofs Complete:**
- Bijection: Every index ↔ exactly one word
- Completeness: Full keyspace coverage, no gaps
- Ordering: Canonical mixed-radix (proven + empirically validated)
- Complexity: Formal time/space analysis

✅ **Professional Quality:**
- Moved from "it works" to "provably correct"
- Academic-level rigor with formal proofs
- Foundation for potential publication
- Reference implementation for future research

---

## Future Enhancements (Post-v1.0)

### Advanced Features
- [ ] **Resume/checkpoint functionality**
  - Save generation state to disk
  - Resume from arbitrary keyspace position
  - Useful for very large keyspaces

- [ ] **Smart mask generation**
  - Integrate with machine learning for mask prediction
  - Generate masks based on leaked password patterns
  - Adaptive charset selection

- [ ] **Distributed coordinator**
  - Master-worker architecture for cluster generation
  - Dynamic load balancing across multiple machines
  - Fault tolerance and worker recovery

- [ ] **GPU compute optimization**
  - Tensor Core utilization (if applicable)
  - Multi-GPU scaling across multiple machines (NVLink)
  - Support for newer CUDA architectures (Hopper, Blackwell)

### Platform Support
- [ ] **OpenCL backend** (AMD, Intel, Apple GPUs)
- [ ] **Metal backend** (Apple Silicon M1/M2/M3)
- [ ] **CPU fallback** (SIMD-optimized)
- [ ] **Cloud GPU support** (AWS, GCP, Azure GPU instances)

### Integration
- [ ] **Hashcat plugin** (generate + hash on GPU)
- [ ] **John the Ripper integration**
- [ ] **HTTP REST API** for web service deployment
- [ ] **Docker container** for easy deployment

## Resources & References

### CUDA Programming
- **NVIDIA CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **Nsight Compute**: https://developer.nvidia.com/nsight-compute

### Rust CUDA Bindings
- **cudarc**: https://github.com/coreylowman/cudarc
- **cust**: https://github.com/Rust-GPU/Rust-CUDA
- **RustaCUDA**: https://github.com/bheisler/RustaCUDA

### Related Projects
- **maskprocessor**: https://github.com/hashcat/maskprocessor
- **hashcat**: https://hashcat.net/hashcat/
- **crunch**: https://sourceforge.net/projects/crunch-wordlist/

### Algorithm Research
- **Mixed-Radix Number Systems**: https://en.wikipedia.org/wiki/Mixed_radix
- **Combinatorial Number System**: https://en.wikipedia.org/wiki/Combinatorial_number_system

## Team & Contributions

### Core Team
- **Lead Developer**: tehw0lf
- **CUDA Expert**: TBD
- **Bindings Maintainer**: TBD

### Contribution Areas
- CUDA kernel optimization
- Language bindings (Python, Node.js, Go, etc.)
- Platform ports (OpenCL, Metal, CPU fallback)
- Documentation and examples
- Testing and validation
- Performance benchmarking

### How to Contribute
See CONTRIBUTING.md (to be created) for:
- Code style guidelines
- Pull request process
- Testing requirements
- Performance regression policies

---

## License

MIT or Apache-2.0 (dual-licensed for maximum compatibility)

## Acknowledgments

- **maskprocessor** - Inspiration for the problem space
- **hashcat** - Motivation for high-performance wordlist generation
- **CUDA ecosystem** - Making GPU computing accessible
- **Rust community** - For excellent tooling and libraries
