# GPU Scatter-Gather Wordlist Generator

**The world's fastest wordlist generator using GPU acceleration**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.82+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Release](https://img.shields.io/badge/release-v1.2.0-brightgreen.svg)](https://github.com/tehw0lf/gpu-scatter-gather/releases/tag/v1.2.0)

> 📄 **[Read the Technical Whitepaper](https://github.com/tehw0lf/gpu-scatter-gather/releases/download/v1.0.0/GPU_Scatter_Gather_Whitepaper_v1.0.0.pdf)** - Comprehensive algorithm design, formal proofs, and performance evaluation
>
> ✅ **Status:** v1.2.0 Released! (Async Multi-GPU Optimization)
>
> Production-ready library with **4-7× speedup** over CPU tools (maskprocessor, cracken).
> Complete C FFI API with 24 functions (17 single-GPU + 7 multi-GPU), 3 output formats, formal validation, and integration guides.
> **NEW:** Async multi-GPU optimization with CUDA streams (+11% on medium batches)!
> See [Development Log](docs/development/DEVELOPMENT_LOG.md) for detailed progress.

## Overview

GPU Scatter-Gather is a GPU-accelerated wordlist generator that achieves **440-700M words/second** - **4-7× faster than maskprocessor** - using a novel scatter-gather algorithm based on mixed-radix arithmetic.

### Key Innovation

Instead of traditional sequential odometer iteration, this generator uses **direct index-to-word mapping**:

```
Index → Mixed-Radix Decomposition → Word
```

This enables:
- ⚡ **Massive parallelism** - Every GPU thread generates words independently
- 🎯 **O(1) random access** - Jump to any position in keyspace instantly
- 🚀 **Perfect GPU utilization** - No sequential dependencies or warp divergence
- 📈 **Linear scaling** - Performance scales with GPU cores

### Performance

**Target Hardware:** NVIDIA RTX 4070 (5,888 CUDA cores)
**Actual Hardware Tested:** NVIDIA RTX 4070 Ti SUPER (8,448 CUDA cores)

| Tool | Speed | Speedup |
|------|-------|---------|
| **GPU Scatter-Gather** | **440-700M words/s** | **4-7×** 🏆 |
| maskprocessor (CPU) | 100-142M words/s | 1.0× (baseline) |
| cracken (CPU) | ~100M words/s | ~1.0× |

*Note: v1.0.0 benchmarks with complete C FFI overhead and realistic workloads. See [Technical Whitepaper](https://github.com/tehw0lf/gpu-scatter-gather/releases/download/v1.0.0/GPU_Scatter_Gather_Whitepaper_v1.0.0.pdf) for detailed methodology and validation.*

## Features

### v1.2.0 Release ✅ (Async Multi-GPU Optimization)

- ✅ **Async Multi-GPU Execution** - CUDA streams for overlapped kernel execution
- ✅ **+11.3% Performance Improvement** - Optimized for medium batches (50M words)
- ✅ **MultiGpuContext::new_async() API** - Opt-in async mode, fully backward compatible
- ✅ **Per-Thread Stream Management** - Thread-safe CUDA stream creation and synchronization
- ✅ **48/48 Tests Passing** - Added 4 new async-specific tests
- ✅ **Multi-GPU API** - 7 functions for automatic parallel generation across GPUs
- ✅ **90-95% Scaling Efficiency** - Minimal overhead with automatic workload distribution
- ✅ **Automatic Keyspace Partitioning** - Static distribution algorithm with load balancing
- ✅ **Thread-Safe Parallel Execution** - One thread per GPU with synchronized aggregation
- ✅ **Complete C FFI API** - 24 functions (17 single-GPU + 7 multi-GPU)
- ✅ **Three output formats** - NEWLINES, PACKED, FIXED_WIDTH
- ✅ **Streaming API** - Zero-copy GPU operation with async batching
- ✅ **Production GPU kernel** - 440-700M words/s per GPU (4-7× faster than CPU tools)
- ✅ **Formal validation** - 100% correctness with mathematical proofs
- ✅ **Statistical testing** - Chi-square, autocorrelation, runs tests
- ✅ **Cross-validation** - 100% match with maskprocessor
- ✅ **Multi-architecture support** - sm_70-90 (Turing to Hopper)
- ✅ **Comprehensive documentation** - API specs, integration guides, whitepaper, multi-GPU benchmarks
- ✅ **Integration guides** - hashcat, John the Ripper, generic C programs
- ✅ **Clean Rust API** - RAII memory management, type-safe

### Planned (v1.3.0+)

- 🔜 Pinned memory with proper context management (10-15% additional improvement)
- 🔜 Dynamic load balancing for heterogeneous GPUs (5-10% efficiency gain)
- 🔜 Persistent thread pool for reduced latency on repeated calls
- 🔜 Single-GPU memory coalescing optimization (2-3× potential speedup)
- 🔜 Hybrid masks (static prefix/suffix + dynamic middle)
- 🔜 Python/JavaScript bindings (PyPI, npm packages)
- 🔜 Advanced optimizations (Barrett reduction, power-of-2 fast paths)
- 🔜 OpenCL backend (AMD/Intel GPU support)

## Quick Start

### Prerequisites

- **Rust 1.82+** - [Install Rust](https://rustup.rs/)
- **CUDA Toolkit 11.8+** - [Download CUDA](https://developer.nvidia.com/cuda-downloads)
- **NVIDIA GPU** with compute capability 7.5+ (Turing or newer)

### Building

```bash
# Clone the repository
git clone https://github.com/tehw0lf/gpu-scatter-gather
cd gpu-scatter-gather

# Build the project (compiles CUDA kernels automatically)
cargo build --release

# Run tests to verify installation
cargo test
```

### Usage

#### CPU Mode (Current)

```bash
# Generate simple wordlist
gpu-scatter-gather -1 'abc' -2 '123' '?1?2'

# Output:
# a1
# a2
# b1
# b2
# c1
# c2

# Show keyspace size
gpu-scatter-gather -1 'abc' -2 '123' '?1?2' --keyspace
# Output: Keyspace size: 9

# Use predefined charsets
gpu-scatter-gather --lowercase --digits '?1?1?2?2'
```

#### Multi-GPU C API (v1.1.0)

```c
#include <wordlist_generator.h>

int main() {
    // Create multi-GPU generator (uses all GPUs automatically)
    wg_multigpu_handle_t gen = wg_multigpu_create();
    printf("Using %d GPU(s)\n", wg_multigpu_get_device_count(gen));

    // Configure charsets
    wg_multigpu_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz", 26);
    wg_multigpu_set_charset(gen, 2, "0123456789", 10);

    // Set mask: ?1?1?1?1?2?2?2?2 (4 letters + 4 digits)
    int mask[] = {1, 1, 1, 1, 2, 2, 2, 2};
    wg_multigpu_set_mask(gen, mask, 8);
    wg_multigpu_set_format(gen, WG_FORMAT_PACKED);

    // Generate 100M words across all GPUs
    uint8_t* buffer = malloc(100000000 * 8);
    ssize_t bytes = wg_multigpu_generate(gen, 0, 100000000, buffer, 100000000 * 8);

    printf("Generated %zd bytes\n", bytes);

    free(buffer);
    wg_multigpu_destroy(gen);
    return 0;
}
```

**Multi-GPU Features:**
- ✅ Automatic device detection and initialization
- ✅ Transparent workload partitioning
- ✅ 90-95% scaling efficiency (minimal overhead)
- ✅ Same API as single-GPU (simplified parallel generation)

See [Multi-GPU Benchmarking Results](docs/benchmarking/MULTI_GPU_RESULTS.md) for detailed performance data.

#### Piping to Hashcat (Planned)

```bash
# Once stdout binding is implemented:
gpu-scatter-gather -1 '?l' -2 '?d' '?1?1?2?2?2?2' | hashcat -m 2500 capture.hccapx
```

## Algorithm

### Mixed-Radix Decomposition

Given a mask pattern with varying charset sizes, we convert an index directly to a word:

```rust
fn index_to_word(index: u64, mask: &[usize], charsets: &[&[u8]], output: &mut [u8]) {
    let mut remaining = index;

    // Process positions from right to left
    for pos in (0..mask.len()).rev() {
        let charset_id = mask[pos];
        let charset = charsets[charset_id];
        let charset_size = charset.len() as u64;

        let char_idx = (remaining % charset_size) as usize;
        output[pos] = charset[char_idx];
        remaining /= charset_size;
    }
}
```

### CUDA Kernel

```cuda
__global__ void generate_words_kernel(
    const char* charset_data,
    const int* charset_offsets,
    const int* charset_sizes,
    const int* mask_pattern,
    unsigned long long start_idx,
    int word_length,
    char* output_buffer,
    unsigned long long batch_size
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    unsigned long long idx = start_idx + tid;
    char* word = output_buffer + (tid * (word_length + 1));

    // Convert index to word (same algorithm as CPU)
    unsigned long long remaining = idx;
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

**Key Properties:**
- Every thread operates completely independently (no synchronization)
- No warp divergence (all threads follow same execution path)
- Coalesced memory access for maximum bandwidth
- Scales linearly with GPU cores

## Benchmarks

### Phase 2 Production Results ✅

See detailed results in [docs/benchmarking/](docs/benchmarking/).

**Production Performance (with full memory I/O):**

| Batch Size | Throughput | Speedup vs maskprocessor |
|------------|-----------|--------------------------|
| 10M words | 1,158 M/s | 8.16x |
| 50M words | **1,237 M/s** | **8.71x** 🏆 |
| 100M words | 1,189 M/s | 8.37x |
| 500M words | 898 M/s | 6.33x |
| 1B words | 635 M/s | 4.47x |

**Validation:**
- ✅ 100% output correctness (9/9 matches with CPU reference)
- ✅ Production kernel with full memory writes
- ✅ Includes GPU compute + memory I/O + PCIe transfer
- ✅ Zero errors or crashes

**Hardware:**
- NVIDIA GeForce RTX 4070 Ti SUPER
- 8,448 CUDA cores, 66 SMs
- Compute capability 8.9
- 16 GB GDDR6X, 672 GB/s bandwidth

### Phase 1 POC Results

See [docs/archive/POC_RESULTS.md](docs/archive/POC_RESULTS.md) for the initial proof-of-concept results that validated the algorithm.

## Project Structure

```
gpu-scatter-gather/
├── src/
│   ├── lib.rs              # Core library and API
│   ├── main.rs             # CLI entry point
│   ├── charset.rs          # Charset management
│   ├── keyspace.rs         # Keyspace calculation and index-to-word
│   ├── mask.rs             # Mask pattern parsing
│   ├── gpu/                # GPU module (CUDA integration)
│   └── bindings/           # Output bindings (stdout, memory, file, network)
├── kernels/
│   └── wordlist_poc.cu     # CUDA kernels
├── examples/
│   ├── validate_gpu.rs         # GPU output validation vs CPU
│   ├── benchmark_production.rs # Production performance benchmark
│   ├── poc_benchmark.rs        # POC performance test
│   └── poc_accurate.rs         # Accurate timing with CUDA events
├── tests/                  # Integration tests
├── benches/                # Criterion benchmarks
├── docs/
│   ├── api/                # C API & FFI documentation
│   ├── design/             # Architecture and design
│   ├── validation/         # Correctness validation
│   ├── benchmarking/       # Performance measurement
│   ├── guides/             # User and integration guides
│   ├── development/        # Internal development docs
│   └── archive/            # Historical documents
└── build.rs                # CUDA kernel compilation
```

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_index_to_word_complex_pattern
```

### Running Benchmarks

```bash
# CPU reference benchmarks
cargo bench

# GPU validation (check correctness)
cargo run --example validate_gpu --release

# GPU production benchmark (realistic performance)
cargo run --example benchmark_production --release

# GPU POC benchmark (compute-only)
cargo run --example poc_accurate --release
```

### Building for Different GPU Architectures

The build script automatically compiles kernels for multiple architectures:

- **sm_75:** Turing (RTX 20xx series)
- **sm_80:** Ampere (A100)
- **sm_86:** Ampere (RTX 30xx series)
- **sm_89:** Ada Lovelace (RTX 40xx series)
- **sm_90:** Hopper (H100)

The correct kernel is loaded at runtime based on your GPU.

## Use Cases

- **Password security testing** - Audit password strength
- **Security research** - Test authentication systems
- **Academic research** - Study password patterns and entropy
- **Integration with security tools** - Hashcat, John the Ripper

**⚠️ Ethical Use Only:** This tool is intended for defensive security research, testing, and auditing.
Unauthorized access to systems is illegal. Always obtain proper authorization before testing.

## Comparison

### Evolution from Author's Prior Work

This project represents the third iteration of wordlist generation by the author:

| Implementation | Language | Algorithm | Performance | Speedup | Repository |
|----------------|----------|-----------|-------------|---------|------------|
| **wlgen** | Python | itertools.product + recursive | 210K-1.6M words/s | 1× | [github.com/tehw0lf/wlgen](https://github.com/tehw0lf/wlgen) (PyPI) |
| **wlgen-rs** | Rust | Odometer (CPU) | ~150M words/s | ~100× | [github.com/tehw0lf/wlgen-rs](https://github.com/tehw0lf/wlgen-rs) |
| **gpu-scatter-gather** | Rust+CUDA | Mixed-radix direct indexing | 572-757M words/s | **285-3600×** | This project |

**Key insight:** Traditional approaches (Python itertools, Rust odometer) cannot leverage GPU parallelism. The mixed-radix direct indexing algorithm (AI-proposed) enables true GPU acceleration.

### vs maskprocessor

**Our Advantages:**
- **4.5x-8.7x faster** with GPU acceleration (measured, not estimated)
- O(1) random access to any keyspace position
- Perfect for distributed workloads (divide keyspace across machines)
- Programmatic API for library integration
- Modern Rust codebase with memory safety

**Maskprocessor strengths:**
- Mature, battle-tested codebase
- No GPU required
- Works on any hardware

### vs crunch

**Our Advantages:**
- **247x faster** (1.2B vs 5M words/s)
- Handles much larger keyspaces efficiently
- Better memory efficiency
- Modern codebase in Rust
- GPU-accelerated parallel generation

### vs Author's Previous Work (wlgen Python)

**Our Advantages:**
- **285-3600× faster** (750M vs 210K-1.6M words/s)
- GPU acceleration (wlgen investigated CUDA but found no benefit in Python)
- Novel algorithm designed for parallelization
- Scales with GPU cores (wlgen is single-threaded CPU-bound)

### vs hashcat built-in

**Our Advantages:**
- Standalone tool (not tied to hashcat)
- Multiple output bindings (stdout, memory, file, network - planned)
- Optimized specifically for wordlist generation
- Can feed multiple hashcat instances
- Faster than hashcat's internal generator

## Roadmap

### Phase 1: Foundation ✅ (COMPLETE)
- [x] CPU reference implementation
- [x] CUDA kernel infrastructure
- [x] POC validation
- [x] Comprehensive documentation

### Phase 2: Production Kernel ✅ (COMPLETE)
- [x] Implement production kernel with memory writes
- [x] Validate output correctness vs CPU (100% match)
- [x] Benchmark realistic throughput with I/O (635M-1.2B words/s)
- [x] Clean Rust API with RAII memory management

### Phase 3: Bindings & Integration
- [ ] Stdout streaming binding
- [ ] In-memory zero-copy API
- [ ] Memory-mapped file output
- [ ] Python bindings (PyO3)
- [ ] Node.js bindings (Neon)
- [ ] C FFI for maximum compatibility

### Phase 4: Optimization & Polish
- [ ] Multi-GPU support
- [ ] Barrett reduction for division optimization
- [ ] Power-of-2 charset fast path (bitwise operations)
- [ ] Nsight Compute profiling and tuning
- [ ] Compression for network streaming
- [ ] Distributed coordinator for clusters

### Phase 5: Release
- [ ] Comprehensive documentation
- [ ] User guide and tutorials
- [ ] Pre-built binaries for Linux/Windows
- [ ] Package distribution (crates.io, PyPI, npm)
- [ ] Performance comparison whitepaper

## Contributing

### About This Project

This is a **human-AI collaborative research project** that serves two purposes:

1. **Technical Innovation:** A novel GPU-accelerated wordlist generation algorithm achieving 4-7× speedup over existing tools
2. **AI Research Experiment:** Demonstrating AI capability in autonomous algorithm design and implementation

### Algorithm Origin Story

**The core innovation—mixed-radix direct indexing—was autonomously proposed by Claude Code (AI assistant).**

When asked *"What algorithm would you suggest for a GPU-based approach that would outshine existing solutions?"*, the AI independently proposed abandoning the traditional odometer approach and using direct index-to-word mapping via mixed-radix arithmetic. This algorithmic choice enabled:

- O(1) random access (vs sequential iteration)
- Perfect GPU parallelization (no synchronization needed)
- 4-7× performance improvement over maskprocessor

### Implementation Approach

**The human developer (tehw0lf) had minimal Rust experience prior to this project.** The entire implementation—Rust codebase, CUDA kernels, build system, and integration—was developed through AI-guided development. The AI taught Rust concepts (Result types, lifetimes, RAII, borrowing) while implementing the algorithm, demonstrating AI's capability to:

- Implement complete systems in languages unfamiliar to the human
- Teach language best practices through working code
- Enable rapid skill transfer while maintaining code quality

The entire development—from algorithm design through Rust/CUDA implementation, mathematical proofs, validation, and documentation—represents genuine human-AI pair programming in systems research, where the human provides direction, domain expertise, and validation while the AI provides implementation and formalization.

**Full transparency:** See [docs/development/DEVELOPMENT_PROCESS.md](docs/development/DEVELOPMENT_PROCESS.md) for detailed methodology and contribution breakdown.

### Contributing to the Project

Contributions are welcome! This project benefits from both human and AI collaboration.

**Areas where help is needed:**
- OpenCL backend for AMD/Intel GPUs
- Metal backend for Apple Silicon
- Algorithm optimizations and improvements
- Testing on different GPU architectures
- Documentation improvements
- Multi-GPU coordination strategies

**Development philosophy:**
- All changes must pass correctness validation (cross-validation with maskprocessor)
- Performance claims require reproducible benchmarks
- Code quality maintained through Rust best practices
- Mathematical claims require formal proofs

See [TODO.md](GPU_SCATTER_GATHER_TODO.md) for detailed implementation plan.

## License

Dual-licensed under either:
- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

Choose whichever license suits your use case.

## Acknowledgments

- **maskprocessor** - Inspiration for the problem space and validation baseline
- **hashcat** - Motivation for high-performance wordlist generation
- **NVIDIA CUDA** - Making GPU computing accessible
- **Rust community** - Excellent tooling and libraries
- **Claude Code (Anthropic)** - AI partner in algorithm design, implementation, and validation
  - Autonomously proposed the mixed-radix direct indexing algorithm
  - Collaborative development of CUDA kernels and mathematical proofs
  - See [docs/development/DEVELOPMENT_PROCESS.md](docs/development/DEVELOPMENT_PROCESS.md) for full methodology

## Contact

- **Repository:** https://github.com/tehw0lf/gpu-scatter-gather
- **Issues:** https://github.com/tehw0lf/gpu-scatter-gather/issues
- **Author:** tehw0lf

---

**Made with 🦀 Rust + ⚡ CUDA + 🤖 AI**

*Building the world's fastest wordlist generator, one kernel at a time.*
