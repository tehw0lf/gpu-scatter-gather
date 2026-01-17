# GPU Scatter-Gather Wordlist Generator

**The world's fastest wordlist generator using GPU acceleration**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/gpu-scatter-gather.svg)](https://crates.io/crates/gpu-scatter-gather)
[![Rust](https://img.shields.io/badge/rust-1.82+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Release](https://img.shields.io/badge/release-v1.7.0-brightgreen.svg)](https://github.com/tehw0lf/gpu-scatter-gather/releases/tag/v1.7.0)

> 📄 **[Read the Technical Whitepaper](https://github.com/tehw0lf/gpu-scatter-gather/releases/download/v1.0.0/GPU_Scatter_Gather_Whitepaper_v1.0.0.pdf)** - Comprehensive algorithm design, formal proofs, and performance evaluation
>
> ✅ **Status:** v1.7.0 Released - Published on [crates.io](https://crates.io/crates/gpu-scatter-gather)!
>
> Production-ready library with **4-15× speedup** over CPU tools (maskprocessor, cracken).
> Complete C FFI API with 24 functions (17 single-GPU + 7 multi-GPU), 3 output formats, formal validation, and integration guides.
> See [Development Log](docs/development/DEVELOPMENT_LOG.md) for detailed progress.

## Overview

GPU Scatter-Gather is a GPU-accelerated wordlist generator that achieves **365-771M words/second** (depending on password length) - **4-15× faster than CPU tools** - using a novel scatter-gather algorithm based on mixed-radix arithmetic.

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

| Tool | 8-char Speed | 16-char Speed | Speedup (16-char) |
|------|--------------|---------------|-------------------|
| **GPU Scatter-Gather** | **771 M/s** | **365 M/s** | **15.3×** 🏆 |
| cracken (CPU) | 201 M/s | 43 M/s | 1.0× (baseline) |
| maskprocessor (CPU) | 100-142M/s | ~50-60M/s | ~6-7× |

*Note: Performance advantage increases with password length due to GPU parallelism scaling better than CPU sequential processing. See [Competitive Results](docs/benchmarking/COMPETITIVE_RESULTS.md) for detailed benchmarks.*

## Installation

### From crates.io (Recommended)

```bash
# Add to your Cargo.toml
[dependencies]
gpu-scatter-gather = "1.7"

# Or install as command-line tool
cargo install gpu-scatter-gather
```

### From Source

```bash
# Clone the repository
git clone https://github.com/tehw0lf/gpu-scatter-gather
cd gpu-scatter-gather

# Build the project (compiles CUDA kernels automatically)
cargo build --release

# Or build without CUDA support (CPU-only reference implementation)
cargo build --release --no-default-features
```

**Prerequisites:**
- **Rust 1.82+** - [Install Rust](https://rustup.rs/)
- **CUDA Toolkit 11.8+** (optional, for GPU acceleration) - [Download CUDA](https://developer.nvidia.com/cuda-downloads)
- **NVIDIA GPU** with compute capability 7.5+ (Turing or newer) - optional for GPU features

### Feature Flags

This crate supports the following Cargo features:

- **`cuda`** (enabled by default) - GPU acceleration support with CUDA
  - Enables GPU-accelerated wordlist generation (365-771M words/s)
  - Requires CUDA Toolkit 11.8+ and NVIDIA GPU
  - Includes C FFI API for integration with hashcat/John the Ripper

**Without GPU support:**
```toml
[dependencies]
gpu-scatter-gather = { version = "1.7", default-features = false }
```

This provides CPU-only reference implementation for development/testing without GPU hardware.

## Features

### Current Release: v1.7.0 ✅ (Published on crates.io)

**Core Features (Production Ready)**:
- ✅ **High-performance GPU kernel** - 365-771M words/s (varies by password length)
- ✅ **Complete C FFI API** - 24 functions for single and multi-GPU operation
- ✅ **Multi-GPU support** - Dynamic load balancing for heterogeneous GPU systems
- ✅ **Pinned memory optimization** - Zero-copy API with callback interface
- ✅ **Three output formats** - NEWLINES, PACKED, FIXED_WIDTH
- ✅ **Stdout streaming** - Pipe directly to hashcat/John the Ripper
- ✅ **Formal mathematical validation** - Proven correctness with statistical tests
- ✅ **Published whitepaper** - Academic-quality documentation
- ✅ **Comprehensive examples** - 16+ examples with detailed documentation
- ✅ **Integration guides** - hashcat, John the Ripper, generic C programs
- ✅ **Multi-architecture support** - sm_75-90 (Turing to Hopper)

**Recent Improvements (v1.3.0-1.7.0)**:
- ✅ Persistent worker threads for multi-GPU systems
- ✅ Pinned memory with 65-75% performance improvement
- ✅ Dynamic load balancing for heterogeneous GPUs
- ✅ Zero-copy callback API (`generate_batch_with()`)
- ✅ Clean compilation without warnings
- ✅ Published to crates.io

### Future Enhancements (Community-Driven)

These features await community interest and contributions:
- 🔜 **Python bindings** (PyO3) - For PyPI distribution
- 🔜 **JavaScript bindings** (Neon) - For npm packages
- 🔜 **Memory-mapped file output** - High-throughput disk writes
- 🔜 **OpenCL backend** - AMD/Intel GPU support
- 🔜 **Metal backend** - Apple Silicon support
- 🔜 **Advanced optimizations** - Barrett reduction, power-of-2 fast paths
- 🔜 **Hybrid masks** - Static prefix/suffix with dynamic middle
- 🔜 **Network streaming** - Distributed generation with compression

## Quick Start

> ⚡ **New to the project?** See [QUICKSTART.md](QUICKSTART.md) for a 5-minute setup guide!
>
> ❓ **Have questions?** Check [FAQ.md](FAQ.md) for common questions and troubleshooting.
>
> 📚 **See [EXAMPLES.md](EXAMPLES.md)** for a complete guide to all 16 examples with detailed explanations!

### Quick Start with Rust

```rust
use gpu_scatter_gather::gpu::GpuContext;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    // Create GPU context
    let gpu = GpuContext::new()?;

    // Define character sets
    let mut charsets = HashMap::new();
    charsets.insert(0, b"abc".to_vec());
    charsets.insert(1, b"123".to_vec());

    // Create mask pattern: ?0?1
    let mask = vec![0, 1];

    // Generate 9 words
    let output = gpu.generate_batch(&charsets, &mask, 0, 9, 2)?;

    // Parse results
    let word_length = mask.len();
    for i in 0..(output.len() / word_length) {
        let start = i * word_length;
        let end = start + word_length;
        let word = String::from_utf8_lossy(&output[start..end]);
        println!("{}", word);
    }

    Ok(())
}
```

**Run the beginner example:**
```bash
cargo run --release --example simple_basic
```

**Run the comprehensive API tour:**
```bash
cargo run --release --example simple_rust_api
```

### Multi-GPU C API

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
- ✅ Transparent workload partitioning with dynamic load balancing
- ✅ 90-95% scaling efficiency (minimal overhead)
- ✅ Same API as single-GPU (simplified parallel generation)

See [Multi-GPU Benchmarking Results](docs/benchmarking/MULTI_GPU_RESULTS.md) for detailed performance data.

### Piping to Hashcat

```bash
# Generate wordlist and pipe to hashcat
cargo run --release --example benchmark_stdout | hashcat -m 2500 capture.hccapx
```

See [examples/benchmark_john_pipe.rs](examples/benchmark_john_pipe.rs) for John the Ripper integration.

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

**For detailed mathematical proofs and formal specification**, see [docs/design/FORMAL_SPECIFICATION.md](docs/design/FORMAL_SPECIFICATION.md).

## Benchmarks

### Current Performance (v1.4.0+)

**Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER
- 8,448 CUDA cores, 66 SMs
- Compute capability 8.9
- 16 GB GDDR6X, 672 GB/s bandwidth

**PACKED Format Performance (50M batch):**

| Password Length | Throughput | PCIe Bandwidth | Notes |
|----------------|-----------|----------------|-------|
| 8-char  | 771 M words/s | 6.2 GB/s | Peak performance |
| 10-char | 576 M words/s | 5.8 GB/s | |
| 12-char | 526 M words/s | 6.3 GB/s | |
| 16-char | 365 M words/s | 5.8 GB/s | Competitive baseline |

**Competitive Comparison (16-char passwords):**

| Tool | Speed | Speedup |
|------|-------|---------|
| **GPU Scatter-Gather** | **365 M/s** | **15.3×** 🏆 |
| cracken (CPU, fastest) | 43 M/s | 1.0× |
| maskprocessor (CPU) | ~50-60 M/s | ~6-7× |

**Validation:**
- ✅ 100% output correctness (validated against maskprocessor)
- ✅ Includes full GPU compute + memory I/O + PCIe transfer
- ✅ Formal mathematical correctness proofs
- ✅ Statistical validation (chi-square, autocorrelation, runs tests)

See [docs/benchmarking/](docs/benchmarking/) for detailed results and methodology.

## Project Structure

```
gpu-scatter-gather/
├── src/
│   ├── lib.rs              # Core library and API
│   ├── ffi.rs              # C FFI (24 functions)
│   ├── multigpu.rs         # Multi-GPU coordination
│   ├── gpu/                # GPU module (CUDA integration)
│   ├── charset.rs          # Charset management
│   ├── keyspace.rs         # Keyspace calculation and index-to-word
│   └── mask.rs             # Mask pattern parsing
├── kernels/
│   └── wordlist_poc.cu     # CUDA kernels (3 variants)
├── examples/               # 16+ comprehensive examples
├── tests/                  # Integration tests (55 tests)
├── docs/
│   ├── api/                # C API & FFI documentation
│   ├── design/             # Architecture and formal specification
│   ├── validation/         # Correctness validation
│   ├── benchmarking/       # Performance measurement
│   ├── guides/             # User and integration guides
│   └── development/        # Internal development docs
└── build.rs                # CUDA kernel compilation
```

## Development

### Running Tests

```bash
# Run all tests (55 tests)
cargo test --lib

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_index_to_word_complex_pattern
```

### Running Benchmarks

```bash
# GPU production benchmark (realistic performance)
cargo run --release --example benchmark_production

# Multi-GPU benchmark
cargo run --release --example benchmark_multigpu

# Competitive comparison with cracken
cargo run --release --example benchmark_cracken_comparison
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
| **gpu-scatter-gather** | Rust+CUDA | Mixed-radix direct indexing | 365-771M words/s | **285-3600×** | This project (crates.io) |

**Key insight:** Traditional approaches (Python itertools, Rust odometer) cannot leverage GPU parallelism. The mixed-radix direct indexing algorithm (AI-proposed) enables true GPU acceleration.

### vs cracken (fastest CPU competitor)

**Our Advantages:**
- **3.8-15.3× faster** with GPU acceleration (validated in competitive benchmarks)
- Performance advantage increases with password length (15.3× for 16-char)
- O(1) random access to any keyspace position
- Perfect for distributed workloads (divide keyspace across machines)
- Multi-GPU support with dynamic load balancing

**cracken strengths:**
- No GPU required
- Works on any hardware
- Lower power consumption

### vs maskprocessor

**Our Advantages:**
- **6-8× faster** for similar workloads
- Modern Rust codebase with memory safety
- Programmatic API for library integration
- Multi-GPU scaling

**Maskprocessor strengths:**
- Mature, battle-tested codebase
- Wider CPU compatibility
- Lower resource requirements

### vs Author's Previous Work (wlgen Python)

**Our Advantages:**
- **285-3600× faster** (771M vs 210K-1.6M words/s)
- GPU acceleration (wlgen investigated CUDA but found no benefit in Python)
- Novel algorithm designed for parallelization
- Scales with GPU cores (wlgen is single-threaded CPU-bound)

### vs hashcat built-in

**Our Advantages:**
- Standalone tool (not tied to hashcat)
- Multiple output bindings (stdout, memory, callback)
- Optimized specifically for wordlist generation
- Can feed multiple hashcat instances
- Programmatic API for custom tools

## Roadmap

### Completed Phases ✅

**Phase 1: Foundation** (COMPLETE)
- [x] CPU reference implementation
- [x] CUDA kernel infrastructure
- [x] POC validation
- [x] Comprehensive documentation

**Phase 2: Production Kernel** (COMPLETE)
- [x] Implement production kernel with memory writes
- [x] Validate output correctness vs CPU (100% match)
- [x] Benchmark realistic throughput with I/O
- [x] Clean Rust API with RAII memory management

**Phase 3: Core Features** (COMPLETE)
- [x] C FFI for maximum compatibility (24 functions)
- [x] Stdout streaming binding
- [x] In-memory zero-copy API (callback interface)
- [x] Multi-GPU support with load balancing
- [x] Pinned memory optimization
- [x] Three output formats (NEWLINES, PACKED, FIXED_WIDTH)

**Phase 4: Production Release** (COMPLETE)
- [x] Comprehensive documentation (100+ pages)
- [x] User guide and tutorials (QUICKSTART, EXAMPLES, FAQ)
- [x] Package distribution (crates.io v1.7.0)
- [x] Performance comparison whitepaper (published v1.0.0)
- [x] Formal mathematical validation
- [x] Integration guides (hashcat, John the Ripper)
- [x] Multi-architecture CUDA support (sm_75-90)

### Future Development (Community-Driven)

The project is feature-complete for its core purpose. Future enhancements depend on community interest:

**Language Bindings:**
- [ ] Python bindings (PyO3) for PyPI
- [ ] JavaScript bindings (Neon) for npm
- [ ] Go bindings (cgo)

**Platform Support:**
- [ ] OpenCL backend (AMD/Intel GPUs)
- [ ] Metal backend (Apple Silicon)
- [ ] CPU fallback (SIMD-optimized)

**Advanced Features:**
- [ ] Memory-mapped file output
- [ ] Network streaming with compression
- [ ] Distributed coordinator for clusters
- [ ] Hybrid masks (static + dynamic components)
- [ ] Advanced optimizations (Barrett reduction, power-of-2 fast paths)
- [ ] Pre-built binaries for Linux/Windows

**Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding features.

## Contributing

### About This Project

This is a **human-AI collaborative research project** that serves two purposes:

1. **Technical Innovation:** A novel GPU-accelerated wordlist generation algorithm achieving 4-15× speedup over existing tools
2. **AI Research Experiment:** Demonstrating AI capability in autonomous algorithm design and implementation

### Algorithm Origin Story

**The core innovation—mixed-radix direct indexing—was autonomously proposed by Claude Code (AI assistant).**

When asked *"What algorithm would you suggest for a GPU-based approach that would outshine existing solutions?"*, the AI independently proposed abandoning the traditional odometer approach and using direct index-to-word mapping via mixed-radix arithmetic. This algorithmic choice enabled:

- O(1) random access (vs sequential iteration)
- Perfect GPU parallelization (no synchronization needed)
- 4-15× performance improvement over existing tools

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
- Python/JavaScript bindings for wider language support
- OpenCL backend for AMD/Intel GPUs
- Metal backend for Apple Silicon
- Algorithm optimizations and improvements
- Testing on different GPU architectures
- Documentation improvements
- Pre-built binary distribution

**Development philosophy:**
- All changes must pass correctness validation (cross-validation with maskprocessor)
- Performance claims require reproducible benchmarks
- Code quality maintained through Rust best practices
- Mathematical claims require formal proofs

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

Dual-licensed under either:
- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

Choose whichever license suits your use case.

## Acknowledgments

- **maskprocessor** - Inspiration for the problem space and validation baseline
- **cracken** - Performance baseline for competitive analysis
- **hashcat** - Motivation for high-performance wordlist generation
- **NVIDIA CUDA** - Making GPU computing accessible
- **Rust community** - Excellent tooling and libraries
- **Claude Code (Anthropic)** - AI partner in algorithm design, implementation, and validation
  - Autonomously proposed the mixed-radix direct indexing algorithm
  - Collaborative development of CUDA kernels and mathematical proofs
  - See [docs/development/DEVELOPMENT_PROCESS.md](docs/development/DEVELOPMENT_PROCESS.md) for full methodology

## Contact

- **Repository:** https://github.com/tehw0lf/gpu-scatter-gather
- **Crates.io:** https://crates.io/crates/gpu-scatter-gather
- **Issues:** https://github.com/tehw0lf/gpu-scatter-gather/issues
- **Author:** tehw0lf

---

**Made with 🦀 Rust + ⚡ CUDA + 🤖 AI**

*Building the world's fastest wordlist generator, one kernel at a time.*
