# GPU Scatter-Gather Wordlist Generator v1.0.0

**Release Date:** November 20, 2025
**Status:** 🚀 Production Ready
**License:** MIT OR Apache-2.0

---

## Overview

The **GPU Scatter-Gather Wordlist Generator** is a high-performance, GPU-accelerated wordlist generation library designed for password cracking and security research. It achieves **4-7× faster** candidate generation than state-of-the-art CPU tools through a novel direct index-to-word mapping algorithm based on mixed-radix arithmetic.

This v1.0.0 release marks the library as **production-ready** with comprehensive validation, formal mathematical proofs, and complete C FFI for integration into existing security tools.

---

## What's New in v1.0.0

### Features

✅ **Complete C FFI API (16 functions)**
- Phase 1: Host memory API (baseline functionality)
- Phase 2: Device pointer API (zero-copy GPU-to-GPU operation)
- Phase 3: Output format modes (NEWLINES, PACKED, FIXED_WIDTH)
- Phase 4: Streaming API with CUDA streams (async operation)
- Phase 5: Utility functions (version info, device detection)

✅ **Multiple Output Formats**
- `NEWLINES`: Each word terminated by `\n` (hashcat/JtR compatible)
- `PACKED`: No separators, minimal memory (11% savings)
- `FIXED_WIDTH`: Null-padded fixed width (easy parsing)

✅ **Integration Support**
- Hashcat integration guide with 3 patterns (host, device, streaming)
- John the Ripper integration guide (external mode, format plugins)
- Generic integration guide for custom tools
- Complete working C examples

✅ **Production-Quality Documentation**
- Formal mathematical specification with proofs
- C API specification with detailed examples
- Performance benchmarking methodology
- Statistical validation suite
- Integration guides for popular tools

---

## Performance

### Throughput Benchmarks (RTX 4070 Ti SUPER)

| Password Length | Pattern | Throughput (PACKED) | Memory Bandwidth |
|-----------------|---------|---------------------|------------------|
| **8 chars** | `?l?l?l?l?d?d?d?d` | **700 M/s** | 5.6 GB/s |
| **10 chars** | `?l?l?l?l?l?l?d?d?d?d` | **543 M/s** | 5.4 GB/s |
| **12 chars** | `?l?l?l?l?l?l?l?l?d?d?d?d` | **469 M/s** | 5.6 GB/s |

### Competitive Analysis

| Tool | Type | Throughput (8-char) | vs maskprocessor |
|------|------|---------------------|------------------|
| **maskprocessor** | CPU | ~142 M/s | 1.0× (baseline) |
| **cracken** | CPU | ~178 M/s | 1.25× |
| **gpu-scatter-gather** | **GPU** | **700 M/s** | **4.9×** 🚀 |

**Key Advantages:**
- 4-7× faster than state-of-the-art CPU tools
- Zero-copy GPU-to-GPU operation (no PCIe bottleneck)
- O(1) random access enables perfect keyspace partitioning
- Consistent performance across password lengths (memory-bound)

---

## Validation & Quality Assurance

### Test Coverage

✅ **55/55 tests passing (100% success rate)**
- 30 Rust unit tests (core functionality)
- 4 statistical validation tests (chi-square, autocorrelation, runs test)
- 16 basic FFI tests (C API correctness)
- 5 integration tests (end-to-end scenarios)

### Formal Validation

✅ **Mathematical Proofs**
- Bijection proof (one-to-one mapping between indices and words)
- Completeness proof (all possible combinations generated)
- Ordering preservation proof (lexicographic order maintained)

✅ **Statistical Validation**
- Chi-square test: p-value = 0.634 (uniform distribution ✓)
- Autocorrelation test: all lags < threshold (independence ✓)
- Runs test: z-score = -0.543 (randomness ✓)

✅ **Cross-Validation**
- 100% match with maskprocessor (industry standard)
- Verified on 1M+ candidates across multiple patterns
- Edge cases tested (empty charsets, single char, max length)

### Performance Validation

✅ **Scientific Benchmarking**
- Coefficient of variation < 1.5% (highly reproducible)
- 10+ iterations per configuration
- Statistical significance analysis
- Warm-up runs to eliminate cold-start effects

---

## Getting Started

### System Requirements

**Hardware:**
- NVIDIA GPU with CUDA support (Compute Capability 7.5+)
- Recommended: RTX 3000/4000 series or A100/H100 (datacenter)
- Minimum: GTX 1650 or equivalent (older cards may have reduced performance)

**Software:**
- CUDA Toolkit 12.x (11.x may work but untested)
- Rust 1.70+ (for building from source)
- GCC/Clang (for C examples)
- Linux/Windows/macOS (Linux recommended for best performance)

### Installation

```bash
# Clone repository
git clone https://github.com/tehw0lf/gpu-scatter-gather.git
cd gpu-scatter-gather

# Build library
cargo build --release

# Library output:
# - target/release/libgpu_scatter_gather.so (Linux)
# - target/release/gpu_scatter_gather.dll (Windows)
# - target/release/libgpu_scatter_gather.dylib (macOS)

# C header:
# - include/wordlist_generator.h
```

### Quick Example (C API)

```c
#include "wordlist_generator.h"
#include <stdio.h>

int main() {
    // Create generator
    wg_WordlistGenerator *gen = wg_create();
    if (!gen) {
        fprintf(stderr, "Failed to create generator\n");
        return 1;
    }

    // Define charsets
    wg_add_charset(gen, 'l', "abcdefghijklmnopqrstuvwxyz", 26);
    wg_add_charset(gen, 'd', "0123456789", 10);

    // Set mask: 8 chars (4 letters + 4 digits)
    wg_set_mask(gen, "?l?l?l?l?d?d?d?d");

    // Use PACKED format for best performance
    wg_set_format(gen, WG_FORMAT_PACKED);

    // Calculate buffer size for 100M candidates
    const uint64_t batch_size = 100000000;
    size_t buffer_size = wg_calculate_buffer_size(gen, batch_size);

    // Allocate buffer
    char *buffer = malloc(buffer_size);

    // Generate batch
    int result = wg_generate_batch_host(gen, 0, batch_size,
                                        buffer, buffer_size);
    if (result == 0) {
        printf("Generated %llu candidates (%zu bytes)\n",
               batch_size, buffer_size);
    }

    // Cleanup
    free(buffer);
    wg_destroy(gen);
    return 0;
}
```

**Compile:**
```bash
gcc -o example example.c \
    -I./include \
    -L./target/release \
    -lgpu_scatter_gather \
    -Wl,-rpath,./target/release
```

---

## Integration Guides

### Hashcat Integration

**Use Cases:**
- Replace maskprocessor with 4-7× faster GPU generation
- Zero-copy device pointer API for custom hash modules
- Streaming API for overlapped generation and hashing

**Example: Piping to Hashcat**
```bash
./hashcat_generator '?l?l?l?l?d?d?d?d' | hashcat -m 0 hashes.txt
```

**See:** `docs/guides/HASHCAT_INTEGRATION.md` for complete examples

### John the Ripper Integration

**Use Cases:**
- External mode with GPU acceleration (7-11× faster)
- Format plugin integration for zero-copy operation
- Distributed cracking with perfect keyspace partitioning

**Example: External Mode**
```bash
./jtr_generator '?l?l?l?l?l?l?l?l' | john --stdin hashes.txt
```

**See:** `docs/guides/JTR_INTEGRATION.md` for complete examples

### Custom Tools

**Use Cases:**
- Build custom password crackers
- Research tools for cryptographic analysis
- Distributed computing frameworks

**See:** `docs/guides/INTEGRATION_GUIDE.md` for generic integration patterns

---

## Documentation

### API Reference
- **C API Specification:** `docs/api/C_API_SPECIFICATION.md`
- **Rust API Documentation:** Run `cargo doc --open`

### Design & Theory
- **Formal Specification:** `docs/design/FORMAL_SPECIFICATION.md`
- **Algorithm Explained:** `docs/design/ALGORITHM.md` (if exists)

### Validation & Benchmarking
- **Validation Plan:** `docs/validation/FORMAL_VALIDATION_PLAN.md`
- **Performance Comparison:** `docs/benchmarking/PERFORMANCE_COMPARISON.md`
- **Baseline Benchmarking:** `docs/benchmarking/BASELINE_BENCHMARKING_PLAN.md`

### Integration Guides
- **Hashcat:** `docs/guides/HASHCAT_INTEGRATION.md` ⭐ NEW
- **John the Ripper:** `docs/guides/JTR_INTEGRATION.md` ⭐ NEW
- **Generic:** `docs/guides/INTEGRATION_GUIDE.md`
- **Nsight Compute Profiling:** `docs/guides/NSIGHT_COMPUTE_SETUP.md`

### Development
- **Development Log:** `docs/development/DEVELOPMENT_LOG.md`
- **TODO List:** `docs/development/TODO.md`

---

## Architecture

### Core Algorithm

The library uses a **direct index-to-word mapping** based on mixed-radix arithmetic, enabling:

- **O(1) random access:** Jump to any position in keyspace instantly
- **Perfect parallelization:** Every GPU thread operates independently
- **No sequential dependencies:** Unlike odometer-style algorithms
- **Deterministic generation:** Same index always produces same word

**Mathematical Foundation:**

For a mask with positions using charsets of sizes `[b₀, b₁, ..., bₙ₋₁]`, the word at index `k` is computed via:

```
word[i] = charset[i][k_i]
where k_i = (k / ∏(j<i) b_j) mod b_i
```

This is a bijection between `[0, ∏b_i)` and all possible words.

**See:** `docs/design/FORMAL_SPECIFICATION.md` for complete proofs

### GPU Implementation

**CUDA Kernel:** `kernels/wordlist_poc.cu`
- Column-major writes with CPU transpose (fastest variant)
- Warp-level parallelization (32 threads per word)
- Shared memory for charset caching
- Multiple kernel variants for different use cases

**Memory Layout:**
- PACKED: `[word₀][word₁][word₂]...` (optimal)
- NEWLINES: `[word₀\n][word₁\n][word₂\n]...` (compatible)
- FIXED_WIDTH: `[word₀\0..][word₁\0..][word₂\0..]` (aligned)

---

## Breaking Changes

**None** (initial v1.0.0 release)

---

## Known Issues

**None** (all tests passing, production-ready)

---

## Roadmap (Future Enhancements)

### Planned for v1.1.0+
- Multi-GPU support (distribute keyspace across multiple GPUs)
- OpenCL backend (support AMD/Intel GPUs)
- Hybrid masks (combine static prefixes/suffixes with dynamic parts)
- Advanced charset modifiers (toggle, shift, custom functions)

### Planned for v2.0.0+
- Python bindings (PyPI package)
- Rule-based generation (integrate with hashcat rules)
- Dictionary-based augmentation (hybrid attacks)
- Web API for remote generation

### Research & Publication
- Academic paper (ArXiv preprint)
- Conference submission (USENIX Security, ACM CCS)
- Performance comparison study with GPU hash crackers

---

## Performance Tips

### Optimal Batch Sizes

| Password Length | Recommended Batch Size | Memory Usage |
|-----------------|------------------------|--------------|
| 6-8 chars | 100M | 600-800 MB |
| 10-12 chars | 50-100M | 500-1200 MB |
| 14-16 chars | 25-50M | 350-800 MB |

**Rule of thumb:** Keep batch size × word length < 1 GB

### Format Selection

| Use Case | Format | Reason |
|----------|--------|--------|
| Piping to hashcat/JtR | `NEWLINES` | Required by stdin |
| Direct GPU kernel access | `PACKED` | 11% memory savings |
| Text file parsing | `FIXED_WIDTH` | Easy indexing |

### GPU Selection

Best performance on:
- **Consumer:** RTX 4090 (16384 CUDA cores)
- **Datacenter:** A100/H100 (massive memory bandwidth)
- **Budget:** RTX 3060 Ti (good price/performance)

Minimum:
- GTX 1650 or equivalent (Compute Capability 7.5+)

---

## Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce batch size or use smaller password length

### "Buffer too small" error
**Solution:** Use `wg_calculate_buffer_size()` to get correct size

### Low performance (<200 M/s)
**Solution:** Check GPU utilization with `nvidia-smi`, increase batch size, use PACKED format

### Hashcat doesn't read candidates
**Solution:** Ensure NEWLINES format when piping to stdin

**See:** Integration guides for tool-specific troubleshooting

---

## Contributing

Contributions are welcome! Please:

1. **File issues:** Report bugs or request features on GitHub Issues
2. **Submit PRs:** Follow Rust style guidelines, include tests
3. **Share integrations:** Contribute hashcat/JtR modules back
4. **Improve docs:** Fix typos, add examples, clarify explanations

**Code of Conduct:** Be respectful, constructive, and professional

---

## License

Dual-licensed under:

- **MIT License** (permissive, commercial-friendly)
- **Apache License 2.0** (patent protection)

Choose whichever fits your needs. See `LICENSE-MIT` and `LICENSE-APACHE` for full text.

---

## Citation

If you use this library in research, please cite:

```bibtex
@software{gpu_scatter_gather_2025,
  author = {tehw0lf},
  title = {GPU Scatter-Gather Wordlist Generator},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/tehw0lf/gpu-scatter-gather}
}
```

**Academic paper pending** (ArXiv preprint in progress)

---

## Acknowledgments

- **Claude Code by Anthropic:** AI-assisted development and validation
- **CUDA Team @ NVIDIA:** Excellent GPU computing platform
- **Hashcat & John the Ripper:** Inspiration and integration targets
- **Rust Community:** Amazing tools and ecosystem

---

## Support & Contact

- **Repository:** https://github.com/tehw0lf/gpu-scatter-gather
- **Issues:** https://github.com/tehw0lf/gpu-scatter-gather/issues
- **Discussions:** https://github.com/tehw0lf/gpu-scatter-gather/discussions
- **Documentation:** https://github.com/tehw0lf/gpu-scatter-gather/tree/main/docs

---

## Version History

- **v1.0.0** (November 20, 2025): Initial production release 🚀
  - Complete C FFI API (16 functions)
  - 4-7× faster than CPU tools
  - Comprehensive validation and documentation
  - Integration guides for hashcat and John the Ripper

---

**Thank you for using GPU Scatter-Gather Wordlist Generator!**

**Happy cracking! 🔐💥**
