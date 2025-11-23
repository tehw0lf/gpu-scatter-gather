# Frequently Asked Questions (FAQ)

Common questions and answers about the GPU Scatter-Gather Wordlist Generator.

## Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Performance](#performance)
- [Usage](#usage)
- [Integration](#integration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## General Questions

### What is GPU Scatter-Gather?

GPU Scatter-Gather is a **GPU-accelerated wordlist generator** that uses a novel scatter-gather algorithm to generate password candidates **4-7× faster than CPU tools** like maskprocessor. It achieves 440-700M words/second on modern GPUs.

### How is this different from maskprocessor or cracken?

**Key differences:**

| Feature | GPU Scatter-Gather | maskprocessor | cracken |
|---------|-------------------|---------------|---------|
| **Speed** | 440-700M words/s | 100-142M words/s | ~100M words/s |
| **Platform** | GPU (CUDA) | CPU | CPU |
| **Algorithm** | Direct index mapping (O(1) random access) | Sequential odometer | Sequential |
| **Multi-GPU** | Native support | N/A | N/A |
| **C API** | Yes (24 functions) | CLI only | CLI only |

**Bottom line**: If you have an NVIDIA GPU, this is significantly faster.

### Do I need a high-end GPU?

**Minimum**: NVIDIA GPU with compute capability 7.5+ (Turing or newer)
- GTX 1650 and newer
- RTX 2000 series and newer
- Data center: T4, A10, A100, H100

**Recommended**: RTX 3060 Ti or better for optimal performance.

**Not supported**: AMD GPUs, Intel GPUs, older NVIDIA GPUs (pre-Turing)

### Is this only for password cracking?

While designed for security research and password auditing, it's a general-purpose wordlist generator. Use cases include:
- ✅ Authorized penetration testing
- ✅ Password policy testing
- ✅ Security research
- ✅ Academic research
- ✅ CTF competitions
- ❌ Unauthorized access attempts (illegal!)

**Always ensure you have authorization** before using this tool.

---

## Installation & Setup

### How do I install CUDA?

1. Check if you already have CUDA:
   ```bash
   nvcc --version
   ```

2. If not installed, download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

3. Install CUDA Toolkit 11.8 or newer

4. Verify installation:
   ```bash
   nvidia-smi  # Check GPU is detected
   nvcc --version  # Check CUDA compiler
   ```

### Do I need to install Rust?

Yes! Install Rust 1.82+ from [rustup.rs](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### The build fails with "CUDA not found"

**Solution**: Set the `CUDA_PATH` environment variable:

```bash
export CUDA_PATH=/usr/local/cuda  # Linux
# or
export CUDA_PATH=/opt/cuda  # Alternative path

# Add to ~/.bashrc or ~/.zshrc to make permanent
echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
```

### The build fails with "sm_70: No such file"

This is normal! The library tries to compile for sm_70 (Volta) but your system may not support it. This warning can be safely ignored as long as other architectures (sm_75, sm_80, sm_86, sm_89, sm_90) compile successfully.

The library will use the best available compiled kernel for your GPU.

---

## Performance

### Why is my performance lower than advertised?

**Common causes:**

1. **PCIe bottleneck** - Are you copying results to host? Use device pointer API for zero-copy.

2. **Small batch size** - Increase batch size to amortize overhead:
   ```rust
   // Bad: Too small
   gpu.generate_batch(&charsets, &mask, 0, 1_000, 2)?;

   // Good: Larger batches
   gpu.generate_batch(&charsets, &mask, 0, 100_000_000, 2)?;
   ```

3. **Wrong GPU** - Check you're using the discrete GPU, not integrated graphics:
   ```bash
   nvidia-smi  # Should show your GPU
   ```

4. **Thermal throttling** - Monitor GPU temperature:
   ```bash
   watch -n 1 nvidia-smi
   ```

5. **Word length** - Longer words = slower (more memory bandwidth):
   - 8-char: ~700 M words/s
   - 10-char: ~550 M words/s
   - 12-char: ~440 M words/s

### What batch size should I use?

**Rule of thumb**:
- **Small patterns** (< 1B keyspace): Use half the keyspace
- **Large patterns** (> 1B keyspace): Use 100M-1B per batch

**Example**:
```rust
let keyspace = calculate_keyspace(&charsets, &mask);

let batch_size = if keyspace < 1_000_000_000 {
    keyspace / 2  // Half for small keyspaces
} else {
    100_000_000  // 100M for large keyspaces
};
```

### Should I use PACKED or NEWLINES format?

**PACKED (format 2)** - Fastest, most bandwidth-efficient:
```rust
let output = gpu.generate_batch(&charsets, &mask, 0, count, 2)?;
```
- No separators between words
- Minimal memory usage
- Parse by word length

**NEWLINES (format 0)** - Easiest for piping:
```rust
let output = gpu.generate_batch(&charsets, &mask, 0, count, 0)?;
```
- One word per line
- Easy to pipe to other tools
- ~10-15% slower due to larger output

**Recommendation**: Use PACKED unless you need line-separated output.

### Does multi-GPU actually help?

**Yes, if you have 2+ GPUs!**

Expected scaling:
- 2 GPUs: 1.8× speedup (90% efficiency)
- 4 GPUs: 3.6× speedup (90% efficiency)

**Single GPU**: Multi-GPU API uses fast path (zero overhead).

**v1.3.0-dev improvement**: Persistent worker threads eliminate context recreation overhead (5-10% faster for multi-GPU systems).

---

## Usage

### How do I generate a simple wordlist?

**Rust (simplest)**:
```bash
cargo run --release --example simple_basic
```

**Rust (custom code)**:
```rust
use gpu_scatter_gather::gpu::GpuContext;
use std::collections::HashMap;

let gpu = GpuContext::new()?;

let mut charsets = HashMap::new();
charsets.insert(0, b"abc".to_vec());
charsets.insert(1, b"123".to_vec());

let mask = vec![0, 1];  // Pattern: ?0?1
let output = gpu.generate_batch(&charsets, &mask, 0, 9, 2)?;

// Parse and print
let word_length = mask.len();
for i in 0..(output.len() / word_length) {
    let start = i * word_length;
    let word = String::from_utf8_lossy(&output[start..start + word_length]);
    println!("{}", word);
}
```

### How do I use it with hashcat?

See the complete [Hashcat Integration Guide](docs/guides/HASHCAT_INTEGRATION.md) for 3 integration patterns.

**Quick example** (pipe pattern):
```bash
# Build a hashcat-compatible tool (future)
./gpu-scatter-gather -1 ?l -2 ?d ?1?1?1?1?2?2?2?2 | hashcat -m 0 hashes.txt
```

### How do I generate only part of the keyspace?

Use `start_idx` and `count` parameters:

```rust
// Generate words 1000-1999 (1000 words starting at index 1000)
let output = gpu.generate_batch(&charsets, &mask, 1000, 1000, 2)?;
```

**Use case**: Distribute work across multiple machines:
- Machine 1: indices 0-999
- Machine 2: indices 1000-1999
- Machine 3: indices 2000-2999

### Can I use custom character sets?

**Yes!** Any byte sequences work:

```rust
let mut charsets = HashMap::new();
charsets.insert(0, b"@#$%^&*".to_vec());  // Special chars
charsets.insert(1, vec![0xC0, 0xC1, 0xC2]);  // Raw bytes
charsets.insert(2, "αβγδ".as_bytes().to_vec());  // Unicode (as UTF-8 bytes)

let mask = vec![0, 1, 2];
```

**Note**: The library treats charsets as byte sequences, not characters. Multi-byte characters (UTF-8) work but count as multiple bytes.

---

## Integration

### Can I call this from Python/JavaScript/other languages?

**Current**: C FFI only (works from any language with C FFI support)

**Planned** (v1.3.0+):
- Python bindings (PyPI package)
- JavaScript bindings (npm package)

**Workaround**: Use C FFI directly:
```python
# Python example using ctypes
import ctypes

lib = ctypes.CDLL('./target/release/libgpu_scatter_gather.so')
# ... call C functions
```

See [docs/api/C_API_SPECIFICATION.md](docs/api/C_API_SPECIFICATION.md) for the complete C API.

### How do I integrate with my existing tool?

See the [Integration Guide](docs/guides/INTEGRATION_GUIDE.md) for generic integration patterns.

**Three patterns**:
1. **Library Mode** - Link as a shared library
2. **Pipe Mode** - Stream wordlists via stdout
3. **Hybrid Mode** - Pre-generate candidate files

Choose based on your tool's architecture.

### Does this work on Windows?

**Currently**: Linux only (tested on Arch, Ubuntu)

**Why**: CUDA toolkit paths and build scripts are Linux-focused

**Planned**: Windows support in future release

**Workaround**: Use WSL2 (Windows Subsystem for Linux) with CUDA support

---

## Troubleshooting

### "error: linker `cc` not found"

**Solution**: Install build tools:

```bash
# Debian/Ubuntu
sudo apt install build-essential

# Arch Linux
sudo pacman -S base-devel

# Fedora
sudo dnf groupinstall "Development Tools"
```

### "CUDA_ERROR_OUT_OF_MEMORY"

**Causes**:
1. Batch size too large for GPU memory
2. Other applications using GPU memory

**Solutions**:
1. Reduce batch size:
   ```rust
   // Try 50M instead of 100M
   let batch_size = 50_000_000;
   ```

2. Free GPU memory:
   ```bash
   nvidia-smi  # Check what's using GPU
   kill <PID>  # Kill memory-hungry processes
   ```

3. Use multiple smaller batches:
   ```rust
   for i in (0..total_keyspace).step_by(50_000_000) {
       let count = std::cmp::min(50_000_000, total_keyspace - i);
       let batch = gpu.generate_batch(&charsets, &mask, i, count, 2)?;
       // Process batch
   }
   ```

### Tests fail with "GPU thread panicked"

**Cause**: GPU context creation failed

**Solutions**:
1. Check GPU is available:
   ```bash
   nvidia-smi
   ```

2. Ensure no other process is using the GPU exclusively

3. Check CUDA version compatibility:
   ```bash
   nvcc --version  # Should be 11.8+
   ```

### "warning: unused variable" during build

**This is normal!** Rust warnings don't affect functionality. To suppress:

```bash
cargo build --release 2>&1 | grep -v warning
```

Or fix them:
```bash
cargo clippy --fix
```

---

## Development

### How do I run benchmarks?

**Recommended benchmark**:
```bash
cargo run --release --example benchmark_realistic
```

**Quick performance test**:
```bash
cargo run --release --example simple_rust_api
```

**Multi-GPU benchmark**:
```bash
cargo run --release --example benchmark_multigpu
```

### How do I profile the kernel?

See the complete [Nsight Compute Setup Guide](docs/guides/NSIGHT_COMPUTE_SETUP.md).

**Quick start**:
```bash
# Build profiling example
cargo build --release --example profile_12char

# Run with Nsight Compute
ncu --set full ./target/release/examples/profile_12char
```

### How do I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Quick steps**:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

**All contributions welcome**:
- Bug fixes
- Performance improvements
- Documentation improvements
- New examples
- Language bindings

### Where do I report bugs?

**GitHub Issues**: https://github.com/tehw0lf/gpu-scatter-gather/issues

**Include**:
- GPU model and driver version (`nvidia-smi`)
- CUDA version (`nvcc --version`)
- Rust version (`rustc --version`)
- Error messages
- Minimal reproduction code

---

## Still Have Questions?

- 📖 **Documentation**: See [docs/README.md](docs/README.md)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/tehw0lf/gpu-scatter-gather/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/tehw0lf/gpu-scatter-gather/issues)
- 📧 **Email**: Open an issue instead (better for community)

---

*Last Updated: November 23, 2025*
*Version: 1.3.0-dev*
