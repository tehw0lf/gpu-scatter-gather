# Quick Start Guide

Get GPU Scatter-Gather running in **5 minutes or less**.

## Prerequisites Checklist

Before starting, ensure you have:

- ✅ **NVIDIA GPU** (GTX 1650 or newer, RTX 2000+ recommended)
- ✅ **CUDA Toolkit 11.8+** installed
- ✅ **Rust 1.82+** installed

### Check Prerequisites

```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check Rust
rustc --version
```

**Don't have these?** See [Installation](#installation) below.

---

## 5-Minute Setup

### Step 1: Clone the Repository (30 seconds)

```bash
git clone https://github.com/tehw0lf/gpu-scatter-gather
cd gpu-scatter-gather
```

### Step 2: Build the Project (2-3 minutes)

```bash
cargo build --release
```

**Expected**: CUDA kernels compile for multiple architectures (sm_75, sm_80, sm_86, sm_89, sm_90).

**Note**: Warning about `sm_70` failing is normal and can be ignored.

### Step 3: Run Your First Example (5 seconds)

```bash
cargo run --release --example simple_basic
```

**Expected output**:
```
🚀 GPU Scatter-Gather - Simple Basic Example

📡 Initializing GPU context...
   ✅ GPU initialized successfully

...

📤 Generated 9 words:
─────────────────────
a1
a2
a3
b1
b2
b3
c1
c2
c3

✨ Done!
```

### Step 4: Verify Performance (10 seconds)

```bash
cargo run --release --example simple_rust_api
```

**Expected**: Should show **500-700 M words/s** throughput depending on your GPU.

---

## ✅ Success!

You now have a working GPU-accelerated wordlist generator!

**Next steps**:
1. 📖 Read [EXAMPLES.md](EXAMPLES.md) to see all 16 examples
2. 🔨 Read [docs/guides/INTEGRATION_GUIDE.md](docs/guides/INTEGRATION_GUIDE.md) to integrate with your tools
3. 📊 Run `benchmark_realistic` for full performance testing

---

## Installation

### Install CUDA Toolkit

**Linux** (Debian/Ubuntu):
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install CUDA
sudo apt install cuda-toolkit-11-8
```

**Linux** (Arch):
```bash
sudo pacman -S cuda
```

**Other Linux**: Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

**Verify**:
```bash
nvidia-smi
nvcc --version
```

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Verify**:
```bash
rustc --version
```

---

## Common Issues

### "CUDA not found" during build

**Solution**: Set CUDA path:
```bash
export CUDA_PATH=/usr/local/cuda
cargo build --release
```

### "linker `cc` not found"

**Solution**: Install build tools:
```bash
# Debian/Ubuntu
sudo apt install build-essential

# Arch
sudo pacman -S base-devel
```

### Low performance (<100 M words/s)

**Check**:
1. Using discrete GPU? (run `nvidia-smi`)
2. Large enough batch size? (use 100M+ words)
3. PACKED format? (format 2)

See [FAQ.md](FAQ.md#why-is-my-performance-lower-than-advertised) for more troubleshooting.

---

## Quick Usage Examples

### Generate Simple Wordlist

```rust
use gpu_scatter_gather::gpu::GpuContext;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::new()?;

    // Define charsets
    let mut charsets = HashMap::new();
    charsets.insert(0, b"abc".to_vec());
    charsets.insert(1, b"123".to_vec());

    // Pattern: ?0?1 (one from charset 0, one from charset 1)
    let mask = vec![0, 1];

    // Generate all 9 combinations
    let output = gpu.generate_batch(&charsets, &mask, 0, 9, 2)?;

    // Print words
    let word_length = mask.len();
    for i in 0..(output.len() / word_length) {
        let start = i * word_length;
        let word = String::from_utf8_lossy(&output[start..start + word_length]);
        println!("{}", word);
    }

    Ok(())
}
```

### Multi-GPU Generation

```rust
use gpu_scatter_gather::multigpu::MultiGpuContext;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    // Automatically detects and uses all GPUs
    let multi_gpu = MultiGpuContext::new()?;

    println!("Using {} GPU(s)", multi_gpu.num_devices());

    let mut charsets = HashMap::new();
    charsets.insert(0, b"abcdefghijklmnopqrstuvwxyz".to_vec());
    charsets.insert(1, b"0123456789".to_vec());

    let mask = vec![0, 0, 0, 0, 1, 1, 1, 1];  // 4 letters + 4 digits

    // Generate 10M words distributed across all GPUs
    let output = multi_gpu.generate_batch(&charsets, &mask, 0, 10_000_000, 2)?;

    println!("Generated {} bytes", output.len());

    Ok(())
}
```

---

## Performance Expectations

| GPU Model | Approximate Speed |
|-----------|------------------|
| RTX 4090 | ~1000 M words/s |
| RTX 4070 Ti SUPER | 550-700 M words/s |
| RTX 3080 | 400-500 M words/s |
| RTX 3060 Ti | 300-400 M words/s |
| GTX 1660 Ti | 200-300 M words/s |

**Factors affecting speed**:
- Word length (longer = slower)
- Output format (PACKED fastest)
- Batch size (larger = better GPU utilization)
- PCIe bandwidth (Gen 4 faster than Gen 3)

---

## Next Steps

### Learning Path

1. **Beginner**: Run `simple_basic.rs` and `simple_rust_api.rs`
2. **Validation**: Run `cross_validate.rs` to verify correctness
3. **Performance**: Run `benchmark_realistic.rs` to measure your GPU
4. **Integration**: Read hashcat or JTR integration guides

### Documentation

- 📖 **Complete examples guide**: [EXAMPLES.md](EXAMPLES.md)
- ❓ **Common questions**: [FAQ.md](FAQ.md)
- 🔧 **Integration patterns**: [docs/guides/INTEGRATION_GUIDE.md](docs/guides/INTEGRATION_GUIDE.md)
- 📊 **API reference**: [docs/api/C_API_SPECIFICATION.md](docs/api/C_API_SPECIFICATION.md)

### Get Help

- 🐛 **Bug reports**: [GitHub Issues](https://github.com/tehw0lf/gpu-scatter-gather/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/tehw0lf/gpu-scatter-gather/discussions)
- 📧 **Questions**: Open an issue (better for community)

---

**🎉 Happy wordlist generating!**

---

*Last Updated: November 23, 2025*
*Version: 1.3.0-dev*
