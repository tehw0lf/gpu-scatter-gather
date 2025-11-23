# Examples Guide

This guide explains all available examples in the `gpu-scatter-gather` project and when to use each one.

## Quick Start Examples

### 🟢 `simple_basic.rs` - Start Here!

**Purpose**: The absolute simplest introduction to the library.

**What it demonstrates**:
- Creating a GPU context
- Defining character sets
- Creating a mask pattern
- Generating a small wordlist (9 words)
- Parsing and displaying results

**When to use**: You're new to the library and want to understand the basics.

**Run it**:
```bash
cargo run --release --example simple_basic
```

**Expected output**: 9 words (a1, a2, a3, b1, b2, b3, c1, c2, c3)

---

### 🟢 `simple_rust_api.rs` - Complete Rust API Tour

**Purpose**: Comprehensive demonstration of the Rust API features.

**What it demonstrates**:
- Different output formats (NEWLINES, PACKED, FIXED_WIDTH)
- Single GPU vs Multi-GPU contexts
- Partial keyspace generation
- Performance measurement
- Real-world use cases

**When to use**: You understand the basics and want to see all available features.

**Run it**:
```bash
cargo run --release --example simple_rust_api
```

**Expected output**: Three examples showing different API features with ~650 M words/s throughput.

---

## Validation & Testing Examples

### 🔵 `cross_validate.rs` - Correctness Verification

**Purpose**: Cross-validate GPU output against CPU reference implementation.

**What it demonstrates**:
- Statistical validation
- Comparing GPU vs CPU implementations
- Ensuring correctness

**When to use**: You want to verify the library produces correct results.

**Run it**:
```bash
cargo run --release --example cross_validate
```

---

### 🔵 `validate_gpu.rs` - GPU-Only Validation

**Purpose**: Validate GPU implementation without CPU comparison.

**What it demonstrates**:
- Self-contained GPU validation
- Quick correctness check

**When to use**: You want to quickly verify GPU functionality.

**Run it**:
```bash
cargo run --release --example validate_gpu
```

---

### 🔵 `test_perf_comparison.rs` - API Performance Comparison

**Purpose**: Compare performance of direct GPU API vs Multi-GPU API on single GPU.

**What it demonstrates**:
- Fast path optimization verification
- Overhead measurement
- Performance regression testing

**When to use**: You want to verify the v1.2.1 fast path optimization is working.

**Run it**:
```bash
cargo run --release --example test_perf_comparison
```

**Expected output**: Should show <5% overhead between direct GPU and Multi-GPU API.

---

## Benchmarking Examples

### 🟡 `benchmark_realistic.rs` - Recommended Benchmark

**Purpose**: Benchmark with realistic password patterns (8, 10, 12 characters).

**What it demonstrates**:
- Real-world performance
- Different word lengths
- Bandwidth utilization

**When to use**: You want to measure performance with typical password patterns.

**Run it**:
```bash
cargo run --release --example benchmark_realistic
```

**Expected output**:
- 8-char: ~700 M words/s
- 10-char: ~550 M words/s
- 12-char: ~440 M words/s

---

### 🟡 `benchmark_production.rs` - Production Performance

**Purpose**: Benchmark the production kernel with various patterns.

**What it demonstrates**:
- Maximum sustained throughput
- Different batch sizes
- Production kernel performance

**When to use**: You want to measure peak performance.

**Run it**:
```bash
cargo run --release --example benchmark_production
```

---

### 🟡 `benchmark_multigpu.rs` - Multi-GPU Scaling

**Purpose**: Benchmark multi-GPU scaling efficiency.

**What it demonstrates**:
- Multi-GPU workload distribution
- Scaling efficiency
- Overhead analysis

**When to use**: You have multiple GPUs and want to measure scaling.

**Run it**:
```bash
cargo run --release --example benchmark_multigpu
```

**Expected output** (single GPU): Fast path should show minimal overhead.

---

### 🟡 `benchmark_multigpu_async.rs` - Async Multi-GPU

**Purpose**: Benchmark async multi-GPU mode with CUDA streams.

**What it demonstrates**:
- CUDA stream usage
- Async kernel launches
- Overlapped execution

**When to use**: You want to measure async mode performance.

**Run it**:
```bash
cargo run --release --example benchmark_multigpu_async
```

---

### 🟡 `benchmark_hybrid.rs` - Hybrid Benchmark

**Purpose**: Compare different kernel variants and modes.

**What it demonstrates**:
- Multiple kernel strategies
- Performance comparison
- Optimization trade-offs

**When to use**: You're exploring kernel optimizations.

**Run it**:
```bash
cargo run --release --example benchmark_hybrid
```

---

## Profiling Examples

### 🔴 `profile_12char.rs` - Profile 12-char Pattern

**Purpose**: Profile 12-character password generation for Nsight Compute analysis.

**What it demonstrates**:
- Long-running workload for profiling
- Realistic 12-char pattern
- Memory bandwidth characteristics

**When to use**: You want to profile the kernel with Nsight Compute.

**Run it with profiling**:
```bash
ncu --set full ./target/release/examples/profile_12char
```

**Run standalone**:
```bash
cargo run --release --example profile_12char
```

---

### 🔴 `profile_transposed.rs` - Profile Transposed Kernel

**Purpose**: Profile the experimental transposed kernel variant.

**What it demonstrates**:
- Alternative memory access pattern
- Transpose overhead
- Experimental optimization

**When to use**: You're analyzing transpose performance.

**Run it**:
```bash
cargo run --release --example profile_transposed
```

---

## Advanced/Experimental Examples

### 🟠 `benchmark_transposed.rs` - Transpose Benchmark

**Purpose**: Benchmark the transposed kernel variant.

**What it demonstrates**:
- Transpose-based optimization
- Memory coalescing attempt
- Performance comparison

**When to use**: You're exploring memory optimization strategies.

**Run it**:
```bash
cargo run --release --example benchmark_transposed
```

**Note**: This is an experimental optimization that didn't improve performance.

---

### 🟠 `test_transpose_perf.rs` - CPU Transpose Performance

**Purpose**: Test CPU-side transpose performance.

**What it demonstrates**:
- CPU transpose overhead
- SIMD optimization (AVX2)
- RAM bandwidth limits

**When to use**: You're analyzing CPU transpose viability.

**Run it**:
```bash
cargo run --release --example test_transpose_perf
```

**Note**: Results show CPU transpose is 5.3× slower than GPU generation.

---

### 🟠 `poc_benchmark.rs` - Original POC Benchmark

**Purpose**: Benchmark from the original proof-of-concept phase.

**What it demonstrates**:
- Historical performance
- Early implementation

**When to use**: You want to compare against original POC performance.

**Run it**:
```bash
cargo run --release --example poc_benchmark
```

---

### 🟠 `poc_accurate.rs` - POC Accuracy Test

**Purpose**: Accuracy testing from POC phase.

**What it demonstrates**:
- Early validation approach
- POC correctness

**When to use**: Historical reference only.

**Run it**:
```bash
cargo run --release --example poc_accurate
```

---

## Example Categories Summary

| Category | Examples | Purpose |
|----------|----------|---------|
| **Beginner** | `simple_basic`, `simple_rust_api` | Learn the API |
| **Validation** | `cross_validate`, `validate_gpu`, `test_perf_comparison` | Verify correctness |
| **Benchmarking** | `benchmark_realistic`, `benchmark_production`, `benchmark_multigpu*` | Measure performance |
| **Profiling** | `profile_12char`, `profile_transposed` | Analyze with Nsight |
| **Experimental** | `benchmark_transposed`, `test_transpose_perf`, `poc_*` | Research & history |

---

## Recommended Learning Path

1. **Start**: `simple_basic.rs` - Understand the basics
2. **Explore**: `simple_rust_api.rs` - See all features
3. **Validate**: `cross_validate.rs` - Verify correctness
4. **Benchmark**: `benchmark_realistic.rs` - Measure performance
5. **Advanced**: Explore profiling and experimental examples

---

## Common Use Cases

### "I want to integrate this into my project"
→ Start with `simple_basic.rs`, then read `docs/guides/INTEGRATION_GUIDE.md`

### "I want to measure performance on my GPU"
→ Run `benchmark_realistic.rs` and `benchmark_multigpu.rs`

### "I want to verify it's working correctly"
→ Run `cross_validate.rs` and `test_perf_comparison.rs`

### "I have multiple GPUs"
→ Run `benchmark_multigpu.rs` and check scaling efficiency

### "I want to optimize the kernel"
→ Use `profile_12char.rs` with Nsight Compute, read `docs/guides/NSIGHT_COMPUTE_SETUP.md`

---

## Build All Examples

```bash
# Build all examples
cargo build --release --examples

# Run a specific example
cargo run --release --example simple_basic

# List all available examples
cargo build --release --examples --message-format=json | \
  jq -r 'select(.target.kind[] | contains("example")) | .target.name' | \
  sort | uniq
```

---

## Getting Help

- **Documentation**: See `docs/README.md` for comprehensive docs
- **API Reference**: Run `cargo doc --open`
- **Integration**: Read `docs/guides/INTEGRATION_GUIDE.md`
- **Issues**: https://github.com/tehw0lf/gpu-scatter-gather/issues

---

*Last Updated: November 23, 2025*
*Version: 1.3.0-dev*
