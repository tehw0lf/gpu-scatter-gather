# CLAUDE.md - Project-Specific Instructions

This file provides guidance to Claude Code when working with the GPU Scatter-Gather Wordlist Generator project.

---

## Project Overview

**Type:** High-performance computing library (GPU-accelerated)
**Language:** Rust + CUDA
**Domain:** Security research, password cracking, wordlist generation
**Performance Target:** 800M-1.2B+ words/second

This is a production-quality library with:
- Formal mathematical specification with proofs
- Scientific validation methodology
- Academic-level documentation
- C FFI for integration into existing tools (hashcat, John the Ripper)

---

## Architecture

### Core Components

```
gpu-scatter-gather/
├── src/
│   ├── lib.rs           # Rust API (WordlistGenerator)
│   ├── ffi.rs           # C FFI layer
│   ├── gpu/mod.rs       # GPU context management
│   ├── charset.rs       # Character set definitions
│   ├── mask.rs          # Mask pattern parsing
│   ├── keyspace.rs      # Mixed-radix arithmetic
│   └── transpose.rs     # SIMD transpose operations
├── kernels/
│   └── wordlist_poc.cu  # CUDA kernels (3 variants)
├── docs/                # Organized documentation (see docs/README.md)
└── tests/               # Integration tests
```

### Algorithm

**Core Innovation:** Direct index-to-word mapping using mixed-radix arithmetic
- No sequential dependencies (unlike odometer algorithms)
- O(1) random access to any position in keyspace
- Perfect parallelization on GPU (every thread independent)

See `docs/design/FORMAL_SPECIFICATION.md` for mathematical foundation.

---

## Development Workflow

### Build & Test

```bash
# Build release (includes CUDA kernel compilation)
cargo build --release

# Run tests
cargo test

# Run C FFI tests
gcc -o test_ffi tests/ffi_basic_test.c -I. -I/opt/cuda/targets/x86_64-linux/include \
    -L./target/release -lgpu_scatter_gather -Wl,-rpath,./target/release
./test_ffi

# Benchmarks
cargo build --release --example benchmark_realistic
./target/release/examples/benchmark_realistic
```

### Pre-Commit Validation

**ALWAYS** run these before committing:

```bash
# 1. Build and test
cargo build --release
cargo test

# 2. Verify FFI works
./test_ffi

# 3. Check formatting
cargo fmt --check

# 4. Linting
cargo clippy -- -D warnings

# 5. Documentation
cargo doc --no-deps
```

### CUDA Development

**Kernels:** Located in `kernels/wordlist_poc.cu`

**Three kernel variants:**
1. `generate_words_kernel` - Original (row-major, uncoalesced)
2. `generate_words_transposed_kernel` - Transposed writes (experimental)
3. `generate_words_columnmajor_kernel` - Column-major + CPU transpose (fastest)

**Build Process:**
- `build.rs` compiles kernels for multiple compute capabilities (sm_70-90)
- PTX files generated at build time
- Runtime selects appropriate PTX based on GPU

**Profiling:**
```bash
# Nsight Compute (detailed metrics)
ncu --set full ./target/release/examples/benchmark_realistic

# nvprof (quick overview)
nvprof --print-gpu-trace ./target/release/examples/benchmark_realistic
```

See `docs/guides/NSIGHT_COMPUTE_SETUP.md` for profiling details.

---

## Code Style & Quality

### Rust Guidelines

- **Safety:** All FFI functions must validate inputs (no panics across boundary)
- **Error handling:** Use `anyhow::Result` for library code, error codes for FFI
- **Documentation:** All public APIs must have doc comments
- **Testing:** Unit tests for all core functions

### CUDA Guidelines

- **Kernel names:** Descriptive, end with `_kernel`
- **Thread safety:** All kernels must be thread-safe
- **Memory:** Check all `cuMem*` calls for errors
- **Optimization:** Profile before optimizing (measure, don't guess)

### Commit Messages

Follow conventional commits:
```
type(scope): description

- feat: New feature
- fix: Bug fix
- docs: Documentation only
- perf: Performance improvement
- refactor: Code refactoring
- test: Adding tests
```

Examples:
```
feat(ffi): Add device pointer API for zero-copy operation

perf(kernel): Implement column-major writes for 2x speedup

docs(validation): Add formal correctness proofs
```

---

## Documentation Organization

See `docs/README.md` for comprehensive documentation index.

**Key Documents:**

| Purpose | Document |
|---------|----------|
| Algorithm | `docs/design/FORMAL_SPECIFICATION.md` |
| C API | `docs/api/C_API_SPECIFICATION.md` |
| Integration | `docs/guides/INTEGRATION_GUIDE.md` |
| Validation | `docs/validation/FORMAL_VALIDATION_PLAN.md` |
| Benchmarking | `docs/benchmarking/BASELINE_BENCHMARKING_PLAN.md` |
| Development | `docs/development/DEVELOPMENT_LOG.md` |

---

## Current State (Phase 2.7)

### Completed ✅

- **Phase 1:** POC - CPU reference + basic CUDA kernel
- **Phase 2:** Production kernel with validation
- **Phase 2.6:** Formal specification + statistical validation
- **Phase 2.7 Phase 1:** C FFI layer (host memory API)

### Current Focus 🎯

**Phase 2.7 Phase 2:** Device pointer API for zero-copy GPU operation
- Expected: 800-1200 M words/s (2-3x improvement over Phase 1)
- See `docs/NEXT_SESSION_PROMPT.md` for implementation plan

### Performance Status

- **Current (host API):** 440 M words/s
- **Bottleneck:** PCIe bandwidth (memory copy from GPU to host)
- **Target (device API):** 800-1200 M words/s (eliminate PCIe overhead)

---

## Testing Philosophy

### Correctness First

1. **Mathematical proofs:** Algorithm is provably correct (bijection proven)
2. **Cross-validation:** All output validated against CPU reference
3. **Statistical tests:** Chi-square, autocorrelation, runs tests
4. **Edge cases:** Boundary conditions, overflow, empty inputs

### Performance Second

Only optimize after proving correctness:
1. Establish baseline
2. Profile to identify bottleneck
3. Implement optimization
4. Validate correctness maintained
5. Measure improvement

See `docs/validation/FORMAL_VALIDATION_PLAN.md` for methodology.

---

## Common Tasks

### Adding a New FFI Function

1. Add to `src/ffi.rs` with `#[no_mangle] pub extern "C"`
2. Document with doc comments (cbindgen will extract)
3. Validate all inputs (null checks, bounds checks)
4. Handle errors gracefully (no panics)
5. Add test to `tests/ffi_basic_test.c`
6. Rebuild - cbindgen auto-generates header

### Adding a New CUDA Kernel

1. Add kernel to `kernels/wordlist_poc.cu`
2. Add kernel function pointer to `GpuContext` struct
3. Load kernel in `GpuContext::new()`
4. Add wrapper method to `GpuContext`
5. Test with benchmark
6. Profile with Nsight Compute

### Performance Optimization

**Process:**
```bash
# 1. Baseline benchmark
./target/release/examples/benchmark_realistic > baseline.txt

# 2. Profile
ncu --set full --export profile.ncu-rep ./target/release/examples/benchmark_realistic

# 3. Identify bottleneck (memory bandwidth? compute? divergence?)

# 4. Implement fix

# 5. Verify correctness maintained
cargo test

# 6. Measure improvement
./target/release/examples/benchmark_realistic > optimized.txt
./scripts/compare_benchmarks.sh baseline.txt optimized.txt
```

---

## Known Gotchas

### Build Issues

**CUDA not found:** Set `CUDA_PATH=/opt/cuda` or install CUDA toolkit
**PTX compilation fails:** Check `nvcc --version` matches build.rs requirements
**Link errors:** Ensure `-L./target/release -Wl,-rpath,./target/release`

### FFI Issues

**Segfaults:** Always validate pointers before dereferencing
**Memory leaks:** Use Box for ownership, track device pointers
**ABI compatibility:** Only use `#[repr(C)]` structs, never Rust enums across FFI

### Performance Issues

**Low throughput:** Check PCIe bandwidth with `nvidia-smi` (16 GB/s expected for PCIe 4.0 x16)
**Slow builds:** `build.rs` compiles 6 CUDA architectures (30-60s expected)
**OOM errors:** Reduce batch size in benchmarks

---

## Publication Readiness

This project is being developed with academic publication in mind.

**Key Assets:**
- Formal mathematical specification with proofs (publication-quality)
- Statistical validation following scientific methodology
- Reproducible benchmarks with raw data
- Fair competitive analysis

**Before Publishing:**
- Review `docs/guides/PUBLICATION_GUIDE.md`
- Ensure all claims are backed by data
- Provide reproducibility package (code, data, scripts)

---

## Memory Management Strategy

### Rust Side
- Use `Box` for heap allocation
- Use `Arc` for shared ownership
- No `unsafe` in library code (only in FFI layer)

### C FFI Side
- Caller allocates buffers, library fills them
- Library manages internal state with opaque handles
- Auto-cleanup on `wg_destroy()`

### CUDA Side
- Track all device pointers
- Free on error paths
- Auto-free on next generation (device API)

---

## Next Session Checklist

Before starting new session:

1. Read `docs/NEXT_SESSION_PROMPT.md`
2. Review `docs/development/TODO.md` for current priorities
3. Check `docs/development/DEVELOPMENT_LOG.md` for recent changes
4. Run tests to verify current state
5. Check git status for uncommitted work

---

## Resources

### Internal Documentation
- **Algorithm:** `docs/design/FORMAL_SPECIFICATION.md`
- **API Reference:** `docs/api/C_API_SPECIFICATION.md`
- **Development Log:** `docs/development/DEVELOPMENT_LOG.md`

### External Resources
- **CUDA Programming:** https://docs.nvidia.com/cuda/
- **Nsight Compute:** https://developer.nvidia.com/nsight-compute
- **Rust FFI:** https://doc.rust-lang.org/nomicon/ffi.html

### Benchmarking
- **Baseline Results:** `benches/scientific/results/baseline_2025-11-09.json`
- **Validation Results:** `benches/scientific/results/validation_2025-11-09.json`

---

**Project Status:** Phase 2.7 (C API) - Phase 1 Complete, Phase 2 Next

**Performance:** 440 M words/s (host API), targeting 800-1200 M words/s (device API)

**Quality:** Production-ready with formal validation and academic rigor

---

*Last Updated: November 19, 2025*
