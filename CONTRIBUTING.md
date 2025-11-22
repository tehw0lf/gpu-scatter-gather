# Contributing to GPU Scatter-Gather

Thank you for your interest in contributing to GPU Scatter-Gather! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Areas for Contribution](#areas-for-contribution)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Performance Validation](#performance-validation)

## Code of Conduct

This project follows a professional and inclusive environment:

- **Be respectful** - Value diverse perspectives and experiences
- **Be constructive** - Provide helpful feedback and suggestions
- **Be collaborative** - Work together towards common goals
- **Be ethical** - Use this tool responsibly for authorized security testing only

Unacceptable behavior includes harassment, discrimination, or malicious use of the software.

## Getting Started

### Prerequisites

- **Rust 1.82+** - [Install Rust](https://rustup.rs/)
- **CUDA Toolkit 11.8+** - [Download CUDA](https://developer.nvidia.com/cuda-downloads)
- **NVIDIA GPU** with compute capability 7.5+ (Turing or newer)
- **Git** for version control

### Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/gpu-scatter-gather
cd gpu-scatter-gather

# Build the project
cargo build --release

# Run tests
cargo test

# Run FFI integration tests
gcc -o test_ffi_integration_simple tests/ffi_integration_test_simple.c \
    -I. -L./target/release -lgpu_scatter_gather \
    -Wl,-rpath,./target/release
./test_ffi_integration_simple
```

## Development Process

### 1. Find or Create an Issue

- Browse [existing issues](https://github.com/tehw0lf/gpu-scatter-gather/issues)
- Comment on an issue to claim it
- For new features, create a feature request first to discuss

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Follow [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 4. Test Thoroughly

```bash
# Run all tests
cargo test

# Run FFI tests
./test_ffi_integration_simple

# Run benchmarks
cargo run --release --example benchmark_realistic

# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings
```

### 5. Submit Pull Request

- Push your branch to your fork
- Create a pull request with a clear description
- Link related issues
- Wait for review and address feedback

## Areas for Contribution

We welcome contributions in these areas:

### High Priority

- **Multi-GPU support** - Distribute keyspace across multiple GPUs
- **OpenCL backend** - Support for AMD/Intel GPUs
- **Performance optimization** - Profiling and kernel improvements
- **Python bindings** - PyO3-based Python API
- **Testing on different GPUs** - Validation across architectures

### Medium Priority

- **Hybrid masks** - Static prefix/suffix with dynamic middle section
- **JavaScript bindings** - WASM/Node.js support
- **Advanced charset modifiers** - Toggle, shift, custom functions
- **Rule-based generation** - Integration with hashcat rules
- **Documentation improvements** - Tutorials, examples, guides

### Low Priority

- **Barrett reduction** - Optimize division operations
- **Power-of-2 charset optimization** - Bitwise operations for special cases
- **Compression for network streaming** - Reduce bandwidth usage
- **Additional output formats** - Custom formatters

See [docs/development/OPTIONAL_ENHANCEMENTS.md](docs/development/OPTIONAL_ENHANCEMENTS.md) for detailed enhancement ideas.

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**: `cargo test` must complete successfully
2. **Run FFI tests**: `./test_ffi_integration_simple` must pass
3. **Format code**: Run `cargo fmt`
4. **Lint code**: Run `cargo clippy -- -D warnings` with no warnings
5. **Update documentation**: Add/update docs for new features
6. **Add tests**: New functionality must have test coverage

### PR Requirements

- **Clear description** - Explain what changes and why
- **Link issues** - Reference related issues (e.g., "Fixes #123")
- **Small, focused changes** - One feature/fix per PR when possible
- **Commit messages** - Follow [conventional commits](https://www.conventionalcommits.org/)
- **No breaking changes** - Unless discussed and approved in advance

### Commit Message Format

```
type(scope): subject

body (optional)

footer (optional)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `docs`: Documentation only
- `test`: Adding tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

**Examples:**
```
feat(ffi): Add multi-GPU device enumeration API

Adds wg_get_device_count() and wg_get_device_info() functions
to support multi-GPU configurations.

Fixes #42
```

```
perf(kernel): Optimize memory coalescing for long passwords

Improves throughput for 16+ character passwords by 15%
through better memory access patterns.
```

## Coding Standards

### Rust Code

- **Style**: Follow `rustfmt` defaults (run `cargo fmt`)
- **Linting**: Pass `cargo clippy` with no warnings
- **Safety**: Minimize `unsafe` code, document when necessary
- **Error handling**: Use `anyhow::Result` for library code
- **Documentation**: All public APIs must have doc comments

### CUDA Code

- **Naming**: Descriptive kernel names ending with `_kernel`
- **Error checking**: Validate all CUDA API calls
- **Thread safety**: Ensure kernels are thread-safe
- **Optimization**: Profile before optimizing (measure first)

### C API (FFI)

- **Input validation**: Always validate pointers and parameters
- **No panics**: FFI functions must never panic
- **Error codes**: Use clear, documented error codes
- **Documentation**: cbindgen will extract doc comments to header

## Testing Requirements

### Unit Tests

All core functionality must have unit tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_name() {
        // Arrange
        let input = ...;

        // Act
        let result = function(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

### Integration Tests

FFI changes require C integration tests:

```c
// tests/ffi_test_feature.c
int main() {
    // Test FFI function
    int result = wg_function(...);
    assert(result == WG_SUCCESS);
    return 0;
}
```

### Performance Tests

Performance improvements require benchmark validation:

```bash
# Before your change
cargo run --release --example benchmark_realistic > before.txt

# After your change
cargo run --release --example benchmark_realistic > after.txt

# Compare results (must show improvement)
```

### Correctness Validation

All changes to core algorithm or output must pass:

```bash
# Cross-validation with maskprocessor
cargo run --release --example validate_correctness

# Statistical validation
cargo run --release --example statistical_validation
```

## Documentation

### Code Documentation

- **Public APIs**: Must have doc comments with examples
- **Complex logic**: Add inline comments explaining "why", not "what"
- **FFI functions**: Document in Rust (cbindgen extracts to header)

### User Documentation

When adding features, update relevant docs:

- `README.md` - For major features
- `docs/api/C_API_SPECIFICATION.md` - For FFI changes
- `docs/guides/` - Add integration guides
- `examples/` - Add code examples

### Example Documentation

```rust
/// Generates wordlist from mask pattern
///
/// # Arguments
///
/// * `mask` - Mask pattern (e.g., "?1?2?3")
/// * `charsets` - Character sets for placeholders
///
/// # Returns
///
/// * `Ok(words)` - Generated wordlist
/// * `Err(e)` - Error if generation fails
///
/// # Example
///
/// ```
/// use gpu_scatter_gather::generate;
///
/// let words = generate("?1?1", &["abc"])?;
/// assert_eq!(words.len(), 9);
/// ```
pub fn generate(mask: &str, charsets: &[&str]) -> Result<Vec<String>> {
    // Implementation
}
```

## Performance Validation

### Required for Performance PRs

1. **Baseline measurement** - Measure performance before changes
2. **Optimization implementation** - Make changes
3. **Correctness validation** - Ensure output is still correct
4. **Performance measurement** - Measure after changes
5. **Document improvement** - Show percentage improvement

### Profiling Tools

- **Nsight Compute** - Detailed GPU kernel profiling
- **nvprof** - Quick GPU performance overview
- **cargo bench** - CPU benchmark comparisons

See [docs/guides/NSIGHT_COMPUTE_SETUP.md](docs/guides/NSIGHT_COMPUTE_SETUP.md) for profiling setup.

### Performance Regression

PRs must not introduce performance regressions without justification:

- **<5% regression**: May be acceptable for correctness/features
- **>5% regression**: Requires discussion and approval
- **Any regression**: Must be documented in PR description

## Project Philosophy

### Correctness First, Performance Second

1. **Prove correctness** - Mathematical proofs and validation
2. **Establish baseline** - Measure current performance
3. **Profile bottleneck** - Identify actual bottleneck
4. **Implement optimization** - Make targeted changes
5. **Validate correctness** - Ensure correctness maintained
6. **Measure improvement** - Confirm performance gain

See [docs/validation/FORMAL_VALIDATION_PLAN.md](docs/validation/FORMAL_VALIDATION_PLAN.md) for validation methodology.

### Development Values

- **Transparency** - Clear documentation of algorithm and methodology
- **Reproducibility** - All benchmarks and validations reproducible
- **Scientific rigor** - Claims backed by data and proofs
- **Collaboration** - Human-AI partnership in development
- **Ethical use** - Tool designed for authorized security testing only

## Human-AI Collaboration

This project is a **human-AI collaborative research effort**:

- **Algorithm design** - AI proposed mixed-radix direct indexing
- **Implementation** - AI-guided Rust/CUDA development
- **Validation** - Formal mathematical proofs and testing
- **Documentation** - Comprehensive technical documentation

See [docs/development/DEVELOPMENT_PROCESS.md](docs/development/DEVELOPMENT_PROCESS.md) for detailed methodology.

We welcome contributions from both humans and AI-assisted development, as long as:
- Code quality is maintained
- Testing is thorough
- Documentation is complete
- Ethical guidelines are followed

## Questions?

- **Discussions**: [GitHub Discussions](https://github.com/tehw0lf/gpu-scatter-gather/discussions)
- **Issues**: [GitHub Issues](https://github.com/tehw0lf/gpu-scatter-gather/issues)
- **Documentation**: [docs/](docs/)

## License

By contributing, you agree that your contributions will be licensed under the same dual license as the project (MIT OR Apache-2.0).

---

**Thank you for contributing to GPU Scatter-Gather!** 🦀⚡🤖
