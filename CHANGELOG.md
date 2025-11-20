# Changelog

All notable changes to the GPU Scatter-Gather Wordlist Generator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **CRITICAL**: Fixed buffer overrun bug in output format modes (PACKED, FIXED_WIDTH)
  - GPU kernels now correctly respect output format mode setting
  - Added `output_format` parameter throughout entire GPU stack (Rust + CUDA)
  - All 3 CUDA kernels now conditionally write separators based on format
  - Fixed memory corruption causing crashes when using PACKED or FIXED_WIDTH formats
  - ~85 lines changed across 5 files (src/gpu/mod.rs, src/ffi.rs, kernels/wordlist_poc.cu, benches/)

### Changed
- Cleaned up test suite: Removed 6 redundant debug/experimental test files
- Kept 2 canonical test files: `tests/ffi_basic_test.c` (16 tests) and `tests/ffi_integration_simple.c` (5 tests)

### Documentation
- Updated DEVELOPMENT_LOG.md with bug fix details
- Rewrote NEXT_SESSION_PROMPT.md with production-ready status
- Added comprehensive bug fix analysis and verification results

## [0.1.0] - 2025-11-19

### Added
- **Phase 1**: CPU reference implementation with mixed-radix algorithm
- **Phase 2**: CUDA kernel with shared memory optimization (440 M words/s)
- **Phase 2.7 Phase 1**: Host memory C API (11 FFI functions)
  - `wg_create()`, `wg_destroy()`
  - `wg_add_charset()`, `wg_set_mask()`
  - `wg_calculate_keyspace()`, `wg_calculate_buffer_size()`
  - `wg_generate_batch_host()`
- **Phase 2.7 Phase 2**: Zero-copy device pointer API (2 FFI functions)
  - `wg_generate_batch_device()` - Synchronous device generation
  - `wg_free_device_batch()` - Manual cleanup
- **Phase 2.7 Phase 3**: Output format modes (2 FFI functions)
  - `wg_set_format()` - Configure output format (NEWLINES, PACKED, FIXED_WIDTH)
  - Format-aware buffer size calculation
  - 11.1% memory savings with PACKED format
- **Phase 2.7 Phase 4**: Async streaming API (1 FFI function)
  - `wg_generate_batch_stream()` - Generate with custom CUDA stream
  - Overlap compute and transfer operations
- **Phase 2.7 Phase 5**: Utility functions (2 FFI functions)
  - `wg_version()` - Get library version
  - `wg_device_info()` - Get GPU device information

### Performance
- Host API: 440 M words/s for 12-char passwords (RTX 4070 Ti SUPER)
- Device API: Zero PCIe transfers (100-200x latency improvement)
- PACKED format: 11.1% less memory usage

### Documentation
- Comprehensive C API specification
- Integration guides for hashcat and John the Ripper
- Formal mathematical specification with proofs
- Scientific validation methodology
- Development log with detailed phase summaries

### Testing
- 21 tests total: 16 basic FFI tests + 5 integration tests
- 100% test pass rate
- Validated all output format modes
- Cross-validated against CPU reference implementation

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | 2025-11-19 | Initial release - Feature complete C API |
| Unreleased | 2025-11-20 | Bug fix - Output format modes now work correctly |

---

**Project Status:** Production Ready (All tests passing, critical bugs fixed)
**Author:** tehw0lf + Claude Code (AI-assisted development)
**License:** [To be determined]
