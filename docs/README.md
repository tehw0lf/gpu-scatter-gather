# Documentation Directory

This directory contains comprehensive documentation for the GPU Scatter-Gather Wordlist Generator.

## Directory Structure

### 📘 API Documentation (`api/`)
C Foreign Function Interface and language bindings:
- **C_API_SPECIFICATION.md** - Complete C API reference
- **FFI_IMPLEMENTATION_GUIDE.md** - Rust-to-C FFI implementation patterns
- **CBINDGEN_SETUP.md** - Automatic C header generation setup
- **PHASE1_SUMMARY.md** - Phase 1 implementation summary (host memory API)

### 🏗️ Design & Architecture (`design/`)
System design and architectural decisions:
- **FORMAL_SPECIFICATION.md** - Mathematical specification with proofs
- **LIBRARY_ARCHITECTURE.md** - System design and component overview
- **PCIE_BOTTLENECK_ANALYSIS.md** - PCIe bandwidth bottleneck analysis

### ✅ Validation & Testing (`validation/`)
Correctness validation and quality assurance:
- **FORMAL_VALIDATION_PLAN.md** - Comprehensive validation methodology
- **STATISTICAL_VALIDATION.md** - Statistical analysis of output properties
- **CROSS_VALIDATION_RESULTS.md** - Cross-validation against reference implementations

### 📊 Benchmarking (`benchmarking/`)
Performance measurement and competitive analysis:
- **BASELINE_BENCHMARKING_PLAN.md** - Scientific benchmarking methodology
- **COMPETITOR_ANALYSIS.md** - Comparison with existing tools

### 📖 User Guides (`guides/`)
Integration and usage documentation:
- **INTEGRATION_GUIDE.md** - How to integrate the library
- **PUBLICATION_GUIDE.md** - Academic publication preparation
- **ENABLE_PROFILING.md** - GPU profiling setup
- **NSIGHT_COMPUTE_SETUP.md** - Nsight Compute profiling guide

### 🔧 Development (`development/`)
Internal development documentation:
- **DEVELOPMENT_LOG.md** - Chronological development history
- **TODO.md** - Current roadmap and remaining tasks
- **DEVELOPMENT_PROCESS.md** - Development methodology
- **OPTIONAL_ENHANCEMENTS.md** - Future enhancement ideas

### 📦 Archive (`archive/`)
Historical documents preserved for reference:
- Phase-specific results and session summaries
- Temporary planning documents
- Superseded analysis documents
- See `archive/README.md` for details

### 🔄 Session Management
- **NEXT_SESSION_PROMPT.md** (root) - Next session objectives (regenerated each session)

## Quick Navigation

### For Users
Start with:
1. Main `README.md` (project root)
2. `guides/INTEGRATION_GUIDE.md` - Integration walkthrough
3. `api/C_API_SPECIFICATION.md` - API reference

### For Contributors
Start with:
1. `development/TODO.md` - Current priorities
2. `development/DEVELOPMENT_LOG.md` - Project history
3. `design/LIBRARY_ARCHITECTURE.md` - System design

### For Researchers
Start with:
1. `design/FORMAL_SPECIFICATION.md` - Mathematical foundation
2. `validation/FORMAL_VALIDATION_PLAN.md` - Validation methodology
3. `benchmarking/BASELINE_BENCHMARKING_PLAN.md` - Performance measurement

### For Publication
Primary documents:
1. `design/FORMAL_SPECIFICATION.md` - Core algorithm
2. `validation/` - All validation documents
3. `benchmarking/` - Performance analysis
4. `guides/PUBLICATION_GUIDE.md` - Publication checklist

## Documentation Standards

### File Naming
- Use UPPERCASE for major documents
- Use descriptive names (avoid abbreviations)
- Date-stamp raw data files (YYYY-MM-DD)

### Content Structure
- Start with TL;DR or Overview section
- Include Table of Contents for long documents (>100 lines)
- Use consistent markdown formatting
- Include code examples where appropriate

### Maintenance
- Archive superseded documents to `archive/`
- Update `NEXT_SESSION_PROMPT.md` after each session
- Keep `DEVELOPMENT_LOG.md` updated with major milestones
- Review and update `TODO.md` regularly

## Size Summary

| Category | Files | Total Size |
|----------|-------|------------|
| API | 4 | ~55 KB |
| Design | 3 | ~62 KB |
| Validation | 3 | ~31 KB |
| Benchmarking | 2 | ~44 KB |
| Guides | 4 | ~57 KB |
| Development | 4 | ~143 KB |
| Archive | 12 | ~106 KB |
| **Total** | **32** | **~498 KB** |

## Recent Changes

**2025-11-19**: Documentation reorganization
- Created logical subdirectories (api/, design/, validation/, etc.)
- Moved 11 historical documents to archive/
- Created this README for better navigation
- Reduced main documentation from ~450KB to ~250KB active docs

---

*Last Updated: November 19, 2025*
