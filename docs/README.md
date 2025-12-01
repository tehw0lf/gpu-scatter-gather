# Documentation Directory

This directory contains comprehensive documentation for the GPU Scatter-Gather Wordlist Generator.

## 📄 Technical Whitepaper

**[GPU_Scatter_Gather_Whitepaper_v1.0.0.pdf](GPU_Scatter_Gather_Whitepaper_v1.0.0.pdf)** - Complete technical whitepaper (23 pages)
- Algorithm design and mathematical proofs
- Performance evaluation and competitive analysis
- Formal validation methodology
- Integration patterns and use cases

**Source:** [WHITEPAPER.md](WHITEPAPER.md) - Markdown source with ASCII diagrams

---

## Directory Structure

### 📘 API Documentation (`api/`)
C Foreign Function Interface and language bindings:
- **C_API_SPECIFICATION.md** - Complete C API reference
- **FFI_IMPLEMENTATION_GUIDE.md** - Rust-to-C FFI implementation patterns
- **CBINDGEN_SETUP.md** - Automatic C header generation setup

### 🏗️ Design & Architecture (`design/`)
System design and architectural decisions:
- **FORMAL_SPECIFICATION.md** - Mathematical specification with proofs
- **LIBRARY_ARCHITECTURE.md** - System design and component overview
- **MULTI_GPU_DESIGN.md** - Multi-GPU architecture and load balancing
- **PINNED_MEMORY_DESIGN.md** - Pinned memory optimization design
- **PCIE_BOTTLENECK_ANALYSIS.md** - PCIe bandwidth bottleneck analysis
- **MEMORY_COALESCING_DECISION.md** - Three-strikes decision on memory coalescing ⭐ NEW

### ✅ Validation & Testing (`validation/`)
Correctness validation and quality assurance:
- **FORMAL_VALIDATION_PLAN.md** - Comprehensive validation methodology
- **STATISTICAL_VALIDATION.md** - Statistical analysis of output properties
- **CROSS_VALIDATION_RESULTS.md** - Cross-validation against reference implementations

### 📊 Benchmarking (`benchmarking/`)
Performance measurement and competitive analysis:
- **BASELINE_BENCHMARKING_PLAN.md** - Scientific benchmarking methodology
- **COMPETITIVE_RESULTS.md** - Complete competitive analysis vs cracken ⭐ NEW
- **COMPETITOR_ANALYSIS.md** - Comparison with existing tools
- **MULTI_GPU_RESULTS.md** - Multi-GPU scaling performance
- **NSIGHT_COMPUTE_PROFILE_2025-11-22.md** - GPU profiling analysis

### 📖 User Guides (`guides/`)
Integration and usage documentation:
- **INTEGRATION_GUIDE.md** - Generic integration patterns for custom tools
- **HASHCAT_INTEGRATION.md** - Complete hashcat integration guide (3 patterns) ⭐ NEW
- **JTR_INTEGRATION.md** - John the Ripper integration guide (external mode, plugins) ⭐ NEW
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
- Phase-specific results and session summaries (PHASE1-5_SUMMARY.md)
- Failed optimization attempts (GPU_TRANSPOSE_ATTEMPT, WRITE_COMBINED_MEMORY_EXPERIMENT)
- Temporary planning documents
- Superseded analysis documents
- Whitepaper support files (Mermaid diagrams, ASCII diagrams reference)
- See `archive/README.md` for details

## Quick Navigation

### For New Users ⭐
Start here:
1. **[QUICKSTART.md](../QUICKSTART.md)** - Get running in 5 minutes
2. **[EXAMPLES.md](../EXAMPLES.md)** - Guide to all 16 examples
3. **[FAQ.md](../FAQ.md)** - Common questions and troubleshooting
4. Main `README.md` (project root) - Complete overview

### For Users
Integration and usage:
1. Tool-specific integration:
   - `guides/HASHCAT_INTEGRATION.md` - For hashcat users
   - `guides/JTR_INTEGRATION.md` - For John the Ripper users
   - `guides/INTEGRATION_GUIDE.md` - For custom tool developers
2. `api/C_API_SPECIFICATION.md` - API reference
3. **[FAQ.md](../FAQ.md)** - Troubleshooting and common questions

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
- Keep `DEVELOPMENT_LOG.md` updated with major milestones
- Review and update `TODO.md` regularly
- Document major decisions (e.g., MEMORY_COALESCING_DECISION.md)

## Size Summary

| Category | Files | Total Size |
|----------|-------|------------|
| API | 8 | ~116 KB |
| Design | 3 | ~68 KB |
| Validation | 3 | ~36 KB |
| Benchmarking | 2 | ~48 KB |
| Guides | 6 | ~112 KB |
| Development | 4 | ~143 KB |
| Archive | 12 | ~106 KB |
| **Total** | **38** | **~629 KB** |

## Recent Changes

**2025-12-01**: v1.6.0 Release & Three-Strikes Decision ⭐
- Released v1.6.0 with competitive benchmarking (3.8-15.3× faster than cracken)
- Added **COMPETITIVE_RESULTS.md** - Complete competitive analysis
- Added **MEMORY_COALESCING_DECISION.md** - Three-strikes optimization decision
- Archived phase summaries and obsolete experiment docs
- Removed NEXT_SESSION_PROMPT.md (no longer needed)
- Documentation cleanup: 7 files archived

**2025-11-23**: User-Facing Documentation Expansion
- Added **[QUICKSTART.md](../QUICKSTART.md)** - 5-minute setup guide for new users
- Added **[FAQ.md](../FAQ.md)** - Comprehensive FAQ with troubleshooting
- Added **[EXAMPLES.md](../EXAMPLES.md)** - Complete guide to all 16 examples
- Created 2 beginner-friendly examples (`simple_basic.rs`, `simple_rust_api.rs`)
- Updated README with prominent links to new guides
- Improved onboarding experience for new users

**2025-11-20**: v1.0.0 Release Documentation
- Added **HASHCAT_INTEGRATION.md** - Complete integration guide with 3 patterns
- Added **JTR_INTEGRATION.md** - John the Ripper integration guide
- Created **RELEASE_NOTES_v1.0.0.md** - Comprehensive release notes
- Updated all API phase summaries (PHASE1-5_SUMMARY.md)

---

*Last Updated: December 1, 2025*
