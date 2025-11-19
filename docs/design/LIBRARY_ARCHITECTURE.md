# Library Architecture: GPU Wordlist Generator as Embeddable Component

**Version**: 2.0 (Library Mode)
**Date**: November 18, 2025
**Purpose**: Design document for transforming GPU wordlist generator into reusable library for password crackers and security tools

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Use Cases & Target Applications](#use-cases--target-applications)
3. [Architecture Overview](#architecture-overview)
4. [API Design](#api-design)
5. [Memory Management](#memory-management)
6. [Output Format Modes](#output-format-modes)
7. [Integration Patterns](#integration-patterns)
8. [Performance Considerations](#performance-considerations)
9. [Security & Licensing](#security--licensing)
10. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

### Current State (v1.0)
- Standalone binary that generates wordlists to stdout/file
- GPU-accelerated: 440 M words/s (3-5x faster than CPU)
- Transfers results to host via PCIe

### Target State (v2.0)
- **Embeddable library** with C API for integration into password crackers
- **Zero-copy GPU-to-GPU** operation (candidates stay on GPU)
- **Streaming interface** for continuous generation
- **Multiple output formats** for different use cases

### Key Advantages for Integration

1. **Performance**: 440 M candidates/s on GPU vs 50-100 M/s on CPU
2. **Zero-Copy**: Eliminate PCIe bottleneck by keeping data on GPU
3. **Perfect Partitioning**: Direct index-to-word mapping enables trivial distributed cracking
4. **No I/O Bottleneck**: Generate on-demand vs reading from disk
5. **Deterministic**: Same index always produces same word (reproducibility)

---

## Use Cases & Target Applications

### Primary Use Cases

#### 1. GPU Password Crackers (hashcat, John the Ripper)
**Current workflow**:
```
CPU generates masks → Transfer to GPU → GPU hashes → Compare
Bottleneck: Mask generation on CPU (50-100 M/s)
```

**With library**:
```
GPU generates candidates → GPU hashes → Compare (all on GPU!)
Bottleneck: Hashing (optimal!)
```

**Integration benefit**: 4-9x faster candidate generation

#### 2. Distributed Password Auditing
**Scenario**: Security team auditing corporate password database across 10 GPUs

```rust
// Keyspace: 36^8 = 2.8 trillion candidates
// 10 GPUs, each gets 280 billion candidates

// GPU 0
generate_batch_device(start_idx: 0, count: 280_000_000_000)

// GPU 1
generate_batch_device(start_idx: 280_000_000_000, count: 280_000_000_000)

// ... GPU 2-9 ...
```

**No coordination needed!** Each GPU works independently on its partition.

#### 3. Password Strength Testing
**Scenario**: Web service checks password strength by seeing if it appears in common patterns

```c
// Initialize with common patterns
wg_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz");
wg_set_charset(gen, 2, "0123456789");
wg_set_mask(gen, [1,1,1,1,2,2,2,2]);  // "aaaa1234" pattern

// Check if user's password matches this pattern
uint64_t idx = wg_word_to_index(gen, user_password);
if (idx < wg_keyspace_size(gen)) {
    warn_user("Password is too common!");
}
```

#### 4. Security Research & Cryptography
- Benchmarking hash functions
- Testing key derivation functions (PBKDF2, Argon2)
- Academic research on password entropy

### Target Applications

1. **Hashcat** - World's fastest password cracker
2. **John the Ripper** - Classic password cracker
3. **Custom penetration testing tools**
4. **Password strength meters**
5. **Academic research frameworks**

---

## Architecture Overview

### Layer Diagram

```
┌─────────────────────────────────────────────────┐
│         User Application (C/C++/Python)         │
│         (hashcat, john, custom tools)           │
└─────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│              C API Layer (FFI)                  │
│  libwordlist_generator.so / wordlist_gen.dll    │
│                                                 │
│  - wg_create(), wg_destroy()                    │
│  - wg_set_charset(), wg_set_mask()              │
│  - wg_generate_batch_device()                   │
│  - wg_generate_batch_host()                     │
└─────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│           Rust Core Library                     │
│                                                 │
│  - GpuContext (CUDA context management)         │
│  - Charset/Mask validation                      │
│  - Keyspace calculation                         │
│  - Memory management                            │
└─────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│              CUDA Kernels                       │
│                                                 │
│  - generate_words_kernel (baseline)             │
│  - generate_words_columnmajor_kernel (v2)       │
│  - Output format variants                       │
└─────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│              GPU Memory                         │
│                                                 │
│  [candidate0][candidate1][candidate2]...        │
│   (stays on GPU for zero-copy hashing)          │
└─────────────────────────────────────────────────┘
```

### Component Responsibilities

#### C API Layer (`src/ffi.rs`)
- **Purpose**: Stable ABI for cross-language integration
- **Responsibilities**:
  - Type marshalling (Rust ↔ C)
  - Error handling (Rust Result → C error codes)
  - Memory lifetime management
  - Thread safety guarantees

#### Rust Core Library (`src/lib.rs`, `src/gpu/mod.rs`)
- **Purpose**: Business logic and CUDA interop
- **Responsibilities**:
  - Charset/mask validation
  - Keyspace calculation
  - CUDA context management
  - Kernel launch orchestration

#### CUDA Kernels (`kernels/wordlist_poc.cu`)
- **Purpose**: High-performance candidate generation
- **Responsibilities**:
  - Mixed-radix index decomposition
  - Character lookup and assembly
  - Memory coalescing optimization
  - Multiple output format support

---

## API Design

### Initialization & Teardown

```c
/**
 * Create a new wordlist generator instance
 *
 * @param ctx CUDA context (optional, pass NULL to create new)
 * @param device CUDA device ID (0 for default GPU)
 * @return Handle to generator, or NULL on error
 */
wg_handle_t wg_create(CUcontext ctx, int device);

/**
 * Destroy generator and free all resources
 */
void wg_destroy(wg_handle_t gen);

/**
 * Get last error message
 */
const char* wg_get_error(wg_handle_t gen);
```

### Configuration

```c
/**
 * Define a charset for use in masks
 *
 * @param gen Generator handle
 * @param charset_id Identifier (1-255)
 * @param chars Character array (e.g., "abc123")
 * @param len Length of character array
 * @return 0 on success, -1 on error
 */
int wg_set_charset(
    wg_handle_t gen,
    int charset_id,
    const char* chars,
    size_t len
);

/**
 * Set the mask pattern
 *
 * @param gen Generator handle
 * @param mask Array of charset IDs (e.g., [1,1,2,2] = ?1?1?2?2)
 * @param length Word length (number of positions)
 * @return 0 on success, -1 on error
 */
int wg_set_mask(
    wg_handle_t gen,
    const int* mask,
    int length
);

/**
 * Set output format mode
 */
typedef enum {
    WG_FORMAT_NEWLINES,    // "password\n" (default)
    WG_FORMAT_FIXED_WIDTH, // "password\0\0\0" (padded to max)
    WG_FORMAT_PACKED,      // "password" (no separator)
} wg_output_format_t;

int wg_set_output_format(
    wg_handle_t gen,
    wg_output_format_t format
);
```

### Keyspace Information

```c
/**
 * Get total keyspace size
 * Returns number of possible candidates given current mask
 */
uint64_t wg_keyspace_size(wg_handle_t gen);

/**
 * Convert word to its index (reverse lookup)
 * Useful for checking if a word matches the mask
 *
 * @return Index if word is valid, UINT64_MAX if not in keyspace
 */
uint64_t wg_word_to_index(
    wg_handle_t gen,
    const char* word,
    size_t word_len
);
```

### Generation: Device Memory (Zero-Copy)

```c
/**
 * Batch generation result (device memory)
 */
typedef struct {
    CUdeviceptr data;       // Device pointer to candidate buffer
    uint64_t count;         // Number of candidates generated
    size_t word_length;     // Length of each word (bytes)
    size_t stride;          // Bytes between word starts
    size_t total_bytes;     // Total buffer size
    int format;             // Output format used
} wg_batch_device_t;

/**
 * Generate batch in GPU memory (zero-copy)
 *
 * Candidates remain on GPU for direct consumption by hash kernels.
 * Caller must NOT free the device pointer - it's managed by generator.
 *
 * @param gen Generator handle
 * @param start_idx Starting index in keyspace
 * @param count Number of candidates to generate
 * @param batch Output batch info
 * @return 0 on success, -1 on error
 */
int wg_generate_batch_device(
    wg_handle_t gen,
    uint64_t start_idx,
    uint64_t count,
    wg_batch_device_t* batch
);

/**
 * Free device batch (if you're done with it before next generation)
 * Optional - automatically freed on next batch or wg_destroy()
 */
void wg_free_batch_device(wg_handle_t gen, wg_batch_device_t* batch);
```

### Generation: Host Memory (Copy)

```c
/**
 * Generate batch and copy to host memory
 *
 * @param gen Generator handle
 * @param start_idx Starting index in keyspace
 * @param count Number of candidates to generate
 * @param output_buffer Pre-allocated host buffer (must be large enough!)
 * @param buffer_size Size of output buffer in bytes
 * @return Number of bytes written, or -1 on error
 */
ssize_t wg_generate_batch_host(
    wg_handle_t gen,
    uint64_t start_idx,
    uint64_t count,
    char* output_buffer,
    size_t buffer_size
);

/**
 * Calculate required buffer size for host generation
 */
size_t wg_calculate_buffer_size(
    wg_handle_t gen,
    uint64_t count
);
```

### Streaming API (Advanced)

```c
/**
 * Generate using CUDA stream (for overlap with other operations)
 *
 * @param gen Generator handle
 * @param stream CUDA stream for async execution
 * @param start_idx Starting index
 * @param count Number to generate
 * @param batch Output batch info
 * @return 0 on success, -1 on error
 */
int wg_generate_batch_stream(
    wg_handle_t gen,
    CUstream stream,
    uint64_t start_idx,
    uint64_t count,
    wg_batch_device_t* batch
);
```

---

## Memory Management

### Ownership Rules

#### Device Memory (GPU)
- **Allocated by**: Library (`wg_generate_batch_device`)
- **Owned by**: Library (generator instance)
- **Lifetime**: Until next generation or `wg_destroy()`
- **Free'd by**: Library automatically

**Caller responsibilities**:
- ✅ DO: Use device pointer for GPU operations
- ✅ DO: Copy data if needed for longer lifetime
- ❌ DON'T: Call `cuMemFree()` on returned pointer
- ❌ DON'T: Store pointer beyond next generation call

#### Host Memory (CPU)
- **Allocated by**: Caller
- **Owned by**: Caller
- **Lifetime**: Managed by caller
- **Free'd by**: Caller

**Caller responsibilities**:
- ✅ DO: Allocate sufficient buffer (use `wg_calculate_buffer_size`)
- ✅ DO: Free buffer when done
- ❌ DON'T: Pass undersized buffer (will fail)

### Memory Layout Examples

#### Newlines Format (WG_FORMAT_NEWLINES)
```
Default for file output / stdout redirection
Each word terminated by '\n'

Memory layout:
┌────────────┬──┬────────────┬──┬────────────┬──┐
│ password1  │\n│ password2  │\n│ password3  │\n│
└────────────┴──┴────────────┴──┴────────────┴──┘

Stride: word_length + 1
Usage: Piping to hashcat stdin, file generation
```

#### Fixed-Width Format (WG_FORMAT_FIXED_WIDTH)
```
Best for GPU hash kernels with fixed-size input
Each word padded to max length with null bytes

Memory layout (8 char max):
┌────────┬────────┬────────┐
│pass\0\0\0\0│word\0\0\0\0│test\0\0\0\0│
└────────┴────────┴────────┘

Stride: max_word_length (constant)
Usage: GPU hash kernels, aligned access
```

#### Packed Format (WG_FORMAT_PACKED)
```
Minimal memory usage, no separators
Caller must know word length to parse

Memory layout:
┌────────┬────────┬────────┐
│password│password│password│
└────────┴────────┴────────┘

Stride: word_length (constant for given mask)
Usage: Custom parsers, maximum throughput
```

---

## Output Format Modes

### Design Rationale

Different consumers need different formats:

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Newlines** | File output, hashcat stdin | Human-readable, standard | Variable stride, parsing overhead |
| **Fixed-Width** | GPU hash kernels | Aligned access, predictable | Wastes space for short words |
| **Packed** | Maximum throughput | Minimal memory, cache-friendly | Requires length info |

### Implementation Strategy

Each format has its own kernel variant:
```cuda
// kernels/wordlist_poc.cu

__global__ void generate_words_newlines(...)    // Current implementation
__global__ void generate_words_fixed_width(...) // New: pad with nulls
__global__ void generate_words_packed(...)      // New: no separator
```

### Format Selection Guide

**Use NEWLINES when**:
- ✅ Piping to stdin of existing tools
- ✅ Writing to file for human inspection
- ✅ Variable-length words in mask (e.g., ?1?1?1?1?2?)

**Use FIXED_WIDTH when**:
- ✅ Feeding GPU hash kernel expecting fixed input
- ✅ Need aligned memory access (performance)
- ✅ Fixed word length in mask (e.g., ?1?1?1?1?1?1?1?1)

**Use PACKED when**:
- ✅ Custom parsing logic
- ✅ Maximum throughput needed
- ✅ Minimal memory footprint
- ✅ Streaming over network

---

## Integration Patterns

### Pattern 1: Simple Host-Side Integration

**Use case**: Existing tool that processes candidates on CPU

```c
#include <wordlist_generator.h>

int main() {
    wg_handle_t gen = wg_create(NULL, 0);

    wg_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz", 26);
    int mask[] = {1, 1, 1, 1, 1, 1, 1, 1};
    wg_set_mask(gen, mask, 8);

    uint64_t batch_size = 100000000;
    size_t buffer_size = wg_calculate_buffer_size(gen, batch_size);
    char* buffer = malloc(buffer_size);

    // Generate batch
    ssize_t bytes = wg_generate_batch_host(gen, 0, batch_size, buffer, buffer_size);

    // Process candidates (hash, compare, etc.)
    process_candidates(buffer, bytes);

    free(buffer);
    wg_destroy(gen);
    return 0;
}
```

**Performance**: ~440 M candidates/s, limited by PCIe transfer

### Pattern 2: GPU-to-GPU Zero-Copy

**Use case**: GPU password cracker, maximum performance

```c
#include <wordlist_generator.h>
#include <cuda.h>

// Custom hash kernel (user-provided)
__global__ void hash_md5_kernel(
    const char* candidates,
    int word_length,
    int stride,
    uint64_t count,
    unsigned char* hashes_out
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const char* word = candidates + (tid * stride);
    compute_md5(word, word_length, hashes_out + tid * 16);
}

int main() {
    wg_handle_t gen = wg_create(NULL, 0);

    // Configure generator
    wg_set_charset(gen, 1, "abc123", 6);
    int mask[] = {1, 1, 1, 1};
    wg_set_mask(gen, mask, 4);
    wg_set_output_format(gen, WG_FORMAT_FIXED_WIDTH);

    // Allocate hash output on GPU
    CUdeviceptr d_hashes;
    cuMemAlloc(&d_hashes, 100000000 * 16);  // 100M MD5 hashes

    // Generate candidates on GPU
    wg_batch_device_t batch;
    wg_generate_batch_device(gen, 0, 100000000, &batch);

    // Hash directly on GPU (zero-copy!)
    hash_md5_kernel<<<grid, block>>>(
        (const char*)batch.data,
        batch.word_length,
        batch.stride,
        batch.count,
        (unsigned char*)d_hashes
    );

    // Compare hashes, find match, etc.
    // ...

    cuMemFree(d_hashes);
    wg_destroy(gen);
    return 0;
}
```

**Performance**: ~440 M candidates/s, NO PCIe bottleneck!

### Pattern 3: Distributed Multi-GPU

**Use case**: Cracking across 4 GPUs on same machine

```c
#include <wordlist_generator.h>
#include <omp.h>

int main() {
    uint64_t keyspace = 36ULL * 36 * 36 * 36 * 36 * 36 * 36 * 36;  // 8 chars
    uint64_t partition = keyspace / 4;  // 4 GPUs

    #pragma omp parallel num_threads(4)
    {
        int gpu_id = omp_get_thread_num();

        // Each thread manages one GPU
        cuDeviceGet(&device, gpu_id);
        cuCtxCreate(&ctx, 0, device);

        wg_handle_t gen = wg_create(ctx, gpu_id);
        wg_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz0123456789", 36);
        int mask[] = {1,1,1,1,1,1,1,1};
        wg_set_mask(gen, mask, 8);

        // Each GPU works on its partition
        uint64_t start = gpu_id * partition;
        uint64_t count = partition;

        wg_batch_device_t batch;
        wg_generate_batch_device(gen, start, count, &batch);

        // Hash and check on this GPU
        crack_passwords(batch);

        wg_destroy(gen);
    }

    return 0;
}
```

**Performance**: Linear scaling! 4 GPUs = 4x throughput

### Pattern 4: Streaming Pipeline

**Use case**: Overlap generation + hashing + transfer

```c
#include <wordlist_generator.h>

int main() {
    wg_handle_t gen = wg_create(NULL, 0);
    // ... configure ...

    // Create multiple CUDA streams for overlap
    CUstream stream[3];
    for (int i = 0; i < 3; i++) {
        cuStreamCreate(&stream[i], 0);
    }

    uint64_t batch_size = 10000000;  // 10M per batch
    wg_batch_device_t batch[3];

    // Pipeline: Generate batch N while processing batch N-1
    for (uint64_t start = 0; start < keyspace; start += batch_size) {
        int slot = (start / batch_size) % 3;

        // Generate next batch (async)
        wg_generate_batch_stream(gen, stream[slot], start, batch_size, &batch[slot]);

        // Process previous batch (async)
        if (start > 0) {
            int prev = (slot + 2) % 3;
            hash_kernel<<<grid, block, 0, stream[prev]>>>(...);
        }

        // Results from batch N-2 (async)
        if (start > batch_size) {
            int done = (slot + 1) % 3;
            cuStreamSynchronize(stream[done]);
            check_results(batch[done]);
        }
    }

    wg_destroy(gen);
    return 0;
}
```

**Performance**: Overlapped execution, minimal stalls

---

## Performance Considerations

### Throughput Benchmarks

**Current Performance** (NVIDIA RTX 4070 Ti SUPER):
```
Word Length | Candidates/s | Memory BW | Time (100M)
------------|-------------|-----------|------------
8 chars     | 679 M/s     | 6.11 GB/s | 147 ms
10 chars    | 534 M/s     | 5.87 GB/s | 187 ms
12 chars    | 441 M/s     | 5.73 GB/s | 227 ms
```

**Comparison to Alternatives**:
```
Tool              | Generation Rate | Notes
------------------|-----------------|--------------------------------
maskprocessor     | 50-100 M/s      | CPU-based, hashcat's default
crunch            | 30-60 M/s       | CPU-based, file output
This library (CPU)| 140 M/s         | Baseline (Phase 1)
This library (GPU)| 440 M/s         | 3-5x faster (Phase 3)
```

### Bottleneck Analysis

**For fast hashes (MD5, SHA1)**:
```
Generation: 441 M/s  ← Potential bottleneck
MD5 hash:   1000 M/s (GPU has headroom)
Comparison: 2000 M/s (trivial)

Conclusion: Generation is bottleneck, but still 4-9x faster than CPU!
```

**For slow hashes (bcrypt cost=10)**:
```
Generation: 441 M/s
bcrypt:     0.5 M/s  ← Clear bottleneck
Comparison: 2000 M/s

Conclusion: Hashing is bottleneck, generation never stalls
```

### Memory Bandwidth Impact

**Zero-copy (device pointer):**
```
GPU generation: 440 M/s
No PCIe transfer needed
Hash kernel reads directly from GPU memory

Benefit: ~30% speedup (eliminate 100ms PCIe transfer time)
```

**Host copy:**
```
GPU generation:  220 ms
PCIe transfer:   100 ms (1.3 GB @ 13 GB/s)
──────────────────────
Total:           320 ms → 312 M/s effective

Loss: ~30% slower due to PCIe bottleneck
```

### Scaling Characteristics

**Multi-GPU (same machine)**:
```
GPUs | Total Throughput | Scaling Efficiency
-----|------------------|-------------------
1    | 440 M/s          | 100%
2    | 880 M/s          | 100% (linear!)
4    | 1760 M/s         | 100%
8    | 3520 M/s         | 100%
```

**Why perfect scaling?**
- Each GPU operates independently
- No shared state or coordination
- Direct index-to-word mapping (no communication)

---

## Security & Licensing

### Responsible Use Statement

This library is a **dual-use security tool**, similar to nmap, Wireshark, or OpenSSL.

**Legitimate use cases**:
- ✅ Authorized penetration testing
- ✅ Password strength auditing (with permission)
- ✅ Security research and academic studies
- ✅ Defensive security tool development
- ✅ Personal password recovery

**Prohibited use cases**:
- ❌ Unauthorized access to systems or accounts
- ❌ Illegal hacking or credential theft
- ❌ Violating computer fraud and abuse laws
- ❌ Any activity without explicit authorization

### Recommended License: MIT or Apache 2.0

**MIT License**:
- Permissive, allows commercial use
- Simple and widely understood
- Common in security tools

**Apache 2.0**:
- Permissive, includes patent grant
- More comprehensive legal protection
- Preferred by many enterprises

### Documentation Requirements

1. **README.md** must include:
   - Responsible use disclaimer
   - Legal notice about authorization requirements
   - Examples of legitimate use cases

2. **API Documentation** must state:
   - "This library generates password candidates for security testing"
   - "User is responsible for obtaining proper authorization"
   - "Unauthorized access is illegal in most jurisdictions"

3. **Example Code** must demonstrate:
   - Authorized testing scenarios
   - Proper permission checks
   - Ethical security research practices

### Comparison to Similar Tools

| Tool | Type | License | Legal Status |
|------|------|---------|--------------|
| **hashcat** | Password cracker | MIT | Legal, widely used |
| **John the Ripper** | Password cracker | GPL | Legal, widely used |
| **nmap** | Port scanner | GPL | Legal (dual-use) |
| **Metasploit** | Exploit framework | BSD | Legal (dual-use) |
| **This library** | Candidate generator | MIT/Apache | Legal (dual-use) |

---

## Implementation Roadmap

### Phase 1: Core C API (Estimated: 8-10 hours)

#### 1.1 FFI Layer (3-4 hours)
**File**: `src/ffi.rs`

Tasks:
- [ ] Define opaque handle type (`struct WordlistGenerator`)
- [ ] Implement `wg_create()` / `wg_destroy()`
- [ ] Implement `wg_set_charset()` / `wg_set_mask()`
- [ ] Implement error handling (thread-local error buffer)
- [ ] Add FFI-safe type conversions

#### 1.2 Header File (1 hour)
**File**: `include/wordlist_generator.h`

Tasks:
- [ ] Define public API declarations
- [ ] Add comprehensive documentation comments
- [ ] Define error codes enum
- [ ] Define output format enum

#### 1.3 Build System (1 hour)
**File**: `build.rs`, `Cargo.toml`

Tasks:
- [ ] Configure cbindgen for header generation
- [ ] Set up cdylib target
- [ ] Add build script for C header generation

#### 1.4 Basic Tests (1-2 hours)
**File**: `tests/ffi_tests.c`

Tasks:
- [ ] Test initialization/teardown
- [ ] Test charset/mask configuration
- [ ] Test error handling

### Phase 2: Device Pointer Support (Estimated: 4-5 hours)

#### 2.1 Batch Device API (2-3 hours)
**File**: `src/ffi.rs`, `src/gpu/mod.rs`

Tasks:
- [ ] Implement `wg_generate_batch_device()`
- [ ] Add device pointer return logic
- [ ] Implement batch metadata struct
- [ ] Add memory lifetime management

#### 2.2 Testing (2 hours)
**File**: `tests/device_pointer_tests.c`

Tasks:
- [ ] Test device pointer validity
- [ ] Test zero-copy hash kernel integration
- [ ] Verify no memory leaks

### Phase 3: Output Formats (Estimated: 6-8 hours)

#### 3.1 Fixed-Width Kernel (2-3 hours)
**File**: `kernels/wordlist_poc.cu`

Tasks:
- [ ] Implement `generate_words_fixed_width_kernel`
- [ ] Add null padding logic
- [ ] Optimize for aligned access

#### 3.2 Packed Kernel (2-3 hours)
**File**: `kernels/wordlist_poc.cu`

Tasks:
- [ ] Implement `generate_words_packed_kernel`
- [ ] Remove separator logic
- [ ] Verify contiguous layout

#### 3.3 Format Selection (1-2 hours)
**File**: `src/gpu/mod.rs`, `src/ffi.rs`

Tasks:
- [ ] Add `wg_set_output_format()` API
- [ ] Implement kernel selection logic
- [ ] Update batch metadata

#### 3.4 Testing (1 hour)
**File**: `tests/output_format_tests.c`

Tasks:
- [ ] Test all three formats
- [ ] Verify stride calculations
- [ ] Validate layout correctness

### Phase 4: Streaming API (Estimated: 3-4 hours)

#### 4.1 Stream Support (2-3 hours)
**File**: `src/gpu/mod.rs`, `src/ffi.rs`

Tasks:
- [ ] Add `wg_generate_batch_stream()` API
- [ ] Implement async kernel launch
- [ ] Handle stream synchronization

#### 4.2 Testing (1 hour)
**File**: `tests/streaming_tests.c`

Tasks:
- [ ] Test multi-stream generation
- [ ] Verify async execution
- [ ] Test overlap scenarios

### Phase 5: Integration Examples (Estimated: 6-8 hours)

#### 5.1 Simple Host Example (1 hour)
**File**: `examples/c_api_simple.c`

Tasks:
- [ ] Demonstrate basic host-side usage
- [ ] Show buffer management
- [ ] Document common patterns

#### 5.2 GPU Hash Example (3-4 hours)
**File**: `examples/c_api_gpu_hash.c`, `examples/hash_kernel.cu`

Tasks:
- [ ] Implement simple MD5 hash kernel
- [ ] Demonstrate zero-copy integration
- [ ] Measure end-to-end performance

#### 5.3 Multi-GPU Example (2-3 hours)
**File**: `examples/c_api_multi_gpu.c`

Tasks:
- [ ] Demonstrate keyspace partitioning
- [ ] Show per-GPU context management
- [ ] Benchmark scaling

### Phase 6: Documentation (Estimated: 8-10 hours)

#### 6.1 API Reference (3-4 hours)
**File**: `docs/C_API_REFERENCE.md`

Tasks:
- [ ] Document all functions
- [ ] Add parameter descriptions
- [ ] Include return value semantics
- [ ] Provide error handling guide

#### 6.2 Integration Guide (3-4 hours)
**File**: `docs/INTEGRATION_GUIDE.md`

Tasks:
- [ ] Write hashcat integration walkthrough
- [ ] Document build/linking process
- [ ] Add troubleshooting section
- [ ] Include performance tuning tips

#### 6.3 Examples Documentation (2 hours)
**File**: `docs/EXAMPLES.md`

Tasks:
- [ ] Explain each example
- [ ] Add build instructions
- [ ] Document expected output
- [ ] Provide modification guides

---

## Total Estimated Effort

| Phase | Hours | Priority |
|-------|-------|----------|
| Phase 1: Core C API | 8-10 | **Critical** |
| Phase 2: Device Pointers | 4-5 | **Critical** |
| Phase 3: Output Formats | 6-8 | High |
| Phase 4: Streaming API | 3-4 | Medium |
| Phase 5: Integration Examples | 6-8 | High |
| Phase 6: Documentation | 8-10 | **Critical** |
| **TOTAL** | **35-45 hours** | |

**Minimum viable library**: Phases 1-2 (12-15 hours)
**Production-ready**: All phases (35-45 hours)

---

## Success Criteria

### Functional Requirements
- ✅ C API compiles without warnings
- ✅ Zero-copy device pointer access works
- ✅ All output formats produce correct results
- ✅ No memory leaks (valgrind clean)
- ✅ Thread-safe for concurrent access

### Performance Requirements
- ✅ Device pointer generation: ≥400 M candidates/s
- ✅ Zero-copy hash integration: No PCIe bottleneck
- ✅ Multi-GPU scaling: >90% efficiency

### Documentation Requirements
- ✅ Every API function documented
- ✅ 3+ working integration examples
- ✅ Troubleshooting guide complete
- ✅ Responsible use disclaimer present

---

## Next Steps

1. **Review this document** - Ensure architecture meets requirements
2. **Get stakeholder approval** - If building for specific use case
3. **Start with Phase 1** - Core C API (minimal viable)
4. **Test early** - Write C test program ASAP
5. **Iterate** - Add phases based on feedback

---

**End of Library Architecture Document**
