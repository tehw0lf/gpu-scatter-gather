# Hashcat Integration Guide

**Version**: 1.0
**Date**: November 20, 2025
**Target Audience**: Hashcat developers, security tool maintainers, custom password cracker builders

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Integration Pattern 1: Host API for Piping](#integration-pattern-1-host-api-for-piping)
4. [Integration Pattern 2: Device Pointer API for Zero-Copy](#integration-pattern-2-device-pointer-api-for-zero-copy)
5. [Integration Pattern 3: Streaming API for Async](#integration-pattern-3-streaming-api-for-async)
6. [Performance Tuning](#performance-tuning)
7. [Complete Working Example](#complete-working-example)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

### Purpose

This guide demonstrates how to integrate `libgpu_scatter_gather` into hashcat or hashcat-based tools to dramatically accelerate candidate generation for mask attacks.

### Use Cases

- **Pre-generation**: Generate wordlists faster than `maskprocessor` for piping to hashcat
- **Direct integration**: Embed library into hashcat for zero-copy GPU-to-GPU operation
- **Custom tools**: Build custom password crackers leveraging hashcat's hash implementations

### Performance Advantages

| Tool | Throughput (8-char) | vs maskprocessor |
|------|---------------------|------------------|
| maskprocessor (CPU) | ~142 M/s | 1.0× (baseline) |
| cracken (CPU) | ~178 M/s | 1.25× |
| **gpu-scatter-gather** | **702 M/s** | **4.9×** |

**Key Benefits:**
- 4-7× faster candidate generation than CPU tools
- Zero-copy operation when integrated directly (no PCIe overhead)
- Perfect keyspace partitioning for distributed cracking
- Deterministic index-to-word mapping enables resume support

---

## Quick Start

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

# Header: include/wordlist_generator.h
```

### Replace maskprocessor Example

**Before (using maskprocessor):**
```bash
maskprocessor -1 ?l?u -2 ?d ?1?1?1?1?2?2?2?2 > wordlist.txt
hashcat -m 0 -a 0 hashes.txt wordlist.txt
```

**After (using gpu-scatter-gather):**
```c
#include "wordlist_generator.h"
#include <stdio.h>

int main() {
    // Create generator
    wg_WordlistGenerator *gen = wg_create();
    if (!gen) return 1;

    // Define charsets (equivalent to maskprocessor -1 ?l?u -2 ?d)
    char alpha[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    char digits[] = "0123456789";
    wg_add_charset(gen, 1, alpha, 52);
    wg_add_charset(gen, 2, digits, 10);

    // Set mask: 4 alpha + 4 digits (?1?1?1?1?2?2?2?2)
    wg_set_mask(gen, "?1?1?1?1?2?2?2?2");

    // Use PACKED format for best performance
    wg_set_format(gen, WG_FORMAT_PACKED);

    // Generate to file
    FILE *fp = fopen("wordlist.txt", "wb");
    // ... (see examples below for generation loop)

    fclose(fp);
    wg_destroy(gen);
    return 0;
}
```

**Performance:** ~5× faster than maskprocessor

---

## Integration Pattern 1: Host API for Piping

### Use Case

Generate wordlists and pipe directly to hashcat's stdin.

```bash
./my_generator | hashcat -m 0 hashes.txt
```

### Complete Example

```c
#include "wordlist_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

volatile sig_atomic_t should_stop = 0;

void sigint_handler(int sig) {
    should_stop = 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <mask>\n", argv[0]);
        fprintf(stderr, "Example: %s '?l?l?l?l?d?d?d?d'\n", argv[0]);
        return 1;
    }

    // Handle Ctrl+C gracefully
    signal(SIGINT, sigint_handler);
    signal(SIGPIPE, SIG_IGN);

    // Create generator
    wg_WordlistGenerator *gen = wg_create();
    if (!gen) {
        fprintf(stderr, "Failed to create generator\n");
        return 1;
    }

    // Define standard hashcat charsets
    wg_add_charset(gen, 'l', "abcdefghijklmnopqrstuvwxyz", 26);
    wg_add_charset(gen, 'u', "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 26);
    wg_add_charset(gen, 'd', "0123456789", 10);
    wg_add_charset(gen, 's', " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", 33);

    // Set mask from command line
    if (wg_set_mask(gen, argv[1]) != 0) {
        fprintf(stderr, "Invalid mask: %s\n", argv[1]);
        wg_destroy(gen);
        return 1;
    }

    // IMPORTANT: Use NEWLINES format for hashcat compatibility
    wg_set_format(gen, WG_FORMAT_NEWLINES);

    // Get keyspace size
    uint64_t keyspace = wg_keyspace_size(gen);
    fprintf(stderr, "Keyspace: %llu candidates\n", keyspace);

    // Calculate buffer size (10M candidates per batch)
    const uint64_t batch_size = 10000000;
    size_t buffer_size = wg_calculate_buffer_size(gen, batch_size);
    char *buffer = malloc(buffer_size);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate buffer\n");
        wg_destroy(gen);
        return 1;
    }

    // Generate in batches
    uint64_t total_generated = 0;
    for (uint64_t start = 0; start < keyspace && !should_stop; start += batch_size) {
        uint64_t count = (start + batch_size > keyspace) ?
                         (keyspace - start) : batch_size;

        // Generate batch
        int result = wg_generate_batch_host(gen, start, count,
                                            buffer, buffer_size);
        if (result != 0) {
            fprintf(stderr, "Generation failed at index %llu\n", start);
            break;
        }

        // Write to stdout (for piping to hashcat)
        size_t bytes_written = fwrite(buffer, 1, buffer_size, stdout);
        if (bytes_written != buffer_size) {
            if (ferror(stdout)) {
                fprintf(stderr, "Write error (pipe closed?)\n");
            }
            break;
        }

        // Flush to ensure hashcat receives data immediately
        fflush(stdout);

        total_generated += count;
        fprintf(stderr, "\rProgress: %llu/%llu (%.1f%%)",
                total_generated, keyspace,
                100.0 * total_generated / keyspace);
    }

    fprintf(stderr, "\nCompleted: %llu candidates generated\n", total_generated);

    free(buffer);
    wg_destroy(gen);
    return 0;
}
```

### Compilation

```bash
gcc -o hashcat_generator hashcat_generator.c \
    -I../include \
    -L../target/release \
    -lgpu_scatter_gather \
    -Wl,-rpath,../target/release \
    -O3
```

### Usage

```bash
# Generate and pipe to hashcat
./hashcat_generator '?l?l?l?l?l?l?l?l' | hashcat -m 0 hashes.txt

# Generate to file
./hashcat_generator '?l?l?l?l?d?d?d?d' > wordlist.txt

# Then use with hashcat
hashcat -m 0 -a 0 hashes.txt wordlist.txt
```

### Key Points

- **Format:** Must use `WG_FORMAT_NEWLINES` for hashcat stdin compatibility
- **Batch size:** 10-100M candidates balances speed and memory
- **Flushing:** Call `fflush(stdout)` after each batch for streaming
- **Signal handling:** Gracefully handle Ctrl+C and SIGPIPE
- **Progress:** Write progress to stderr (stdout is for wordlist data)

---

## Integration Pattern 2: Device Pointer API for Zero-Copy

### Use Case

Integrate directly into hashcat modules for GPU-to-GPU operation with **zero PCIe overhead**.

### Architecture

```
┌────────────────────────────────────┐
│  libgpu_scatter_gather             │
│  - Generates candidates on GPU     │
│  - Returns CUdeviceptr             │
└────────────────┬───────────────────┘
                 │ (zero-copy)
                 ▼
┌────────────────────────────────────┐
│  Hashcat Hash Kernel               │
│  - Reads from device pointer       │
│  - No PCIe transfer!               │
└────────────────────────────────────┘
```

### Example: Custom Hashcat Module

```c
#include "wordlist_generator.h"
#include <cuda.h>
#include <stdio.h>

// Your hash kernel (example: MD5)
__global__ void md5_kernel_packed(
    const char *candidates,    // Device pointer from library
    size_t stride,             // Bytes between candidates
    size_t word_length,        // Length of each word
    uint64_t count,            // Number of candidates
    const uint8_t *target,     // Target hash
    int *found_index           // Output: index of match (-1 if none)
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    // Read candidate (packed format: no separators)
    const char *word = candidates + (tid * stride);

    // Compute MD5
    uint8_t hash[16];
    md5_gpu(word, word_length, hash);

    // Compare
    bool match = true;
    for (int i = 0; i < 16; i++) {
        if (hash[i] != target[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        atomicExch(found_index, (int)tid);
    }
}

int crack_md5_with_mask(const char *mask, const uint8_t *target_hash) {
    // Initialize CUDA
    cuInit(0);
    CUdevice device;
    CUcontext context;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Create generator with our CUDA context
    wg_WordlistGenerator *gen = wg_create_with_context(context);
    if (!gen) {
        fprintf(stderr, "Failed to create generator\n");
        return -1;
    }

    // Configure charsets and mask
    wg_add_charset(gen, 'l', "abcdefghijklmnopqrstuvwxyz", 26);
    wg_add_charset(gen, 'd', "0123456789", 10);
    wg_set_mask(gen, mask);

    // IMPORTANT: Use PACKED format for best performance
    // - 11% memory savings vs NEWLINES
    // - Better cache utilization
    // - Stride = word_length (no separators)
    wg_set_format(gen, WG_FORMAT_PACKED);

    uint64_t keyspace = wg_keyspace_size(gen);
    fprintf(stderr, "Cracking MD5 with mask %s (%llu candidates)\n",
            mask, keyspace);

    // Allocate device memory for target and result
    CUdeviceptr d_target, d_found_index;
    cuMemAlloc(&d_target, 16);
    cuMemAlloc(&d_found_index, sizeof(int));
    cuMemcpyHtoD(d_target, target_hash, 16);

    // Process in batches
    const uint64_t batch_size = 100000000;  // 100M per batch
    int found_idx = -1;

    for (uint64_t start = 0; start < keyspace && found_idx < 0;
         start += batch_size) {

        uint64_t count = (start + batch_size > keyspace) ?
                         (keyspace - start) : batch_size;

        // Generate batch on GPU (zero-copy!)
        wg_BatchDevice batch;
        if (wg_generate_batch_device(gen, start, count, &batch) != 0) {
            fprintf(stderr, "Generation failed\n");
            break;
        }

        fprintf(stderr, "\rTesting batch at index %llu...", start);

        // Reset found flag
        int not_found = -1;
        cuMemcpyHtoD(d_found_index, &not_found, sizeof(int));

        // Launch hash kernel (reads directly from batch.data)
        dim3 block(256);
        dim3 grid((count + 255) / 256);

        md5_kernel_packed<<<grid, block>>>(
            (const char*)batch.data,     // Device pointer (zero-copy!)
            batch.stride,                 // = word_length (PACKED format)
            batch.word_length,
            batch.count,
            (const uint8_t*)d_target,
            (int*)d_found_index
        );

        cuCtxSynchronize();

        // Check result
        cuMemcpyDtoH(&found_idx, d_found_index, sizeof(int));

        if (found_idx >= 0) {
            // Copy the matching candidate
            char password[64] = {0};
            cuMemcpyDtoH(password, batch.data + found_idx * batch.stride,
                         batch.word_length);

            fprintf(stderr, "\n\nPassword found: %.*s\n",
                    (int)batch.word_length, password);
            break;
        }
    }

    if (found_idx < 0) {
        fprintf(stderr, "\n\nPassword not found in keyspace.\n");
    }

    // Cleanup
    cuMemFree(d_target);
    cuMemFree(d_found_index);
    wg_destroy(gen);
    cuCtxDestroy(context);

    return found_idx >= 0 ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <mask> <md5_hex>\n", argv[0]);
        fprintf(stderr, "Example: %s '?l?l?l?l?l?l?l?l' 5f4dcc3b5aa765d61d8327deb882cf99\n",
                argv[0]);
        return 1;
    }

    // Parse target hash
    uint8_t target[16];
    // ... (hex parsing code)

    return crack_md5_with_mask(argv[1], target);
}
```

### Key Points

- **Format:** Use `WG_FORMAT_PACKED` for optimal performance
  - 11% memory savings vs NEWLINES
  - `stride = word_length` (no separators between words)
- **Context sharing:** Pass your CUDA context to `wg_create_with_context()`
- **Zero-copy:** Device pointer is directly usable in kernels
- **Synchronization:** Call `cuCtxSynchronize()` before reading results
- **Memory lifetime:** Device pointer valid until next `wg_generate_batch_device()` call

### Performance Advantage

```
Host API (Pattern 1):
  Generation: 702 M/s
  PCIe transfer: ~16 GB/s = ~2000 M/s for 8-byte words
  Bottleneck: Generation (702 M/s)

Device API (Pattern 2):
  Generation: 702 M/s
  PCIe transfer: 0 (zero-copy!)
  Bottleneck: Hash computation or generation

Speedup: No PCIe overhead = 100% of GPU bandwidth available for hashing
```

---

## Integration Pattern 3: Streaming API for Async

### Use Case

Overlap candidate generation with hash computation using CUDA streams for maximum throughput.

### Architecture

```
Stream 1: [Generate Batch 1] → [Hash Batch 1] → [Check Results 1]
                                      ↓
Stream 2:                    [Generate Batch 2] → [Hash Batch 2] → ...
```

### Example: Double-Buffered Pipeline

```c
#include "wordlist_generator.h"
#include <cuda_runtime.h>
#include <stdio.h>

void crack_with_streaming(wg_WordlistGenerator *gen,
                         const uint8_t *target_hash) {
    // Create two CUDA streams for double-buffering
    CUstream stream1, stream2;
    cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING);
    cuStreamCreate(&stream2, CU_STREAM_NON_BLOCKING);

    uint64_t keyspace = wg_keyspace_size(gen);
    const uint64_t batch_size = 50000000;  // 50M per batch

    // Allocate device memory for target hash
    CUdeviceptr d_target, d_found_index;
    cuMemAlloc(&d_target, 16);
    cuMemAlloc(&d_found_index, sizeof(int));
    cuMemcpyHtoD(d_target, target_hash, 16);

    wg_BatchDevice batch1 = {0}, batch2 = {0};
    CUstream streams[] = {stream1, stream2};
    wg_BatchDevice *batches[] = {&batch1, &batch2};

    int found_idx = -1;
    uint64_t start = 0;

    // Prime the pipeline: start first generation
    if (start < keyspace) {
        wg_generate_batch_stream(gen, stream1, start,
                                 batch_size, &batch1);
        start += batch_size;
    }

    // Pipeline loop
    int current = 1;  // Next stream to use
    while (start < keyspace && found_idx < 0) {
        CUstream current_stream = streams[current];
        wg_BatchDevice *current_batch = batches[current];
        CUstream prev_stream = streams[1 - current];
        wg_BatchDevice *prev_batch = batches[1 - current];

        // Start generating next batch (async)
        uint64_t count = (start + batch_size > keyspace) ?
                         (keyspace - start) : batch_size;
        wg_generate_batch_stream(gen, current_stream, start,
                                 count, current_batch);

        // Wait for previous batch to finish generating
        cuStreamSynchronize(prev_stream);

        // Hash previous batch (async on previous stream)
        dim3 block(256);
        dim3 grid((prev_batch->count + 255) / 256);

        int not_found = -1;
        cuMemcpyHtoDAsync(d_found_index, &not_found, sizeof(int), prev_stream);

        md5_kernel_packed<<<grid, block, 0, prev_stream>>>(
            (const char*)prev_batch->data,
            prev_batch->stride,
            prev_batch->word_length,
            prev_batch->count,
            (const uint8_t*)d_target,
            (int*)d_found_index
        );

        // Check result from previous batch
        cuStreamSynchronize(prev_stream);
        cuMemcpyDtoH(&found_idx, d_found_index, sizeof(int));

        if (found_idx >= 0) {
            // Found! Copy the password
            char password[64] = {0};
            cuMemcpyDtoH(password,
                         prev_batch->data + found_idx * prev_batch->stride,
                         prev_batch->word_length);
            printf("Password found: %.*s\n",
                   (int)prev_batch->word_length, password);
            break;
        }

        start += batch_size;
        current = 1 - current;  // Swap streams
    }

    // Cleanup
    cuStreamDestroy(stream1);
    cuStreamDestroy(stream2);
    cuMemFree(d_target);
    cuMemFree(d_found_index);
}
```

### Key Points

- **Streams:** Use 2+ streams for overlapped execution
- **Synchronization:** `cuStreamSynchronize()` before accessing results
- **Non-blocking:** Create streams with `CU_STREAM_NON_BLOCKING`
- **Batch size:** Smaller batches (10-50M) for better overlap
- **Default stream:** Passing `NULL` stream = synchronous (default stream)

### Performance Advantage

```
Without streaming:
  Time = T_generate + T_hash + T_check

With double-buffering:
  Time ≈ max(T_generate, T_hash) + T_check

Speedup: Up to 2× if generation and hashing are balanced
```

---

## Performance Tuning

### Batch Size Recommendations

| Password Length | Batch Size | Memory Usage | Notes |
|-----------------|------------|--------------|-------|
| 6 chars | 100-200M | 600-1200 MB | Maximize throughput |
| 8 chars | 100M | 800 MB | Optimal for most GPUs |
| 10 chars | 50-100M | 500-1000 MB | Balance speed/memory |
| 12 chars | 50M | 600 MB | Watch GPU memory |
| 16 chars | 25-50M | 400-800 MB | Avoid OOM |

**Rule of thumb:** `batch_size * word_length < 1 GB`

### Format Mode Selection

| Use Case | Format | Stride | Memory Overhead | Best For |
|----------|--------|--------|-----------------|----------|
| Piping to hashcat stdin | `NEWLINES` | `word_length + 1` | +12.5% (8-char) | Compatibility |
| Direct GPU kernel access | `PACKED` | `word_length` | 0% (optimal) | Performance |
| Fixed-width text processing | `FIXED_WIDTH` | `word_length` | 0% (null-padded) | Parsing |

**Recommendation:**
- **Pattern 1 (piping):** `WG_FORMAT_NEWLINES` (required by hashcat)
- **Pattern 2 (device API):** `WG_FORMAT_PACKED` (11% faster)
- **Pattern 3 (streaming):** `WG_FORMAT_PACKED` (best throughput)

### Common Pitfalls

❌ **Don't:** Use PACKED format when piping to hashcat stdin
```c
wg_set_format(gen, WG_FORMAT_PACKED);
// Then write to stdout for hashcat → WRONG! Hashcat expects newlines
```

✅ **Do:** Use NEWLINES format for hashcat stdin
```c
wg_set_format(gen, WG_FORMAT_NEWLINES);
fwrite(buffer, 1, buffer_size, stdout);  // Correct!
```

❌ **Don't:** Exceed GPU memory with oversized batches
```c
uint64_t batch_size = 1000000000;  // 1B × 12 bytes = 12 GB → OOM!
```

✅ **Do:** Check memory requirements first
```c
size_t buffer_size = wg_calculate_buffer_size(gen, batch_size);
fprintf(stderr, "Buffer size: %.2f MB\n", buffer_size / 1024.0 / 1024.0);
```

❌ **Don't:** Use device pointer after next generation call
```c
wg_generate_batch_device(gen, 0, 1000, &batch1);
wg_generate_batch_device(gen, 1000, 1000, &batch2);
use_pointer(batch1.data);  // ERROR: batch1.data was freed!
```

✅ **Do:** Use device pointer immediately
```c
wg_generate_batch_device(gen, 0, 1000, &batch);
hash_kernel<<<grid, block>>>(batch.data, ...);  // Use immediately
cuCtxSynchronize();  // Finish before next generation
```

---

## Complete Working Example

See `examples/hashcat_pipeline.c` in the repository for a complete, compilable example demonstrating:

- Generator setup with hashcat-style charsets
- Batch generation loop with progress reporting
- Writing to file or stdout
- Error handling and cleanup
- Performance monitoring

**Build:**
```bash
gcc -o hashcat_pipeline hashcat_pipeline.c \
    -I../include \
    -L../target/release \
    -lgpu_scatter_gather \
    -Wl,-rpath,../target/release \
    -O3
```

**Usage:**
```bash
# Generate to file
./hashcat_pipeline '?l?l?l?l?l?l?l?l' > wordlist.txt

# Pipe to hashcat
./hashcat_pipeline '?l?l?l?l?d?d?d?d' | hashcat -m 0 hashes.txt

# Benchmark mode
./hashcat_pipeline '?l?l?l?l?l?l?l?l' > /dev/null
```

---

## Troubleshooting

### Issue 1: "Buffer too small" Error

**Symptom:**
```
Error: Buffer size (800000000) too small, need 900000000
```

**Cause:** Calculated buffer size incorrect or batch size changed

**Solution:**
```c
// Always recalculate buffer size after changing format
wg_set_format(gen, WG_FORMAT_NEWLINES);
size_t buffer_size = wg_calculate_buffer_size(gen, batch_size);
buffer = realloc(buffer, buffer_size);
```

### Issue 2: Hashcat Doesn't Read Words

**Symptom:** Hashcat shows "0 words" or hangs

**Cause:** Using PACKED format instead of NEWLINES

**Solution:**
```c
// For piping to hashcat, MUST use NEWLINES
wg_set_format(gen, WG_FORMAT_NEWLINES);
```

### Issue 3: Segmentation Fault

**Symptom:** Crash when generating

**Cause:** NULL generator handle or uninitialized charset

**Solution:**
```c
wg_WordlistGenerator *gen = wg_create();
if (!gen) {
    fprintf(stderr, "Failed to create generator\n");
    return 1;
}

// Always check return values
if (wg_add_charset(gen, 'l', "abc...", 26) != 0) {
    fprintf(stderr, "Failed to add charset\n");
    wg_destroy(gen);
    return 1;
}
```

### Issue 4: Lower Performance Than Expected

**Symptom:** Throughput much lower than 400-700 M/s

**Checklist:**
- [ ] Check batch size (too small = kernel launch overhead)
- [ ] Check GPU utilization with `nvidia-smi` (should be >90%)
- [ ] Verify PACKED format for device API (11% faster)
- [ ] Ensure no PCIe transfers in hot path
- [ ] Check CUDA compute capability (need ≥7.5 for optimal performance)

**Profiling:**
```bash
# Quick profiling
nvprof --print-gpu-trace ./your_program

# Detailed analysis
ncu --set full ./your_program
```

### Issue 5: Words Don't Match Expected Pattern

**Symptom:** Generated candidates incorrect

**Debug:**
```c
// Copy first 10 candidates to host for inspection
char sample[1000];
wg_generate_batch_host(gen, 0, 10, sample, 1000);

printf("First 10 candidates:\n");
for (int i = 0; i < 10; i++) {
    // Adjust offset based on format
    size_t offset = i * (word_length + 1);  // For NEWLINES
    printf("%2d: %.*s\n", i, word_length, sample + offset);
}
```

### Issue 6: CUDA Out of Memory

**Symptom:**
```
CUDA error: out of memory
```

**Solution:** Reduce batch size
```c
// Check available GPU memory
size_t free_mem, total_mem;
cuMemGetInfo(&free_mem, &total_mem);
fprintf(stderr, "GPU memory: %.2f GB free / %.2f GB total\n",
        free_mem / 1e9, total_mem / 1e9);

// Adjust batch size to use ~50% of free memory
uint64_t max_batch = (free_mem / 2) / (word_length + 1);
batch_size = (batch_size > max_batch) ? max_batch : batch_size;
```

---

## Next Steps

1. **Study the complete example:** `examples/hashcat_pipeline.c`
2. **Read C API specification:** `docs/api/C_API_SPECIFICATION.md`
3. **Profile your integration:** Use `nvprof` or `nsys` to identify bottlenecks
4. **Contribute back:** Share your hashcat integration via pull request!

---

## Additional Resources

- **Generic integration guide:** `docs/guides/INTEGRATION_GUIDE.md`
- **John the Ripper integration:** `docs/guides/JTR_INTEGRATION.md`
- **Performance benchmarks:** `docs/benchmarking/PERFORMANCE_COMPARISON.md`
- **Formal specification:** `docs/design/FORMAL_SPECIFICATION.md`

---

**Questions?** File an issue at https://github.com/tehw0lf/gpu-scatter-gather/issues

**End of Hashcat Integration Guide**
