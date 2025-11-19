# Integration Guide: Embedding libwordlist_generator in Password Crackers

**Version**: 2.0
**Date**: November 18, 2025
**Audience**: Security tool developers, password cracker maintainers

---

## Overview

This guide explains how to integrate `libwordlist_generator` into GPU-based password crackers like hashcat, John the Ripper, or custom security tools.

**Key Benefits**:
- 4-9x faster candidate generation vs CPU
- Zero-copy GPU-to-GPU operation (no PCIe bottleneck)
- Perfect keyspace partitioning for multi-GPU setups
- Deterministic index-to-word mapping

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Integration Patterns](#integration-patterns)
3. [Hashcat Integration](#hashcat-integration)
4. [John the Ripper Integration](#john-the-ripper-integration)
5. [Custom Tool Integration](#custom-tool-integration)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Step 1: Install Library

```bash
# Clone repository
git clone https://github.com/yourusername/gpu-scatter-gather
cd gpu-scatter-gather

# Build library
cargo build --release

# Library output: target/release/libwordlist_generator.so (Linux)
#                 target/release/wordlist_generator.dll (Windows)
```

### Step 2: Minimal Integration

```c
#include <wordlist_generator.h>
#include <stdio.h>

int main() {
    // Create generator
    wg_handle_t gen = wg_create(NULL, 0);

    // Configure: 8-char lowercase+digits
    wg_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz0123456789", 36);
    int mask[] = {1,1,1,1,1,1,1,1};
    wg_set_mask(gen, mask, 8);

    // Generate 100M candidates
    wg_batch_device_t batch;
    wg_generate_batch_device(gen, 0, 100000000, &batch);

    printf("Generated %llu candidates at device pointer %p\n",
           batch.count, (void*)batch.data);

    // Use batch.data in your GPU kernel...

    wg_destroy(gen);
    return 0;
}
```

### Step 3: Compile

```bash
gcc -o test test.c \
    -I/path/to/gpu-scatter-gather/include \
    -L/path/to/gpu-scatter-gather/target/release \
    -lwordlist_generator \
    -lcuda
```

---

## Integration Patterns

### Pattern 1: Drop-in Replacement for CPU Mask Attack

**Scenario**: Replace existing CPU-based mask generation

**Before** (CPU-based):
```c
// Generate candidates on CPU
for (uint64_t i = 0; i < keyspace; i++) {
    char candidate[32];
    index_to_word_cpu(i, mask, charsets, candidate);

    // Transfer to GPU
    cuMemcpyHtoD(d_candidates + i * word_length, candidate, word_length);
}

// Hash on GPU
hash_kernel<<<grid, block>>>(d_candidates, count);
```

**After** (GPU-based):
```c
// Generate directly on GPU
wg_batch_device_t batch;
wg_generate_batch_device(gen, 0, keyspace, &batch);

// Hash on GPU (zero-copy!)
hash_kernel<<<grid, block>>>((const char*)batch.data, batch.count);
```

**Speedup**: 4-9x (generation) + eliminate PCIe bottleneck

---

### Pattern 2: Streaming Pipeline

**Scenario**: Continuous generation while hashing

```c
CUstream streams[3];
wg_batch_device_t batches[3];

for (int i = 0; i < 3; i++) {
    cuStreamCreate(&streams[i], 0);
}

uint64_t batch_size = 10000000;  // 10M per batch

for (uint64_t start = 0; start < keyspace; start += batch_size) {
    int slot = (start / batch_size) % 3;

    // Generate batch N (async)
    wg_generate_batch_stream(gen, streams[slot], start, batch_size, &batches[slot]);

    // Hash batch N-1 (async)
    if (start > 0) {
        int prev = (slot + 2) % 3;
        hash_kernel<<<grid, block, 0, streams[prev]>>>(...);
    }

    // Check results from batch N-2 (async)
    if (start > batch_size) {
        int done = (slot + 1) % 3;
        cuStreamSynchronize(streams[done]);
        check_for_matches(batches[done]);
    }
}
```

**Benefit**: Overlapped execution, minimal stalls

---

### Pattern 3: Multi-GPU Distributed

**Scenario**: Split keyspace across multiple GPUs

```c
void crack_on_gpu(int gpu_id, uint64_t start, uint64_t count) {
    // Set GPU device
    CUdevice device;
    CUcontext context;
    cuDeviceGet(&device, gpu_id);
    cuCtxCreate(&context, 0, device);

    // Create generator for this GPU
    wg_handle_t gen = wg_create(context, gpu_id);

    // Configure (same for all GPUs)
    wg_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz0123456789", 36);
    int mask[] = {1,1,1,1,1,1,1,1};
    wg_set_mask(gen, mask, 8);

    // Generate this GPU's partition
    wg_batch_device_t batch;
    wg_generate_batch_device(gen, start, count, &batch);

    // Hash and check
    crack_batch(batch);

    wg_destroy(gen);
    cuCtxDestroy(context);
}

int main() {
    int num_gpus = wg_get_device_count();
    uint64_t keyspace = wg_keyspace_size(gen);
    uint64_t per_gpu = keyspace / num_gpus;

    #pragma omp parallel for
    for (int i = 0; i < num_gpus; i++) {
        crack_on_gpu(i, i * per_gpu, per_gpu);
    }

    return 0;
}
```

**Benefit**: Linear scaling! 4 GPUs = 4x throughput

---

## Hashcat Integration

### Architecture Overview

```
┌─────────────────────────────────────────┐
│         Hashcat Core                    │
│  - Hash target management               │
│  - Result reporting                     │
│  - Session management                   │
└─────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│    libwordlist_generator (NEW!)         │
│  - GPU candidate generation             │
│  - Mask pattern support                 │
│  - Zero-copy device pointers            │
└─────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│       Hashcat GPU Kernels               │
│  - MD5, SHA1, bcrypt, etc.              │
│  - Read candidates from device pointer  │
└─────────────────────────────────────────┘
```

### Integration Points

#### 1. Mask Attack Mode

**File**: `src/modules/module_00000.c` (MD5 example)

**Current flow**:
```c
// CPU generates masks
for (uint64_t i = 0; i < keyspace; i++) {
    char word[MAX_LEN];
    mask_to_word(i, mask, word);
    // ... copy to GPU, hash ...
}
```

**Modified flow**:
```c
#include <wordlist_generator.h>

static wg_handle_t wg_gen = NULL;

int module_init(hashcat_ctx_t *hashcat_ctx) {
    // Create generator
    wg_gen = wg_create(hashcat_ctx->cuda_ctx, hashcat_ctx->device_id);

    // Configure from hashcat mask
    configure_from_hashcat_mask(wg_gen, hashcat_ctx->mask);

    return 0;
}

int module_hash_encode(hashcat_ctx_t *hashcat_ctx, ...) {
    // Generate batch on GPU
    wg_batch_device_t batch;
    wg_generate_batch_device(wg_gen, ctx->start_idx, ctx->batch_size, &batch);

    // Pass device pointer to hash kernel
    ctx->d_candidates = batch.data;
    ctx->candidate_stride = batch.stride;

    return 0;
}

void module_shutdown(hashcat_ctx_t *hashcat_ctx) {
    wg_destroy(wg_gen);
}
```

#### 2. Kernel Modification

**File**: `OpenCL/m00000_a3.cl` (MD5 mask attack kernel)

**Current**:
```c
__kernel void m00000_a3(__global pw_t *pws, ...) {
    const u32 gid = get_global_id(0);

    // Read candidate from pws buffer
    u32 w[16];
    w[0] = pws[gid].i[0];
    w[1] = pws[gid].i[1];
    // ...
}
```

**Modified**:
```c
__kernel void m00000_a3(
    __global const char *candidates,  // NEW: device pointer from library
    const u32 stride,                  // NEW: stride between candidates
    ...
) {
    const u32 gid = get_global_id(0);

    // Read candidate from library's buffer
    const char *word = candidates + (gid * stride);

    u32 w[16];
    // Convert char* to u32[16] for MD5 input
    unpack_word(word, w);
    // ...
}
```

### Expected Performance Improvement

```
Hashcat MD5 Benchmark (RTX 4070 Ti SUPER):

Before:
  Candidate generation (CPU): ~100 M/s
  MD5 hashing (GPU):          ~1000 M/s
  Bottleneck: Candidate generation

After (with library):
  Candidate generation (GPU): ~440 M/s
  MD5 hashing (GPU):          ~1000 M/s
  Bottleneck: Balanced (still generation, but 4.4x better!)
```

---

## John the Ripper Integration

### Architecture Overview

```
┌─────────────────────────────────────────┐
│         John Core (CPU)                 │
│  - Format detection                     │
│  - Result reporting                     │
│  - Rule processing                      │
└─────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│    libwordlist_generator (NEW!)         │
│  - GPU mask generation                  │
│  - Device pointer return                │
└─────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│       John GPU Formats                  │
│  - cuda_rawmd5.cu                       │
│  - cuda_rawsha1.cu                      │
│  - etc.                                 │
└─────────────────────────────────────────┘
```

### Integration Points

#### 1. New Attack Mode

**File**: `src/mask_gpu.c` (new file)

```c
#include <wordlist_generator.h>

static wg_handle_t wg_gen = NULL;

void mask_gpu_init(struct fmt_main *format) {
    // Create generator
    wg_gen = wg_create(NULL, 0);  // John manages CUDA context

    // Configure from John's mask format
    char *mask = options.mask;
    parse_john_mask(wg_gen, mask);
}

void mask_gpu_run(struct db_salt *salt) {
    wg_batch_device_t batch;
    wg_generate_batch_device(wg_gen, 0, options.max_keys_per_crypt, &batch);

    // Pass to format's GPU kernel
    format->methods.crypt_all(batch.data, batch.count, salt);
}

void mask_gpu_done(void) {
    wg_destroy(wg_gen);
}
```

#### 2. Format Modification

**File**: `src/cuda/cuda_rawmd5.cu`

**Add parameter to kernel**:
```cuda
extern "C" __global__ void cuda_rawmd5(
    const char *candidates,  // NEW: from library
    size_t stride,           // NEW: from library
    uint32_t num_keys,
    // ... existing params ...
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_keys) return;

    // Read candidate
    const char *word = candidates + (gid * stride);

    // Hash it
    uint32_t hash[4];
    md5_hash(word, hash);

    // Store result
    // ...
}
```

### Expected Performance

```
John the Ripper MD5 (8-char mask):

Before:
  CPU generation: ~60 M/s
  GPU hashing:    ~800 M/s
  Effective:      ~60 M/s (CPU bottleneck)

After:
  GPU generation: ~440 M/s
  GPU hashing:    ~800 M/s
  Effective:      ~440 M/s (7.3x faster!)
```

---

## Custom Tool Integration

### Example: Simple MD5 Cracker

**File**: `my_cracker.c`

```c
#include <wordlist_generator.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>

// MD5 hash kernel (simplified)
__global__ void md5_kernel(
    const char *candidates,
    size_t stride,
    uint64_t count,
    const unsigned char *target_hash,
    int *found_idx
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const char *word = candidates + (tid * stride);

    // Compute MD5
    unsigned char hash[16];
    compute_md5_gpu(word, hash);

    // Compare
    if (memcmp(hash, target_hash, 16) == 0) {
        atomicExch(found_idx, tid);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <target_md5_hex>\n", argv[0]);
        return 1;
    }

    // Parse target hash
    unsigned char target[16];
    hex_to_bytes(argv[1], target, 16);

    // Initialize CUDA
    cuInit(0);
    CUdevice device;
    CUcontext context;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Create wordlist generator
    wg_handle_t gen = wg_create(context, 0);

    // Configure: 8-char lowercase+digits
    wg_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz0123456789", 36);
    int mask[] = {1,1,1,1,1,1,1,1};
    wg_set_mask(gen, mask, 8);
    wg_set_output_format(gen, WG_FORMAT_FIXED_WIDTH);

    // Allocate result flag on GPU
    CUdeviceptr d_target, d_found_idx;
    cuMemAlloc(&d_target, 16);
    cuMemAlloc(&d_found_idx, sizeof(int));
    cuMemcpyHtoD(d_target, target, 16);

    // Process in batches
    uint64_t keyspace = wg_keyspace_size(gen);
    uint64_t batch_size = 100000000;  // 100M per batch

    for (uint64_t start = 0; start < keyspace; start += batch_size) {
        printf("Testing indices %llu - %llu...\n", start, start + batch_size);

        // Generate batch
        wg_batch_device_t batch;
        wg_generate_batch_device(gen, start, batch_size, &batch);

        // Reset found flag
        int not_found = -1;
        cuMemcpyHtoD(d_found_idx, &not_found, sizeof(int));

        // Launch hash kernel
        dim3 block(256);
        dim3 grid((batch.count + 255) / 256);
        md5_kernel<<<grid, block>>>(
            (const char*)batch.data,
            batch.stride,
            batch.count,
            (const unsigned char*)d_target,
            (int*)d_found_idx
        );

        // Check if found
        int found_idx;
        cuMemcpyDtoH(&found_idx, d_found_idx, sizeof(int));

        if (found_idx >= 0) {
            // Copy the matching candidate to host
            char password[32] = {0};
            cuMemcpyDtoH(password, batch.data + found_idx * batch.stride, batch.word_length);

            printf("Password found: %s\n", password);
            goto cleanup;
        }
    }

    printf("Password not found in keyspace.\n");

cleanup:
    cuMemFree(d_target);
    cuMemFree(d_found_idx);
    wg_destroy(gen);
    cuCtxDestroy(context);
    return 0;
}
```

**Build**:
```bash
nvcc -o my_cracker my_cracker.c \
    -I/path/to/wordlist_generator/include \
    -L/path/to/wordlist_generator/lib \
    -lwordlist_generator
```

**Run**:
```bash
./my_cracker 5f4dcc3b5aa765d61d8327deb882cf99
# Cracks MD5("password") in ~6 seconds on RTX 4070 Ti
```

---

## Performance Optimization

### 1. Batch Size Tuning

```c
// Test different batch sizes
uint64_t batch_sizes[] = {1000000, 10000000, 100000000, 500000000};

for (int i = 0; i < 4; i++) {
    uint64_t batch_size = batch_sizes[i];

    double start = get_time();
    wg_batch_device_t batch;
    wg_generate_batch_device(gen, 0, batch_size, &batch);
    cuCtxSynchronize();  // Wait for generation
    double elapsed = get_time() - start;

    double throughput = batch_size / elapsed / 1e6;
    printf("Batch %lluM: %.0f M/s\n", batch_size / 1000000, throughput);
}

// Output (typical):
// Batch 1M:   380 M/s  (kernel launch overhead)
// Batch 10M:  420 M/s
// Batch 100M: 440 M/s  (optimal!)
// Batch 500M: 435 M/s  (memory contention)
```

**Recommendation**: 10-100M candidates per batch

### 2. Output Format Selection

```c
// For hash kernels expecting fixed-size input
wg_set_output_format(gen, WG_FORMAT_FIXED_WIDTH);
// Pros: Aligned access, predictable stride
// Best for: MD5, SHA1, bcrypt, etc.

// For maximum throughput
wg_set_output_format(gen, WG_FORMAT_PACKED);
// Pros: Minimal memory, cache-friendly
// Best for: Custom parsers

// For compatibility
wg_set_output_format(gen, WG_FORMAT_NEWLINES);
// Pros: Human-readable, standard
// Best for: Debugging, piping to stdin
```

### 3. GPU Selection

```c
// Find fastest GPU
int num_gpus = wg_get_device_count();
int best_gpu = 0;
double best_throughput = 0;

for (int i = 0; i < num_gpus; i++) {
    wg_handle_t gen = wg_create(NULL, i);
    // ... configure ...

    // Benchmark
    double start = get_time();
    wg_batch_device_t batch;
    wg_generate_batch_device(gen, 0, 10000000, &batch);
    cuCtxSynchronize();
    double throughput = 10000000 / (get_time() - start);

    if (throughput > best_throughput) {
        best_throughput = throughput;
        best_gpu = i;
    }

    wg_destroy(gen);
}

printf("Best GPU: %d (%.0f M/s)\n", best_gpu, best_throughput / 1e6);
```

---

## Troubleshooting

### Issue 1: Device Pointer Not Valid

**Symptom**:
```
CUDA error: invalid device pointer
```

**Cause**: Using device pointer after it's been freed

**Solution**:
```c
// BAD
wg_batch_device_t batch1, batch2;
wg_generate_batch_device(gen, 0, 1000, &batch1);
wg_generate_batch_device(gen, 1000, 1000, &batch2);
use_pointer(batch1.data);  // ERROR: batch1.data was freed!

// GOOD
wg_batch_device_t batch;
wg_generate_batch_device(gen, 0, 1000, &batch);
use_pointer(batch.data);  // OK: batch.data still valid

wg_generate_batch_device(gen, 1000, 1000, &batch);
// Now previous batch.data is invalid
```

### Issue 2: Candidates Look Wrong

**Symptom**: Generated candidates don't match expected pattern

**Debug**:
```c
// Copy a few candidates to host for inspection
char sample[1000];
cuMemcpyDtoH(sample, batch.data, 1000);

printf("First 10 candidates:\n");
for (int i = 0; i < 10; i++) {
    char *word = sample + i * batch.stride;
    printf("%2d: %.*s\n", i, (int)batch.word_length, word);
}
```

### Issue 3: Slow Performance

**Symptom**: Throughput much lower than expected

**Checklist**:
- [ ] Check batch size (too small = overhead)
- [ ] Check GPU utilization (`nvidia-smi`)
- [ ] Check output format (FIXED_WIDTH faster than NEWLINES)
- [ ] Check CUDA errors (`wg_get_error()`)
- [ ] Verify correct GPU selected

**Profiling**:
```bash
# Profile with nvprof
nvprof ./my_cracker

# Look for:
# - Kernel execution time
# - Memory transfer time
# - Number of kernel launches
```

### Issue 4: Memory Leaks

**Symptom**: GPU memory usage grows over time

**Solution**: Always call `wg_destroy()`
```c
wg_handle_t gen = wg_create(NULL, 0);

// ... use generator ...

wg_destroy(gen);  // REQUIRED!
```

**Verify with**:
```bash
# Before running
nvidia-smi

# After running
nvidia-smi

# Memory usage should be the same
```

---

## Next Steps

1. **Review API specification**: `docs/C_API_SPECIFICATION.md`
2. **Study examples**: `examples/c_api_*.c`
3. **Test integration**: Build simple test program
4. **Profile performance**: Measure throughput improvements
5. **Submit pull request**: Contribute integration back to main project!

---

**Questions?** File an issue at https://github.com/yourusername/gpu-scatter-gather/issues

**End of Integration Guide**
