# C API Specification: libwordlist_generator

**Version**: 2.0
**Date**: November 19, 2025
**Status**: ✅ PRODUCTION READY (All Phases Complete)

---

## Implementation Status

| Phase | Status | Functions | Description |
|-------|--------|-----------|-------------|
| **Phase 1** | ✅ **COMPLETE** | 8 functions | Host memory API (baseline) |
| **Phase 2** | ✅ **COMPLETE** | 3 functions | Device pointer API (zero-copy) |
| **Phase 3** | ✅ **COMPLETE** | 1 function | Output format modes |
| **Phase 4** | ✅ **COMPLETE** | 1 function | Streaming API (async) |
| **Phase 5** | ✅ **COMPLETE** | 4 functions | Utility functions (includes device enumeration) |

**Total Implemented**: 17 functions (ALL PHASES COMPLETE + Device Enumeration)
**Test Coverage**: 16/16 tests passing
**Documentation**: Complete (see docs/api/PHASE*_SUMMARY.md)

🎉 **LIBRARY FEATURE-COMPLETE AND PRODUCTION-READY** 🎉

---

## Overview

This document specifies the complete C API for `libwordlist_generator`, a high-performance GPU-accelerated wordlist generation library designed for integration into password crackers and security tools.

---

## Header File

**Location**: `include/wordlist_generator.h`

```c
#ifndef WORDLIST_GENERATOR_H
#define WORDLIST_GENERATOR_H

#include <stddef.h>
#include <stdint.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 * Types and Constants
 * ===========================================================================
 */

/**
 * Opaque handle to wordlist generator instance
 */
typedef struct WordlistGenerator* wg_handle_t;

/**
 * Output format modes
 */
typedef enum {
    WG_FORMAT_NEWLINES = 0,     /**< Each word terminated by '\n' */
    WG_FORMAT_FIXED_WIDTH = 1,  /**< Fixed-width, null-padded */
    WG_FORMAT_PACKED = 2,       /**< No separators, packed */
} wg_output_format_t;

/**
 * Error codes
 */
typedef enum {
    WG_SUCCESS = 0,              /**< Operation successful */
    WG_ERROR_INVALID_HANDLE = -1,/**< NULL or invalid handle */
    WG_ERROR_INVALID_PARAM = -2, /**< Invalid parameter */
    WG_ERROR_CUDA = -3,          /**< CUDA runtime error */
    WG_ERROR_OUT_OF_MEMORY = -4, /**< Allocation failed */
    WG_ERROR_NOT_CONFIGURED = -5,/**< Generator not configured */
    WG_ERROR_BUFFER_TOO_SMALL = -6,/**< Output buffer too small */
    WG_ERROR_KEYSPACE_OVERFLOW = -7,/**< Index out of keyspace */
} wg_error_t;

/**
 * Device batch result
 */
typedef struct {
    CUdeviceptr data;           /**< Device pointer to candidates */
    uint64_t count;             /**< Number of candidates */
    size_t word_length;         /**< Length of each word (chars) */
    size_t stride;              /**< Bytes between word starts */
    size_t total_bytes;         /**< Total buffer size */
    wg_output_format_t format;  /**< Format used */
} wg_batch_device_t;

/*
 * ===========================================================================
 * Initialization and Teardown
 * ===========================================================================
 */

/**
 * Create a new wordlist generator instance
 *
 * @param ctx CUDA context to use (NULL = create new context)
 * @param device_id CUDA device ID (e.g., 0 for first GPU)
 * @return Generator handle, or NULL on error
 *
 * @note If ctx is NULL, library creates and owns context
 * @note If ctx is provided, caller must keep context alive
 *
 * Example:
 *   wg_handle_t gen = wg_create(NULL, 0);
 *   if (!gen) {
 *       fprintf(stderr, "Failed to create generator\n");
 *       return -1;
 *   }
 */
wg_handle_t wg_create(CUcontext ctx, int device_id);

/**
 * Destroy generator and free all resources
 *
 * @param gen Generator handle
 *
 * @note Safe to call with NULL handle (no-op)
 * @note Automatically frees all device memory
 * @note If library created context, destroys it
 *
 * Example:
 *   wg_destroy(gen);
 */
void wg_destroy(wg_handle_t gen);

/*
 * ===========================================================================
 * Configuration
 * ===========================================================================
 */

/**
 * Define a charset for use in masks
 *
 * @param gen Generator handle
 * @param charset_id Charset identifier (1-255)
 * @param chars Null-terminated string of characters
 * @param len Length of chars (or 0 for strlen)
 * @return WG_SUCCESS or error code
 *
 * @note charset_id must be unique and non-zero
 * @note Maximum 512 total bytes across all charsets
 * @note String is copied; caller retains ownership
 *
 * Example:
 *   wg_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz", 26);
 *   wg_set_charset(gen, 2, "0123456789", 10);
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
 * @param mask Array of charset IDs
 * @param length Number of positions (word length)
 * @return WG_SUCCESS or error code
 *
 * @note All charset IDs must be previously defined
 * @note Maximum word length: 32 characters
 * @note Array is copied; caller retains ownership
 *
 * Example:
 *   int mask[] = {1, 1, 1, 1, 2, 2, 2, 2};  // "aaaa1234"
 *   wg_set_mask(gen, mask, 8);
 */
int wg_set_mask(
    wg_handle_t gen,
    const int* mask,
    int length
);

/**
 * Set output format mode
 *
 * @param gen Generator handle
 * @param format Output format
 * @return WG_SUCCESS or error code
 *
 * @note Default format is WG_FORMAT_NEWLINES
 * @note Format affects stride and buffer size calculations
 *
 * Example:
 *   wg_set_output_format(gen, WG_FORMAT_FIXED_WIDTH);
 */
int wg_set_output_format(
    wg_handle_t gen,
    wg_output_format_t format
);

/*
 * ===========================================================================
 * Keyspace Information
 * ===========================================================================
 */

/**
 * Get total keyspace size
 *
 * @param gen Generator handle
 * @return Number of possible candidates, or 0 on error
 *
 * @note Returns 0 if generator not configured
 * @note May overflow for very large keyspaces
 *
 * Example:
 *   uint64_t total = wg_keyspace_size(gen);
 *   printf("Keyspace: %llu candidates\n", total);
 */
uint64_t wg_keyspace_size(wg_handle_t gen);

/**
 * Convert word to its index (reverse lookup)
 *
 * @param gen Generator handle
 * @param word Word to look up
 * @param word_len Length of word
 * @return Index if valid, UINT64_MAX if not in keyspace
 *
 * @note Useful for checking if word matches mask
 * @note Word must exactly match mask pattern
 *
 * Example:
 *   uint64_t idx = wg_word_to_index(gen, "pass1234", 8);
 *   if (idx != UINT64_MAX) {
 *       printf("Word is candidate #%llu\n", idx);
 *   }
 */
uint64_t wg_word_to_index(
    wg_handle_t gen,
    const char* word,
    size_t word_len
);

/**
 * Calculate required buffer size for host generation
 *
 * @param gen Generator handle
 * @param count Number of candidates
 * @return Required buffer size in bytes, or 0 on error
 *
 * @note Accounts for current output format
 * @note Always returns size for worst-case alignment
 *
 * Example:
 *   size_t size = wg_calculate_buffer_size(gen, 1000000);
 *   char* buffer = malloc(size);
 */
size_t wg_calculate_buffer_size(
    wg_handle_t gen,
    uint64_t count
);

/*
 * ===========================================================================
 * Generation: Device Memory (Zero-Copy)
 * ===========================================================================
 */

/**
 * Generate batch in GPU memory (zero-copy)
 *
 * Candidates remain on GPU for direct consumption by hash kernels.
 * Device pointer is valid until next generation or wg_destroy().
 *
 * @param gen Generator handle
 * @param start_idx Starting index in keyspace
 * @param count Number of candidates to generate
 * @param batch [out] Batch result info
 * @return WG_SUCCESS or error code
 *
 * @note Caller must NOT free device pointer
 * @note Device pointer invalidated on next generation
 * @note Returns WG_ERROR_KEYSPACE_OVERFLOW if start_idx+count > keyspace
 *
 * Example:
 *   wg_batch_device_t batch;
 *   if (wg_generate_batch_device(gen, 0, 100000000, &batch) == WG_SUCCESS) {
 *       // Use batch.data in your GPU kernel
 *       my_hash_kernel<<<grid, block>>>((const char*)batch.data, ...);
 *   }
 */
int wg_generate_batch_device(
    wg_handle_t gen,
    uint64_t start_idx,
    uint64_t count,
    wg_batch_device_t* batch
);

/**
 * Free device batch (optional)
 *
 * Releases device memory early if you're done with it.
 * Not required - automatically freed on next generation or destroy.
 *
 * @param gen Generator handle
 * @param batch Batch to free
 *
 * @note Safe to call multiple times (idempotent)
 * @note batch->data set to 0 after freeing
 *
 * Example:
 *   wg_free_batch_device(gen, &batch);
 */
void wg_free_batch_device(
    wg_handle_t gen,
    wg_batch_device_t* batch
);

/*
 * ===========================================================================
 * Generation: Host Memory (Copy)
 * ===========================================================================
 */

/**
 * Generate batch and copy to host memory
 *
 * @param gen Generator handle
 * @param start_idx Starting index in keyspace
 * @param count Number of candidates to generate
 * @param output_buffer Pre-allocated host buffer
 * @param buffer_size Size of output buffer in bytes
 * @return Number of bytes written, or negative error code
 *
 * @note Buffer must be allocated by caller
 * @note Use wg_calculate_buffer_size() to get required size
 * @note Returns WG_ERROR_BUFFER_TOO_SMALL if buffer too small
 *
 * Example:
 *   size_t size = wg_calculate_buffer_size(gen, 1000000);
 *   char* buffer = malloc(size);
 *   ssize_t bytes = wg_generate_batch_host(gen, 0, 1000000, buffer, size);
 *   if (bytes < 0) {
 *       fprintf(stderr, "Generation failed: %d\n", (int)bytes);
 *   }
 */
ssize_t wg_generate_batch_host(
    wg_handle_t gen,
    uint64_t start_idx,
    uint64_t count,
    char* output_buffer,
    size_t buffer_size
);

/*
 * ===========================================================================
 * Streaming API (Advanced)
 * ===========================================================================
 */

/**
 * Generate batch using CUDA stream (async)
 *
 * Allows overlapping generation with other GPU operations.
 * Kernel launch returns immediately; use cuStreamSynchronize()
 * to wait for completion.
 *
 * @param gen Generator handle
 * @param stream CUDA stream for async execution
 * @param start_idx Starting index in keyspace
 * @param count Number of candidates to generate
 * @param batch [out] Batch result info
 * @return WG_SUCCESS or error code
 *
 * @note Caller must synchronize stream before using batch.data
 * @note Device pointer lifetime same as wg_generate_batch_device()
 *
 * Example:
 *   CUstream stream;
 *   cuStreamCreate(&stream, 0);
 *
 *   wg_batch_device_t batch;
 *   wg_generate_batch_stream(gen, stream, 0, 100000000, &batch);
 *
 *   // Do other work...
 *
 *   cuStreamSynchronize(stream);  // Wait for generation
 *   // Now batch.data is valid
 */
int wg_generate_batch_stream(
    wg_handle_t gen,
    CUstream stream,
    uint64_t start_idx,
    uint64_t count,
    wg_batch_device_t* batch
);

/*
 * ===========================================================================
 * Error Handling
 * ===========================================================================
 */

/**
 * Get last error message for this thread
 *
 * @param gen Generator handle
 * @return Error message string, or NULL if no error
 *
 * @note Thread-local error storage
 * @note String valid until next API call or wg_destroy()
 * @note Safe to call with NULL handle
 *
 * Example:
 *   if (wg_set_charset(gen, 1, "abc", 3) != WG_SUCCESS) {
 *       fprintf(stderr, "Error: %s\n", wg_get_error(gen));
 *   }
 */
const char* wg_get_error(wg_handle_t gen);

/**
 * Get CUDA device properties
 *
 * @param gen Generator handle
 * @param prop [out] CUDA device properties
 * @return WG_SUCCESS or error code
 *
 * @note Useful for checking GPU capabilities
 *
 * Example:
 *   CUdevprop prop;
 *   if (wg_get_device_properties(gen, &prop) == WG_SUCCESS) {
 *       printf("GPU: %s\n", prop.name);
 *   }
 */
int wg_get_device_properties(
    wg_handle_t gen,
    CUdevprop* prop
);

/*
 * ===========================================================================
 * Utility Functions
 * ===========================================================================
 */

/**
 * Get library version
 *
 * @param major [out] Major version
 * @param minor [out] Minor version
 * @param patch [out] Patch version
 *
 * Example:
 *   int major, minor, patch;
 *   wg_get_version(&major, &minor, &patch);
 *   printf("libwordlist_generator v%d.%d.%d\n", major, minor, patch);
 */
void wg_get_version(int* major, int* minor, int* patch);

/**
 * Check if CUDA is available
 *
 * @return 1 if CUDA available, 0 otherwise
 *
 * Example:
 *   if (!wg_cuda_available()) {
 *       fprintf(stderr, "CUDA not available\n");
 *       return -1;
 *   }
 */
int wg_cuda_available(void);

/**
 * Get number of CUDA devices
 *
 * @return Number of CUDA devices, or -1 on error
 *
 * Example:
 *   int num_gpus = wg_get_device_count();
 *   printf("Found %d GPUs\n", num_gpus);
 */
int wg_get_device_count(void);

/**
 * Get device information
 *
 * Retrieves detailed information about a specific CUDA device.
 *
 * @param device_id         Device index (0 to wg_get_device_count() - 1)
 * @param name_out          [out] Buffer for device name (at least 256 bytes)
 * @param compute_cap_major_out [out] Major compute capability
 * @param compute_cap_minor_out [out] Minor compute capability
 * @param total_memory_out  [out] Total device memory in bytes
 *
 * @return WG_SUCCESS or error code
 *
 * Example:
 *   int count = wg_get_device_count();
 *   for (int i = 0; i < count; i++) {
 *       char name[256];
 *       int major, minor;
 *       uint64_t memory;
 *
 *       if (wg_get_device_info(i, name, &major, &minor, &memory) == WG_SUCCESS) {
 *           printf("Device %d: %s (sm_%d%d, %lu MB)\n",
 *                  i, name, major, minor, memory / (1024*1024));
 *       }
 *   }
 */
int wg_get_device_info(
    int device_id,
    char* name_out,
    int* compute_cap_major_out,
    int* compute_cap_minor_out,
    uint64_t* total_memory_out
);

#ifdef __cplusplus
}
#endif

#endif /* WORDLIST_GENERATOR_H */
```

---

## API Usage Patterns

### Pattern 1: Minimal Usage (Host Memory)

```c
#include <wordlist_generator.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 1. Create generator
    wg_handle_t gen = wg_create(NULL, 0);
    if (!gen) {
        fprintf(stderr, "Failed to create generator\n");
        return 1;
    }

    // 2. Configure
    wg_set_charset(gen, 1, "abc", 3);
    int mask[] = {1, 1, 1, 1};
    wg_set_mask(gen, mask, 4);

    // 3. Generate
    uint64_t count = 1000;
    size_t size = wg_calculate_buffer_size(gen, count);
    char* buffer = malloc(size);

    ssize_t bytes = wg_generate_batch_host(gen, 0, count, buffer, size);
    if (bytes < 0) {
        fprintf(stderr, "Error: %s\n", wg_get_error(gen));
        free(buffer);
        wg_destroy(gen);
        return 1;
    }

    // 4. Process
    printf("Generated %zd bytes:\n%.*s\n", bytes, (int)bytes, buffer);

    // 5. Cleanup
    free(buffer);
    wg_destroy(gen);
    return 0;
}
```

### Pattern 2: Zero-Copy GPU Integration

```c
#include <wordlist_generator.h>
#include <cuda.h>

// Your hash kernel
__global__ void hash_md5_kernel(
    const char* candidates,
    size_t stride,
    uint64_t count,
    unsigned char* hashes_out
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const char* word = candidates + (tid * stride);
    // Hash word, store in hashes_out[tid * 16]
}

int main() {
    // Initialize CUDA
    cuInit(0);
    CUdevice device;
    CUcontext context;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Create generator with existing context
    wg_handle_t gen = wg_create(context, 0);

    // Configure
    wg_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz0123456789", 36);
    int mask[] = {1,1,1,1,1,1,1,1};
    wg_set_mask(gen, mask, 8);
    wg_set_output_format(gen, WG_FORMAT_FIXED_WIDTH);

    // Allocate hash output
    CUdeviceptr d_hashes;
    cuMemAlloc(&d_hashes, 100000000 * 16);

    // Generate on GPU
    wg_batch_device_t batch;
    wg_generate_batch_device(gen, 0, 100000000, &batch);

    // Hash on GPU (zero-copy!)
    dim3 block(256);
    dim3 grid((batch.count + 255) / 256);
    hash_md5_kernel<<<grid, block>>>(
        (const char*)batch.data,
        batch.stride,
        batch.count,
        (unsigned char*)d_hashes
    );

    // Check hashes, etc.
    // ...

    cuMemFree(d_hashes);
    wg_destroy(gen);
    cuCtxDestroy(context);
    return 0;
}
```

### Pattern 3: Multi-GPU Distributed

```c
#include <wordlist_generator.h>
#include <omp.h>

int main() {
    int num_gpus = wg_get_device_count();
    uint64_t keyspace = 36ULL * 36 * 36 * 36 * 36 * 36 * 36 * 36;
    uint64_t per_gpu = keyspace / num_gpus;

    #pragma omp parallel num_threads(num_gpus)
    {
        int gpu_id = omp_get_thread_num();

        // Each thread gets its own generator
        wg_handle_t gen = wg_create(NULL, gpu_id);

        // Configure (same for all)
        wg_set_charset(gen, 1, "abcdefghijklmnopqrstuvwxyz0123456789", 36);
        int mask[] = {1,1,1,1,1,1,1,1};
        wg_set_mask(gen, mask, 8);

        // Each GPU works on its partition
        uint64_t start = gpu_id * per_gpu;

        wg_batch_device_t batch;
        wg_generate_batch_device(gen, start, per_gpu, &batch);

        // Process on this GPU
        process_batch(gpu_id, &batch);

        wg_destroy(gen);
    }

    return 0;
}
```

---

## Error Handling Best Practices

### Always Check Return Values

```c
// BAD
wg_set_charset(gen, 1, "abc", 3);
wg_set_mask(gen, mask, 4);

// GOOD
if (wg_set_charset(gen, 1, "abc", 3) != WG_SUCCESS) {
    fprintf(stderr, "Error: %s\n", wg_get_error(gen));
    wg_destroy(gen);
    return 1;
}
if (wg_set_mask(gen, mask, 4) != WG_SUCCESS) {
    fprintf(stderr, "Error: %s\n", wg_get_error(gen));
    wg_destroy(gen);
    return 1;
}
```

### Cleanup on Error

```c
wg_handle_t gen = wg_create(NULL, 0);
char* buffer = NULL;

if (!gen) goto error;

if (wg_set_charset(gen, 1, "abc", 3) != WG_SUCCESS) goto error;

size_t size = wg_calculate_buffer_size(gen, 1000);
buffer = malloc(size);
if (!buffer) goto error;

// ... success path ...

free(buffer);
wg_destroy(gen);
return 0;

error:
    fprintf(stderr, "Error: %s\n", wg_get_error(gen));
    free(buffer);
    wg_destroy(gen);
    return 1;
```

---

## Thread Safety

### Rules

1. **Handle creation/destruction**: Not thread-safe
   - Create one handle per thread
   - Or protect with mutex

2. **Configuration**: Not thread-safe
   - Configure before multi-threading
   - Or protect with mutex

3. **Generation**: Thread-safe (different handles)
   - Multiple threads can generate simultaneously
   - Each must use its own handle

4. **Error messages**: Thread-local
   - Each thread has its own error buffer

### Multi-Threaded Example

```c
#include <pthread.h>

void* worker_thread(void* arg) {
    int thread_id = *(int*)arg;

    // Each thread creates its own generator
    wg_handle_t gen = wg_create(NULL, 0);

    // Configure (safe because handle is thread-local)
    wg_set_charset(gen, 1, "abc", 3);
    int mask[] = {1, 1, 1, 1};
    wg_set_mask(gen, mask, 4);

    // Generate (safe because handle is thread-local)
    wg_batch_device_t batch;
    wg_generate_batch_device(gen, thread_id * 1000000, 1000000, &batch);

    // Process...

    wg_destroy(gen);
    return NULL;
}

int main() {
    pthread_t threads[4];
    int thread_ids[4] = {0, 1, 2, 3};

    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &thread_ids[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
```

---

## Memory Management

### Device Memory Lifetime

```c
wg_batch_device_t batch1, batch2;

// Generate batch 1
wg_generate_batch_device(gen, 0, 1000, &batch1);
// batch1.data is valid

// Generate batch 2
wg_generate_batch_device(gen, 1000, 1000, &batch2);
// batch1.data is now INVALID (freed automatically)
// batch2.data is valid

// Destroy generator
wg_destroy(gen);
// batch2.data is now INVALID (freed automatically)
```

### Host Memory Ownership

```c
// Caller allocates
char* buffer = malloc(1000000);

// Library writes to it
wg_generate_batch_host(gen, 0, 10000, buffer, 1000000);

// Caller frees
free(buffer);
```

---

## Performance Tuning

### Batch Size Selection

```c
// Too small: Kernel launch overhead dominates
wg_generate_batch_device(gen, 0, 1000, &batch);  // BAD: 1K candidates

// Too large: May exhaust GPU memory
wg_generate_batch_device(gen, 0, 10000000000, &batch);  // BAD: 10B candidates

// Optimal: 10M - 100M candidates per batch
wg_generate_batch_device(gen, 0, 100000000, &batch);  // GOOD: 100M candidates
```

### Output Format Selection

```c
// Use FIXED_WIDTH for hash kernels (aligned access)
wg_set_output_format(gen, WG_FORMAT_FIXED_WIDTH);

// Use NEWLINES for file output / piping
wg_set_output_format(gen, WG_FORMAT_NEWLINES);

// Use PACKED for maximum throughput
wg_set_output_format(gen, WG_FORMAT_PACKED);
```

---

## Build and Linking

### CMake Example

```cmake
find_package(CUDA REQUIRED)

add_executable(my_cracker main.c)

target_link_libraries(my_cracker
    wordlist_generator  # Our library
    ${CUDA_LIBRARIES}
)

target_include_directories(my_cracker PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)
```

### Manual Compilation

```bash
# Compile your code
gcc -c main.c -I/path/to/wordlist_generator/include

# Link with library
gcc main.o -o my_cracker \
    -L/path/to/wordlist_generator/lib \
    -lwordlist_generator \
    -lcuda
```

---

**End of C API Specification**
