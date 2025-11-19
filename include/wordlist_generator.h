/* libwordlist_generator - Auto-generated C API */

#ifndef WORDLIST_GENERATOR_H
#define WORDLIST_GENERATOR_H

/* WARNING: Auto-generated from src/ffi.rs - DO NOT EDIT */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <cuda.h>

#define wg_WG_SUCCESS 0

#define wg_WG_ERROR_INVALID_HANDLE -1

#define wg_WG_ERROR_INVALID_PARAM -2

#define wg_WG_ERROR_CUDA -3

#define wg_WG_ERROR_OUT_OF_MEMORY -4

#define wg_WG_ERROR_NOT_CONFIGURED -5

#define wg_WG_ERROR_BUFFER_TOO_SMALL -6

#define wg_WG_ERROR_KEYSPACE_OVERFLOW -7

#define wg_WG_FORMAT_NEWLINES 0

#define wg_WG_FORMAT_FIXED_WIDTH 1

#define wg_WG_FORMAT_PACKED 2

/**
 * Opaque handle to wordlist generator (exported to C)
 */
typedef struct wg_WordlistGenerator {
    uint8_t _private[0];
} wg_WordlistGenerator;

/**
 * Device batch result (zero-copy GPU memory access)
 *
 * Contains a GPU device pointer for direct kernel-to-kernel data passing.
 * Memory is automatically freed on next generation or wg_destroy().
 */
typedef struct wg_BatchDevice {
    /**
     * Device pointer to candidates (CUdeviceptr)
     */
    uint64_t data;
    /**
     * Number of candidates generated
     */
    uint64_t count;
    /**
     * Length of each word in characters
     */
    uintptr_t word_length;
    /**
     * Bytes between word starts (stride)
     */
    uintptr_t stride;
    /**
     * Total buffer size in bytes
     */
    uintptr_t total_bytes;
    /**
     * Output format used (WG_FORMAT_*)
     */
    int32_t format;
} wg_BatchDevice;

/**
 * Create a new wordlist generator
 *
 * # Arguments
 * * `ctx` - CUDA context (NULL to create new, non-NULL to use existing)
 * * `device_id` - CUDA device ID (0 for first GPU, ignored if ctx provided)
 *
 * # Returns
 * Generator handle, or NULL on error
 */
struct wg_WordlistGenerator *wg_create(void *ctx, int32_t _device_id);

/**
 * Destroy generator and free all resources
 *
 * # Safety
 * Safe to call with NULL (no-op)
 */
void wg_destroy(struct wg_WordlistGenerator *gen);

/**
 * Define a charset for use in masks
 *
 * # Arguments
 * * `gen` - Generator handle
 * * `charset_id` - Identifier (1-255)
 * * `chars` - Character array
 * * `len` - Length of character array
 *
 * # Returns
 * WG_SUCCESS or error code
 */
int32_t wg_set_charset(struct wg_WordlistGenerator *gen,
                       int32_t charset_id,
                       const char *chars,
                       uintptr_t len);

/**
 * Set the mask pattern
 *
 * # Arguments
 * * `gen` - Generator handle
 * * `mask` - Array of charset IDs
 * * `length` - Number of positions (word length)
 *
 * # Returns
 * WG_SUCCESS or error code
 */
int32_t wg_set_mask(struct wg_WordlistGenerator *gen, const int32_t *mask, int32_t length);

/**
 * Set output format mode
 *
 * # Arguments
 * * `gen` - Generator handle
 * * `format` - Output format (WG_FORMAT_*)
 *
 * # Returns
 * WG_SUCCESS or error code
 */
int32_t wg_set_format(struct wg_WordlistGenerator *gen, int32_t format);

/**
 * Get total keyspace size
 *
 * # Returns
 * Number of possible candidates, or 0 on error
 */
uint64_t wg_keyspace_size(struct wg_WordlistGenerator *gen);

/**
 * Calculate required buffer size for host generation
 *
 * # Returns
 * Required buffer size in bytes, or 0 on error
 */
uintptr_t wg_calculate_buffer_size(struct wg_WordlistGenerator *gen, uint64_t count);

/**
 * Get last error message for this thread
 *
 * # Returns
 * Error message string, or NULL if no error
 */
const char *wg_get_error(struct wg_WordlistGenerator *_gen);

/**
 * Generate batch and copy to host memory
 *
 * # Returns
 * Number of bytes written, or negative error code
 */
intptr_t wg_generate_batch_host(struct wg_WordlistGenerator *gen,
                                uint64_t start_idx,
                                uint64_t count,
                                uint8_t *output_buffer,
                                uintptr_t buffer_size);

/**
 * Generate batch in GPU memory (zero-copy)
 *
 * This function generates candidates directly in GPU memory without copying to host.
 * The device pointer remains valid until the next generation call or wg_destroy().
 *
 * # Arguments
 * * `gen` - Generator handle
 * * `start_idx` - Starting index in keyspace
 * * `count` - Number of candidates to generate
 * * `batch` - Output structure to fill with device pointer info
 *
 * # Returns
 * WG_SUCCESS or error code
 */
int32_t wg_generate_batch_device(struct wg_WordlistGenerator *gen,
                                 uint64_t start_idx,
                                 uint64_t count,
                                 struct wg_BatchDevice *batch);

/**
 * Free device batch memory early (optional)
 *
 * Device memory is automatically freed on next generation or wg_destroy(),
 * but this function allows explicit early cleanup.
 *
 * # Arguments
 * * `gen` - Generator handle
 * * `batch` - Batch to free (data pointer will be set to 0)
 */
void wg_free_batch_device(struct wg_WordlistGenerator *gen, struct wg_BatchDevice *batch);

/**
 * Generate batch using CUDA stream (async)
 *
 * Allows overlapping generation with other GPU operations.
 * Kernel launch returns immediately; use cuStreamSynchronize()
 * to wait for completion.
 *
 * # Arguments
 * * `gen` - Generator handle
 * * `stream` - CUDA stream for async execution (null for default stream)
 * * `start_idx` - Starting index in keyspace
 * * `count` - Number of candidates to generate
 * * `batch` - [out] Batch result info
 *
 * # Returns
 * WG_SUCCESS or error code
 *
 * # Safety
 * Caller must synchronize stream before using batch.data.
 * Device pointer lifetime same as wg_generate_batch_device().
 *
 * # Example
 * ```c
 * CUstream stream;
 * cuStreamCreate(&stream, 0);
 *
 * wg_batch_device_t batch;
 * wg_generate_batch_stream(gen, stream, 0, 100000000, &batch);
 *
 * // Do other work...
 *
 * cuStreamSynchronize(stream);  // Wait for generation
 * // Now batch.data is valid
 * ```
 */
int32_t wg_generate_batch_stream(struct wg_WordlistGenerator *gen,
                                 CUstream stream,
                                 uint64_t start_idx,
                                 uint64_t count,
                                 struct wg_BatchDevice *batch);

/**
 * Get library version string
 *
 * Returns a static string with the library version.
 * This function never fails and always returns a valid pointer.
 *
 * # Returns
 * Pointer to static version string (e.g., "0.1.0")
 *
 * # Example
 * ```c
 * const char* version = wg_get_version();
 * printf("Library version: %s\n", version);
 * ```
 */
const char *wg_get_version(void);

/**
 * Check if CUDA is available
 *
 * Attempts to initialize CUDA and check for devices.
 * This function can be called before creating a generator
 * to verify CUDA is available.
 *
 * # Returns
 * 1 if CUDA is available and working, 0 otherwise
 *
 * # Example
 * ```c
 * if (!wg_cuda_available()) {
 *     fprintf(stderr, "CUDA not available\n");
 *     return -1;
 * }
 * // Safe to create generator
 * ```
 */
int32_t wg_cuda_available(void);

/**
 * Get number of CUDA devices
 *
 * Returns the count of CUDA-capable devices in the system.
 * Returns -1 if CUDA is not available or on error.
 *
 * # Returns
 * Number of CUDA devices (>= 0) or -1 on error
 *
 * # Example
 * ```c
 * int count = wg_get_device_count();
 * if (count < 0) {
 *     fprintf(stderr, "CUDA error\n");
 * } else if (count == 0) {
 *     fprintf(stderr, "No CUDA devices found\n");
 * } else {
 *     printf("Found %d CUDA device(s)\n", count);
 * }
 * ```
 */
int32_t wg_get_device_count(void);

#endif /* WORDLIST_GENERATOR_H */
