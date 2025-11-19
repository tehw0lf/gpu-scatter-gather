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

/**
 * Opaque handle to wordlist generator (exported to C)
 */
typedef struct wg_WordlistGenerator {
    uint8_t _private[0];
} wg_WordlistGenerator;

/**
 * Create a new wordlist generator
 *
 * # Arguments
 * * `ctx` - CUDA context (NULL to create new)
 * * `device_id` - CUDA device ID (0 for first GPU)
 *
 * # Returns
 * Generator handle, or NULL on error
 */
struct wg_WordlistGenerator *wg_create(void *_ctx, int32_t _device_id);

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

#endif /* WORDLIST_GENERATOR_H */
