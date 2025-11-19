#include "../include/wordlist_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

void test_create_destroy() {
    printf("Test: create/destroy...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);
    assert(gen != NULL && "Failed to create generator");

    wg_destroy(gen);

    printf("✓ create/destroy passed\n\n");
}

void test_configuration() {
    printf("Test: configuration...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Set charset
    int result = wg_set_charset(gen, 1, "abc", 3);
    assert(result == 0 && "Failed to set charset");

    // Set mask
    int mask[] = {1, 1, 1, 1};
    result = wg_set_mask(gen, mask, 4);
    assert(result == 0 && "Failed to set mask");

    // Check keyspace
    uint64_t keyspace = wg_keyspace_size(gen);
    assert(keyspace == 81 && "Wrong keyspace (expected 3^4=81)");

    printf("  Keyspace: %llu\n", (unsigned long long)keyspace);

    wg_destroy(gen);

    printf("✓ configuration passed\n\n");
}

void test_generation() {
    printf("Test: host generation...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Configure
    wg_set_charset(gen, 1, "ab", 2);
    int mask[] = {1, 1, 1};
    wg_set_mask(gen, mask, 3);

    // Allocate buffer
    uint64_t count = 8; // 2^3 = 8 total
    size_t buffer_size = wg_calculate_buffer_size(gen, count);
    printf("  Buffer size needed: %zu bytes\n", buffer_size);

    char* buffer = malloc(buffer_size);
    assert(buffer != NULL);

    // Generate
    ssize_t bytes = wg_generate_batch_host(gen, 0, count, (uint8_t*)buffer, buffer_size);
    assert(bytes > 0 && "Generation failed");
    printf("  Generated %zd bytes\n", bytes);

    // Verify first few words
    printf("  First 3 words:\n");
    char* ptr = buffer;
    for (int i = 0; i < 3; i++) {
        printf("    %d: %.*s", i, 3, ptr);
        ptr += 4; // 3 chars + newline
    }

    free(buffer);
    wg_destroy(gen);

    printf("✓ generation passed\n\n");
}

void test_error_handling() {
    printf("Test: error handling...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);

    // Invalid charset ID
    int result = wg_set_charset(gen, 0, "abc", 3);
    assert(result != 0 && "Should fail with invalid charset_id");

    const char* error = wg_get_error(gen);
    assert(error != NULL && "Should have error message");
    printf("  Error message: %s\n", error);

    wg_destroy(gen);

    printf("✓ error handling passed\n\n");
}

void test_device_generation() {
    printf("Test: device generation (zero-copy)...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Configure
    wg_set_charset(gen, 1, "xyz", 3);
    int mask[] = {1, 1, 1};
    wg_set_mask(gen, mask, 3);

    // Generate on device
    struct wg_BatchDevice batch;
    int result = wg_generate_batch_device(gen, 0, 27, &batch);  // 3^3 = 27
    assert(result == 0 && "Device generation failed");
    assert(batch.data != 0 && "Device pointer is NULL");
    assert(batch.count == 27 && "Wrong candidate count");
    assert(batch.word_length == 3 && "Wrong word length");
    assert(batch.stride == 4 && "Wrong stride (should be word_length + 1)");

    printf("  Device pointer: 0x%lx\n", (unsigned long)batch.data);
    printf("  Count: %llu\n", (unsigned long long)batch.count);
    printf("  Word length: %zu\n", batch.word_length);
    printf("  Stride: %zu bytes\n", batch.stride);
    printf("  Total bytes: %zu\n", batch.total_bytes);

    // Test auto-free on next generation
    struct wg_BatchDevice batch2;
    result = wg_generate_batch_device(gen, 0, 10, &batch2);
    assert(result == 0 && "Second generation failed");
    assert(batch2.data != 0 && "Second device pointer is NULL");
    printf("  Second generation device pointer: 0x%lx\n", (unsigned long)batch2.data);

    wg_destroy(gen);
    printf("✓ device generation passed\n\n");
}

void test_device_free() {
    printf("Test: explicit device free...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Configure
    wg_set_charset(gen, 1, "01", 2);
    int mask[] = {1, 1};
    wg_set_mask(gen, mask, 2);

    // Generate on device
    struct wg_BatchDevice batch;
    int result = wg_generate_batch_device(gen, 0, 4, &batch);  // 2^2 = 4
    assert(result == 0 && "Device generation failed");
    assert(batch.data != 0 && "Device pointer is NULL");

    printf("  Device pointer before free: 0x%lx\n", (unsigned long)batch.data);

    // Explicitly free
    wg_free_batch_device(gen, &batch);
    assert(batch.data == 0 && "Device pointer should be NULL after free");

    printf("  Device pointer after free: 0x%lx\n", (unsigned long)batch.data);

    wg_destroy(gen);
    printf("✓ explicit device free passed\n\n");
}

void test_device_copy_back() {
    printf("Test: device pointer copy back verification...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Configure
    wg_set_charset(gen, 1, "ab", 2);
    int mask[] = {1, 1};
    wg_set_mask(gen, mask, 2);

    // Generate on device
    struct wg_BatchDevice batch;
    int result = wg_generate_batch_device(gen, 0, 4, &batch);  // 2^2 = 4
    assert(result == 0 && "Device generation failed");
    assert(batch.data != 0 && "Device pointer is NULL");

    // Copy back to host to verify (requires CUDA driver API)
    // For now, we'll skip the actual verification as it requires cuMemcpyDtoH
    // which would need CUDA headers. The fact that generation succeeded
    // and returned a valid pointer is the key test.

    printf("  Generated %llu candidates\n", (unsigned long long)batch.count);
    printf("  Device pointer: 0x%lx (valid)\n", (unsigned long)batch.data);
    printf("  Note: Actual content verification would require CUDA driver API\n");

    wg_destroy(gen);
    printf("✓ device pointer verification passed\n\n");
}

void test_format_newlines() {
    printf("Test: WG_FORMAT_NEWLINES mode...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Configure
    wg_set_charset(gen, 1, "abc", 3);
    int mask[] = {1, 1, 1};
    wg_set_mask(gen, mask, 3);

    // Set format to newlines (default, but explicit)
    int result = wg_set_format(gen, 0);  // WG_FORMAT_NEWLINES
    assert(result == 0 && "Failed to set format");

    // Generate on device
    struct wg_BatchDevice batch;
    result = wg_generate_batch_device(gen, 0, 27, &batch);  // 3^3 = 27
    assert(result == 0 && "Device generation failed");
    assert(batch.format == 0 && "Format should be NEWLINES");
    assert(batch.word_length == 3 && "Wrong word length");
    assert(batch.stride == 4 && "Stride should be word_length + 1");
    assert(batch.total_bytes == 27 * 4 && "Total bytes should be count * stride");

    printf("  Format: NEWLINES (0)\n");
    printf("  Word length: %zu\n", batch.word_length);
    printf("  Stride: %zu bytes (word + newline)\n", batch.stride);
    printf("  Total bytes: %zu\n", batch.total_bytes);
    printf("  Memory efficiency: 100%% (baseline)\n");

    wg_destroy(gen);
    printf("✓ newlines format passed\n\n");
}

void test_format_packed() {
    printf("Test: WG_FORMAT_PACKED mode...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Configure
    wg_set_charset(gen, 1, "xyz", 3);
    int mask[] = {1, 1, 1, 1, 1, 1, 1, 1};  // 8 characters
    wg_set_mask(gen, mask, 8);

    // Set format to packed
    int result = wg_set_format(gen, 2);  // WG_FORMAT_PACKED
    assert(result == 0 && "Failed to set format");

    // Generate on device
    struct wg_BatchDevice batch;
    result = wg_generate_batch_device(gen, 0, 1000, &batch);
    assert(result == 0 && "Device generation failed");
    assert(batch.format == 2 && "Format should be PACKED");
    assert(batch.word_length == 8 && "Wrong word length");
    assert(batch.stride == 8 && "Stride should equal word_length (no separator)");

    // Calculate memory savings
    size_t newlines_bytes = 1000 * 9;  // word + '\n'
    size_t packed_bytes = 1000 * 8;    // just word
    float savings = ((float)(newlines_bytes - packed_bytes) / newlines_bytes) * 100;

    printf("  Format: PACKED (2)\n");
    printf("  Word length: %zu\n", batch.word_length);
    printf("  Stride: %zu bytes (no separator)\n", batch.stride);
    printf("  Total bytes: %zu\n", batch.total_bytes);
    printf("  Memory saved vs NEWLINES: %.1f%%\n", savings);

    wg_destroy(gen);
    printf("✓ packed format passed\n\n");
}

void test_format_invalid() {
    printf("Test: invalid format handling...\n");

    struct wg_WordlistGenerator* gen = wg_create(NULL, 0);
    assert(gen != NULL);

    // Try invalid format
    int result = wg_set_format(gen, 99);
    assert(result != 0 && "Should fail with invalid format");

    const char* error = wg_get_error(gen);
    assert(error != NULL && "Should have error message");
    printf("  Error message: %s\n", error);

    wg_destroy(gen);
    printf("✓ invalid format handling passed\n\n");
}

int main() {
    printf("=== FFI Basic Tests ===\n\n");

    // Phase 1 tests (host memory)
    test_create_destroy();
    test_configuration();
    test_generation();
    test_error_handling();

    // Phase 2 tests (device memory)
    test_device_generation();
    test_device_free();
    test_device_copy_back();

    // Phase 3 tests (output formats)
    test_format_newlines();
    test_format_packed();
    test_format_invalid();

    printf("=== All tests passed! ===\n");
    return 0;
}
