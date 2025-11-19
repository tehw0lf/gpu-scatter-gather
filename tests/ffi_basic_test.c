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

int main() {
    printf("=== FFI Basic Tests ===\n\n");

    test_create_destroy();
    test_configuration();
    test_generation();
    test_error_handling();

    printf("=== All tests passed! ===\n");
    return 0;
}
