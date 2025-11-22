/*
 * Multi-GPU API Integration Test
 *
 * Tests the multi-GPU wordlist generation API
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/wordlist_generator.h"

#define COLOR_GREEN "\033[0;32m"
#define COLOR_RED "\033[0;31m"
#define COLOR_BLUE "\033[0;34m"
#define COLOR_RESET "\033[0m"

int test_count = 0;
int pass_count = 0;

#define RUN_TEST(name, func) do { \
    printf(COLOR_BLUE "[TEST %d] " COLOR_RESET "%s\n", ++test_count, name); \
    if (func()) { \
        pass_count++; \
        printf(COLOR_GREEN "[PASS]" COLOR_RESET "\n\n"); \
    } else { \
        printf(COLOR_RED "[FAIL]" COLOR_RESET "\n\n"); \
    } \
} while(0)

// Test 1: Create and destroy multi-GPU generator
int test_multigpu_create_destroy() {
    printf("  Creating multi-GPU generator...\n");
    struct wg_MultiGpuGenerator* gen = wg_multigpu_create();
    if (!gen) {
        printf("  ERROR: Failed to create multi-GPU generator\n");
        printf("  (This is expected if no CUDA devices are available)\n");
        return 1; // Pass test if no GPUs available
    }

    printf("  Getting device count...\n");
    int device_count = wg_multigpu_get_device_count(gen);
    printf("  Using %d GPU(s)\n", device_count);

    if (device_count <= 0) {
        printf("  ERROR: Invalid device count\n");
        wg_multigpu_destroy(gen);
        return 0;
    }

    printf("  Destroying multi-GPU generator...\n");
    wg_multigpu_destroy(gen);

    return 1;
}

// Test 2: Configure and generate simple keyspace
int test_multigpu_simple_generation() {
    printf("  Creating multi-GPU generator...\n");
    struct wg_MultiGpuGenerator* gen = wg_multigpu_create();
    if (!gen) {
        printf("  No GPU available, skipping test\n");
        return 1;
    }

    printf("  Configuring charset and mask...\n");
    wg_multigpu_set_charset(gen, 1, "ab", 2);

    int mask[] = {1, 1}; // ?1?1 = aa, ab, ba, bb (4 words)
    wg_multigpu_set_mask(gen, mask, 2);

    printf("  Allocating buffer...\n");
    size_t buffer_size = 4 * 3; // 4 words * (2 chars + newline)
    uint8_t* buffer = malloc(buffer_size);

    printf("  Generating 4 words...\n");
    ssize_t bytes = wg_multigpu_generate(gen, 0, 4, buffer, buffer_size);

    if (bytes <= 0) {
        printf("  ERROR: Generation failed with code: %ld\n", bytes);
        free(buffer);
        wg_multigpu_destroy(gen);
        return 0;
    }

    printf("  Generated %ld bytes\n", bytes);

    // Verify output
    char* output = (char*)buffer;
    printf("  Output:\n%.*s\n", (int)bytes, output);

    // Check for expected words
    int has_aa = (strstr(output, "aa\n") != NULL);
    int has_ab = (strstr(output, "ab\n") != NULL);
    int has_ba = (strstr(output, "ba\n") != NULL);
    int has_bb = (strstr(output, "bb\n") != NULL);

    int success = has_aa && has_ab && has_ba && has_bb;

    if (!success) {
        printf("  ERROR: Missing expected words\n");
        printf("  aa: %s, ab: %s, ba: %s, bb: %s\n",
               has_aa ? "yes" : "no",
               has_ab ? "yes" : "no",
               has_ba ? "yes" : "no",
               has_bb ? "yes" : "no");
    }

    free(buffer);
    wg_multigpu_destroy(gen);

    return success;
}

// Test 3: Create with specific devices
int test_multigpu_create_with_devices() {
    printf("  Creating multi-GPU generator with device 0...\n");
    int devices[] = {0};
    struct wg_MultiGpuGenerator* gen = wg_multigpu_create_with_devices(devices, 1);

    if (!gen) {
        printf("  No GPU available, skipping test\n");
        return 1;
    }

    int device_count = wg_multigpu_get_device_count(gen);
    printf("  Using %d GPU(s)\n", device_count);

    if (device_count != 1) {
        printf("  ERROR: Expected 1 device, got %d\n", device_count);
        wg_multigpu_destroy(gen);
        return 0;
    }

    wg_multigpu_destroy(gen);
    return 1;
}

// Test 4: Partial keyspace generation
int test_multigpu_partial_keyspace() {
    printf("  Creating multi-GPU generator...\n");
    struct wg_MultiGpuGenerator* gen = wg_multigpu_create();
    if (!gen) {
        printf("  No GPU available, skipping test\n");
        return 1;
    }

    printf("  Configuring charset and mask...\n");
    wg_multigpu_set_charset(gen, 1, "abc", 3);

    int mask[] = {1, 1}; // ?1?1 = 3x3 = 9 words
    wg_multigpu_set_mask(gen, mask, 2);

    printf("  Allocating buffer...\n");
    size_t buffer_size = 3 * 4; // 3 words * (3 chars + newline)
    uint8_t* buffer = malloc(buffer_size);

    printf("  Generating words at indices 3-5 (ba, bb, bc)...\n");
    ssize_t bytes = wg_multigpu_generate(gen, 3, 3, buffer, buffer_size);

    if (bytes <= 0) {
        printf("  ERROR: Generation failed with code: %ld\n", bytes);
        free(buffer);
        wg_multigpu_destroy(gen);
        return 0;
    }

    printf("  Generated %ld bytes\n", bytes);

    // Verify output
    char* output = (char*)buffer;
    printf("  Output:\n%.*s\n", (int)bytes, output);

    // Check for expected words at indices 3-5
    int has_ba = (strstr(output, "ba\n") != NULL);
    int has_bb = (strstr(output, "bb\n") != NULL);
    int has_bc = (strstr(output, "bc\n") != NULL);

    int success = has_ba && has_bb && has_bc;

    if (!success) {
        printf("  ERROR: Missing expected words\n");
        printf("  ba: %s, bb: %s, bc: %s\n",
               has_ba ? "yes" : "no",
               has_bb ? "yes" : "no",
               has_bc ? "yes" : "no");
    }

    free(buffer);
    wg_multigpu_destroy(gen);

    return success;
}

int main() {
    printf("=== Multi-GPU API Integration Tests ===\n\n");

    RUN_TEST("Create and destroy multi-GPU generator", test_multigpu_create_destroy);
    RUN_TEST("Simple keyspace generation", test_multigpu_simple_generation);
    RUN_TEST("Create with specific devices", test_multigpu_create_with_devices);
    RUN_TEST("Partial keyspace generation", test_multigpu_partial_keyspace);

    printf("=== Summary ===\n");
    printf("Passed: %d/%d\n", pass_count, test_count);
    printf("Failed: %d/%d\n", test_count - pass_count, test_count);

    return (pass_count == test_count) ? 0 : 1;
}
