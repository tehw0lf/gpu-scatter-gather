/*
 * Test program for device enumeration API
 *
 * Tests:
 * - wg_get_device_count()
 * - wg_get_device_info()
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "../include/wordlist_generator.h"

#define TEST_PASS "\033[32m[PASS]\033[0m"
#define TEST_FAIL "\033[31m[FAIL]\033[0m"

// Convenience macros for constants
#define WG_SUCCESS wg_WG_SUCCESS

int main() {
    printf("=== Device Enumeration API Tests ===\n\n");

    int passed = 0;
    int failed = 0;

    // Test 1: Get device count
    printf("Test 1: wg_get_device_count()\n");
    int device_count = wg_get_device_count();
    if (device_count >= 0) {
        printf("%s Device count: %d\n", TEST_PASS, device_count);
        passed++;
    } else {
        printf("%s Failed to get device count\n", TEST_FAIL);
        failed++;
        return 1;  // Cannot continue without devices
    }

    if (device_count == 0) {
        printf("\nNo CUDA devices found. Skipping device info tests.\n");
        printf("\nSummary: %d passed, %d failed\n", passed, failed);
        return 0;
    }

    printf("\n");

    // Test 2: Get info for each device
    for (int i = 0; i < device_count; i++) {
        char name[256];
        int major, minor;
        uint64_t memory;

        printf("Test 2.%d: wg_get_device_info() for device %d\n", i, i);

        int result = wg_get_device_info(i, name, &major, &minor, &memory);

        if (result == WG_SUCCESS) {
            printf("%s Device %d:\n", TEST_PASS, i);
            printf("  Name: %s\n", name);
            printf("  Compute Capability: sm_%d%d\n", major, minor);
            printf("  Total Memory: %lu MB (%.2f GB)\n",
                   memory / (1024*1024),
                   memory / (1024.0*1024.0*1024.0));

            // Validate values
            if (strlen(name) == 0) {
                printf("%s Device name is empty\n", TEST_FAIL);
                failed++;
            } else if (major < 3 || major > 20) {
                printf("%s Invalid compute capability major: %d\n", TEST_FAIL, major);
                failed++;
            } else if (minor < 0 || minor > 9) {
                printf("%s Invalid compute capability minor: %d\n", TEST_FAIL, minor);
                failed++;
            } else if (memory == 0) {
                printf("%s Total memory is 0\n", TEST_FAIL);
                failed++;
            } else {
                passed++;
            }
        } else {
            printf("%s Failed to get device info (error code: %d)\n", TEST_FAIL, result);
            failed++;
        }

        printf("\n");
    }

    // Test 3: Invalid device ID
    printf("Test 3: wg_get_device_info() with invalid device ID\n");
    char name[256];
    int major, minor;
    uint64_t memory;

    int result = wg_get_device_info(device_count, name, &major, &minor, &memory);
    if (result != WG_SUCCESS) {
        printf("%s Correctly rejected invalid device ID %d\n", TEST_PASS, device_count);
        passed++;
    } else {
        printf("%s Should have rejected invalid device ID %d\n", TEST_FAIL, device_count);
        failed++;
    }

    printf("\n");

    // Test 4: Null pointer checks
    printf("Test 4: wg_get_device_info() with null pointers\n");

    // Test null name pointer
    result = wg_get_device_info(0, NULL, &major, &minor, &memory);
    if (result != WG_SUCCESS) {
        printf("%s Correctly rejected null name pointer\n", TEST_PASS);
        passed++;
    } else {
        printf("%s Should have rejected null name pointer\n", TEST_FAIL);
        failed++;
    }

    // Test null major pointer
    result = wg_get_device_info(0, name, NULL, &minor, &memory);
    if (result != WG_SUCCESS) {
        printf("%s Correctly rejected null major pointer\n", TEST_PASS);
        passed++;
    } else {
        printf("%s Should have rejected null major pointer\n", TEST_FAIL);
        failed++;
    }

    // Test null minor pointer
    result = wg_get_device_info(0, name, &major, NULL, &memory);
    if (result != WG_SUCCESS) {
        printf("%s Correctly rejected null minor pointer\n", TEST_PASS);
        passed++;
    } else {
        printf("%s Should have rejected null minor pointer\n", TEST_FAIL);
        failed++;
    }

    // Test null memory pointer
    result = wg_get_device_info(0, name, &major, &minor, NULL);
    if (result != WG_SUCCESS) {
        printf("%s Correctly rejected null memory pointer\n", TEST_PASS);
        passed++;
    } else {
        printf("%s Should have rejected null memory pointer\n", TEST_FAIL);
        failed++;
    }

    printf("\n");

    // Summary
    printf("=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
