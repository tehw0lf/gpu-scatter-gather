# John the Ripper Integration Guide

**Version**: 1.0
**Date**: November 20, 2025
**Target Audience**: John the Ripper developers, security researchers, custom cracking tool builders

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Integration Pattern 1: External Mode](#integration-pattern-1-external-mode)
4. [Integration Pattern 2: Format Plugin](#integration-pattern-2-format-plugin)
5. [Integration Pattern 3: Distributed Cracking](#integration-pattern-3-distributed-cracking)
6. [Performance Tuning](#performance-tuning)
7. [Complete Working Example](#complete-working-example)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

### Purpose

This guide demonstrates how to integrate `libgpu_scatter_gather` into John the Ripper (JtR) or custom password cracking tools to dramatically accelerate mask-based attacks.

### Use Cases

- **External mode**: Generate wordlists for JtR's external mode with GPU acceleration
- **Format plugins**: Embed library into custom JtR format modules for zero-copy operation
- **Distributed cracking**: Perfect keyspace partitioning for multi-node clusters
- **Custom tools**: Build standalone crackers using JtR hash implementations

### Performance Advantages

| Tool | Throughput (8-char) | vs CPU |
|------|---------------------|--------|
| John the Ripper (CPU) | ~60-100 M/s | 1.0× |
| cracken (CPU) | ~178 M/s | 1.8-3.0× |
| **gpu-scatter-gather** | **702 M/s** | **7-11×** |

**Key Benefits:**
- 7-11× faster candidate generation than CPU-based JtR modes
- O(1) random access enables perfect keyspace partitioning for distributed cracking
- Zero-copy GPU operation when integrated into format plugins
- Deterministic generation allows resume support and reproducibility

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

### Prerequisites

- NVIDIA GPU with CUDA support (Compute Capability 7.5+)
- CUDA Toolkit 12.x
- John the Ripper (jumbo version recommended)
- GCC or Clang compiler

---

## Integration Pattern 1: External Mode

### Use Case

Generate wordlists with GPU acceleration and feed them to JtR's external mode via stdin.

```bash
./jtr_generator '?l?l?l?l?l?l?l?l' | john --external=GPU_Mask hashes.txt
```

### Complete Generator Example

```c
/*
 * jtr_generator.c - GPU-accelerated wordlist generator for John the Ripper
 *
 * Usage: ./jtr_generator <mask> [options] | john --stdin hashes.txt
 */

#include "wordlist_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <getopt.h>

volatile sig_atomic_t should_stop = 0;

void sigint_handler(int sig) {
    should_stop = 1;
}

typedef struct {
    char *mask;
    uint64_t start_index;
    uint64_t end_index;
    uint64_t batch_size;
    int show_progress;
} config_t;

void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [OPTIONS] <mask>\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -s, --start INDEX    Start at keyspace index (default: 0)\n");
    fprintf(stderr, "  -e, --end INDEX      End at keyspace index (default: keyspace size)\n");
    fprintf(stderr, "  -b, --batch SIZE     Batch size in millions (default: 10M)\n");
    fprintf(stderr, "  -q, --quiet          Suppress progress output\n");
    fprintf(stderr, "  -h, --help           Show this help\n");
    fprintf(stderr, "\nMask syntax (JtR-compatible):\n");
    fprintf(stderr, "  ?l  lowercase letters (a-z)\n");
    fprintf(stderr, "  ?u  uppercase letters (A-Z)\n");
    fprintf(stderr, "  ?d  digits (0-9)\n");
    fprintf(stderr, "  ?s  special characters\n");
    fprintf(stderr, "  ?a  all printable ASCII\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s '?l?l?l?l?l?l?l?l' | john --stdin hashes.txt\n", prog);
    fprintf(stderr, "  %s -s 0 -e 1000000000 '?l?l?l?l?d?d?d?d' | john --stdin hashes.txt\n", prog);
    fprintf(stderr, "  %s --batch 50 '?u?l?l?l?l?l?d?d' | john --stdin hashes.txt\n", prog);
}

int main(int argc, char **argv) {
    config_t config = {
        .mask = NULL,
        .start_index = 0,
        .end_index = UINT64_MAX,
        .batch_size = 10000000,  // 10M default
        .show_progress = 1
    };

    // Parse command line options
    static struct option long_options[] = {
        {"start",   required_argument, 0, 's'},
        {"end",     required_argument, 0, 'e'},
        {"batch",   required_argument, 0, 'b'},
        {"quiet",   no_argument,       0, 'q'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "s:e:b:qh", long_options, NULL)) != -1) {
        switch (opt) {
            case 's':
                config.start_index = strtoull(optarg, NULL, 10);
                break;
            case 'e':
                config.end_index = strtoull(optarg, NULL, 10);
                break;
            case 'b':
                config.batch_size = strtoull(optarg, NULL, 10) * 1000000;
                break;
            case 'q':
                config.show_progress = 0;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Error: Mask pattern required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    config.mask = argv[optind];

    // Signal handling
    signal(SIGINT, sigint_handler);
    signal(SIGPIPE, SIG_IGN);

    // Create generator
    wg_WordlistGenerator *gen = wg_create();
    if (!gen) {
        fprintf(stderr, "Error: Failed to create generator\n");
        return 1;
    }

    // Define JtR-standard charsets
    wg_add_charset(gen, 'l', "abcdefghijklmnopqrstuvwxyz", 26);
    wg_add_charset(gen, 'u', "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 26);
    wg_add_charset(gen, 'd', "0123456789", 10);
    wg_add_charset(gen, 's', " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", 33);

    // ?a = all printable ASCII (0x20-0x7E)
    char all_ascii[96];
    for (int i = 0; i < 95; i++) {
        all_ascii[i] = 0x20 + i;
    }
    all_ascii[95] = '\0';
    wg_add_charset(gen, 'a', all_ascii, 95);

    // Set mask
    if (wg_set_mask(gen, config.mask) != 0) {
        fprintf(stderr, "Error: Invalid mask pattern: %s\n", config.mask);
        wg_destroy(gen);
        return 1;
    }

    // Use NEWLINES format (required for JtR stdin)
    wg_set_format(gen, WG_FORMAT_NEWLINES);

    // Get keyspace size
    uint64_t keyspace = wg_keyspace_size(gen);
    if (config.end_index == UINT64_MAX || config.end_index > keyspace) {
        config.end_index = keyspace;
    }

    if (config.start_index >= config.end_index) {
        fprintf(stderr, "Error: Invalid range (start >= end)\n");
        wg_destroy(gen);
        return 1;
    }

    if (config.show_progress) {
        fprintf(stderr, "Mask: %s\n", config.mask);
        fprintf(stderr, "Keyspace: %llu candidates\n", keyspace);
        fprintf(stderr, "Range: %llu to %llu\n",
                config.start_index, config.end_index);
        fprintf(stderr, "Batch size: %llu\n", config.batch_size);
        fprintf(stderr, "\n");
    }

    // Calculate buffer size
    size_t buffer_size = wg_calculate_buffer_size(gen, config.batch_size);
    char *buffer = malloc(buffer_size);
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate %zu bytes\n", buffer_size);
        wg_destroy(gen);
        return 1;
    }

    // Generate and output
    uint64_t total_generated = 0;
    uint64_t total_count = config.end_index - config.start_index;

    for (uint64_t start = config.start_index;
         start < config.end_index && !should_stop;
         start += config.batch_size) {

        uint64_t count = (start + config.batch_size > config.end_index) ?
                         (config.end_index - start) : config.batch_size;

        // Generate batch
        int result = wg_generate_batch_host(gen, start, count,
                                            buffer, buffer_size);
        if (result != 0) {
            fprintf(stderr, "\nError: Generation failed at index %llu\n", start);
            break;
        }

        // Calculate actual bytes written (depends on format and count)
        size_t word_length = wg_word_length(gen);
        size_t stride = word_length + 1;  // NEWLINES format
        size_t bytes_to_write = count * stride;

        // Write to stdout
        size_t bytes_written = fwrite(buffer, 1, bytes_to_write, stdout);
        if (bytes_written != bytes_to_write) {
            if (ferror(stdout)) {
                fprintf(stderr, "\nError: Write failed (pipe closed?)\n");
            }
            break;
        }

        fflush(stdout);

        total_generated += count;

        if (config.show_progress) {
            double progress = 100.0 * total_generated / total_count;
            fprintf(stderr, "\rProgress: %llu/%llu (%.1f%%) - %.2f M/s",
                    total_generated, total_count, progress,
                    total_generated / 1e6);  // Rough throughput estimate
        }
    }

    if (config.show_progress) {
        fprintf(stderr, "\n\nCompleted: %llu candidates generated\n",
                total_generated);
    }

    free(buffer);
    wg_destroy(gen);
    return 0;
}
```

### Compilation

```bash
gcc -o jtr_generator jtr_generator.c \
    -I../include \
    -L../target/release \
    -lgpu_scatter_gather \
    -Wl,-rpath,../target/release \
    -O3
```

### Usage with John the Ripper

```bash
# Basic usage (8-char lowercase)
./jtr_generator '?l?l?l?l?l?l?l?l' | john --stdin hashes.txt

# With progress monitoring
./jtr_generator '?l?l?l?l?d?d?d?d' | john --stdin --format=raw-md5 hashes.txt

# Keyspace partitioning (for distributed cracking)
# Node 1:
./jtr_generator --start 0 --end 1000000000 '?l?l?l?l?l?l?l?l' | john --stdin hashes.txt

# Node 2:
./jtr_generator --start 1000000000 --end 2000000000 '?l?l?l?l?l?l?l?l' | john --stdin hashes.txt

# Quiet mode (no progress to stderr)
./jtr_generator --quiet '?u?l?l?l?l?l?d?d' | john --stdin hashes.txt
```

### JtR Configuration (Optional)

Add to `john.conf` for convenience:

```ini
[List.External:GPU_Mask]
# External mode that reads from stdin (our generator)
void init()
{
    # No initialization needed
}

void generate()
{
    word = $STDIN;
}
```

**Usage:**
```bash
./jtr_generator '?l?l?l?l?l?l?l?l' | john --external=GPU_Mask hashes.txt
```

### Key Points

- **Format:** Must use `WG_FORMAT_NEWLINES` for JtR stdin compatibility
- **Flushing:** Call `fflush(stdout)` after each batch for streaming
- **Progress:** Write progress to stderr (stdout is for candidates)
- **Partitioning:** Use `--start` and `--end` for distributed cracking
- **Signal handling:** Gracefully handle Ctrl+C and SIGPIPE

---

## Integration Pattern 2: Format Plugin

### Use Case

Integrate directly into a JtR format plugin for zero-copy GPU operation.

### Architecture

```
┌────────────────────────────────────┐
│  John the Ripper Core              │
│  - Load hashes                     │
│  - Call format methods             │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│  Custom Format Plugin              │
│  - Uses libgpu_scatter_gather      │
│  - GPU candidate generation        │
│  - GPU hash computation            │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│  libgpu_scatter_gather             │
│  - Device pointer API              │
│  - Zero-copy operation             │
└────────────────────────────────────┘
```

### Example: Custom GPU MD5 Format

**File:** `src/cuda/fmt_rawmd5_gpu_mask.c`

```c
/*
 * John the Ripper format plugin: Raw MD5 with GPU mask generation
 *
 * Integrates libgpu_scatter_gather for zero-copy candidate generation
 */

#if FMT_EXTERNS_H
extern struct fmt_main fmt_rawmd5_gpu_mask;
#elif FMT_REGISTERS_H
john_register_one(&fmt_rawmd5_gpu_mask);
#else

#include <string.h>
#include <cuda.h>
#include "arch.h"
#include "misc.h"
#include "common.h"
#include "formats.h"
#include "wordlist_generator.h"

#define FORMAT_LABEL            "raw-md5-gpu-mask"
#define FORMAT_NAME             "Raw MD5 (GPU Mask)"
#define ALGORITHM_NAME          "MD5 CUDA"
#define BENCHMARK_COMMENT       ""
#define BENCHMARK_LENGTH        0x107
#define PLAINTEXT_LENGTH        32
#define BINARY_SIZE             16
#define SALT_SIZE               0
#define MIN_KEYS_PER_CRYPT      1000000  // 1M per batch
#define MAX_KEYS_PER_CRYPT      100000000  // 100M per batch

// Format-specific state
static struct {
    wg_WordlistGenerator *gen;
    wg_BatchDevice current_batch;
    uint64_t current_index;
    uint64_t keyspace_size;
    char *mask_pattern;
    int initialized;
} gpu_state = {0};

static struct fmt_tests tests[] = {
    {"5f4dcc3b5aa765d61d8327deb882cf99", "password"},
    {"098f6bcd4621d373cade4e832627b4f6", "test"},
    {NULL}
};

// CUDA kernel (simplified - use your actual MD5 kernel)
__global__ void md5_kernel_packed(
    const char *candidates,
    size_t stride,
    size_t word_length,
    uint64_t count,
    const uint8_t *target_hashes,
    int num_hashes,
    int *found_indices
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const char *word = candidates + (tid * stride);

    // Compute MD5
    uint8_t hash[16];
    // ... (your MD5 implementation)

    // Compare against all loaded hashes
    for (int i = 0; i < num_hashes; i++) {
        if (memcmp(hash, target_hashes + i * 16, 16) == 0) {
            atomicExch(&found_indices[i], (int)tid);
        }
    }
}

static void init(struct fmt_main *self)
{
    // Initialize CUDA
    cuInit(0);
    CUdevice device;
    CUcontext context;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Create generator
    gpu_state.gen = wg_create_with_context(context);
    if (!gpu_state.gen) {
        fprintf(stderr, "Failed to create wordlist generator\n");
        error();
    }

    // Default mask (can be overridden by user)
    gpu_state.mask_pattern = "?l?l?l?l?l?l?l?l";

    // Configure charsets
    wg_add_charset(gpu_state.gen, 'l', "abcdefghijklmnopqrstuvwxyz", 26);
    wg_add_charset(gpu_state.gen, 'u', "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 26);
    wg_add_charset(gpu_state.gen, 'd', "0123456789", 10);

    // Set mask
    if (wg_set_mask(gpu_state.gen, gpu_state.mask_pattern) != 0) {
        fprintf(stderr, "Failed to set mask\n");
        error();
    }

    // Use PACKED format for optimal performance
    wg_set_format(gpu_state.gen, WG_FORMAT_PACKED);

    gpu_state.keyspace_size = wg_keyspace_size(gpu_state.gen);
    gpu_state.current_index = 0;
    gpu_state.initialized = 1;

    fprintf(stderr, "GPU Mask format initialized: %s (%llu candidates)\n",
            gpu_state.mask_pattern, gpu_state.keyspace_size);
}

static void done(void)
{
    if (gpu_state.gen) {
        wg_destroy(gpu_state.gen);
        gpu_state.gen = NULL;
    }
}

static void reset(struct db_main *db)
{
    gpu_state.current_index = 0;
}

static int valid(char *ciphertext, struct fmt_main *self)
{
    // Standard MD5 hex validation
    if (strlen(ciphertext) != 32) return 0;
    for (int i = 0; i < 32; i++) {
        if (!ishex(ciphertext[i])) return 0;
    }
    return 1;
}

static char *prepare(char *split_fields[10], struct fmt_main *self)
{
    return split_fields[1];
}

static void *get_binary(char *ciphertext)
{
    static uint8_t binary[16];
    // Convert hex string to binary
    for (int i = 0; i < 16; i++) {
        sscanf(ciphertext + i * 2, "%2hhx", &binary[i]);
    }
    return binary;
}

static int crypt_all(int *pcount, struct db_salt *salt)
{
    int count = *pcount;

    if (!gpu_state.initialized || !gpu_state.gen) {
        return 0;
    }

    // Generate next batch
    uint64_t batch_size = MIN(count, gpu_state.keyspace_size - gpu_state.current_index);
    if (batch_size == 0) {
        return 0;  // Exhausted keyspace
    }

    // Generate candidates on GPU
    if (wg_generate_batch_device(gpu_state.gen,
                                  gpu_state.current_index,
                                  batch_size,
                                  &gpu_state.current_batch) != 0) {
        fprintf(stderr, "Generation failed\n");
        return 0;
    }

    // Launch hash kernel
    // ... (allocate target hashes, found indices, etc.)
    // ... (launch md5_kernel_packed with current_batch.data)
    // ... (synchronize and check results)

    gpu_state.current_index += batch_size;
    return batch_size;
}

static int cmp_all(void *binary, int count)
{
    // Implement hash comparison
    // ... (check if any generated candidate matches binary hash)
    return 0;  // Simplified
}

static int cmp_one(void *binary, int index)
{
    // Implement single hash comparison
    return 0;  // Simplified
}

static int cmp_exact(char *source, int index)
{
    return 1;  // MD5 is exact (no false positives)
}

static char *get_key(int index)
{
    // Return the candidate at index from current batch
    static char key[PLAINTEXT_LENGTH + 1];

    if (index >= gpu_state.current_batch.count) {
        return "";
    }

    // Copy from device memory to host
    cuMemcpyDtoH(key,
                 gpu_state.current_batch.data + index * gpu_state.current_batch.stride,
                 gpu_state.current_batch.word_length);
    key[gpu_state.current_batch.word_length] = '\0';

    return key;
}

struct fmt_main fmt_rawmd5_gpu_mask = {
    {
        FORMAT_LABEL,
        FORMAT_NAME,
        ALGORITHM_NAME,
        BENCHMARK_COMMENT,
        BENCHMARK_LENGTH,
        0,
        PLAINTEXT_LENGTH,
        BINARY_SIZE,
        DEFAULT_ALIGN,
        SALT_SIZE,
        DEFAULT_ALIGN,
        MIN_KEYS_PER_CRYPT,
        MAX_KEYS_PER_CRYPT,
        FMT_CASE | FMT_8_BIT,
        { NULL },
        { NULL },
        tests
    }, {
        init,
        done,
        reset,
        prepare,
        valid,
        NULL,  // split
        get_binary,
        NULL,  // salt
        { NULL },
        NULL,  // source
        {
            NULL,  // binary_hash_0
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL
        },
        NULL,  // salt_hash
        NULL,  // set_salt
        NULL,  // set_key
        get_key,
        NULL,  // clear_keys
        crypt_all,
        {
            NULL,  // get_hash_0
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL
        },
        cmp_all,
        cmp_one,
        cmp_exact
    }
};

#endif  // FMT_EXTERNS_H
```

### Building with John the Ripper

**Makefile additions:**
```makefile
# Add to John's Makefile
CUDA_LIBS = -L/path/to/gpu-scatter-gather/target/release -lgpu_scatter_gather
CUDA_INCLUDES = -I/path/to/gpu-scatter-gather/include

# Link the format
john: ... fmt_rawmd5_gpu_mask.o
    $(CC) $(LDFLAGS) ... fmt_rawmd5_gpu_mask.o ... $(CUDA_LIBS) -o john
```

### Usage

```bash
# Use the new GPU mask format
john --format=raw-md5-gpu-mask hashes.txt

# The format will automatically generate candidates using GPU
```

### Key Points

- **Initialization:** Create generator in `init()`, destroy in `done()`
- **Format:** Use `WG_FORMAT_PACKED` for optimal GPU memory usage
- **Batch size:** Match JtR's `MAX_KEYS_PER_CRYPT` setting
- **get_key():** Copy candidates from device memory on demand
- **Zero-copy:** Candidates stay on GPU until explicitly copied

---

## Integration Pattern 3: Distributed Cracking

### Use Case

Perfect keyspace partitioning for distributed John the Ripper across multiple nodes.

### Key Advantage

Unlike sequential generators (odometer algorithms), `libgpu_scatter_gather` supports **O(1) random access** to any position in the keyspace, enabling perfect load balancing with zero overlap.

### Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Node 1     │    │   Node 2     │    │   Node 3     │
│ Indices      │    │ Indices      │    │ Indices      │
│ 0 - 1B       │    │ 1B - 2B      │    │ 2B - 3B      │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                  ┌─────────▼────────┐
                  │  Shared Hash File │
                  │  (networked)      │
                  └───────────────────┘
```

### Complete Distributed Example

```c
/*
 * jtr_distributed.c - Distributed cracking with perfect keyspace partitioning
 */

#include "wordlist_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void crack_partition(int worker_id, int total_workers,
                     const char *mask, const char *hash_file) {
    // Create generator
    wg_WordlistGenerator *gen = wg_create();
    if (!gen) {
        fprintf(stderr, "[Worker %d] Failed to create generator\n", worker_id);
        return;
    }

    // Configure charsets
    wg_add_charset(gen, 'l', "abcdefghijklmnopqrstuvwxyz", 26);
    wg_add_charset(gen, 'u', "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 26);
    wg_add_charset(gen, 'd', "0123456789", 10);

    if (wg_set_mask(gen, mask) != 0) {
        fprintf(stderr, "[Worker %d] Invalid mask\n", worker_id);
        wg_destroy(gen);
        return;
    }

    wg_set_format(gen, WG_FORMAT_NEWLINES);

    // Calculate this worker's partition
    uint64_t keyspace = wg_keyspace_size(gen);
    uint64_t per_worker = keyspace / total_workers;
    uint64_t start = worker_id * per_worker;
    uint64_t count = per_worker;

    // Last worker handles remainder
    if (worker_id == total_workers - 1) {
        count = keyspace - start;
    }

    fprintf(stderr, "[Worker %d] Processing indices %llu to %llu (%llu candidates)\n",
            worker_id, start, start + count, count);

    // Generate this worker's partition
    const uint64_t batch_size = 10000000;  // 10M per batch
    size_t buffer_size = wg_calculate_buffer_size(gen, batch_size);
    char *buffer = malloc(buffer_size);

    // Pipe to John the Ripper
    // In production, use MPI or network communication
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "john --stdin --format=raw-md5 --session=worker%d %s",
             worker_id, hash_file);
    FILE *pipe = popen(cmd, "w");
    if (!pipe) {
        fprintf(stderr, "[Worker %d] Failed to open pipe to john\n", worker_id);
        free(buffer);
        wg_destroy(gen);
        return;
    }

    uint64_t processed = 0;
    for (uint64_t i = start; i < start + count; i += batch_size) {
        uint64_t this_batch = (i + batch_size > start + count) ?
                              (start + count - i) : batch_size;

        if (wg_generate_batch_host(gen, i, this_batch, buffer, buffer_size) != 0) {
            fprintf(stderr, "[Worker %d] Generation failed\n", worker_id);
            break;
        }

        size_t word_length = wg_word_length(gen);
        size_t bytes_to_write = this_batch * (word_length + 1);
        fwrite(buffer, 1, bytes_to_write, pipe);
        fflush(pipe);

        processed += this_batch;
        fprintf(stderr, "\r[Worker %d] Progress: %.1f%%",
                worker_id, 100.0 * processed / count);
    }

    fprintf(stderr, "\n[Worker %d] Completed\n", worker_id);

    pclose(pipe);
    free(buffer);
    wg_destroy(gen);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <worker_id> <total_workers> <mask>\n", argv[0]);
        fprintf(stderr, "Example: %s 0 4 '?l?l?l?l?l?l?l?l'\n", argv[0]);
        return 1;
    }

    int worker_id = atoi(argv[1]);
    int total_workers = atoi(argv[2]);
    char *mask = argv[3];

    if (worker_id < 0 || worker_id >= total_workers) {
        fprintf(stderr, "Error: worker_id must be 0 <= id < total_workers\n");
        return 1;
    }

    crack_partition(worker_id, total_workers, mask, "hashes.txt");
    return 0;
}
```

### Usage: Distributed Cracking

**On 4 nodes:**

```bash
# Node 1
./jtr_distributed 0 4 '?l?l?l?l?l?l?l?l'

# Node 2
./jtr_distributed 1 4 '?l?l?l?l?l?l?l?l'

# Node 3
./jtr_distributed 2 4 '?l?l?l?l?l?l?l?l'

# Node 4
./jtr_distributed 3 4 '?l?l?l?l?l?l?l?l'
```

**Properties:**
- ✅ **Zero overlap:** Each node generates distinct candidates
- ✅ **Complete coverage:** All keyspace covered exactly once
- ✅ **Perfect load balancing:** Each node gets equal work (±1)
- ✅ **Fault tolerance:** Failed nodes can restart their partition
- ✅ **Resume support:** Deterministic generation allows safe resume

### Integration with MPI (Optional)

For production distributed cracking:

```c
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each MPI process handles its partition
    crack_partition(rank, size, argv[1], "hashes.txt");

    MPI_Finalize();
    return 0;
}
```

**Run with MPI:**
```bash
mpirun -n 8 -hostfile nodes.txt ./jtr_distributed_mpi '?l?l?l?l?l?l?l?l'
```

---

## Performance Tuning

### Batch Size Recommendations for JtR

| Password Length | Batch Size | Memory | JtR Compatibility |
|-----------------|------------|--------|-------------------|
| 6 chars | 10-50M | 60-300 MB | Optimal |
| 8 chars | 10-50M | 80-400 MB | Optimal |
| 10 chars | 10-30M | 100-300 MB | Recommended |
| 12 chars | 5-20M | 60-240 MB | Safe |

**Tuning:**
```c
// Test different batch sizes
uint64_t batch_sizes[] = {1000000, 5000000, 10000000, 50000000};
for (int i = 0; i < 4; i++) {
    // Benchmark and find optimal batch size
}
```

### Format Mode Selection

| Use Case | Format | Reason |
|----------|--------|--------|
| External mode (stdin) | `NEWLINES` | Required by JtR |
| Format plugin (GPU) | `PACKED` | 11% memory savings |
| Distributed cracking | `NEWLINES` | Pipe compatibility |

### JtR-Specific Optimizations

**1. Match JtR's batch size:**
```c
// In format plugin:
#define MIN_KEYS_PER_CRYPT  1000000
#define MAX_KEYS_PER_CRYPT  100000000

// Generate exactly MAX_KEYS_PER_CRYPT candidates per call
wg_generate_batch_device(gen, index, MAX_KEYS_PER_CRYPT, &batch);
```

**2. Pre-generate next batch:**
```c
// While JtR processes current batch, generate next batch
wg_BatchDevice batch1, batch2;
wg_generate_batch_device(gen, 0, MAX_KEYS, &batch1);

while (!done) {
    // Generate next batch (async)
    wg_generate_batch_device(gen, index + MAX_KEYS, MAX_KEYS, &batch2);

    // Process current batch
    john_crypt_all(batch1);

    // Swap
    batch1 = batch2;
}
```

**3. Reuse buffers:**
```c
// Allocate once, reuse many times
char *buffer = malloc(buffer_size);

for (uint64_t i = 0; i < keyspace; i += batch_size) {
    wg_generate_batch_host(gen, i, batch_size, buffer, buffer_size);
    // Process buffer...
}

free(buffer);
```

---

## Complete Working Example

See `examples/jtr_external_mode.c` in the repository for:

- Complete external mode generator with all features
- Command-line argument parsing
- Progress reporting
- Distributed cracking support
- Error handling

**Build:**
```bash
gcc -o jtr_external_mode jtr_external_mode.c \
    -I../include \
    -L../target/release \
    -lgpu_scatter_gather \
    -Wl,-rpath,../target/release \
    -O3
```

**Usage:**
```bash
./jtr_external_mode --help
./jtr_external_mode '?l?l?l?l?l?l?l?l' | john --stdin hashes.txt
```

---

## Troubleshooting

### Issue 1: JtR Shows "No password hashes loaded"

**Cause:** Hash file format incorrect

**Solution:**
```bash
# Verify hash file format
cat hashes.txt
# Should be: hash_value or username:hash_value

# Test with known hash
echo "5f4dcc3b5aa765d61d8327deb882cf99" > test.txt
./jtr_generator '?l?l?l?l?l?l?l?l' | john --stdin --format=raw-md5 test.txt
```

### Issue 2: Candidates Not Reaching JtR

**Symptom:** Generator runs but JtR shows no progress

**Debug:**
```bash
# Test pipeline
./jtr_generator '?l?l?l?l' | head -10
# Should show candidates

# Check JtR stdin mode
echo -e "test\npassword\nhello" | john --stdin --format=raw-md5 hashes.txt
```

### Issue 3: Slow Performance

**Symptom:** Throughput < 400 M/s

**Checklist:**
- [ ] Batch size too small (increase to 10-50M)
- [ ] GPU underutilized (check `nvidia-smi`)
- [ ] Pipe buffer full (JtR not consuming fast enough)
- [ ] Disk I/O bottleneck (if writing to file)

**Solution:**
```bash
# Benchmark generator alone
./jtr_generator '?l?l?l?l?l?l?l?l' > /dev/null

# Check throughput (should be 400-700 M/s)
```

### Issue 4: Distributed Workers Overlap

**Symptom:** Same candidates generated by multiple workers

**Cause:** Incorrect partition calculation

**Solution:**
```c
// Ensure integer division is correct
uint64_t per_worker = keyspace / total_workers;
uint64_t start = worker_id * per_worker;

// Last worker handles remainder
if (worker_id == total_workers - 1) {
    count = keyspace - start;  // Include remainder
}
```

### Issue 5: Format Plugin Crashes

**Symptom:** Segfault in custom format

**Debug:**
```c
// Add safety checks
if (!gpu_state.initialized || !gpu_state.gen) {
    fprintf(stderr, "Generator not initialized\n");
    return 0;
}

// Validate batch
if (gpu_state.current_batch.count == 0) {
    fprintf(stderr, "Empty batch\n");
    return 0;
}
```

---

## Next Steps

1. **Try the examples:** Build and test `jtr_external_mode.c`
2. **Study the format plugin:** Understand zero-copy integration
3. **Experiment with distributed cracking:** Test keyspace partitioning
4. **Profile your setup:** Measure end-to-end throughput
5. **Contribute back:** Share your JtR integration!

---

## Additional Resources

- **Generic integration guide:** `docs/guides/INTEGRATION_GUIDE.md`
- **Hashcat integration:** `docs/guides/HASHCAT_INTEGRATION.md`
- **C API specification:** `docs/api/C_API_SPECIFICATION.md`
- **Performance benchmarks:** `docs/benchmarking/PERFORMANCE_COMPARISON.md`
- **John the Ripper docs:** https://www.openwall.com/john/doc/

---

**Questions?** File an issue at https://github.com/tehw0lf/gpu-scatter-gather/issues

**End of John the Ripper Integration Guide**
