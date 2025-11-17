/**
 * GPU Scatter-Gather Wordlist Generator - POC Kernel
 *
 * This is a proof-of-concept kernel that measures PURE COMPUTE throughput
 * without any I/O overhead. It generates words but keeps them in registers.
 *
 * Goal: Prove we can achieve 1.47 BILLION words/second on RTX 4070!
 *
 * Theoretical maximum:
 * - RTX 4070: 5,888 CUDA cores @ 2.5 GHz
 * - Assume 10 cycles per word
 * - Max throughput: 5,888 * 2.5 GHz / 10 = 1.47 billion words/s
 */

extern "C" __global__ void poc_generate_words_compute_only(
    const char* __restrict__ charset_data,      // Flat array of all charset chars
    const int* __restrict__ charset_offsets,    // Start index for each charset
    const int* __restrict__ charset_sizes,      // Size of each charset
    const int* __restrict__ mask_pattern,       // Which charset for each position
    unsigned long long start_idx,               // Starting combination index
    int word_length,                            // Number of positions
    unsigned long long batch_size,              // Number of words to generate
    unsigned long long* __restrict__ checksum   // Output checksum (prevents dead code elimination)
) {
    // Calculate global thread ID
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    unsigned long long idx = start_idx + tid;

    // Local word buffer - stays in registers!
    char word[32];  // Max 32 character words

    // Convert index to word using mixed-radix arithmetic
    unsigned long long remaining = idx;

    #pragma unroll
    for (int pos = word_length - 1; pos >= 0; pos--) {
        int charset_id = mask_pattern[pos];
        int cs_size = charset_sizes[charset_id];
        int cs_offset = charset_offsets[charset_id];

        // Mixed-radix decomposition
        int char_idx = remaining % cs_size;
        word[pos] = charset_data[cs_offset + char_idx];
        remaining /= cs_size;
    }

    // Prevent compiler from optimizing away the computation
    // We compute a checksum that will never actually match but forces computation
    if (word[0] == 255) {  // This will never happen with ASCII
        atomicAdd(checksum, (unsigned long long)word[0]);
    }
}

/**
 * Production kernel - Optimized with shared memory caching
 *
 * Memory optimization: Cache charset metadata in shared memory to reduce
 * global memory reads from 16/word (4-char) to ~0.06/word (cooperative loading).
 * This should reduce L2 cache pressure from 97.6% to ~30-40% and unlock compute.
 */
extern "C" __global__ void generate_words_kernel(
    const char* __restrict__ charset_data,      // Flat array of all charset chars
    const int* __restrict__ charset_offsets,    // Start index for each charset
    const int* __restrict__ charset_sizes,      // Size of each charset
    const int* __restrict__ mask_pattern,       // Which charset for each position
    unsigned long long start_idx,               // Starting combination index
    int word_length,                            // Number of positions
    char* __restrict__ output_buffer,           // Output buffer for words
    unsigned long long batch_size               // Number of words to generate
) {
    // Shared memory for charset metadata (fast access, ~15 TB/s vs L2 ~2.3 TB/s)
    __shared__ int s_charset_sizes[32];    // Max 32 unique charsets
    __shared__ int s_charset_offsets[32];
    __shared__ int s_mask_pattern[32];     // Max 32 character word length
    __shared__ char s_charset_data[512];   // Cache up to 512 bytes of charset chars
    __shared__ int s_num_charsets;         // Number of charsets (shared across block)
    __shared__ int s_total_charset_size;   // Total bytes of charset data (shared across block)

    // Cooperative loading: each thread loads one element
    // This reduces global memory reads from N*threads to N (where N = metadata size)
    int tid_local = threadIdx.x;

    // Load mask pattern (word_length elements)
    if (tid_local < word_length) {
        s_mask_pattern[tid_local] = mask_pattern[tid_local];
    }

    // Count unique charsets and total size (only thread 0)
    if (tid_local == 0) {
        // Find max charset ID to know how many charsets we have
        int max_id = 0;
        for (int i = 0; i < word_length; i++) {
            if (mask_pattern[i] > max_id) max_id = mask_pattern[i];
        }
        s_num_charsets = max_id + 1;

        // Calculate total charset data size
        int total_size = 0;
        for (int i = 0; i < s_num_charsets; i++) {
            total_size += charset_sizes[i];
        }
        s_total_charset_size = total_size;
    }
    __syncthreads();

    // Load charset sizes and offsets cooperatively
    if (tid_local < s_num_charsets) {
        s_charset_sizes[tid_local] = charset_sizes[tid_local];
        s_charset_offsets[tid_local] = charset_offsets[tid_local];
    }

    // Load charset data cooperatively (multiple threads working together)
    // Each thread loads one byte until all charset data is loaded
    for (int i = tid_local; i < s_total_charset_size && i < 512; i += blockDim.x) {
        s_charset_data[i] = charset_data[i];
    }

    __syncthreads();  // Ensure all shared memory is loaded before computation

    // Now each thread generates its word using only shared memory (fast!)
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    unsigned long long idx = start_idx + tid;
    char* word = output_buffer + (tid * (word_length + 1)); // +1 for newline

    // Convert index to word - now using SHARED MEMORY (no global reads!)
    unsigned long long remaining = idx;

    #pragma unroll
    for (int pos = word_length - 1; pos >= 0; pos--) {
        int charset_id = s_mask_pattern[pos];          // Shared memory read (fast)
        int cs_size = s_charset_sizes[charset_id];     // Shared memory read (fast)
        int cs_offset = s_charset_offsets[charset_id]; // Shared memory read (fast)

        int char_idx = remaining % cs_size;
        word[pos] = s_charset_data[cs_offset + char_idx]; // Shared memory read (fast)
        remaining /= cs_size;
    }

    word[word_length] = '\n';
}
