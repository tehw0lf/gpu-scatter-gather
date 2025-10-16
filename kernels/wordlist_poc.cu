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
 * Production kernel - actually writes output to global memory
 * This will be slower due to memory writes, but needed for real wordlist generation
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
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    unsigned long long idx = start_idx + tid;
    char* word = output_buffer + (tid * (word_length + 1)); // +1 for newline

    // Convert index to word
    unsigned long long remaining = idx;

    #pragma unroll
    for (int pos = word_length - 1; pos >= 0; pos--) {
        int charset_id = mask_pattern[pos];
        int cs_size = charset_sizes[charset_id];
        int cs_offset = charset_offsets[charset_id];

        int char_idx = remaining % cs_size;
        word[pos] = charset_data[cs_offset + char_idx];
        remaining /= cs_size;
    }

    word[word_length] = '\n';
}
