//! Test transpose performance in isolation

use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    let num_words = 100_000_000;
    let word_length = 13; // 12 chars + newline

    println!("Transpose Performance Test");
    println!("Words: {} M", num_words / 1_000_000);
    println!("Word length: {word_length}");
    println!("Data size: {} MB\n", (num_words * word_length) / 1_000_000);

    // Create column-major test data
    println!("Generating test data...");
    let mut column_major = vec![0u8; num_words * word_length];
    for char_idx in 0..word_length {
        for word_idx in 0..num_words {
            column_major[char_idx * num_words + word_idx] = ((word_idx + char_idx) % 256) as u8;
        }
    }
    println!("Done.\n");

    // Warm up
    let _ = gpu_scatter_gather::transpose::transpose_to_rowmajor(
        &column_major[0..1000 * word_length],
        1000,
        word_length,
    )?;

    // Benchmark
    println!("Benchmarking transpose...");
    let start = Instant::now();
    let _row_major = gpu_scatter_gather::transpose::transpose_to_rowmajor(
        &column_major,
        num_words,
        word_length,
    )?;
    let duration = start.elapsed();

    let throughput_gb_s = (num_words * word_length) as f64 / duration.as_secs_f64() / 1e9;

    println!("Time: {:.2} ms", duration.as_secs_f64() * 1000.0);
    println!("Throughput: {throughput_gb_s:.2} GB/s");
    println!();

    // Compare to memcpy baseline
    println!("Baseline: memcpy same-size data...");
    let mut dest = vec![0u8; num_words * word_length];
    let start = Instant::now();
    dest.copy_from_slice(&column_major);
    let memcpy_duration = start.elapsed();
    let memcpy_throughput = (num_words * word_length) as f64 / memcpy_duration.as_secs_f64() / 1e9;

    println!("Time: {:.2} ms", memcpy_duration.as_secs_f64() * 1000.0);
    println!("Throughput: {memcpy_throughput:.2} GB/s");
    println!();

    println!(
        "Transpose overhead: {:.1}x slower than memcpy",
        throughput_gb_s / memcpy_throughput
    );

    Ok(())
}
