//! Stdout Benchmark - Test Piping Performance
//!
//! Measures throughput when writing to stdout for piping to other tools.
//! Example: ./benchmark_stdout | pv -r > /dev/null

use anyhow::Result;
use gpu_scatter_gather::ffi::WG_FORMAT_PACKED;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use gpu_scatter_gather::Charset;
use std::collections::HashMap;
use std::io::{self, Write};

fn main() -> Result<()> {
    // Write info to stderr so it doesn't interfere with stdout wordlist
    eprintln!("🚀 Stdout Piping Benchmark");
    eprintln!("{}", "=".repeat(70));
    eprintln!();

    // Create MultiGpuContext
    let mut ctx = MultiGpuContext::new()?;
    eprintln!("✅ MultiGpuContext initialized");
    eprintln!();

    // Charset: lowercase only
    let lowercase = Charset::new(b"abcdefghijklmnopqrstuvwxyz".to_vec());

    let mut charsets = HashMap::new();
    charsets.insert(0, lowercase.as_bytes().to_vec());

    // Mask: 16 lowercase characters (same as all other benchmarks)
    let mask = vec![0; 16];
    let batch_size = 50_000_000u64; // 50M words per batch
    let total_batches = 100; // 5 billion words total

    eprintln!("Configuration:");
    eprintln!("  Mask: ?l?l?l?l?l?l?l?l?l?l?l?l?l?l?l?l (16 lowercase)");
    eprintln!("  Batch size: {batch_size} words");
    eprintln!("  Total batches: {total_batches}");
    eprintln!(
        "  Total words: {:.2} billion",
        (batch_size * total_batches) as f64 / 1e9
    );
    eprintln!();

    eprintln!("📝 Writing to stdout...");
    eprintln!("   Pipe this to another tool: ./benchmark_stdout | pv -r > /dev/null");
    eprintln!();

    // Use stdout with large buffer for maximum throughput
    let stdout = io::stdout();
    let mut writer = io::BufWriter::with_capacity(16 * 1024 * 1024, stdout.lock());

    // Generate and write to stdout
    for batch in 0..total_batches {
        let start_index = batch * batch_size;

        ctx.generate_batch_with(
            &charsets,
            &mask,
            start_index,
            batch_size,
            WG_FORMAT_PACKED,
            |data| writer.write_all(data),
        )?;

        // Progress to stderr every 10 batches
        if (batch + 1) % 10 == 0 {
            eprintln!(
                "  Batch {}/{} complete ({:.2} billion words written)",
                batch + 1,
                total_batches,
                ((batch + 1) * batch_size) as f64 / 1e9
            );
        }
    }

    writer.flush()?;

    eprintln!();
    eprintln!(
        "✅ Complete! Generated {:.2} billion words",
        (batch_size * total_batches) as f64 / 1e9
    );

    Ok(())
}
