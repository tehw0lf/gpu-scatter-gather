//! John the Ripper Piping Benchmark
//!
//! Generates a limited wordlist to stdout for piping to John.
//! Test: ./benchmark_john_pipe | john --stdin --format=raw-md5 /data/claude/test_hashes.txt

use anyhow::Result;
use gpu_scatter_gather::ffi::WG_FORMAT_PACKED;
use gpu_scatter_gather::multigpu::MultiGpuContext;
use gpu_scatter_gather::Charset;
use std::collections::HashMap;
use std::env;
use std::io::{self, Write};

fn main() -> Result<()> {
    // Get batch count from args, default to 20 (1 billion words)
    let args: Vec<String> = env::args().collect();
    let total_batches: u64 = if args.len() > 1 {
        args[1].parse().unwrap_or(20)
    } else {
        20
    };

    // Write info to stderr
    eprintln!("🚀 John the Ripper Piping Benchmark");
    eprintln!("{}", "=".repeat(70));
    eprintln!();

    let mut ctx = MultiGpuContext::new()?;
    eprintln!("✅ MultiGpuContext initialized");
    eprintln!();

    // Charset: lowercase only
    let lowercase = Charset::new(b"abcdefghijklmnopqrstuvwxyz".to_vec());
    let mut charsets = HashMap::new();
    charsets.insert(0, lowercase.as_bytes().to_vec());

    // Shorter mask for faster testing: 8 chars
    let mask = vec![0; 8]; // ?l?l?l?l?l?l?l?l
    let batch_size = 50_000_000u64;

    eprintln!("Configuration:");
    eprintln!("  Mask: ?l?l?l?l?l?l?l?l (8 lowercase)");
    eprintln!("  Batch size: {batch_size} words");
    eprintln!("  Total batches: {total_batches}");
    eprintln!(
        "  Total words: {:.2} billion",
        (batch_size * total_batches) as f64 / 1e9
    );
    eprintln!();

    eprintln!("📝 Piping to John the Ripper...");
    eprintln!();

    // Large buffer for stdout
    let stdout = io::stdout();
    let mut writer = io::BufWriter::with_capacity(16 * 1024 * 1024, stdout.lock());

    // Generate and pipe to stdout
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

        if (batch + 1) % 10 == 0 {
            eprintln!(
                "  Batch {}/{} ({:.2} M words)",
                batch + 1,
                total_batches,
                ((batch + 1) * batch_size) as f64 / 1e6
            );
        }
    }

    writer.flush()?;
    eprintln!();
    eprintln!(
        "✅ Done! Piped {:.2} billion words",
        (batch_size * total_batches) as f64 / 1e9
    );

    Ok(())
}
