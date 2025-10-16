//! GPU Scatter-Gather Wordlist Generator CLI
//!
//! Command-line interface for the world's fastest wordlist generator.

use anyhow::Result;
use clap::Parser;
use gpu_scatter_gather::{Charset, WordlistGenerator};

/// GPU-accelerated wordlist generator
#[derive(Parser, Debug)]
#[command(name = "gpu-scatter-gather")]
#[command(about = "World's fastest wordlist generator using GPU acceleration", long_about = None)]
#[command(version)]
struct Args {
    /// Mask pattern (e.g., "?1?2?1" for charset1 + charset2 + charset1)
    #[arg(value_name = "MASK")]
    mask: String,

    /// Charset 1
    #[arg(short = '1', long, value_name = "CHARSET")]
    charset1: Option<String>,

    /// Charset 2
    #[arg(short = '2', long, value_name = "CHARSET")]
    charset2: Option<String>,

    /// Charset 3
    #[arg(short = '3', long, value_name = "CHARSET")]
    charset3: Option<String>,

    /// Charset 4
    #[arg(short = '4', long, value_name = "CHARSET")]
    charset4: Option<String>,

    /// Use lowercase letters for charset (can specify ID with =N)
    #[arg(long)]
    lowercase: Option<Option<usize>>,

    /// Use uppercase letters for charset (can specify ID with =N)
    #[arg(long)]
    uppercase: Option<Option<usize>>,

    /// Use digits for charset (can specify ID with =N)
    #[arg(long)]
    digits: Option<Option<usize>>,

    /// Show keyspace size and exit
    #[arg(short = 'k', long)]
    keyspace: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logger
    if args.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    log::info!("GPU Scatter-Gather Wordlist Generator v{}", env!("CARGO_PKG_VERSION"));

    // Build charset map
    let mut builder = WordlistGenerator::builder();

    // Add explicit charsets
    if let Some(cs1) = args.charset1 {
        builder = builder.charset(1, Charset::from(cs1));
    }
    if let Some(cs2) = args.charset2 {
        builder = builder.charset(2, Charset::from(cs2));
    }
    if let Some(cs3) = args.charset3 {
        builder = builder.charset(3, Charset::from(cs3));
    }
    if let Some(cs4) = args.charset4 {
        builder = builder.charset(4, Charset::from(cs4));
    }

    // Add predefined charsets
    if let Some(id) = args.lowercase {
        let id = id.unwrap_or(1);
        builder = builder.charset(id, Charset::lowercase());
    }
    if let Some(id) = args.uppercase {
        let id = id.unwrap_or(1);
        builder = builder.charset(id, Charset::uppercase());
    }
    if let Some(id) = args.digits {
        let id = id.unwrap_or(1);
        builder = builder.charset(id, Charset::digits());
    }

    // Parse mask
    let mask = gpu_scatter_gather::Mask::parse(&args.mask)?;
    builder = builder.mask(mask.pattern());

    // Build generator
    let generator = builder.build()?;

    // Show keyspace if requested
    let keyspace_size = generator.keyspace_size();
    if args.keyspace {
        println!("Keyspace size: {}", keyspace_size);
        return Ok(());
    }

    log::info!("Generating {} words...", keyspace_size);
    log::info!("Mask: {}", args.mask);

    // Generate wordlist to stdout (CPU reference implementation for now)
    for word in generator.iter() {
        println!("{}", String::from_utf8_lossy(&word));
    }

    log::info!("Generation complete!");

    Ok(())
}
