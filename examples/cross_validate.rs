//! Cross-Validation Against External Tools
//!
//! This validates our GPU output against external wordlist generators
//! (maskprocessor, hashcat, etc.) to ensure we haven't just perfectly
//! reproduced bugs from our CPU reference implementation.
//!
//! This is CRITICAL because:
//! - GPU vs CPU only validates internal consistency
//! - External tools provide independent ground truth
//! - Catches algorithmic bugs in our implementation

use anyhow::Result;
use gpu_scatter_gather::gpu::GpuContext;
use gpu_scatter_gather::Charset;
use std::process::Command;

fn main() -> Result<()> {
    println!("🔍 Cross-Validation Against External Tools");
    println!("{}", "=".repeat(70));
    println!();

    // Test pattern that external tools can generate
    // Using simple pattern that's widely supported
    let pattern = "?1?2";
    let charset1 = "abc";
    let charset2 = "123";

    println!("Test pattern: {}", pattern);
    println!("  Charset 1 (?1): {}", charset1);
    println!("  Charset 2 (?2): {}", charset2);
    println!("  Expected: 9 words (3 × 3)");
    println!();

    // Generate with our GPU implementation
    println!("Generating with GPU implementation...");
    let gpu = GpuContext::new()?;

    let mut charsets = std::collections::HashMap::new();
    charsets.insert(0, charset1.as_bytes().to_vec());
    charsets.insert(1, charset2.as_bytes().to_vec());

    let gpu_output = gpu.generate_batch(&charsets, &[0, 1], 0, 9)?;
    let gpu_words: Vec<&[u8]> = gpu_output
        .split(|&b| b == b'\n')
        .filter(|w| !w.is_empty())
        .collect();

    println!("✅ GPU generated {} words", gpu_words.len());
    println!();

    // Try to generate with maskprocessor if available
    println!("Checking for maskprocessor...");
    match Command::new("maskprocessor")
        .arg("-1")
        .arg(charset1)
        .arg("-2")
        .arg(charset2)
        .arg(pattern)
        .output()
    {
        Ok(output) if output.status.success() => {
            println!("✅ maskprocessor found!");
            println!();

            let mp_output = String::from_utf8_lossy(&output.stdout);
            let mp_words: Vec<&str> = mp_output.lines().collect();

            println!("maskprocessor generated {} words", mp_words.len());
            println!();

            // Compare outputs
            println!("Comparing GPU vs maskprocessor:");
            println!("{}", "-".repeat(70));

            let mut matches = 0;
            let mut mismatches = 0;

            for (i, (gpu_word, mp_word)) in gpu_words.iter().zip(mp_words.iter()).enumerate() {
                let gpu_str = String::from_utf8_lossy(gpu_word);

                if gpu_str == *mp_word {
                    println!("[{}] ✅ {} == {} (GPU vs maskprocessor)", i, gpu_str, mp_word);
                    matches += 1;
                } else {
                    println!("[{}] ❌ {} != {} (MISMATCH!)", i, gpu_str, mp_word);
                    mismatches += 1;
                }
            }

            println!("{}", "-".repeat(70));
            println!();

            if gpu_words.len() != mp_words.len() {
                println!("❌ LENGTH MISMATCH!");
                println!("   GPU: {} words", gpu_words.len());
                println!("   maskprocessor: {} words", mp_words.len());
                anyhow::bail!("Output length mismatch");
            }

            if mismatches > 0 {
                println!("❌ VALIDATION FAILED!");
                println!("   Matches: {}", matches);
                println!("   Mismatches: {}", mismatches);
                anyhow::bail!("GPU output does not match maskprocessor");
            }

            println!("🎉 SUCCESS! GPU matches maskprocessor perfectly!");
            println!("   All {} words validated against external ground truth", matches);
        }
        Ok(output) => {
            println!("⚠️  maskprocessor exited with error: {:?}", output.status);
            println!("   stderr: {}", String::from_utf8_lossy(&output.stderr));
            println!();
            println!("Skipping maskprocessor validation...");
        }
        Err(e) => {
            println!("⚠️  maskprocessor not found: {}", e);
            println!();
            println!("To install maskprocessor:");
            println!("  git clone https://github.com/hashcat/maskprocessor");
            println!("  cd maskprocessor");
            println!("  make");
            println!("  sudo make install");
            println!();
            println!("Skipping maskprocessor validation...");
        }
    }

    println!();

    // Try hashcat if available
    println!("Checking for hashcat...");
    match Command::new("hashcat")
        .arg("--stdout")
        .arg("-a")
        .arg("3")
        .arg("-1")
        .arg(charset1)
        .arg("-2")
        .arg(charset2)
        .arg(pattern)
        .output()
    {
        Ok(output) if output.status.success() => {
            println!("✅ hashcat found!");
            println!();

            let hc_output = String::from_utf8_lossy(&output.stdout);
            let hc_words: Vec<&str> = hc_output.lines().collect();

            println!("hashcat generated {} words", hc_words.len());
            println!();

            // Compare outputs
            println!("Comparing GPU vs hashcat:");
            println!("{}", "-".repeat(70));

            let mut matches = 0;
            let mut mismatches = 0;

            for (i, (gpu_word, hc_word)) in gpu_words.iter().zip(hc_words.iter()).enumerate() {
                let gpu_str = String::from_utf8_lossy(gpu_word);

                if gpu_str == *hc_word {
                    println!("[{}] ✅ {} == {} (GPU vs hashcat)", i, gpu_str, hc_word);
                    matches += 1;
                } else {
                    println!("[{}] ❌ {} != {} (MISMATCH!)", i, gpu_str, hc_word);
                    mismatches += 1;
                }
            }

            println!("{}", "-".repeat(70));
            println!();

            if gpu_words.len() != hc_words.len() {
                println!("❌ LENGTH MISMATCH!");
                println!("   GPU: {} words", gpu_words.len());
                println!("   hashcat: {} words", hc_words.len());
                anyhow::bail!("Output length mismatch");
            }

            if mismatches > 0 {
                println!("❌ VALIDATION FAILED!");
                println!("   Matches: {}", matches);
                println!("   Mismatches: {}", mismatches);
                anyhow::bail!("GPU output does not match hashcat");
            }

            println!("🎉 SUCCESS! GPU matches hashcat perfectly!");
            println!("   All {} words validated against external ground truth", matches);
        }
        Ok(output) => {
            println!("⚠️  hashcat exited with error: {:?}", output.status);
            println!("   stderr: {}", String::from_utf8_lossy(&output.stderr));
            println!();
            println!("Skipping hashcat validation...");
        }
        Err(e) => {
            println!("⚠️  hashcat not found: {}", e);
            println!();
            println!("To install hashcat:");
            println!("  # Arch Linux:");
            println!("  sudo pacman -S hashcat");
            println!("  # Ubuntu/Debian:");
            println!("  sudo apt install hashcat");
            println!();
            println!("Skipping hashcat validation...");
        }
    }

    println!();
    println!("{}", "=".repeat(70));
    println!("CROSS-VALIDATION SUMMARY");
    println!("{}", "=".repeat(70));
    println!();
    println!("This validation provides independent verification that our");
    println!("implementation is correct, not just internally consistent.");
    println!();
    println!("If both maskprocessor and hashcat match our output, we can");
    println!("be confident we haven't reproduced bugs from our CPU reference.");
    println!();

    Ok(())
}
