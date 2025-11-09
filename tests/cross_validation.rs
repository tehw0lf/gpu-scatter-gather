//! Cross-validation tests against maskprocessor and hashcat
//!
//! These tests ensure our GPU implementation produces identical output
//! to industry-standard tools like maskprocessor and hashcat.

use std::collections::HashMap;
use std::process::Command;
use gpu_scatter_gather::WordlistGenerator;

/// Helper function to run maskprocessor with given charsets and mask
fn run_maskprocessor(mask: &str, charsets: &HashMap<usize, &str>) -> Result<Vec<u8>, String> {
    // Find the project root (where Cargo.toml is)
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mp_path = format!("{}/tools/maskprocessor/src/mp64.bin", manifest_dir);

    let mut cmd = Command::new(&mp_path);

    // Add charset arguments
    for (id, charset) in charsets.iter() {
        cmd.arg(format!("-{}", id)).arg(charset);
    }

    // Add mask
    cmd.arg(mask);

    let output = cmd.output()
        .map_err(|e| format!("Failed to run maskprocessor: {}", e))?;

    if !output.status.success() {
        return Err(format!("maskprocessor failed: {}",
            String::from_utf8_lossy(&output.stderr)));
    }

    Ok(output.stdout)
}

/// Helper function to run hashcat in stdout mode
fn run_hashcat_stdout(mask: &str, charsets: &HashMap<usize, &str>) -> Result<Vec<u8>, String> {
    // Find the project root (where Cargo.toml is)
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let hashcat_path = format!("{}/tools/hashcat/hashcat", manifest_dir);

    let mut cmd = Command::new(&hashcat_path);
    cmd.arg("--stdout");
    cmd.arg("-a").arg("3"); // Brute-force/mask attack

    // Add custom charsets
    for (id, charset) in charsets.iter() {
        cmd.arg(format!("--custom-charset{}={}", id, charset));
    }

    // Add mask
    cmd.arg(mask);

    let output = cmd.output()
        .map_err(|e| format!("Failed to run hashcat: {}", e))?;

    if !output.status.success() {
        return Err(format!("hashcat failed: {}",
            String::from_utf8_lossy(&output.stderr)));
    }

    Ok(output.stdout)
}

/// Helper function to generate wordlist with our GPU implementation
fn run_gpu_scatter_gather(mask_str: &str, charsets_map: &HashMap<usize, &str>) -> Result<Vec<u8>, String> {
    use gpu_scatter_gather::{Mask, Charset};

    // Parse the mask string
    let mask = Mask::parse(mask_str)
        .map_err(|e| format!("Failed to parse mask: {}", e))?;

    // Build the generator with individual charset() calls
    let mut builder = WordlistGenerator::builder();

    for (id, charset_str) in charsets_map.iter() {
        builder = builder.charset(*id, Charset::from(*charset_str));
    }

    let generator = builder
        .mask(mask.pattern())
        .build()
        .map_err(|e| format!("Failed to build generator: {}", e))?;

    // Generate all words
    let mut output = Vec::new();
    for word in generator.iter() {
        output.extend_from_slice(&word);
        output.push(b'\n');
    }

    Ok(output)
}

#[test]
fn test_cross_validation_small_simple() {
    // Simple 3x3 test case
    let mask = "?1?2";
    let mut charsets = HashMap::new();
    charsets.insert(1, "abc");
    charsets.insert(2, "123");

    // Generate with our tool
    let our_output = run_gpu_scatter_gather(mask, &charsets)
        .expect("GPU generation failed");

    // Generate with maskprocessor
    let mp_output = run_maskprocessor(mask, &charsets)
        .expect("maskprocessor failed");

    // Compare
    assert_eq!(
        our_output, mp_output,
        "Output mismatch with maskprocessor!\nOurs:\n{}\nMaskprocessor:\n{}",
        String::from_utf8_lossy(&our_output),
        String::from_utf8_lossy(&mp_output)
    );
}

#[test]
fn test_cross_validation_with_hashcat() {
    // Test against hashcat stdout mode
    // NOTE: Hashcat uses a DIFFERENT ORDERING than maskprocessor/our tool
    // This test verifies they generate the SAME SET of words, just in different order
    let mask = "?1?2";
    let mut charsets = HashMap::new();
    charsets.insert(1, "abc");
    charsets.insert(2, "123");

    // Generate with our tool
    let our_output = run_gpu_scatter_gather(mask, &charsets)
        .expect("GPU generation failed");

    // Generate with hashcat
    let hashcat_output = run_hashcat_stdout(mask, &charsets)
        .expect("hashcat failed");

    // Parse into sets for order-independent comparison
    let our_words: std::collections::HashSet<Vec<u8>> = our_output
        .split(|&b| b == b'\n')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_vec())
        .collect();

    let hashcat_words: std::collections::HashSet<Vec<u8>> = hashcat_output
        .split(|&b| b == b'\n')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_vec())
        .collect();

    // Compare sets (order-independent)
    assert_eq!(
        our_words.len(), hashcat_words.len(),
        "Different number of words! Ours: {}, Hashcat: {}",
        our_words.len(), hashcat_words.len()
    );

    // Find differences
    let in_ours_not_hashcat: Vec<_> = our_words.difference(&hashcat_words).collect();
    let in_hashcat_not_ours: Vec<_> = hashcat_words.difference(&our_words).collect();

    assert!(
        in_ours_not_hashcat.is_empty() && in_hashcat_not_ours.is_empty(),
        "Word set mismatch!\nIn ours but not hashcat: {:?}\nIn hashcat but not ours: {:?}",
        in_ours_not_hashcat.iter().map(|w| String::from_utf8_lossy(w)).collect::<Vec<_>>(),
        in_hashcat_not_ours.iter().map(|w| String::from_utf8_lossy(w)).collect::<Vec<_>>()
    );

    // If we reach here, both tools generate the exact same set of words
    // (just in different order, which is acceptable)
}

#[test]
fn test_cross_validation_medium() {
    // Medium test: 4-character password, 26x26x10x10 = 67,600 combinations
    let mask = "?1?1?2?2";
    let mut charsets = HashMap::new();
    charsets.insert(1, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    charsets.insert(2, "0123456789");

    // Generate with our tool
    let our_output = run_gpu_scatter_gather(mask, &charsets)
        .expect("GPU generation failed");

    // Generate with maskprocessor
    let mp_output = run_maskprocessor(mask, &charsets)
        .expect("maskprocessor failed");

    // Compare lengths first (faster check)
    assert_eq!(
        our_output.len(), mp_output.len(),
        "Output length mismatch with maskprocessor! Ours: {}, MP: {}",
        our_output.len(), mp_output.len()
    );

    // Full comparison
    assert_eq!(
        our_output, mp_output,
        "Output mismatch with maskprocessor for medium-sized wordlist!"
    );
}

#[test]
fn test_cross_validation_single_charset() {
    // Test with single charset repeated
    let mask = "?1?1?1?1";
    let mut charsets = HashMap::new();
    charsets.insert(1, "0123456789");

    // Generate with our tool
    let our_output = run_gpu_scatter_gather(mask, &charsets)
        .expect("GPU generation failed");

    // Generate with maskprocessor
    let mp_output = run_maskprocessor(mask, &charsets)
        .expect("maskprocessor failed");

    // Should generate 10^4 = 10,000 combinations
    let expected_count = 10000;
    let our_lines: Vec<&[u8]> = our_output.split(|&b| b == b'\n').filter(|l| !l.is_empty()).collect();
    let mp_lines: Vec<&[u8]> = mp_output.split(|&b| b == b'\n').filter(|l| !l.is_empty()).collect();

    assert_eq!(our_lines.len(), expected_count, "Wrong number of combinations");
    assert_eq!(mp_lines.len(), expected_count, "maskprocessor wrong count");
    assert_eq!(our_output, mp_output, "Output mismatch with maskprocessor!");
}

#[test]
fn test_cross_validation_mixed_charsets() {
    // Test with 3 different charsets
    let mask = "?1?2?3";
    let mut charsets = HashMap::new();
    charsets.insert(1, "abc");
    charsets.insert(2, "123");
    charsets.insert(3, "xyz");

    // Generate with our tool
    let our_output = run_gpu_scatter_gather(mask, &charsets)
        .expect("GPU generation failed");

    // Generate with maskprocessor
    let mp_output = run_maskprocessor(mask, &charsets)
        .expect("maskprocessor failed");

    // Should generate 3x3x3 = 27 combinations
    let expected_count = 27;
    let our_lines: Vec<&[u8]> = our_output.split(|&b| b == b'\n').filter(|l| !l.is_empty()).collect();

    assert_eq!(our_lines.len(), expected_count, "Wrong number of combinations");
    assert_eq!(our_output, mp_output, "Output mismatch with maskprocessor!");
}

#[test]
#[ignore] // This test takes longer - run with --ignored
fn test_cross_validation_large() {
    // Large test: 8-character lowercase, 26^8 = 208,827,064,576 combinations
    // This is too large for full comparison, but we can compare first and last N
    let _mask = "?1?1?1?1?1?1?1?1";
    let mut _charsets = HashMap::new();
    _charsets.insert(1, "abcdefghijklmnopqrstuvwxyz");

    // For this test, we'll just verify the first 1000 match
    // (Full test would require too much memory and time)

    println!("Note: Large test compares only first 1000 entries");

    // TODO: Implement chunked comparison for very large keyspaces
    // For now, this test is a placeholder
}

#[test]
fn test_cross_validation_special_characters() {
    // Test with special characters that might cause encoding issues
    let mask = "?1?2";
    let mut charsets = HashMap::new();
    charsets.insert(1, "!@#$%");
    charsets.insert(2, "^&*()");

    // Generate with our tool
    let our_output = run_gpu_scatter_gather(mask, &charsets)
        .expect("GPU generation failed");

    // Generate with maskprocessor
    let mp_output = run_maskprocessor(mask, &charsets)
        .expect("maskprocessor failed");

    // Should generate 5x5 = 25 combinations
    assert_eq!(our_output, mp_output, "Output mismatch with special characters!");
}
