//! Keyspace calculation and index-to-word conversion
//!
//! This module implements the core algorithm: direct index-to-word mapping using
//! mixed-radix arithmetic. This is what enables massive GPU parallelism.
//!
//! # Algorithm
//!
//! Given an index `i` and a mask pattern with varying charset sizes, we use
//! mixed-radix decomposition to convert the index directly to a word:
//!
//! ```text
//! For mask [charset_0, charset_1, ..., charset_n] with sizes [s0, s1, ..., sn]:
//!
//! word[n] = charsets[n][i % sn]
//! i = i / sn
//! word[n-1] = charsets[n-1][i % sn-1]
//! i = i / sn-1
//! ...
//! ```
//!
//! This allows O(1) random access to any position in the keyspace without
//! sequential iteration.

/// Calculate the total keyspace size for a given mask pattern
///
/// # Arguments
///
/// * `mask` - Array of charset IDs defining the pattern
/// * `charset_sizes` - Size of each charset referenced in the mask
///
/// # Returns
///
/// Total number of possible combinations (product of all charset sizes)
///
/// # Example
///
/// ```
/// use gpu_scatter_gather::keyspace::calculate_keyspace;
///
/// let mask = vec![0, 1, 0];  // Pattern using 2 charsets
/// let charset_sizes = vec![3, 2];  // First charset has 3 chars, second has 2
///
/// assert_eq!(calculate_keyspace(&mask, &charset_sizes), 18); // 3 * 2 * 3 = 18
/// ```
pub fn calculate_keyspace(mask: &[usize], charset_sizes: &[usize]) -> u128 {
    mask.iter()
        .map(|&charset_id| charset_sizes[charset_id] as u128)
        .product()
}

/// Convert a single index to a word using mixed-radix arithmetic
///
/// This is the CPU reference implementation. The GPU kernel implements the same
/// algorithm but processes millions of indices in parallel.
///
/// # Arguments
///
/// * `index` - The combination index to convert (0 to keyspace_size - 1)
/// * `mask` - Array of charset IDs defining the pattern
/// * `charsets` - Array of charset byte arrays
/// * `output` - Output buffer to write the word into (must be at least mask.len() bytes)
///
/// # Algorithm
///
/// Uses mixed-radix decomposition, processing positions from right to left:
/// 1. For each position (rightmost first):
///    - char_index = index % charset_size
///    - output[pos] = charset[char_index]
///    - index = index / charset_size
///
/// # Example
///
/// ```
/// use gpu_scatter_gather::keyspace::index_to_word;
///
/// let mask = vec![0, 1];
/// let charsets = vec![b"abc".as_ref(), b"12".as_ref()];
/// let mut output = vec![0u8; 2];
///
/// index_to_word(0, &mask, &charsets, &mut output);
/// assert_eq!(&output, b"a1");
///
/// index_to_word(1, &mask, &charsets, &mut output);
/// assert_eq!(&output, b"a2");
///
/// index_to_word(2, &mask, &charsets, &mut output);
/// assert_eq!(&output, b"b1");
/// ```
pub fn index_to_word(index: u64, mask: &[usize], charsets: &[&[u8]], output: &mut [u8]) {
    let mut remaining = index;

    // Process positions from right to left (least significant to most significant)
    for pos in (0..mask.len()).rev() {
        let charset_id = mask[pos];
        let charset = charsets[charset_id];
        let charset_size = charset.len() as u64;

        // Mixed-radix decomposition
        let char_idx = (remaining % charset_size) as usize;
        output[pos] = charset[char_idx];
        remaining /= charset_size;
    }
}

/// Validate that a given index is within the keyspace
pub fn validate_index(index: u64, keyspace_size: u128) -> bool {
    (index as u128) < keyspace_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyspace_calculation() {
        // Single position, single charset
        assert_eq!(calculate_keyspace(&[0], &[3]), 3);

        // Two positions, same charset
        assert_eq!(calculate_keyspace(&[0, 0], &[3]), 9); // 3^2

        // Two positions, different charsets
        assert_eq!(calculate_keyspace(&[0, 1], &[3, 2]), 6); // 3 * 2

        // Three positions, mixed
        assert_eq!(calculate_keyspace(&[0, 1, 0], &[3, 2]), 18); // 3 * 2 * 3
    }

    #[test]
    fn test_index_to_word_simple() {
        let mask = vec![0];
        let charsets = vec![b"abc".as_ref()];
        let mut output = vec![0u8; 1];

        // Test all 3 combinations
        index_to_word(0, &mask, &charsets, &mut output);
        assert_eq!(&output, b"a");

        index_to_word(1, &mask, &charsets, &mut output);
        assert_eq!(&output, b"b");

        index_to_word(2, &mask, &charsets, &mut output);
        assert_eq!(&output, b"c");
    }

    #[test]
    fn test_index_to_word_two_positions() {
        let mask = vec![0, 0]; // Both positions use same charset
        let charsets = vec![b"abc".as_ref()];
        let mut output = vec![0u8; 2];

        // Test systematic generation
        let expected = [
            b"aa", b"ab", b"ac", // a + (a,b,c)
            b"ba", b"bb", b"bc", // b + (a,b,c)
            b"ca", b"cb", b"cc", // c + (a,b,c)
        ];

        for (i, expected_word) in expected.iter().enumerate() {
            index_to_word(i as u64, &mask, &charsets, &mut output);
            assert_eq!(
                &output,
                expected_word.as_ref(),
                "Index {} should produce {:?}",
                i,
                std::str::from_utf8(*expected_word).unwrap()
            );
        }
    }

    #[test]
    fn test_index_to_word_mixed_charsets() {
        let mask = vec![0, 1]; // Two different charsets
        let charsets = vec![b"abc".as_ref(), b"12".as_ref()];
        let mut output = vec![0u8; 2];

        // Keyspace: abc × 12 = 6 combinations
        let expected = [b"a1", b"a2", b"b1", b"b2", b"c1", b"c2"];

        for (i, expected_word) in expected.iter().enumerate() {
            index_to_word(i as u64, &mask, &charsets, &mut output);
            assert_eq!(
                &output,
                expected_word.as_ref(),
                "Index {} should produce {:?}",
                i,
                std::str::from_utf8(*expected_word).unwrap()
            );
        }
    }

    #[test]
    fn test_index_to_word_complex_pattern() {
        // Pattern: ?1?2?1 where ?1=abc, ?2=12
        let mask = vec![0, 1, 0];
        let charsets = vec![b"abc".as_ref(), b"12".as_ref()];
        let mut output = vec![0u8; 3];

        // First few combinations
        index_to_word(0, &mask, &charsets, &mut output);
        assert_eq!(&output, b"a1a");

        index_to_word(1, &mask, &charsets, &mut output);
        assert_eq!(&output, b"a1b");

        index_to_word(2, &mask, &charsets, &mut output);
        assert_eq!(&output, b"a1c");

        index_to_word(3, &mask, &charsets, &mut output);
        assert_eq!(&output, b"a2a");

        // Last combination (index 17, keyspace is 3*2*3=18)
        index_to_word(17, &mask, &charsets, &mut output);
        assert_eq!(&output, b"c2c");
    }

    #[test]
    fn test_bijection_property() {
        // Verify that all indices map to unique words (bijection)
        let mask = vec![0, 1];
        let charsets = vec![b"ab".as_ref(), b"12".as_ref()];
        let mut output = vec![0u8; 2];

        let keyspace = calculate_keyspace(&mask, &[2, 2]) as u64;
        let mut seen = std::collections::HashSet::new();

        for i in 0..keyspace {
            index_to_word(i, &mask, &charsets, &mut output);
            let word = output.clone();
            assert!(
                seen.insert(word.clone()),
                "Duplicate word generated: {:?}",
                std::str::from_utf8(&word).unwrap()
            );
        }

        assert_eq!(seen.len(), keyspace as usize);
    }

    #[test]
    fn test_validate_index() {
        let keyspace = calculate_keyspace(&[0, 0], &[3]);
        assert!(validate_index(0, keyspace));
        assert!(validate_index(8, keyspace)); // Last valid index (9 total)
        assert!(!validate_index(9, keyspace)); // Out of range
        assert!(!validate_index(100, keyspace));
    }
}
