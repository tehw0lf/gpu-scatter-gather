//! Mask pattern parsing and management
//!
//! A mask defines the pattern of a wordlist using charset references.
//! For example: "?1?2?1" means position 0 uses charset 1, position 1 uses charset 2,
//! and position 2 uses charset 1 again.

use anyhow::{bail, Result};

/// A mask pattern defining which charset to use at each position
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    /// Pattern of charset IDs
    pattern: Vec<usize>,
}

impl Mask {
    /// Create a new mask from a pattern
    pub fn new(pattern: Vec<usize>) -> Self {
        Self { pattern }
    }

    /// Parse a mask string like "?1?2?1" into a pattern
    ///
    /// # Format
    ///
    /// - `?1`, `?2`, ... `?9` - Reference to charset ID 1-9
    /// - Multiple references can be concatenated: "?1?2?3"
    ///
    /// # Example
    ///
    /// ```
    /// use gpu_scatter_gather::Mask;
    ///
    /// let mask = Mask::parse("?1?2?1").unwrap();
    /// assert_eq!(mask.pattern(), &[1, 2, 1]);
    /// ```
    pub fn parse(s: &str) -> Result<Self> {
        let mut pattern = Vec::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '?' {
                // Expect a digit after '?'
                if let Some(digit_char) = chars.next() {
                    if let Some(digit) = digit_char.to_digit(10) {
                        if digit == 0 {
                            bail!("Charset ID cannot be 0 (must be 1-9)");
                        }
                        pattern.push(digit as usize);
                    } else {
                        bail!("Expected digit after '?', found '{digit_char}'");
                    }
                } else {
                    bail!("Unexpected end of string after '?'");
                }
            } else {
                bail!("Unexpected character '{ch}' in mask (expected '?')");
            }
        }

        if pattern.is_empty() {
            bail!("Mask pattern cannot be empty");
        }

        Ok(Self { pattern })
    }

    /// Get the pattern
    pub fn pattern(&self) -> &[usize] {
        &self.pattern
    }

    /// Get the length (number of positions)
    pub fn len(&self) -> usize {
        self.pattern.len()
    }

    /// Check if the mask is empty
    pub fn is_empty(&self) -> bool {
        self.pattern.is_empty()
    }

    /// Get the maximum charset ID referenced
    pub fn max_charset_id(&self) -> Option<usize> {
        self.pattern.iter().max().copied()
    }

    /// Convert back to string representation
    pub fn to_string_representation(&self) -> String {
        self.pattern.iter().map(|&id| format!("?{id}")).collect()
    }
}

impl From<Vec<usize>> for Mask {
    fn from(pattern: Vec<usize>) -> Self {
        Self::new(pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_parse_simple() {
        let mask = Mask::parse("?1?2?3").unwrap();
        assert_eq!(mask.pattern(), &[1, 2, 3]);
        assert_eq!(mask.len(), 3);
    }

    #[test]
    fn test_mask_parse_repeated() {
        let mask = Mask::parse("?1?1?1").unwrap();
        assert_eq!(mask.pattern(), &[1, 1, 1]);
    }

    #[test]
    fn test_mask_parse_mixed() {
        let mask = Mask::parse("?1?2?1?3").unwrap();
        assert_eq!(mask.pattern(), &[1, 2, 1, 3]);
    }

    #[test]
    fn test_mask_parse_all_digits() {
        let mask = Mask::parse("?1?2?3?4?5?6?7?8?9").unwrap();
        assert_eq!(mask.pattern(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_mask_parse_empty() {
        let result = Mask::parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_mask_parse_invalid_no_question() {
        let result = Mask::parse("123");
        assert!(result.is_err());
    }

    #[test]
    fn test_mask_parse_invalid_no_digit() {
        let result = Mask::parse("?a");
        assert!(result.is_err());
    }

    #[test]
    fn test_mask_parse_invalid_zero() {
        let result = Mask::parse("?0");
        assert!(result.is_err());
    }

    #[test]
    fn test_mask_parse_incomplete() {
        let result = Mask::parse("?1?");
        assert!(result.is_err());
    }

    #[test]
    fn test_max_charset_id() {
        let mask = Mask::parse("?1?3?2").unwrap();
        assert_eq!(mask.max_charset_id(), Some(3));

        let mask = Mask::parse("?5").unwrap();
        assert_eq!(mask.max_charset_id(), Some(5));
    }

    #[test]
    fn test_to_string_representation() {
        let mask = Mask::parse("?1?2?3").unwrap();
        assert_eq!(mask.to_string_representation(), "?1?2?3");

        let mask = Mask::new(vec![1, 1, 2]);
        assert_eq!(mask.to_string_representation(), "?1?1?2");
    }

    #[test]
    fn test_from_vec() {
        let mask = Mask::from(vec![1, 2, 3]);
        assert_eq!(mask.pattern(), &[1, 2, 3]);
    }
}
