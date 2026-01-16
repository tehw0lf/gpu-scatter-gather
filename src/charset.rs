//! Charset management for wordlist generation
//!
//! A charset is a set of characters that can appear at specific positions in generated words.
//! Multiple charsets can be defined and referenced by ID in a mask pattern.

use std::fmt;

/// A charset is a collection of bytes (typically ASCII characters)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Charset {
    bytes: Vec<u8>,
}

impl Charset {
    /// Create a new charset from a byte vector
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Get the length of the charset
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Check if the charset is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Get the underlying bytes
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get a character at a specific index
    #[inline]
    pub fn get(&self, index: usize) -> Option<u8> {
        self.bytes.get(index).copied()
    }

    /// Create a charset from a string slice
    pub fn from_string(s: &str) -> Self {
        Self {
            bytes: s.as_bytes().to_vec(),
        }
    }

    // Common predefined charsets

    /// Lowercase letters: abcdefghijklmnopqrstuvwxyz
    pub fn lowercase() -> Self {
        Self::from_string("abcdefghijklmnopqrstuvwxyz")
    }

    /// Uppercase letters: ABCDEFGHIJKLMNOPQRSTUVWXYZ
    pub fn uppercase() -> Self {
        Self::from_string("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    }

    /// Digits: 0123456789
    pub fn digits() -> Self {
        Self::from_string("0123456789")
    }

    /// Lowercase + uppercase letters
    pub fn alpha() -> Self {
        Self::from_string("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    }

    /// Letters + digits
    pub fn alphanumeric() -> Self {
        Self::from_string("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    }

    /// Common special characters
    pub fn special() -> Self {
        Self::from_string("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    }

    /// All printable ASCII (0x20-0x7E)
    pub fn printable_ascii() -> Self {
        Self::new((0x20u8..=0x7E).collect())
    }
}

impl From<&str> for Charset {
    fn from(s: &str) -> Self {
        Self::from_string(s)
    }
}

impl From<String> for Charset {
    fn from(s: String) -> Self {
        Self {
            bytes: s.into_bytes(),
        }
    }
}

impl From<Vec<u8>> for Charset {
    fn from(bytes: Vec<u8>) -> Self {
        Self::new(bytes)
    }
}

impl fmt::Display for Charset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charset_creation() {
        let charset = Charset::from("abc");
        assert_eq!(charset.len(), 3);
        assert_eq!(charset.get(0), Some(b'a'));
        assert_eq!(charset.get(2), Some(b'c'));
        assert_eq!(charset.get(3), None);
    }

    #[test]
    fn test_predefined_charsets() {
        assert_eq!(Charset::lowercase().len(), 26);
        assert_eq!(Charset::uppercase().len(), 26);
        assert_eq!(Charset::digits().len(), 10);
        assert_eq!(Charset::alpha().len(), 52);
        assert_eq!(Charset::alphanumeric().len(), 62);
    }

    #[test]
    fn test_empty_charset() {
        let charset = Charset::from("");
        assert!(charset.is_empty());
        assert_eq!(charset.len(), 0);
    }

    #[test]
    fn test_charset_display() {
        let charset = Charset::from("abc123");
        assert_eq!(charset.to_string(), "abc123");
    }
}
