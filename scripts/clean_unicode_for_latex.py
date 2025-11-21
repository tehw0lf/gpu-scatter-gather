#!/usr/bin/env python3
"""Clean unicode characters from markdown for LaTeX compatibility"""

import re
import sys

def clean_unicode(text):
    """Replace unicode with LaTeX-friendly alternatives"""

    # Emojis
    replacements = {
        '✅': '[YES]',
        '❌': '[NO]',
        '⚠️': '[WARNING]',
        '🚀': '',
        '🔄': '',
        '⚡': '',
        '🦀': '',
        '🤖': '',
        '🏆': '',
        '📈': '',
        '🎯': '',
        '✓': '[x]',
        '✗': '[ ]',

        # Math symbols
        '∏': 'product',
        'Σ': 'Sum',
        '∑': 'sum',
        '∀': 'forall',
        '∃': 'exists',
        '⟹': '=>',
        '⟨': '<',
        '⟩': '>',
        '∈': 'in',
        'ℕ': 'N',
        '×': 'x',
        '→': '->',
        '⌊': 'floor(',
        '⌋': ')',
        '≤': '<=',
        '≥': '>=',
        '≠': '!=',
        '≈': '~=',
        '±': '+/-',
        '·': '*',
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove combining diacritics and other unicode
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'docs/WHITEPAPER.md'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'docs/WHITEPAPER_PDF.md'

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    cleaned = clean_unicode(content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    print(f"Cleaned {input_file} -> {output_file}")
