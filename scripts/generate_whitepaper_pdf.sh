#!/bin/bash
# Generate PDF whitepaper from Markdown with diagrams

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_ROOT/docs"
OUTPUT_DIR="$PROJECT_ROOT/whitepaper_output"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}GPU Scatter-Gather Whitepaper Generator${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓${NC} Created output directory: $OUTPUT_DIR"

# Check dependencies
echo ""
echo -e "${BLUE}Checking dependencies...${NC}"

check_dependency() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

MISSING_DEPS=0

if ! check_dependency "pandoc"; then
    echo -e "${YELLOW}  Install: apt-get install pandoc (Debian/Ubuntu) or brew install pandoc (macOS)${NC}"
    MISSING_DEPS=1
fi

if ! check_dependency "pdflatex"; then
    echo -e "${YELLOW}  Install: apt-get install texlive-xetex (Debian/Ubuntu) or brew install mactex (macOS)${NC}"
    MISSING_DEPS=1
fi

# Optional: mermaid-cli for diagram rendering
MERMAID_AVAILABLE=0
if check_dependency "mmdc"; then
    MERMAID_AVAILABLE=1
else
    echo -e "${YELLOW}  Optional: npm install -g @mermaid-js/mermaid-cli${NC}"
    echo -e "${YELLOW}  Diagrams will be included as code blocks if not installed${NC}"
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo -e "${RED}Error: Missing required dependencies. Please install them and try again.${NC}"
    exit 1
fi

# Render Mermaid diagrams if mermaid-cli is available
if [ $MERMAID_AVAILABLE -eq 1 ]; then
    echo ""
    echo -e "${BLUE}Rendering Mermaid diagrams...${NC}"

    # Create diagrams directory
    mkdir -p "$OUTPUT_DIR/diagrams"

    # Extract mermaid blocks from WHITEPAPER_VISUALS.md and render them
    # This is a simplified approach - in practice you'd parse the markdown properly
    echo -e "${YELLOW}Note: Automated Mermaid rendering not yet implemented.${NC}"
    echo -e "${YELLOW}To render diagrams:${NC}"
    echo -e "${YELLOW}  1. Visit https://mermaid.live/${NC}"
    echo -e "${YELLOW}  2. Copy mermaid code from docs/WHITEPAPER_VISUALS.md${NC}"
    echo -e "${YELLOW}  3. Export as PNG and save to $OUTPUT_DIR/diagrams/${NC}"
fi

# Generate PDF with Pandoc
echo ""
echo -e "${BLUE}Generating PDF...${NC}"

pandoc "$DOCS_DIR/WHITEPAPER.md" \
    -o "$OUTPUT_DIR/GPU_Scatter_Gather_Whitepaper_v1.0.0.pdf" \
    --from markdown \
    --to pdf \
    --pdf-engine=pdflatex \
    --variable geometry:margin=1in \
    --variable fontsize=11pt \
    --variable documentclass=article \
    --variable colorlinks=true \
    --variable linkcolor=blue \
    --variable urlcolor=blue \
    --variable toccolor=black \
    --toc \
    --toc-depth=3 \
    --number-sections \
    --highlight-style=tango \
    --metadata title="GPU-Accelerated Wordlist Generation: A Novel Approach Using Direct Index-to-Word Mapping" \
    --metadata author="tehw0lf" \
    --metadata date="November 21, 2025" \
    --metadata version="1.0.0"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PDF generated successfully!"
    echo ""
    echo -e "${GREEN}Output: $OUTPUT_DIR/GPU_Scatter_Gather_Whitepaper_v1.0.0.pdf${NC}"
    echo ""
else
    echo -e "${RED}✗${NC} PDF generation failed!"
    exit 1
fi

# Generate HTML version as well
echo -e "${BLUE}Generating HTML version...${NC}"

pandoc "$DOCS_DIR/WHITEPAPER.md" \
    -o "$OUTPUT_DIR/GPU_Scatter_Gather_Whitepaper_v1.0.0.html" \
    --from markdown \
    --to html5 \
    --standalone \
    --toc \
    --toc-depth=3 \
    --number-sections \
    --highlight-style=tango \
    --css=https://cdn.jsdelivr.net/gh/kognise/water.css@latest/dist/light.min.css \
    --metadata title="GPU-Accelerated Wordlist Generation: Technical Whitepaper" \
    --metadata author="tehw0lf" \
    --metadata date="November 21, 2025"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} HTML generated successfully!"
    echo -e "${GREEN}Output: $OUTPUT_DIR/GPU_Scatter_Gather_Whitepaper_v1.0.0.html${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠${NC} HTML generation failed (non-critical)"
fi

# Generate summary
echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "${GREEN}Generated files:${NC}"
ls -lh "$OUTPUT_DIR"/*.pdf "$OUTPUT_DIR"/*.html 2>/dev/null || echo "No files generated"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Review the PDF: $OUTPUT_DIR/GPU_Scatter_Gather_Whitepaper_v1.0.0.pdf"
echo -e "  2. Optionally add diagrams from docs/WHITEPAPER_VISUALS.md"
echo -e "  3. Share on GitHub releases: https://github.com/tehw0lf/gpu-scatter-gather/releases"
echo ""
echo -e "${BLUE}Recommended workflow for diagrams:${NC}"
echo -e "  1. Visit https://mermaid.live/"
echo -e "  2. Copy Mermaid code from docs/WHITEPAPER_VISUALS.md"
echo -e "  3. Export as PNG (transparent background)"
echo -e "  4. Edit WHITEPAPER.md to include images: ![Diagram](diagrams/diagram1.png)"
echo -e "  5. Re-run this script to regenerate PDF with images"
echo ""
echo -e "${GREEN}Done!${NC}"
