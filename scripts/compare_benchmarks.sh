#!/bin/bash
set -euo pipefail

# Compare two benchmark results
# Usage: ./scripts/compare_benchmarks.sh baseline_2025-11-09.json optimized_2025-11-10.json

if [ $# -ne 2 ]; then
    echo "Usage: $0 <baseline.json> <optimized.json>"
    exit 1
fi

BASELINE="$1"
OPTIMIZED="$2"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "❌ Error: jq is required for JSON processing"
    echo "Install with: sudo apt install jq (Ubuntu/Debian) or sudo pacman -S jq (Arch)"
    exit 1
fi

# Check if bc is installed
if ! command -v bc &> /dev/null; then
    echo "❌ Error: bc is required for calculations"
    echo "Install with: sudo apt install bc (Ubuntu/Debian) or sudo pacman -S bc (Arch)"
    exit 1
fi

echo "=== Benchmark Comparison ==="
echo "Baseline:  $BASELINE"
echo "Optimized: $OPTIMIZED"
echo ""

# Check if files exist
if [ ! -f "$BASELINE" ]; then
    echo "❌ Error: Baseline file not found: $BASELINE"
    exit 1
fi

if [ ! -f "$OPTIMIZED" ]; then
    echo "❌ Error: Optimized file not found: $OPTIMIZED"
    exit 1
fi

echo "🔍 Throughput Comparison:"
echo ""

# Extract pattern names
PATTERNS=$(jq -r 'keys[]' "$BASELINE")

for pattern in $PATTERNS; do
    baseline_mean=$(jq -r ".\"$pattern\".mean_throughput" "$BASELINE")
    optimized_mean=$(jq -r ".\"$pattern\".mean_throughput" "$OPTIMIZED" 2>/dev/null || echo "0")

    if [ "$optimized_mean" = "0" ] || [ "$optimized_mean" = "null" ]; then
        echo "Pattern: $pattern"
        echo "  ⚠️  Not found in optimized results"
        echo ""
        continue
    fi

    # Calculate improvement percentage
    improvement=$(echo "scale=2; (($optimized_mean - $baseline_mean) / $baseline_mean) * 100" | bc)

    # Format throughput values
    baseline_formatted=$(echo "scale=2; $baseline_mean / 1000000" | bc)
    optimized_formatted=$(echo "scale=2; $optimized_mean / 1000000" | bc)

    echo "Pattern: $pattern"
    echo "  Baseline:  ${baseline_formatted}M words/s"
    echo "  Optimized: ${optimized_formatted}M words/s"

    # Color-code the improvement
    if (( $(echo "$improvement > 0" | bc -l) )); then
        echo "  Change:    +${improvement}% 🚀"
    elif (( $(echo "$improvement < 0" | bc -l) )); then
        echo "  Change:    ${improvement}% ⚠️  REGRESSION"
    else
        echo "  Change:    ${improvement}% (no change)"
    fi
    echo ""
done

echo "✅ Comparison complete!"
