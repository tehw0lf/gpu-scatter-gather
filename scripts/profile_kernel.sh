#!/bin/bash
set -euo pipefail

# GPU Scatter-Gather Kernel Profiler
# Uses Nsight Compute to analyze GPU kernel performance

echo "=== GPU Kernel Profiling with Nsight Compute ==="
echo ""

# Create results directory
mkdir -p profiling/results

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
OUTPUT_FILE="profiling/results/profile_${TIMESTAMP}"

echo "📊 Profiling kernel with Nsight Compute..."
echo "   Output: ${OUTPUT_FILE}.ncu-rep"
echo ""

# Profile the generate_words_kernel with comprehensive metrics
# Focus on the production kernel, not the POC kernel
ncu \
    --set full \
    --target-processes all \
    --kernel-name "generate_words_kernel" \
    --launch-skip 2 \
    --launch-count 3 \
    --export "${OUTPUT_FILE}" \
    --force-overwrite \
    ./target/release/examples/benchmark_production

echo ""
echo "✅ Profiling complete!"
echo ""
echo "📁 Results saved to: ${OUTPUT_FILE}.ncu-rep"
echo ""
echo "To view the results:"
echo "  1. GUI: ncu-ui ${OUTPUT_FILE}.ncu-rep"
echo "  2. CLI: ncu --import ${OUTPUT_FILE}.ncu-rep --print-summary per-kernel"
echo ""
echo "Key metrics to check:"
echo "  - Compute throughput (% of theoretical)"
echo "  - Memory throughput (% of peak bandwidth)"
echo "  - Occupancy"
echo "  - Warp execution efficiency"
echo "  - Memory access patterns (coalescing)"
echo "  - Instruction mix (division/modulo operations)"
echo ""

# Generate text summary
echo "Generating text summary..."
ncu --import "${OUTPUT_FILE}.ncu-rep" --page details --print-summary per-kernel > "${OUTPUT_FILE}_summary.txt" 2>&1 || true

echo "✅ Summary saved to: ${OUTPUT_FILE}_summary.txt"
