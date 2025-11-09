#!/bin/bash
#
# Cross-Tool Benchmark Script
# Compares gpu-scatter-gather performance against maskprocessor and hashcat
#
# Usage: ./cross_tool_benchmark.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Tool paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MP_BIN="$PROJECT_ROOT/tools/maskprocessor/src/mp64.bin"
HASHCAT_BIN="$PROJECT_ROOT/tools/hashcat/hashcat"
GPU_SG_BIN="$PROJECT_ROOT/target/release/gpu-scatter-gather"

# Check if tools exist
check_tools() {
    echo -e "${BLUE}Checking for required tools...${NC}"

    if [[ ! -f "$MP_BIN" ]]; then
        echo -e "${RED}ERROR: maskprocessor not found at $MP_BIN${NC}"
        echo "Run: ./scripts/install_test_tools.sh"
        exit 1
    fi

    if [[ ! -f "$HASHCAT_BIN" ]]; then
        echo -e "${RED}ERROR: hashcat not found at $HASHCAT_BIN${NC}"
        echo "Run: ./scripts/install_test_tools.sh"
        exit 1
    fi

    if [[ ! -f "$GPU_SG_BIN" ]]; then
        echo -e "${YELLOW}WARNING: gpu-scatter-gather not built in release mode${NC}"
        echo "Building now..."
        cd "$PROJECT_ROOT"
        cargo build --release
    fi

    echo -e "${GREEN}✓ All tools found${NC}\n"
}

# Benchmark function
# Args: tool_name, command, description
benchmark_tool() {
    local tool_name="$1"
    local command="$2"
    local description="$3"

    echo -e "${BLUE}Testing $tool_name: $description${NC}"

    # Warm-up run
    eval "$command" > /dev/null 2>&1 || true

    # Actual benchmark (3 runs, take average)
    local total_time=0
    local runs=3

    for i in $(seq 1 $runs); do
        local start=$(date +%s.%N)
        eval "$command" > /dev/null 2>&1
        local end=$(date +%s.%N)
        local elapsed=$(echo "$end - $start" | bc)
        total_time=$(echo "$total_time + $elapsed" | bc)
    done

    local avg_time=$(echo "scale=4; $total_time / $runs" | bc)
    echo "$avg_time"
}

# Count words in output
count_words() {
    local command="$1"
    eval "$command" 2>/dev/null | wc -l
}

echo "=================================================="
echo "  GPU Scatter-Gather Cross-Tool Benchmark"
echo "=================================================="
echo ""

check_tools

# Test configurations
declare -A TESTS=(
    ["small"]="?1?2|1=abc|2=123|3x3=9 words"
    ["medium"]="?1?1?2?2|1=ABCDEFGHIJKLMNOPQRSTUVWXYZ|2=0123456789|26²×10²=67,600 words"
    ["large"]="?1?1?2?2?2?2?2?2|1=ABCDEFGHIJKLMNOPQRSTUVWXYZ|2=0123456789|26²×10⁶=6,760,000 words"
)

# Results storage
declare -A RESULTS_MP
declare -A RESULTS_HASHCAT
declare -A RESULTS_GPU_SG

# Run benchmarks
for test_name in "small" "medium" "large"; do
    IFS='|' read -r mask charset1 charset2 description <<< "${TESTS[$test_name]}"

    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Test: $test_name - $description${NC}"
    echo -e "${YELLOW}Mask: $mask${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    # Extract charset values
    cs1_value="${charset1#*=}"
    cs2_value="${charset2#*=}"

    # Test 1: maskprocessor
    echo -e "${BLUE}[1/3] maskprocessor (CPU)${NC}"
    mp_cmd="'$MP_BIN' -1 '$cs1_value' -2 '$cs2_value' '$mask'"
    mp_time=$(benchmark_tool "maskprocessor" "$mp_cmd" "CPU-based generation")
    mp_count=$(count_words "$mp_cmd")
    mp_throughput=$(echo "scale=2; $mp_count / $mp_time / 1000000" | bc)
    echo -e "  Time: ${GREEN}${mp_time}s${NC}"
    echo -e "  Throughput: ${GREEN}${mp_throughput}M words/s${NC}"
    echo -e "  Words: $mp_count"
    RESULTS_MP[$test_name]="$mp_time|$mp_throughput|$mp_count"

    # Test 2: hashcat stdout (CPU)
    echo ""
    echo -e "${BLUE}[2/3] hashcat --stdout (CPU)${NC}"
    hc_cmd="'$HASHCAT_BIN' --stdout -a 3 --custom-charset1='$cs1_value' --custom-charset2='$cs2_value' '$mask'"
    hc_time=$(benchmark_tool "hashcat" "$hc_cmd" "CPU-based generation")
    hc_count=$(count_words "$hc_cmd")
    hc_throughput=$(echo "scale=2; $hc_count / $hc_time / 1000000" | bc)
    echo -e "  Time: ${GREEN}${hc_time}s${NC}"
    echo -e "  Throughput: ${GREEN}${hc_throughput}M words/s${NC}"
    echo -e "  Words: $hc_count"
    RESULTS_HASHCAT[$test_name]="$hc_time|$hc_throughput|$hc_count"

    # Test 3: gpu-scatter-gather (GPU)
    echo ""
    echo -e "${BLUE}[3/3] gpu-scatter-gather (GPU)${NC}"
    gpu_cmd="'$GPU_SG_BIN' -1 '$cs1_value' -2 '$cs2_value' '$mask'"
    gpu_time=$(benchmark_tool "gpu-scatter-gather" "$gpu_cmd" "GPU-accelerated generation")
    gpu_count=$(count_words "$gpu_cmd")
    gpu_throughput=$(echo "scale=2; $gpu_count / $gpu_time / 1000000" | bc)
    echo -e "  Time: ${GREEN}${gpu_time}s${NC}"
    echo -e "  Throughput: ${GREEN}${gpu_throughput}M words/s${NC}"
    echo -e "  Words: $gpu_count"
    RESULTS_GPU_SG[$test_name]="$gpu_time|$gpu_throughput|$gpu_count"

    # Calculate speedups
    echo ""
    echo -e "${YELLOW}Speedup Analysis:${NC}"
    mp_speedup=$(echo "scale=2; $gpu_throughput / $mp_throughput" | bc)
    hc_speedup=$(echo "scale=2; $gpu_throughput / $hc_throughput" | bc)
    echo -e "  vs maskprocessor: ${GREEN}${mp_speedup}x faster${NC}"
    echo -e "  vs hashcat --stdout: ${GREEN}${hc_speedup}x faster${NC}"
done

# Summary table
echo ""
echo ""
echo -e "${YELLOW}=================================================="
echo "                 SUMMARY TABLE"
echo -e "==================================================${NC}"
echo ""
printf "%-12s %-20s %-12s %-15s\n" "Test" "Tool" "Time (s)" "Throughput (M/s)"
echo "------------------------------------------------------------"

for test_name in "small" "medium" "large"; do
    IFS='|' read -r mp_time mp_throughput mp_count <<< "${RESULTS_MP[$test_name]}"
    IFS='|' read -r hc_time hc_throughput hc_count <<< "${RESULTS_HASHCAT[$test_name]}"
    IFS='|' read -r gpu_time gpu_throughput gpu_count <<< "${RESULTS_GPU_SG[$test_name]}"

    printf "%-12s %-20s %-12s %-15s\n" "$test_name" "maskprocessor" "$mp_time" "$mp_throughput"
    printf "%-12s %-20s %-12s %-15s\n" "" "hashcat --stdout" "$hc_time" "$hc_throughput"
    printf "%-12s %-20s %-12s %-15s\n" "" "gpu-scatter-gather" "$gpu_time" "$gpu_throughput"
    echo "------------------------------------------------------------"
done

# Final verdict
echo ""
echo -e "${YELLOW}=================================================="
echo "                 KEY FINDINGS"
echo -e "==================================================${NC}"
echo ""
echo "1. All tools produce identical output (validated via tests)"
echo "2. maskprocessor and hashcat --stdout are CPU-based"
echo "3. gpu-scatter-gather leverages GPU for massive parallelism"
echo ""
echo "For fair GPU-to-GPU comparison, note that:"
echo "  - Hashcat uses GPU during hash cracking, not wordlist generation"
echo "  - Our tool focuses on GPU-accelerated wordlist generation"
echo "  - Direct comparison should be: our generation + hashcat hashing"
echo ""
echo -e "${GREEN}✓ Benchmark complete!${NC}"
