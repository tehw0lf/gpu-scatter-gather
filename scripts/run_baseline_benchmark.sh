#!/bin/bash
set -euo pipefail

# GPU Scatter-Gather Baseline Benchmark Runner
# Purpose: Establish pre-optimization performance baseline

echo "=== GPU Scatter-Gather Baseline Benchmark Runner ==="
echo ""

# Check if running on correct hardware
if ! nvidia-smi &> /dev/null; then
    echo "❌ Error: NVIDIA GPU not detected"
    exit 1
fi

echo "📊 Detected GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Set CPU governor to performance mode (requires sudo)
echo "⚙️  Setting CPU governor to performance mode..."
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g performance || echo "⚠️  Failed to set CPU governor (continuing anyway)"
else
    echo "⚠️  cpupower not found, skipping CPU governor setup"
fi

# Lock GPU clocks to prevent boosting/throttling (optional, requires nvidia-smi)
echo "⚙️  Locking GPU clocks..."
sudo nvidia-smi -pm 1 || echo "⚠️  Failed to enable persistence mode"
sudo nvidia-smi -lgc 2610,2610 || echo "⚠️  Failed to lock GPU clocks (continuing anyway)"

# Create results directory
mkdir -p benches/scientific/results

# Kill background processes that might interfere
echo "🧹 Minimizing background processes..."
# (Optional: add commands to stop services)

# Build benchmarks in release mode
echo "🔨 Building benchmarks..."
cargo build --release --bin baseline_benchmark

# Run benchmarks
echo ""
echo "🚀 Running baseline benchmarks..."
echo "   This will take approximately 10-20 minutes..."
echo ""

./target/release/baseline_benchmark

# Restore CPU governor
echo ""
echo "⚙️  Restoring CPU governor to ondemand..."
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g ondemand || true
fi

# Unlock GPU clocks
echo "⚙️  Unlocking GPU clocks..."
sudo nvidia-smi -rgc || true
sudo nvidia-smi -pm 0 || true

echo ""
echo "✅ Baseline benchmarking complete!"
echo ""
echo "📁 Results saved to: benches/scientific/results/"
ls -lh benches/scientific/results/baseline_*
