# Reboot Checklist - Enable GPU Profiling

## What You Need to Do Before Rebooting

Run these three commands:

```bash
# 1. Add profiling permission to NVIDIA module config
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee -a /etc/modprobe.d/nvidia.conf

# 2. Regenerate initramfs (loads module with new settings)
sudo mkinitcpio -P

# 3. Reboot
sudo reboot
```

## After Reboot - Verification Steps

### Step 1: Verify Profiling Permission is Enabled

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```

**Expected output:**
```
RmProfilingAdminOnly: 0
```

**If you see `1` instead of `0`, the setting didn't apply.** Check troubleshooting below.

### Step 2: Navigate to Project Directory

```bash
cd /path/to/gpu-scatter-gather
```

### Step 3: Run Kernel Profiling

```bash
./scripts/profile_kernel.sh
```

This will:
- Profile the `generate_words_kernel` with Nsight Compute
- Run 3 kernel launches (skip first 2 warmup runs)
- Save results to `profiling/results/profile_YYYY-MM-DD_HHMMSS.ncu-rep`
- Generate text summary: `profiling/results/profile_YYYY-MM-DD_HHMMSS_summary.txt`

**Expected:** No `ERR_NVGPUCTRPERM` error, profiling completes successfully.

### Step 4: View Profiling Results

**Option 1: Text Summary (Quick)**
```bash
cat profiling/results/profile_*_summary.txt | less
```

**Option 2: Detailed CLI Report**
```bash
ncu --import profiling/results/profile_*.ncu-rep --page details
```

**Option 3: GUI (if available)**
```bash
ncu-ui profiling/results/profile_*.ncu-rep
```

## What to Look For in Profiling Results

### Key Metrics

1. **Compute Throughput**
   - What % of theoretical GPU compute are we achieving?
   - If low (<50%): Likely compute-bound → Focus on arithmetic optimization

2. **Memory Bandwidth**
   - What % of peak memory bandwidth are we using?
   - If high (>80%): Memory-bound → Focus on memory optimization

3. **Occupancy**
   - How many warps are active?
   - Target: >50% for good utilization

4. **Instruction Mix**
   - Look for expensive instructions (division, modulo)
   - These are prime candidates for optimization

5. **Memory Access Patterns**
   - Are global memory accesses coalesced?
   - Cache hit rates?

### Expected Bottleneck

Based on kernel analysis, we expect:
- **High division/modulo count** (70-80% of kernel time)
- **Compute-bound** (arithmetic is expensive)
- **Good memory coalescing** (already optimized in current kernel)

**Likely optimization:** Barrett reduction to replace div/mod with fast multiplication.

## Troubleshooting

### Problem: RmProfilingAdminOnly still shows 1 after reboot

**Solution 1:** Check if the config was added correctly
```bash
cat /etc/modprobe.d/nvidia.conf
```

Should contain:
```
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

**Solution 2:** Manually verify initramfs was regenerated
```bash
ls -lth /boot/initramfs-*.img | head -5
```

The most recent one should be from after you ran `mkinitcpio -P`.

**Solution 3:** Check module parameters at runtime
```bash
systool -m nvidia -A | grep RmProfilingAdminOnly
```

### Problem: Profiling script fails with "command not found"

```bash
chmod +x scripts/profile_kernel.sh
```

### Problem: Nsight Compute not found

```bash
which ncu
# Should show: /opt/cuda/nsight_compute/ncu
```

If not found:
```bash
export PATH=$PATH:/opt/cuda/nsight_compute
```

## Next Steps After Successful Profiling

1. **Analyze Results**
   - Identify primary bottleneck (compute vs memory vs launch)
   - Quantify impact of division/modulo operations

2. **Implement Optimization**
   - If compute-bound (expected): Implement Barrett reduction
   - If memory-bound: Optimize memory access patterns
   - If launch-bound: Tune block size and occupancy

3. **Benchmark Improvement**
   ```bash
   ./scripts/run_baseline_benchmark.sh
   ```

4. **Compare Results**
   ```bash
   ./scripts/compare_benchmarks.sh \
       benches/scientific/results/baseline_2025-11-17.json \
       benches/scientific/results/optimized_YYYY-MM-DD.json
   ```

5. **Document**
   - Update `docs/DEVELOPMENT_LOG.md`
   - Commit changes with performance data

## Reference Documents

- **Phase 3 Plan:** `docs/PHASE3_KICKOFF.md`
- **Profiling Setup:** `docs/ENABLE_PROFILING.md`
- **Nsight Compute Help:** `docs/NSIGHT_COMPUTE_SETUP.md`
- **Next Session:** `docs/NEXT_SESSION_PROMPT.md`

## Quick Copy-Paste for Next Claude Session

After successful profiling, start next session with:

```
I've rebooted and enabled GPU profiling. Profiling completed successfully.

Profiling results: profiling/results/profile_<timestamp>.ncu-rep

Please analyze the profiling results and identify the primary bottleneck:
- Review profiling summary
- Identify whether we're compute-bound, memory-bound, or launch-bound
- Recommend the highest-impact optimization to implement first
- Help me implement that optimization

Working directory: /path/to/gpu-scatter-gather
Current baseline: ~684M words/s average
Goal: Identify and fix performance bottlenecks
```

---

**Checklist Summary:**

- [ ] Run commands before reboot (nvidia.conf + mkinitcpio + reboot)
- [ ] After reboot: Verify RmProfilingAdminOnly: 0
- [ ] Run: `./scripts/profile_kernel.sh`
- [ ] Review profiling results
- [ ] Start next Claude session with profiling analysis
