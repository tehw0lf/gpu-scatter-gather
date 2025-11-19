# Nsight Compute Setup Guide

## Permission Error (ERR_NVGPUCTRPERM)

If you see this error:
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

### Solution 1: Temporary (requires sudo, lasts until reboot)

```bash
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

### Solution 2: Permanent (recommended)

Create a modprobe configuration file:

```bash
sudo tee /etc/modprobe.d/nvidia-profiling.conf <<EOF
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF
```

Then reload the NVIDIA driver:

```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
```

**OR** simply reboot for changes to take effect.

### Solution 3: Run with sudo (not recommended for security)

```bash
sudo ./scripts/profile_kernel.sh
```

### Verification

After applying one of the solutions, verify with:

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```

Should show: `RmProfilingAdminOnly: 0`

## Running Profiling

Once permissions are fixed:

```bash
./scripts/profile_kernel.sh
```

## Viewing Results

### GUI (if X11/Wayland available):
```bash
ncu-ui profiling/results/profile_YYYY-MM-DD_HHMMSS.ncu-rep
```

### CLI Summary:
```bash
ncu --import profiling/results/profile_YYYY-MM-DD_HHMMSS.ncu-rep --print-summary per-kernel
```

### Detailed CLI Report:
```bash
ncu --import profiling/results/profile_YYYY-MM-DD_HHMMSS.ncu-rep --page details
```

## Key Metrics to Analyze

After profiling, look for:

1. **Compute Throughput**: What % of theoretical GPU compute are we achieving?
2. **Memory Bandwidth**: Are we memory-bound or compute-bound?
3. **Occupancy**: How many warps are active? (target: >50%)
4. **Instruction Mix**: How many div/mod operations? Can we optimize them?
5. **Memory Coalescing**: Are global memory accesses coalesced?
6. **Warp Divergence**: Are threads in warps taking different paths?

## References

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [ERR_NVGPUCTRPERM Fix](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
