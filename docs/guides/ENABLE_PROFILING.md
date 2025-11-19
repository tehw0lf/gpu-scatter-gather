# Enable NVIDIA GPU Profiling for Nsight Compute

## Problem
Nsight Compute requires access to GPU performance counters, which are restricted by default.

Error message:
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

## Current System State

Your system has:
- `nvidia_drm` loaded at boot (used by display server)
- Existing NVIDIA module config at `/etc/modprobe.d/nvidia.conf`
- Cannot unload driver without dropping to TTY

## Solution: Add Profiling Option to Existing Config

### Step 1: Edit NVIDIA modprobe configuration

```bash
sudo nano /etc/modprobe.d/nvidia.conf
```

Add this line:
```
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

The file should look like:
```
options nvidia NVreg_PreserveVideoMemoryAllocations=1
options nvidia NVreg_TemporaryFilePath=/tmp
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

### Step 2: Regenerate initramfs (Arch Linux)

```bash
sudo mkinitcpio -P
```

This rebuilds the initial ramdisk with the new module parameters.

### Step 3: Reboot

```bash
sudo reboot
```

## Verification After Reboot

Check that profiling is enabled:

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```

Should show:
```
RmProfilingAdminOnly: 0
```

If it shows `1`, the setting didn't apply correctly.

## Alternative: One-Line Command

```bash
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee -a /etc/modprobe.d/nvidia.conf
sudo mkinitcpio -P
sudo reboot
```

## Testing Profiling After Reboot

```bash
cd /path/to/gpu-scatter-gather
./scripts/profile_kernel.sh
```

Should run without permission errors.

## Security Note

This allows non-root users to access GPU performance counters. This is generally safe on a personal development machine, but be aware that performance counters can theoretically be used for side-channel attacks in multi-user environments.

## Troubleshooting

### If profiling still fails after reboot:

1. Verify module parameter is loaded:
   ```bash
   systool -m nvidia -A
   ```
   Look for `NVreg_RestrictProfilingToAdminUsers = 0`

2. Check kernel command line (shouldn't be needed, but verify):
   ```bash
   cat /proc/cmdline | grep nvidia
   ```

3. Try manual module reload (requires TTY):
   - Switch to TTY (Ctrl+Alt+F2)
   - Login
   - Stop display manager: `sudo systemctl stop gdm` (or sddm/lightdm)
   - Unload: `sudo modprobe -r nvidia_drm nvidia_uvm nvidia_modeset nvidia`
   - Reload: `sudo modprobe nvidia`
   - Start display: `sudo systemctl start gdm`

## References

- [NVIDIA Profiling Permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
- [Arch Linux Kernel Module Parameters](https://wiki.archlinux.org/title/Kernel_module#Setting_module_options)
