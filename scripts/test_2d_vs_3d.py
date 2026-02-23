#!/usr/bin/env python
"""Compare 2D and 3D operators on a phantom — visual + numerical comparison.

Run with:
    uv run python scripts/test_2d_vs_3d.py

Requires: ASTRA with CUDA support, matplotlib.
"""
import numpy as np
import tomosipo as ts
import matplotlib.pyplot as plt
import time


def create_phantom(shape):
    """Create a simple 3D phantom with a box and sphere."""
    Z, Y, X = shape
    phantom = np.zeros(shape, dtype=np.float32)

    # Outer box
    s = [s // 5 for s in shape]
    phantom[s[0]:-s[0], s[1]:-s[1], s[2]:-s[2]] = 1.0

    # Inner hole
    s2 = [s // 3 for s in shape]
    phantom[s2[0]:-s2[0], s2[1]:-s2[1], s2[2]:-s2[2]] = 0.0

    # Small sphere
    zz, yy, xx = np.mgrid[:Z, :Y, :X]
    center = np.array([Z // 3, Y // 2, X // 2])
    radius = min(shape) // 8
    mask = (zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2 < radius**2
    phantom[mask] = 0.5

    return phantom


def main():
    print("=" * 70)
    print("Comparing 2D vs 3D operators on a parallel-beam phantom")
    print("=" * 70)

    # Setup
    vol_shape = (32, 64, 64)
    n_angles = 96
    det_shape = (32, 64)  # (V, U) — V must match Z

    vg = ts.volume(shape=vol_shape)
    pg = ts.parallel(angles=n_angles, shape=det_shape)

    phantom = create_phantom(vol_shape)
    print(f"Volume shape: {vol_shape}")
    print(f"Detector shape: {det_shape}")
    print(f"Number of angles: {n_angles}")
    print()

    # 3D operator
    print("Creating 3D operator...")
    t0 = time.time()
    A_3d = ts.operator(vg, pg)
    t_create_3d = time.time() - t0

    t0 = time.time()
    sino_3d = A_3d(phantom)
    t_fp_3d = time.time() - t0

    t0 = time.time()
    bp_3d = A_3d.T(sino_3d)
    t_bp_3d = time.time() - t0

    print(f"  3D create: {t_create_3d:.4f}s")
    print(f"  3D FP:     {t_fp_3d:.4f}s")
    print(f"  3D BP:     {t_bp_3d:.4f}s")
    print()

    # 2D operator
    print("Creating 2D operator...")
    t0 = time.time()
    A_2d = ts.operator_2d(vg, pg)
    t_create_2d = time.time() - t0

    t0 = time.time()
    sino_2d = A_2d(phantom)
    t_fp_2d = time.time() - t0

    t0 = time.time()
    bp_2d = A_2d.T(sino_2d)
    t_bp_2d = time.time() - t0

    print(f"  2D create: {t_create_2d:.4f}s")
    print(f"  2D FP:     {t_fp_2d:.4f}s")
    print(f"  2D BP:     {t_bp_2d:.4f}s")
    print()

    # Comparison
    fp_diff = np.abs(sino_3d - sino_2d)
    bp_diff = np.abs(bp_3d - bp_2d)
    print("Numerical comparison:")
    print(f"  FP max diff:  {fp_diff.max():.6f}")
    print(f"  FP mean diff: {fp_diff.mean():.6f}")
    print(f"  FP relative:  {fp_diff.max() / (np.abs(sino_3d).max() + 1e-8):.6f}")
    print(f"  BP max diff:  {bp_diff.max():.6f}")
    print(f"  BP mean diff: {bp_diff.mean():.6f}")
    print(f"  BP relative:  {bp_diff.max() / (np.abs(bp_3d).max() + 1e-8):.6f}")

    match_fp = np.allclose(sino_3d, sino_2d, atol=1e-4, rtol=1e-3)
    match_bp = np.allclose(bp_3d, bp_2d, atol=1e-4, rtol=1e-3)
    print(f"  FP match: {'PASS' if match_fp else 'FAIL'}")
    print(f"  BP match: {'PASS' if match_bp else 'FAIL'}")
    print()

    # Plotting
    mid_z = vol_shape[0] // 2
    mid_angle = n_angles // 2

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("2D vs 3D Operator Comparison (Parallel Beam)", fontsize=14)

    # Row 1: Phantom
    axes[0, 0].imshow(phantom[mid_z], cmap='gray')
    axes[0, 0].set_title(f"Phantom (z={mid_z})")
    axes[0, 1].imshow(phantom[:, vol_shape[1]//2, :], cmap='gray')
    axes[0, 1].set_title("Phantom (y=mid)")
    axes[0, 2].set_visible(False)
    axes[0, 3].set_visible(False)

    # Row 2: Sinograms
    axes[1, 0].imshow(sino_3d[mid_z], cmap='gray', aspect='auto')
    axes[1, 0].set_title(f"Sinogram 3D (z={mid_z})")
    axes[1, 1].imshow(sino_2d[mid_z], cmap='gray', aspect='auto')
    axes[1, 1].set_title(f"Sinogram 2D (z={mid_z})")
    im = axes[1, 2].imshow(fp_diff[mid_z], cmap='hot', aspect='auto')
    axes[1, 2].set_title(f"|FP diff| (z={mid_z})")
    plt.colorbar(im, ax=axes[1, 2])
    axes[1, 3].plot(sino_3d[mid_z, mid_angle, :], label='3D')
    axes[1, 3].plot(sino_2d[mid_z, mid_angle, :], '--', label='2D')
    axes[1, 3].set_title(f"FP profile (angle={mid_angle})")
    axes[1, 3].legend()

    # Row 3: Backprojections
    axes[2, 0].imshow(bp_3d[mid_z], cmap='gray')
    axes[2, 0].set_title(f"BP 3D (z={mid_z})")
    axes[2, 1].imshow(bp_2d[mid_z], cmap='gray')
    axes[2, 1].set_title(f"BP 2D (z={mid_z})")
    im = axes[2, 2].imshow(bp_diff[mid_z], cmap='hot')
    axes[2, 2].set_title(f"|BP diff| (z={mid_z})")
    plt.colorbar(im, ax=axes[2, 2])
    axes[2, 3].plot(bp_3d[mid_z, vol_shape[1]//2, :], label='3D')
    axes[2, 3].plot(bp_2d[mid_z, vol_shape[1]//2, :], '--', label='2D')
    axes[2, 3].set_title("BP profile (y=mid)")
    axes[2, 3].legend()

    plt.tight_layout()
    plt.savefig("scripts/comparison_2d_vs_3d.png", dpi=150)
    print("Saved comparison plot to scripts/comparison_2d_vs_3d.png")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
