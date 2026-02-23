#!/usr/bin/env python
"""Demonstrate Operator2D handling volumes exceeding 3D CUDA texture limits.

Run with:
    uv run python scripts/test_2d_large_volume.py

Requires: ASTRA with CUDA support.
"""
import numpy as np
import tomosipo as ts
import time
import sys


def main():
    print("=" * 70)
    print("Large Volume Test: 2D Slice-Based Operator")
    print("=" * 70)
    print()

    # Large volume along Z axis — exceeds 3D CUDA texture limits
    # 3D CUDA textures are limited to ~2048 in each dimension.
    # We use a large Z dimension to demonstrate the bypass.
    Z = 4096  # Exceeds 3D texture limit on most GPUs
    Y = 64
    X = 64
    n_angles = 96

    print(f"Volume shape: ({Z}, {Y}, {X})")
    print(f"Total voxels: {Z * Y * X:,}")
    print(f"Number of angles: {n_angles}")
    print()

    vg = ts.volume(shape=(Z, Y, X))
    pg = ts.parallel(angles=n_angles, shape=(Z, X))

    # Create a simple phantom (only fill a few slices to save memory)
    phantom = np.zeros((Z, Y, X), dtype=np.float32)
    # Place a box in the middle
    phantom[Z//4:3*Z//4, Y//4:3*Y//4, X//4:3*X//4] = 1.0
    phantom[Z//3:2*Z//3, Y//3:2*Y//3, X//3:2*X//3] = 0.0
    print(f"Phantom memory: {phantom.nbytes / 1e9:.2f} GB")

    # -- Test with 2D operator (should succeed) --
    print("\n--- 2D Operator (slice-by-slice) ---")
    try:
        t0 = time.time()
        A_2d = ts.operator_2d(vg, pg)
        print(f"  Created in {time.time()-t0:.3f}s")

        t0 = time.time()
        sino = A_2d(phantom)
        t_fp = time.time() - t0
        print(f"  FP: {t_fp:.3f}s ({Z/t_fp:.0f} slices/sec)")
        print(f"  Sinogram shape: {sino.shape}")
        print(f"  Sinogram sum: {sino.sum():.4f}")

        t0 = time.time()
        bp = A_2d.T(sino)
        t_bp = time.time() - t0
        print(f"  BP: {t_bp:.3f}s ({Z/t_bp:.0f} slices/sec)")
        print(f"  BP shape: {bp.shape}")
        print(f"  BP sum: {bp.sum():.4f}")
        print("  Status: SUCCESS ✓")
    except Exception as e:
        print(f"  Status: FAILED ✗  ({e})")
        sys.exit(1)

    # -- Optional: Test with 3D operator (expected to fail for large Z) --
    print("\n--- 3D Operator (for comparison) ---")
    try:
        A_3d = ts.operator(vg, pg)
        sino_3d = A_3d(phantom)
        print(f"  Status: SUCCESS (unexpected for Z={Z})")
    except Exception as e:
        print(f"  Status: FAILED (expected for large volume)")
        print(f"  Error: {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print("Test complete!")


if __name__ == "__main__":
    main()
