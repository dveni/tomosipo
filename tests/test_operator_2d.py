#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Operator2D (2D slice-based CUDA operator)."""

import pytest
from pytest import approx
import numpy as np
import tomosipo as ts
from tomosipo.Operator import Operator2D
from tests import skip_if_no_cuda


N = 64
N_angles = N * 3 // 2
M = N * 3 // 2


@skip_if_no_cuda
def test_operator_2d_basic():
    """Test basic forward and back projection with Operator2D."""
    vg = ts.volume(shape=N)
    pg = ts.parallel(angles=N_angles, shape=N)
    A = ts.operator_2d(vg, pg)

    x = np.ones(A.domain_shape, dtype=np.float32)
    y = A(x)
    bp = A.T(y)

    assert y.shape == A.range_shape
    assert bp.shape == A.domain_shape
    assert y.sum() > 0
    assert bp.sum() > 0


@skip_if_no_cuda
def test_operator_2d_shapes():
    """Test domain_shape and range_shape match expectations."""
    vg = ts.volume(shape=(10, 20, 30))
    pg = ts.parallel(angles=50, shape=(10, 30))
    A = ts.operator_2d(vg, pg)

    assert A.domain_shape == (10, 20, 30)
    assert A.range_shape == (10, 50, 30)

    assert A.T.domain_shape == A.range_shape
    assert A.T.range_shape == A.domain_shape


@skip_if_no_cuda
def test_operator_2d_transpose_identity():
    """Test that A.T is A.T.T.T (same as 3D operator)."""
    vg = ts.volume(shape=10)
    pg = ts.parallel(angles=10, shape=10)
    A = ts.operator_2d(vg, pg)
    assert A.T is A.T.T.T


@skip_if_no_cuda
def test_operator_2d_domain_range():
    """Test that domain and range properties return correct geometries."""
    vg = ts.volume(shape=10)
    pg = ts.parallel(angles=10, shape=10)
    A = ts.operator_2d(vg, pg)

    assert A.domain == vg
    assert A.range == pg
    assert A.T.domain == pg
    assert A.T.range == vg


@skip_if_no_cuda
def test_operator_2d_with_data():
    """Test Operator2D with ts.Data input."""
    vg = ts.volume(shape=N)
    pg = ts.parallel(angles=N_angles, shape=N)
    A = ts.operator_2d(vg, pg)

    vd = ts.data(vg)
    ts.phantom.hollow_box(vd)

    y = A(vd)
    assert hasattr(y, 'data')
    assert y.data.sum() > 0

    bp = A.T(y)
    assert hasattr(bp, 'data')
    assert bp.data.sum() > 0


@skip_if_no_cuda
def test_operator_2d_numpy_input():
    """Test Operator2D with raw numpy array input."""
    vg = ts.volume(shape=N)
    pg = ts.parallel(angles=N_angles, shape=N)
    A = ts.operator_2d(vg, pg)

    x = np.random.rand(*A.domain_shape).astype(np.float32)
    y = A(x)
    bp = A.T(y)

    assert isinstance(y, np.ndarray)
    assert isinstance(bp, np.ndarray)
    assert y.shape == A.range_shape
    assert bp.shape == A.domain_shape


@skip_if_no_cuda
def test_operator_2d_consistency_with_3d():
    """Test that Operator2D gives same results as Operator for parallel beam.

    For parallel-beam, slices are independent so 2D and 3D should match.
    """
    vg = ts.volume(shape=(8, N, N))
    pg = ts.parallel(angles=N_angles, shape=(8, N))

    A_3d = ts.operator(vg, pg)
    A_2d = ts.operator_2d(vg, pg)

    x = np.random.rand(*A_3d.domain_shape).astype(np.float32)

    y_3d = A_3d(x)
    y_2d = A_2d(x)

    # They should be very close (same underlying ASTRA projection)
    assert np.allclose(y_3d, y_2d, atol=1e-4, rtol=1e-3), (
        f"Max diff FP: {np.max(np.abs(y_3d - y_2d))}"
    )

    bp_3d = A_3d.T(y_3d)
    bp_2d = A_2d.T(y_2d)

    assert np.allclose(bp_3d, bp_2d, atol=1e-4, rtol=1e-3), (
        f"Max diff BP: {np.max(np.abs(bp_3d - bp_2d))}"
    )


@skip_if_no_cuda
def test_operator_2d_rejects_cone():
    """Test that Operator2D raises ValueError for cone-beam geometry."""
    vg = ts.volume(shape=10)
    pg = ts.cone(angles=10, shape=10, cone_angle=0.5)

    with pytest.raises(ValueError, match="parallel-beam"):
        ts.operator_2d(vg, pg)


@skip_if_no_cuda
def test_operator_2d_vec_geometry():
    """Test Operator2D with vector parallel geometry."""
    vg = ts.volume(shape=N)
    pg = ts.parallel(angles=N_angles, shape=N).to_vec()
    A = ts.operator_2d(vg, pg)

    x = np.ones(A.domain_shape, dtype=np.float32)
    y = A(x)
    bp = A.T(y)

    assert y.sum() > 0
    assert bp.sum() > 0


@skip_if_no_cuda
def test_operator_2d_non_square():
    """Test Operator2D with non-square volume and detector."""
    vg = ts.volume(shape=(16, 32, 48))
    pg = ts.parallel(angles=60, shape=(16, 48))
    A = ts.operator_2d(vg, pg)

    x = np.ones(A.domain_shape, dtype=np.float32)
    y = A(x)
    bp = A.T(y)

    assert y.shape == A.range_shape
    assert bp.shape == A.domain_shape
    assert y.sum() > 0


try:
    import torch
    torch_present = True
except ImportError:
    torch_present = False

skip_if_no_torch = pytest.mark.skipif(not torch_present, reason="Pytorch not installed")


@skip_if_no_torch
@skip_if_no_cuda
def test_operator_2d_torch_cpu():
    """Test Operator2D with CPU torch tensor."""
    vg = ts.volume(shape=N)
    pg = ts.parallel(angles=N_angles, shape=N)
    A = ts.operator_2d(vg, pg)

    x = torch.ones(A.domain_shape, dtype=torch.float32)
    y = A(x)
    bp = A.T(y)

    assert isinstance(y, torch.Tensor)
    assert isinstance(bp, torch.Tensor)
    assert y.sum() > 0
    assert bp.sum() > 0


@skip_if_no_torch
@skip_if_no_cuda
def test_operator_2d_torch_gpu():
    """Test Operator2D with GPU torch tensor."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    vg = ts.volume(shape=N)
    pg = ts.parallel(angles=N_angles, shape=N)
    A = ts.operator_2d(vg, pg)

    x = torch.ones(A.domain_shape, dtype=torch.float32, device='cuda')
    y = A(x)
    bp = A.T(y)

    # Result should be back on GPU
    assert isinstance(y, torch.Tensor)
    assert y.device.type == 'cuda'
    assert bp.device.type == 'cuda'
    assert y.sum() > 0


@skip_if_no_torch
@skip_if_no_cuda
def test_autograd_operator_2d_cuda():
    """Test autograd_operator with use_2d_cuda=True."""
    from tomosipo.torch_support import autograd_operator

    vg = ts.volume(shape=(1, N, N))
    pg = ts.parallel(angles=N_angles, shape=(1, N))

    A_ag = autograd_operator(vg, pg, use_2d_cuda=True)

    x = torch.ones(A_ag.domain_shape, dtype=torch.float32, requires_grad=True)
    y = A_ag(x)
    assert y.sum() > 0

    # Test backward pass
    y.backward(torch.ones_like(y))
    assert x.grad is not None
    assert x.grad.sum() > 0
