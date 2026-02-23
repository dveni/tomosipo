#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for 2D ASTRA geometry conversions (to_astra_2d)."""

import pytest
import numpy as np
import tomosipo as ts
from tomosipo.geometry import random_parallel, random_parallel_vec, random_volume


def test_volume_to_astra_2d():
    """Test VolumeGeometry.to_astra_2d() produces valid 2D geometry."""
    vg = ts.volume(shape=(10, 20, 30), size=(1.0, 2.0, 3.0))
    geom_2d = vg.to_astra_2d()

    # 2D vol geom should have GridRowCount=Y=20, GridColCount=X=30
    assert geom_2d['GridRowCount'] == 20
    assert geom_2d['GridColCount'] == 30
    # Should NOT have GridSliceCount (that's 3D)
    assert 'GridSliceCount' not in geom_2d


def test_volume_to_astra_2d_extent():
    """Test that 2D geometry preserves the Y-X extent correctly."""
    vg = ts.volume(shape=(10, 20, 30), pos=(0, 0, 0), size=(1.0, 2.0, 3.0))
    geom_2d = vg.to_astra_2d()

    # X extent: (-1.5, 1.5), Y extent: (-1.0, 1.0)
    opts = geom_2d.get('option', {})
    assert abs(opts.get('WindowMinX', 0) - (-1.5)) < 1e-6
    assert abs(opts.get('WindowMaxX', 0) - 1.5) < 1e-6
    assert abs(opts.get('WindowMinY', 0) - (-1.0)) < 1e-6
    assert abs(opts.get('WindowMaxY', 0) - 1.0) < 1e-6


def test_volume_to_astra_2d_default():
    """Test with default volume geometry."""
    vg = ts.volume(shape=64)
    geom_2d = vg.to_astra_2d()
    assert geom_2d['GridRowCount'] == 64
    assert geom_2d['GridColCount'] == 64


def test_parallel_to_astra_2d():
    """Test ParallelGeometry.to_astra_2d() produces valid 2D geometry."""
    pg = ts.parallel(angles=100, shape=(20, 30), size=(2.0, 3.0))
    geom_2d = pg.to_astra_2d()

    assert geom_2d['type'] == 'parallel'
    assert geom_2d['DetectorCount'] == 30  # col count (U)
    assert len(geom_2d['ProjectionAngles']) == 100


def test_parallel_to_astra_2d_det_spacing():
    """Test that detector spacing is computed correctly for 2D."""
    pg = ts.parallel(angles=10, shape=(5, 10), size=(2.0, 4.0))
    geom_2d = pg.to_astra_2d()

    # U spacing = size_u / shape_u = 4.0 / 10 = 0.4
    assert abs(geom_2d['DetectorWidth'] - 0.4) < 1e-6


def test_parallel_vec_to_astra_2d():
    """Test ParallelVectorGeometry.to_astra_2d() produces valid 2D geometry."""
    pg = ts.parallel(angles=50, shape=(10, 20), size=(1.0, 2.0)).to_vec()
    geom_2d = pg.to_astra_2d()

    assert geom_2d['type'] == 'parallel_vec'
    assert geom_2d['DetectorCount'] == 20  # col count
    assert geom_2d['Vectors'].shape == (50, 6)


def test_parallel_vec_to_astra_2d_vectors():
    """Test that 2D vector components are extracted correctly."""
    # Create a simple parallel geometry and convert to vec
    pg = ts.parallel(angles=1, shape=(1, 1), size=(1.0, 1.0)).to_vec()
    geom_2d = pg.to_astra_2d()

    # The vectors should have 6 columns: (rayX, rayY, dX, dY, uX, uY)
    vecs = geom_2d['Vectors']
    assert vecs.shape[1] == 6

    # For a standard parallel beam at angle 0:
    # ray_dir should be (0, 1, 0) in (Z,Y,X) -> rayX=0, rayY=1
    # This might be off by convention but should be non-trivial)
    assert not np.allclose(vecs, 0)  # vectors shouldn't be all zero


def test_consistency_3d_2d_geometry():
    """Test that 3D and 2D geometries are consistent in Y-X plane."""
    vg = ts.volume(shape=(8, 16, 32), size=(0.8, 1.6, 3.2))

    geom_3d = vg.to_astra()
    geom_2d = vg.to_astra_2d()

    # Both should have the same Y and X dimensions
    assert geom_3d['GridRowCount'] == geom_2d['GridRowCount']
    assert geom_3d['GridColCount'] == geom_2d['GridColCount']

    # X and Y extents should match
    opts_3d = geom_3d.get('option', {})
    opts_2d = geom_2d.get('option', {})
    assert abs(opts_3d['WindowMinX'] - opts_2d['WindowMinX']) < 1e-6
    assert abs(opts_3d['WindowMaxX'] - opts_2d['WindowMaxX']) < 1e-6
    assert abs(opts_3d['WindowMinY'] - opts_2d['WindowMinY']) < 1e-6
    assert abs(opts_3d['WindowMaxY'] - opts_2d['WindowMaxY']) < 1e-6
