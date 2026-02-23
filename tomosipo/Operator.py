import numpy as np
import tomosipo as ts
from tomosipo.Data import Data
from tomosipo.astra import (
    create_astra_projector,
    direct_fp,
    direct_bp,
    fp_2d_slice,
    bp_2d_slice,
)


def to_astra_compatible_operator_geometry(vg, pg):
    """Convert volume vector geometry to volume geometry (if necessary)

    ASTRA does not support arbitrarily oriented volume geometries. If
    `vg` is a VolumeVectorGeometry, we rotate and translate both `vg`
    and `pg` such that `vg` is axis-aligned, and positioned on the
    origin, which makes it ASTRA-compatible.

    Parameters
    ----------
    vg:
        volume geometry
    pg:
        projection geometry

    Returns
    -------
    (VolumeGeometry, ProjectionGeometry)
        A non-vector volume geometry centered on the origin and its
        corresponding projection geometry.

    """
    if isinstance(vg, ts.geometry.VolumeGeometry):
        return (vg, pg)

    if not isinstance(vg, ts.geometry.VolumeVectorGeometry):
        raise TypeError(f"Expected volume geometry. Got {type(vg)}. ")

    vg = vg.to_vec()
    # Change perspective *without* changing the voxel volume.
    P = ts.from_perspective(
        pos=vg.pos,
        w=vg.w / ts.vector_calc.norm(vg.w)[:, None],
        v=vg.v / ts.vector_calc.norm(vg.v)[:, None],
        u=vg.u / ts.vector_calc.norm(vg.u)[:, None],
    )
    # Move vg to perspective:
    vg = P * vg
    pg = P * pg.to_vec()

    # Assert that vg is now axis-aligned and positioned on the origin:
    voxel_size = vg.voxel_size
    assert np.allclose(vg.pos, np.array([0, 0, 0]))
    assert np.allclose(vg.w, voxel_size[0] * np.array([1, 0, 0]))
    assert np.allclose(vg.v, voxel_size[1] * np.array([0, 1, 0]))
    assert np.allclose(vg.u, voxel_size[2] * np.array([0, 0, 1]))

    axis_aligned_vg = ts.volume(shape=vg.shape, pos=0, size=vg.size)

    return axis_aligned_vg, pg


def operator(
    volume_geometry,
    projection_geometry,
    voxel_supersampling=1,
    detector_supersampling=1,
    additive=False,
):
    """Create a new tomographic operator

    Parameters:
    -----------
    volume_geometry: `VolumeGeometry`
        The domain of the operator.

    projection_geometry:  `ProjectionGeometry`
        The range of the operator.

    voxel_supersampling: `int` (optional)
        Specifies the amount of voxel supersampling, i.e., how
        many (one dimension) subvoxels are generated from a single
        parent voxel. The default is 1.

    detector_supersampling: `int` (optional)
        Specifies the amount of detector supersampling, i.e., how
        many rays are cast per detector. The default is 1.

    additive: `bool` (optional)
        Specifies whether the operator should overwrite its range
        (forward) and domain (transpose). When `additive=True`,
        the operator adds instead of overwrites. The default is
        `additive=False`.

    Returns
    -------
    Operator
        A linear tomographic projection operator
    """
    return Operator(
        volume_geometry,
        projection_geometry,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
        additive=additive,
    )


def _to_link(geometry, x):
    if isinstance(x, Data):
        return x.link
    else:
        return ts.link(geometry, x)


class Operator:
    """A linear tomographic projection operator

    An operator describes and computes the projection from a volume onto a
    projection geometry.
    """

    def __init__(
        self,
        volume_geometry,
        projection_geometry,
        voxel_supersampling=1,
        detector_supersampling=1,
        additive=False,
    ):
        """Create a new tomographic operator

        Parameters
        ----------
        volume_geometry: `VolumeGeometry`
            The domain of the operator.

        projection_geometry:  `ProjectionGeometry`
            The range of the operator.

        voxel_supersampling: `int` (optional)
            Specifies the amount of voxel supersampling, i.e., how
            many (one dimension) subvoxels are generated from a single
            parent voxel. The default is 1.

        detector_supersampling: `int` (optional)
            Specifies the amount of detector supersampling, i.e., how
            many rays are cast per detector. The default is 1.

        additive: `bool` (optional)
            Specifies whether the operator should overwrite its range
            (forward) and domain (transpose). When `additive=True`,
            the operator adds instead of overwrites. The default is
            `additive=False`.

        """
        super(Operator, self).__init__()
        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry

        vg, pg = to_astra_compatible_operator_geometry(
            volume_geometry, projection_geometry
        )
        self.astra_compat_vg = vg
        self.astra_compat_pg = pg

        self.astra_projector = create_astra_projector(
            self.astra_compat_vg,
            self.astra_compat_pg,
            voxel_supersampling=voxel_supersampling,
            detector_supersampling=detector_supersampling,
        )
        self.additive = additive
        self._transpose = BackprojectionOperator(self)

    def _fp(self, volume, out=None):
        vlink = _to_link(self.astra_compat_vg, volume)

        if out is not None:
            plink = _to_link(self.astra_compat_pg, out)
        else:
            if self.additive:
                plink = vlink.new_zeros(self.range_shape)
            else:
                plink = vlink.new_empty(self.range_shape)

        direct_fp(self.astra_projector, vlink, plink, additive=self.additive)

        if isinstance(volume, Data):
            return ts.data(self.projection_geometry, plink.data)
        else:
            return plink.data

    def _bp(self, projection, out=None):
        """Apply backprojection

        :param projection: `np.array` or `Data`
            An input projection dataset. If a numpy array, the shape
            must match the operator geometry. If the projection dataset is
            an instance of `Data`, its geometry must match the
            operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A volume dataset on which the projection dataset has been
            backprojected.
        :rtype: `Data`

        """
        plink = _to_link(self.astra_compat_pg, projection)

        if out is not None:
            vlink = _to_link(self.astra_compat_vg, out)
        else:
            if self.additive:
                vlink = plink.new_zeros(self.domain_shape)
            else:
                vlink = plink.new_empty(self.domain_shape)

        direct_bp(
            self.astra_projector,
            vlink,
            plink,
            additive=self.additive,
        )

        if isinstance(projection, Data):
            return ts.data(self.volume_geometry, vlink.data)
        else:
            return vlink.data

    def __call__(self, volume, out=None):
        """Apply operator

        :param volume: `np.array` or `Data`
            An input volume. If a numpy array, the shape must match
            the operator geometry. If the input volume is an instance
            of `Data`, its geometry must match the operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A projection dataset on which the volume has been forward
            projected.
        :rtype: `Data`

        """
        return self._fp(volume, out)

    def transpose(self):
        """Return backprojection operator"""
        return self._transpose

    @property
    def T(self):
        """The transpose operator

        This property returns the transpose (backprojection) operator.
        """
        return self.transpose()

    @property
    def domain(self):
        """The domain (volume geometry) of the operator"""
        return self.volume_geometry

    @property
    def range(self):
        """The range (projection geometry) of the operator"""
        return self.projection_geometry

    @property
    def domain_shape(self):
        """The expected shape of the input (volume) data"""
        return ts.links.geometry_shape(self.astra_compat_vg)

    @property
    def range_shape(self):
        """The expected shape of the output (projection) data"""
        return ts.links.geometry_shape(self.astra_compat_pg)


class BackprojectionOperator:
    """Transpose of the Forward operator

    The idea of having a dedicated class for the backprojection
    operator, which just saves a link to the "real" operator has
    been shamelessly ripped from OpTomo.

    We have the following property:

    >>> import tomosipo as ts
    >>> vg = ts.volume(shape=10)
    >>> pg = ts.parallel(angles=10, shape=10)
    >>> A = ts.operator(vg, pg)
    >>> A.T is A.T.T.T
    True

    It is nice that we do not allocate a new object every time we use
    `A.T`. If we did, users might save the transpose in a separate
    variable for 'performance reasons', writing

    >>> A = ts.operator(vg, pg)
    >>> A_T = A.T

    This is a waste of time.
    """

    def __init__(
        self,
        parent,
    ):
        """Create a new tomographic operator"""
        super(BackprojectionOperator, self).__init__()
        self.parent = parent

    def __call__(self, projection, out=None):
        """Apply operator

        :param projection: `np.array` or `Data`
            An input projection. If a numpy array, the shape must match
            the operator geometry. If the input volume is an instance
            of `Data`, its geometry must match the operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A projection dataset on which the volume has been forward
            projected.
        :rtype: `Data`

        """
        return self.parent._bp(projection, out)

    def transpose(self):
        """Return forward projection operator"""
        return self.parent

    @property
    def T(self):
        """The transpose of the backprojection operator

        This property returns the transpose (forward projection) operator.
        """
        return self.transpose()

    @property
    def domain(self):
        """The domain (projection geometry) of the operator"""
        return self.parent.range

    @property
    def range(self):
        """The range (volume geometry) of the operator"""
        return self.parent.domain

    @property
    def domain_shape(self):
        """The expected shape of the input (projection) data"""
        return self.parent.range_shape

    @property
    def range_shape(self):
        """The expected shape of the output (volume) data"""
        return self.parent.domain_shape


###############################################################################
#                       2D Slice-based Operator                               #
###############################################################################


def _to_numpy(x):
    """Convert input to a numpy array (handles Data, torch.Tensor, np.ndarray)."""
    if isinstance(x, Data):
        return np.ascontiguousarray(x.data, dtype=np.float32)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        pass
    return np.ascontiguousarray(x, dtype=np.float32)


def _result_like(input_val, result_np, geometry):
    """Return result in the same type as input (Data, torch.Tensor, or np.ndarray)."""
    if isinstance(input_val, Data):
        return ts.data(geometry, result_np)
    try:
        import torch
        if isinstance(input_val, torch.Tensor):
            result_tensor = torch.from_numpy(result_np)
            return result_tensor.to(device=input_val.device, dtype=torch.float32)
    except ImportError:
        pass
    return result_np


def operator_2d(
    volume_geometry,
    projection_geometry,
    voxel_supersampling=1,
    detector_supersampling=1,
    additive=False,
):
    """Create a 2D slice-based tomographic operator

    This operator uses ASTRA's 2D CUDA operators (FP_CUDA / BP_CUDA)
    instead of 3D CUDA operators, bypassing the 3D CUDA texture size
    limits. It processes the volume slice-by-slice along the Z axis.

    This is exact for parallel-beam geometries (slices are independent).
    Only parallel-beam geometries are supported.

    Parameters
    ----------
    volume_geometry: `VolumeGeometry`
        The domain of the operator.

    projection_geometry: `ParallelGeometry` or `ParallelVectorGeometry`
        The range of the operator. Must be a parallel-beam geometry.

    voxel_supersampling: `int` (optional)
        Specifies the amount of voxel supersampling. Default is 1.

    detector_supersampling: `int` (optional)
        Specifies the amount of detector supersampling. Default is 1.

    additive: `bool` (optional)
        If True, the operator adds instead of overwrites. Default is False.

    Returns
    -------
    Operator2D
        A 2D slice-based tomographic operator.
    """
    return Operator2D(
        volume_geometry,
        projection_geometry,
        voxel_supersampling=voxel_supersampling,
        detector_supersampling=detector_supersampling,
        additive=additive,
    )


class Operator2D:
    """A 2D slice-based tomographic operator

    Uses ASTRA's 2D CUDA operators (FP_CUDA / BP_CUDA) to process
    a 3D volume slice-by-slice along the Z axis. This bypasses the
    3D CUDA texture size limits.

    Only parallel-beam geometries are supported (slices are independent
    in parallel-beam CT).
    """

    def __init__(
        self,
        volume_geometry,
        projection_geometry,
        voxel_supersampling=1,
        detector_supersampling=1,
        additive=False,
    ):
        """Create a 2D slice-based tomographic operator

        Parameters
        ----------
        volume_geometry: `VolumeGeometry`
            The domain of the operator.

        projection_geometry: `ParallelGeometry` or `ParallelVectorGeometry`
            The range of the operator.

        voxel_supersampling: `int` (optional)
            Specifies the amount of voxel supersampling. Default is 1.

        detector_supersampling: `int` (optional)
            Specifies the amount of detector supersampling. Default is 1.

        additive: `bool` (optional)
            If True, adds to existing data. Default is False.
        """
        super(Operator2D, self).__init__()

        if not ts.geometry.is_parallel(projection_geometry):
            raise ValueError(
                f"Operator2D only supports parallel-beam projection geometries. "
                f"Got: {type(projection_geometry).__name__}. "
                f"Use ts.operator() for cone-beam geometries."
            )

        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry
        self.voxel_supersampling = voxel_supersampling
        self.detector_supersampling = detector_supersampling
        self.additive = additive

        vg, pg = to_astra_compatible_operator_geometry(
            volume_geometry, projection_geometry
        )
        self.astra_compat_vg = vg
        self.astra_compat_pg = pg

        # Pre-compute 2D ASTRA geometries
        self._vol_geom_2d = vg.to_astra_2d()
        self._proj_geom_2d = pg.to_astra_2d()

        self._transpose = BackprojectionOperator(self)

    def _fp(self, volume, out=None):
        """Forward project slice-by-slice using FP_CUDA."""
        vol_np = _to_numpy(volume)
        Z = vol_np.shape[0]

        if out is not None:
            sino_np = _to_numpy(out)
        else:
            sino_shape = self.range_shape
            if self.additive:
                sino_np = np.zeros(sino_shape, dtype=np.float32)
            else:
                sino_np = np.empty(sino_shape, dtype=np.float32)

        for z in range(Z):
            slice_sino = fp_2d_slice(
                vol_np[z],
                self._proj_geom_2d,
                self._vol_geom_2d,
                detector_supersampling=self.detector_supersampling,
            )
            if self.additive:
                sino_np[z] += slice_sino
            else:
                sino_np[z] = slice_sino

        if out is not None and isinstance(volume, Data):
            out.data[:] = sino_np
            return out

        return _result_like(volume, sino_np, self.projection_geometry)

    def _bp(self, projection, out=None):
        """Backproject slice-by-slice using BP_CUDA."""
        sino_np = _to_numpy(projection)
        Z = sino_np.shape[0]

        if out is not None:
            vol_np = _to_numpy(out)
        else:
            vol_shape = self.domain_shape
            if self.additive:
                vol_np = np.zeros(vol_shape, dtype=np.float32)
            else:
                vol_np = np.empty(vol_shape, dtype=np.float32)

        for z in range(Z):
            slice_bp = bp_2d_slice(
                sino_np[z],
                self._proj_geom_2d,
                self._vol_geom_2d,
                voxel_supersampling=self.voxel_supersampling,
            )
            if self.additive:
                vol_np[z] += slice_bp
            else:
                vol_np[z] = slice_bp

        if out is not None and isinstance(projection, Data):
            out.data[:] = vol_np
            return out

        return _result_like(projection, vol_np, self.volume_geometry)

    def __call__(self, volume, out=None):
        """Apply forward projection

        :param volume: `np.array`, `torch.Tensor`, or `Data`
            An input volume.
        :param out: (optional) output buffer.
        :returns: forward projected sinogram data.
        """
        return self._fp(volume, out)

    def transpose(self):
        """Return backprojection operator"""
        return self._transpose

    @property
    def T(self):
        """The transpose (backprojection) operator."""
        return self.transpose()

    @property
    def domain(self):
        """The domain (volume geometry) of the operator"""
        return self.volume_geometry

    @property
    def range(self):
        """The range (projection geometry) of the operator"""
        return self.projection_geometry

    @property
    def domain_shape(self):
        """The expected shape of the input (volume) data"""
        return ts.links.geometry_shape(self.astra_compat_vg)

    @property
    def range_shape(self):
        """The expected shape of the output (projection) data"""
        return ts.links.geometry_shape(self.astra_compat_pg)
