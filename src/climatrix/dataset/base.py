from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Self

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from climatrix.dataset.axis import Axis, AxisType
from climatrix.dataset.domain import (
    Domain,
    SamplingNaNPolicy,
    ensure_single_var,
)
from climatrix.dataset.utils import ensure_list_or_slice
from climatrix.decorators import cm_arithmetic_binary_operator
from climatrix.exceptions import (
    DomainMismatchError,
    LongitudeConventionMismatch,
    SubsettingByNonDimensionAxisError,
)
from climatrix.types import Latitude, Longitude

if TYPE_CHECKING:
    from climatrix.reconstruct.type import ReconstructionType


def drop_scalar_coords_and_dims(da: xr.DataArray) -> xr.DataArray:
    coords_and_dims = {*da.dims, *da.coords.keys()}
    for coord in coords_and_dims:
        if len(da[coord].shape) == 0:
            da = da.drop_vars(coord)
    return da


log = logging.getLogger(__name__)


@xr.register_dataset_accessor("cm")
class BaseClimatrixDataset:
    """
    Base class for Climatrix workflows.

    This class provides a set of methods for manipulating
    xarray datasets. It is designed to be used as an
    xarray accessor, allowing you to call its methods
    directly on xarray datasets.

    The class supports basic arithmetic operations, including:
    addition, subtraction, multiplication, and division.

    Attributes
    ----------
    da : xarray.DataArray
        The underlying `xarray.DataArray` object (if
        single-variable `xarray.Dataset` was passed, it is squeezed
        to `xarray.DataArray`).
    domain : Domain
        The domain object representing the spatial
        and temporal dimensions of the dataset.
        See [`SparseDomain`][climatrix.dataset.domain.SparseDomain] and
        [`DenseDomain`][climatrix.dataset.domain.DenseDomain]
        for more details.
    """

    __slots__ = (
        "da",
        "domain",
    )

    da: xr.DataArray
    domain: Domain

    def __init__(self, xarray_obj: xr.DataArray):
        # NOTE: At the moment we only support DataArray or
        # Dataset with a single variable
        xarray_obj = ensure_single_var(xarray_obj)
        xarray_obj = drop_scalar_coords_and_dims(xarray_obj)
        self.domain = Domain(xarray_obj)
        self.da = xarray_obj

    # ###############################
    # Properties
    # ###############################
    @property
    def dims(self) -> tuple[AxisType, ...]:
        """
        Get the dimensions of the dataset.

        Returns
        -------
        tuple[AxisType, ...]
            A tuple of `AxisType` objects representing the dimensions
            of the dataset.
        """
        return self.domain.dims

    # ###############################
    #  Operators
    # ###############################
    @cm_arithmetic_binary_operator
    def __add__(self, other: Any) -> Self: ...  # noqa: E704

    @cm_arithmetic_binary_operator
    def __sub__(self, other: Any) -> Self: ...  # noqa: E704

    @cm_arithmetic_binary_operator
    def __mul__(self, other: Any) -> Self: ...  # noqa: E704

    @cm_arithmetic_binary_operator
    def __truediv__(self, other: Any) -> Self: ...  # noqa: E704

    # ###############################
    #  Rolling
    # ###############################
    def to_signed_longitude(self) -> Self:
        """
        Convert the dataset to signed longitude convention.

        The longitude values are converted to be in
        the range (-180 to 180 degrees).

        Examples
        --------
        >>> import climatrix as cm
        >>> dset = xr.open_dataset("path/to/dataset.nc").cm
        >>> dset.da
        <xarray.DataArray 'temperature' (time: 1, latitude: 180, longitude: 360)>
        ...
        Dimensions:  (time: 1, latitude: 180, longitude: 360)
        Coordinates:
          * time     (time) datetime64[ns] 2020-01-01
          * latitude (latitude) float64 -90.0 -89.0 -88.0 ... 88.0 89.0
          * longitude (longitude) float64 0.0 1.0 2.0 ... 357.0 358.0 359.0
        Data variables:
            temperature (time, latitude, longitude) float64 ...
        >>> dset2 = cm.to_signed_longitude()
        >>> dset2.da
        <xarray.DataArray 'temperature' (time: 1, latitude: 180, longitude: 360)>
        ...
        Dimensions:  (time: 1, latitude: 180, longitude: 360)
        Coordinates:
          * time     (time) datetime64[ns] 2020-01-01
          * latitude (latitude) float64 -90.0 -89.0 -88.0 ... 88.0 89.0
          * longitude (longitude) float64 -180.0 -179.0 -178.0 ... 177.0 178.0 179.0

        References
        ----------
        [1] Mancini, M., Walczak, J. Stojiljkovic, M., geokube: A Python
           library for geospatial data processing, 2024,
           [https://doi.org/10.5281/zenodo.10597965](https://doi.org/10.5281/zenodo.10597965)
           [https://github.com/CMCC-Foundation/geokube](https://github.com/CMCC-Foundation/geokube)
        """
        roll_value = (self.da[self.domain.longitude.name] >= 180).sum().item()
        res = self.da.assign_coords(
            {
                self.domain.longitude.name: (
                    ((self.da[self.domain.longitude.name] + 180) % 360) - 180
                )
            }
        ).roll(**{self.domain.longitude.name: roll_value}, roll_coords=True)
        res[self.domain.longitude.name].attrs.update(
            self.da[self.domain.longitude.name].attrs
        )
        return type(self)(res)

    def to_positive_longitude(self) -> Self:
        """
        Convert the dataset to positive longitude convention.

        The longitude values are converted to be in
        the range (0 to 360 degrees).

        Examples
        --------
        >>> import climatrix as cm
        >>> dset = xr.open_dataset("path/to/dataset.nc").cm
        >>> dset.da
        <xarray.DataArray 'temperature' (time: 1, latitude: 180,
        longitude: 360)>
        ...
        Dimensions:  (time: 1, latitude: 180, longitude: 360)
        Coordinates:
          * time     (time) datetime64[ns] 2020-01-01
          * latitude (latitude) float64 -90.0 -89.0 -88.0 ... 88.0 89.0
          * longitude (longitude) float64 -180.0 ... 178.0 179.0
        Data variables:
            temperature (time, latitude, longitude) float64 ...
        >>> dset2 = dset.to_positive_longitude()
        >>> dset2.da
        <xarray.DataArray 'temperature' (time: 1, latitude: 180,
          longitude: 360)>
        ...
        Dimensions:  (time: 1, latitude: 180, longitude: 360)
        Coordinates:
          * time     (time) datetime64[ns] 2020-01-01
          * latitude (latitude) float64 -90.0 -89.0 -88.0 ... 88.0 89.0
          * longitude (longitude) float64 0.0 1.0 ... 357.0 358.0 359.0
        Data variables:
            temperature (time, latitude, longitude) float64 ...

        References
        ----------
        [1] Mancini, M., Walczak, J. Stojiljkovic, M., geokube: A Python
           library for geospatial data processing, 2024,
           [https://doi.org/10.5281/zenodo.10597965](https://doi.org/10.5281/zenodo.10597965)
           [https://github.com/CMCC-Foundation/geokube](https://github.com/CMCC-Foundation/geokube)
        """
        roll_value = (self.da[self.domain.longitude.name] <= 0).sum().item()
        res = (
            self.da.assign_coords(
                {
                    self.domain.longitude.name: (
                        self.da[self.domain.longitude.name] % 360
                    )
                }
            )
            .roll(
                **{self.domain.longitude.name: -roll_value}, roll_coords=True
            )
            .assign_attrs(**self.da[self.domain.longitude.name].attrs)
        )
        res[self.domain.longitude.name].attrs.update(
            self.da[self.domain.longitude.name].attrs
        )
        return type(self)(res)

    # ###############################
    # Utility methods
    # ###############################

    def squeeze(self) -> Self:
        """
        Squeeze the dataset to remove dimensions of size 1.

        Returns
        -------
        Self
            The squeezed dataset.
        """
        da = self.da.squeeze()
        return type(self)(da)

    def profile_along_axes(
        self, *axes: AxisType | str
    ) -> Iterable[BaseClimatrixDataset]:
        """
        Generate profiles along the specified axes.

        Parameters
        ----------
        *axes : AxisType | str
            The axes along which to generate profiles.

        Yields
        ------
        BaseClimatrixDataset
            A dataset containing the profile along the specified axes.
        """
        axes = [AxisType.get(axis) for axis in axes]
        axis_values = []
        for axis in axes:
            axis_obj = self.domain.get_axis(axis)
            if axis_obj is None or axis_obj.values is None:
                continue
            axis_values.append(range(len(axis_obj.values)))

        for combination in product(*axis_values):
            profile = self.isel(
                {axis: value for axis, value in zip(axes, combination)}
            )
            yield profile

    def mask_nan(self, source: Self) -> Self:
        """
        Apply NaN values from another dataset to the current one.

        Parameters
        ----------
        source : BaseClimatrixDataset
            Dataset whose NaN values will be applied to the current one.

        Returns
        -------
        BaseClimatrixDataset
            A new dataset with NaN values applied.

        Raises
        ------
        TypeError
            If the `source` argument is not a BaseClimatrixDataset.
        ValueError
            If the domain of the `source` or the current dataset is sparse.
        DomainMismatchError
            If the domains of the `source` and the current dataset
            do not match.

        Examples
        --------
        >>> import climatrix as cm
        >>> dset1 = xr.open_dataset("path/to/dataset1.nc").cm
        >>> dset2 = xr.open_dataset("path/to/dataset2.nc").cm
        >>> dset1.mask_nan(dset2)
        """
        if not isinstance(source, BaseClimatrixDataset):
            raise TypeError("Argument `source` must be a BaseClimatrixDataset")
        if source.domain.is_sparse or self.domain.is_sparse:
            raise ValueError(
                "Masking NaN values is only supported for dense domain."
            )
        if self.domain != source.domain:
            log.error("Domains of the datasets do not match.")
            raise DomainMismatchError("Domains of the datasets do not match.")

        source = source.transpose(*self.dims)
        # NOTE: there can be slight differences (E.g. due to overflow
        # or floating point precision) in the coordinates of the source,
        source_da = source.da.reindex_like(
            self.da, method="nearest", tolerance=1e-5
        )
        da = xr.where(source_da.isnull(), np.nan, self.da)
        return type(self)(da)

    def transpose(self, *axes: AxisType | str) -> Self:
        """
        Transpose the dataset along the specified dimensions.

        Parameters
        ----------
        *axes : AxisType or str
            The axes along which to transpose the dataset.

        Returns
        -------
        Self
            The transposed dataset.

        Examples
        --------
        >>> import climatrix as cm
        >>> dset = xr.open_dataset("path/to/dataset.nc").cm
        >>> dset2 = dset.transpose("longitude", "latitude")
        """
        dim_names = [self.domain.get_axis(ax).name for ax in axes]
        da = self.da.transpose(*dim_names)
        da = da.compute() if hasattr(da.data, "dask") else da
        return type(self)(da)

    def sel(self, query: dict[AxisType | str, Any]) -> Self:
        """
        Select data along the specified axes.

        Parameters
        ----------
        query : dict
            Dictionary of axes and values to select.
            The keys can be either `AxisType` objects or strings
            associated to standard names of axis type.

        Returns
        -------
        Self
            The selected dataset.

        Examples
        --------
        >>> import climatrix as cm
        >>> from climatrix import AxisType
        >>> dset = xr.open_dataset("path/to/dataset.nc").cm
        >>> dset2 = dset.sel({"latitude": 10.0, "longitude": 20.0})
        >>> dset3 = dset.sel({AxisType.TIME: [datetime(2020, 1, 1)]})
        """
        query_ = {}
        for axis_type, value in query.items():
            axis_type = AxisType.get(axis_type)
            corresponding_axis = self.domain.get_axis(axis_type)
            if corresponding_axis is None:
                log.warning(
                    "Axis for '%s' not found in the dataset", axis_type
                )
                warnings.warn(
                    f"Axis for '{axis_type}' not found in the dataset."
                    "Skipping."
                )
                continue
            if not corresponding_axis.is_dimension:
                log.error(
                    "Axis '%s' is not a dimension in this dataset. ", axis_type
                )
                raise SubsettingByNonDimensionAxisError(
                    f"Axis '{axis_type}' is not a dimension in this dataset. "
                    "Subsetting by non-dimension axis is not supported."
                )
            query_[corresponding_axis.name] = ensure_list_or_slice(value)

        da = self.da.sel(query_)
        return type(self)(da)

    def isel(self, query: dict[Axis | str, Any]) -> Self:
        """
        Select data along the specified axes by index.

        Parameters
        ----------
        query : dict
            Dictionary of axes and values to select.
            The keys can be either `AxisType` objects or strings
            associated to standard names of axis type.
            The values must be integers, list of integers, or slices.

        Returns
        -------
        Self
            The selected dataset.

        Examples
        --------
        >>> import climatrix as cm
        >>> from climatrix import AxisType
        >>> dset = xr.open_dataset("path/to/dataset.nc").cm
        >>> dset2 = dset.isel({"latitude": 0, "longitude": 1})
        >>> dset3 = dset.isel({AxisType.TIME: slice(0, 2)})
        """
        query_ = {}
        for axis_type, value in query.items():
            axis_type = AxisType.get(axis_type)
            corresponding_axis = self.domain.get_axis(axis_type)
            if corresponding_axis is None:
                log.warning(
                    "Axis for '%s' not found in the dataset", axis_type
                )
                warnings.warn(
                    f"Axis for '{axis_type}' not found in the dataset."
                    "Skipping."
                )
                continue
            if not corresponding_axis.is_dimension:
                log.error(
                    "Axis '%s' is not a dimension in this dataset. ", axis_type
                )
                raise SubsettingByNonDimensionAxisError(
                    f"Axis '{axis_type}' is not a dimension in this dataset. "
                    "Subsetting by non-dimension axis is not supported."
                )
            query_[corresponding_axis.name] = ensure_list_or_slice(value)
        da = self.da.isel(query_)
        return type(self)(da)

    # ###############################
    #  Subsetting
    # ###############################
    def subset(
        self,
        north: float | None = None,
        south: float | None = None,
        west: float | None = None,
        east: float | None = None,
    ) -> Self:
        r"""
        Subset data with the specified bounding box.

        If an argument is not provided, it means no bounds set
        in that direction. For example, if `north` is not provided,
        it means that the maximum latitude of the dataset will be used.
        If `north` and `south` are provided, the dataset will be
        subsetted to the area between these two latitudes.

        Parameters
        ----------
        north : float, optional
            North latitude of the bounding box.
        south : float, optional
            South latitude of the bounding box.
        west : float, optional
            West longitude of the bounding box.
        east : float, optional
            East longitude of the bounding box.

        Returns
        -------
        Self
            The subsetted dataset.

        Raises
        ------
        LongitudeConventionMismatch
            - If the dataset is in positive-only convention (longitude $\lambda \in [0, 360]$) and
            negative values are requested, or vice versa.
            - If the dataset is in signed-longitude convention (longitude $\lambda \in [-180, 180]$) and
            positive values greater than 360 are requested.

        Examples
        --------
        >>> import climatrix as cm
        >>> globe_dset = xr.open_dataset("path/to/dataset.nc")
        >>> globe_dset
        <xarray.Dataset>
        Dimensions:  (time: 1, latitude: 180, longitude: 360)
        Coordinates:
          * time     (time) datetime64[ns] 2020-01-01
          * latitude (latitude) float64 -90.0 -89.0 -88.0 ... 88.0 89.0
          * longitude (longitude) float64 0.0 1.0 2.0 ... 357.0 358.0 359.0
        Data variables:
            temperature (time, latitude, longitude) float64 ...
        >>> dset2 = globe_dset.cm.subset(
        ...     north=10.0,
        ...     south=5.0,
        ...     west=20.0,
        ...     east=25.0,
        ... )
        >>> dset2 = globe_dset.cm.subset(
        ...     north=10.0,
        ...     south=5.0,
        ...     west=-50.0,
        ...     east=25.0,
        ... )
        LongitudeConventionMismatch: The dataset is in positive-only convention
        (longitude goes from 0 to 360) while you are
        requesting negative values (longitude goes from -180 to 180).
        """
        idx, start, stop = self.domain._compute_subset_indexers(
            north=north,
            south=south,
            west=west,
            east=east,
        )
        first_el, _ = (
            self.domain.longitude.values.min(),
            self.domain.longitude.values.max(),
        )
        start = 0 if start is None else start
        stop = 0 if stop is None else stop
        sel_neg_conv = (start < 0) | (stop < 0)
        sel_pos_conv = (start > 180) | (stop > 180)

        dset_neg_conv = first_el < 0
        dset_pos_conv = first_el >= 0
        if dset_pos_conv and sel_neg_conv:
            raise LongitudeConventionMismatch(
                "The dataset is in positive-only convention "
                "(longitude goes from 0 to 360) while you are "
                "requesting negative values (longitude goes "
                "from -180 to 180). Run `to_signed_longitude()` "
                "first to convert dataset properly."
            )
        if dset_neg_conv and sel_pos_conv:
            raise LongitudeConventionMismatch(
                "The dataset is in signed-longitude convention "
                "(longitude goes from -180 to 180) while you are "
                "requesting values from 0 to 360. "
                "Run `to_positive_longitude()` first to convert "
                "dataset properly."
            )
        da = self.da.sel(idx)
        if da.size == 0:
            logging.warning(
                "No data found in the specified bounding box. "
                "Returning empty dataset."
            )
            warnings.warn(
                "No data found in the specified bounding box "
                f"(north={north}, south={south}, west={west}, east={east}). "
                "Returning empty dataset."
            )

        return type(self)(da)

    def time(
        self, time: datetime | np.datetime64 | slice | list | np.ndarray
    ) -> Self:
        """
        Select data at a specific time or times.

        Parameters
        ----------
        time : datetime, np.datetime64, slice, list, or np.ndarray
            Time or times to be selected.

        Returns
        -------
        Self
            The dataset with the selected time or times.

        Examples
        --------
        >>> import climatrix as cm
        >>> dset = xr.open_dataset("path/to/dataset.nc")

        Selecting by `datetime` object:
        >>> dset.cm.time(datetime(2020, 1, 1))

        Selecting by `np.datetime64` object:
        >>> dset.cm.time(np.datetime64("2020-01-01"))

        Selecting by `str` object:
        >>> dset.cm.time(slice("2020-01-01"))

        Selecting by `list` of any of the above:
        >>> dset.cm.time([datetime(2020, 1, 1), np.datetime64("2020-01-02")])

        Selecting by `slice` object:
        >>> dset.cm.time(slice(datetime(2020, 1, 1), datetime(2020, 1, 2)))
        """
        time = ensure_list_or_slice(time)
        return type(self)(
            self.da.sel({self.domain.time.name: time}, method="nearest")
        )

    def itime(self, time: int | list[int] | np.ndarray | slice) -> Self:
        """
        Select time value by index.

        Parameters
        ----------
        time : int, list[int], np.ndarray, or slice
            Time index or indices to be selected.

        Returns
        -------
        Self
            The dataset with the selected time or times.

        Examples
        --------
        >>> import climatrix as cm
        >>> dset = xr.open_dataset("path/to/dataset.nc")

        Selecting by `int` object:
        >>> dset.cm.itime(0)

        Selecting by `list` of `int`s:
        >>> dset.cm.itime([0, 1])

        Selecting by `slice` object:
        >>> dset.cm.itime(slice(0, 2))
        """
        if self.domain.is_dynamic:
            time = ensure_list_or_slice(time)
            return type(self)(self.da.isel({self.domain.time.name: time}))
        return self

    # ##############################
    #  Sampling
    # ###############################
    def sample_uniform(
        self,
        portion: float | None = None,
        number: int | None = None,
        nan: SamplingNaNPolicy | str = "ignore",
    ) -> Self:
        """
        Sample the dataset using a uniform distribution.

        Parameters
        ----------
        portion : float, optional
            Portion of the dataset to be sampled.
        number : int, optional
            Number of points to be sampled.
        nan : SamplingNaNPolicy | str, optional
            Policy for handling NaN values.

        Notes
        -----
        At least one of `portion` or `number` must be provided.
        Cannot be provided both at the same time.

        Warns
        -----
        TooLargeSamplePortionWarning
            If the portion exceeds 1.0 or number of points exceeds
            the number of spatial points in the Domain

        Raises
        ------
        ValueError
            If the dataset contains NaN values and `nan` parameter
             (NaN handling policy) is set to `SamplingNaNPolicy.RAISE`.

        Examples
        --------
        >>> import climatrix as cm
        >>> dset = xr.open_dataset("path/to/dataset.nc")
        >>> sparse_dset = dset.cm.sample_uniform(portion=0.1)
        """
        nan = SamplingNaNPolicy.get(nan)
        if nan in (SamplingNaNPolicy.RAISE, SamplingNaNPolicy.IGNORE):
            idx = self.domain._compute_sample_uniform_indexers(
                portion=portion, number=number
            )
        elif nan == SamplingNaNPolicy.RESAMPLE:
            idx = self.domain._compute_sample_no_nans_indexers(
                self.da, portion=portion, number=number
            )
        da = self.da.isel(idx)
        if nan == SamplingNaNPolicy.RAISE and da.isnull().any():
            raise ValueError("Not all points have data")
        return type(self)(da)

    def sample_normal(
        self,
        portion: float | None = None,
        number: int | None = None,
        center_point: tuple[Longitude, Latitude] = None,
        sigma: float = 10.0,
        nan: SamplingNaNPolicy | str = "ignore",
    ) -> Self:
        """
        Sample the dataset using a normal distribution.

        Parameters
        ----------
        portion : float, optional
            Portion of the dataset to be sampled.
        number : int, optional
            Number of points to be sampled.
        center_point : tuple[Longitude, Latitude], optional
            Center point for the normal distribution.
        sigma : float, optional
            Standard deviation for the normal distribution.
        nan : SamplingNaNPolicy | str, optional
            Policy for handling NaN values.

        Notes
        -----
        At least one of `portion` or `number` must be provided.
        Cannot be provided both at the same time.

        Warns
        -----
        TooLargeSamplePortionWarning
            If the portion exceeds 1.0 or number of points exceeds
            the number of spatial points in the Domain

        Raises
        ------
        ValueError
            If the dataset contains NaN values and `nan` parameter
             (NaN handling policy) is set to `SamplingNaNPolicy.RAISE`.

        Examples
        --------
        >>> import climatrix as cm
        >>> dset = xr.open_dataset("path/to/dataset.nc")
        >>> sparse_dset = dset.cm.sample_normal(
        ...     number=1_000,
        ...     center_point=(10.0, 20.0),
        ...     sigma=5.0,
        ... )
        """
        nan = SamplingNaNPolicy.get(nan)
        idx = self.domain._compute_sample_normal_indexers(
            portion=portion,
            number=number,
            center_point=center_point,
            sigma=sigma,
        )
        da = self.da.isel(idx)
        if nan == SamplingNaNPolicy.RAISE and da.isnull().any():
            raise ValueError("Not all points have data")
        return type(self)(da)

    # ###############################
    #  Reconstruction
    # ###############################
    def reconstruct(
        self,
        target: Domain,
        *,
        method: ReconstructionType | str,
        **recon_kwargs,
    ) -> Self:
        """
        Reconstruct the dataset to a target domain.

        If target domain is sparse, the reconstruction will be sparse
        too. If target domain is dense, the reconstruction will be dense
        too. The reconstruction will be done using the method specified
        in the `method` argument.

        The method can be one of the following:
            Inverse Distance Weightining (`idw`),
            Ordinary Kriging (`ok`).

        Parameters
        ----------
        target : Domain
            The target domain to reconstruct the dataset to.
        method : ReconstructionType | str
            The method to use for reconstruction. Can be one of the
            following: 'idw', 'ok'.
        recon_kwargs : dict
            Additional keyword arguments to pass to the reconstruction
            method.

        See Also
        --------
        - [`climatrix.reconstruct.type.ReconstructionType`](/climatrix/api/#climatrix.reconstruct.type.ReconstructionType)
        - [`climatrix.reconstruct.idw.IDWReconstructor`](/climatrix/api/#climatrix.reconstruct.idw.IDWReconstructor)
        - [`climatrix.reconstruct.kriging.OrdinaryKrigingReconstructor`](/climatrix/api/#climatrix.reconstruct.kriging.OrdinaryKrigingReconstructor)
        - [`climatrix.reconstruct.siren.siren.SIRENReconstructor`](/climatrix/api/#climatrix.reconstruct.siren.siren.SIRENReconstructor)

        Returns
        -------
        Self
            The reconstructed dataset.
        """
        from climatrix.reconstruct.type import ReconstructionType

        method = ReconstructionType.get(method)
        return (
            ReconstructionType.get(method)
            .value(self, target_domain=target, **recon_kwargs)
            .reconstruct()
        )

    # ###############################
    #  Plotting
    # ###############################
    def plot(
        self,
        title: str | None = None,
        target: str | os.PathLike | Path | None = None,
        show: bool = True,
        **kwargs,
    ) -> Axes:
        """
        Plot the dataset on a map.

        The dataset is plotted using Cartopy and Matplotlib.

        Parameters
        ----------
        title : str, optional
            Title of the plot. If not provided, the name of the dataset
            will be used. If the dataset has no name, "Climatrix Dataset" will be used.
        target : str, os.PathLike, Path, or None, optional
            Path to save the plot. If not provided, the plot
            will not be saved.
        show : bool, optional
            Whether to show the plot. Default is True.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plotting function.

            - `figsize`: tuple, optional
                Size of the figure. Default is (12, 6).
            - `vmin`: float, optional
                Minimum value for the color scale. Default is None.
            - `vmax`: float, optional
                Maximum value for the color scale. Default is None.
            - `cmap`: str, optional
                Colormap to use for the plot. Default is "seismic".
            - `ax`: Axes, optional
                Axes to plot on. If not provided, a new figure and axes
                will be created.
            - `size`: int, optional
                Size of the points for sparse datasets. Default is 10.

        Returns
        -------
        Axes
            The axes object containing the plot.

        Raises
        ------
        NotImplementedError
            If the dataset is dynamic (contains time dimension
            with more than one value).
        """
        for axis in self.domain.all_axes_types:
            if axis in [AxisType.LATITUDE, AxisType.LONGITUDE, AxisType.POINT]:
                continue
            elif (
                self.domain.get_axis(axis).is_dimension
                and self.domain.get_size(axis) > 1
            ):
                warnings.warn(
                    f"Dataset contains axis '{axis}' with more than one "
                    "value. It is not yet supported for plotting. "
                    "The first value will be used."
                )
                self.isel({axis: 0}).plot()
                return
        figsize = kwargs.pop("figsize", (12, 6))
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        cmap = kwargs.pop("cmap", "seismic")
        ax = kwargs.pop("ax", None)
        cbar_name = kwargs.pop("cbar_name", None)
        title = title or self.da.name or "Climatrix Dataset"

        lat = self.da[self.domain.latitude.name]
        lon = self.da[self.domain.longitude.name]
        proj = ccrs.PlateCarree()

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": proj}
            )
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            ax.gridlines(draw_labels=True, linewidth=0.2, linestyle="--")

            ax.text(
                -0.07,
                0.55,
                "latitude",
                va="bottom",
                ha="center",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                -0.1,
                "longitude",
                va="bottom",
                ha="center",
                rotation="horizontal",
                rotation_mode="anchor",
                transform=ax.transAxes,
            )

            ax.set_aspect("equal")

        if self.domain.is_sparse:
            size = kwargs.pop("size", 10)
            actor = ax.scatter(
                lon,
                lat,
                c=self.da.squeeze(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=proj,
                marker="x",
                s=size,
            )
        else:
            actor = ax.pcolormesh(
                lon,
                lat,
                self.da.squeeze(),
                transform=proj,
                cmap=cmap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )

        cbar = plt.colorbar(
            actor, ax=ax, orientation="vertical", shrink=0.7, pad=0.05
        )
        cbar.set_label(cbar_name or self.da.name or "Value")

        if title:
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        if target is not None:
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(target, dpi=300)
        if show:
            plt.show()
        return ax
