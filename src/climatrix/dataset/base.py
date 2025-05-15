from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from climatrix.dataset.domain import (
    Domain,
    SamplingNaNPolicy,
    ensure_single_var,
)
from climatrix.decorators import cm_arithmetic_binary_operator
from climatrix.exceptions import LongitudeConventionMismatch
from climatrix.types import Latitude, Longitude

if TYPE_CHECKING:
    from climatrix.reconstruct.type import ReconstructionType


def drop_scalar_coords_and_dims(da: xr.DataArray) -> xr.DataArray:
    coords_and_dims = {*da.dims, *da.coords.keys()}
    for coord in coords_and_dims:
        if len(da[coord].shape) == 0:
            da = da.drop(coord)
    return da


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
        roll_value = (self.da[self.domain.longitude_name] >= 180).sum().item()
        res = self.da.assign_coords(
            {
                self.domain.longitude_name: (
                    ((self.da[self.domain.longitude_name] + 180) % 360) - 180
                )
            }
        ).roll(**{self.domain.longitude_name: roll_value}, roll_coords=True)
        res[self.domain.longitude_name].attrs.update(
            self.da[self.domain.longitude_name].attrs
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
        roll_value = (self.da[self.domain.longitude_name] <= 0).sum().item()
        res = (
            self.da.assign_coords(
                {
                    self.domain.longitude_name: (
                        self.da[self.domain.longitude_name] % 360
                    )
                }
            )
            .roll(
                **{self.domain.longitude_name: -roll_value}, roll_coords=True
            )
            .assign_attrs(**self.da[self.domain.longitude_name].attrs)
        )
        res[self.domain.longitude_name].attrs.update(
            self.da[self.domain.longitude_name].attrs
        )
        return type(self)(res)

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

        da = xr.where(source.da.isnull(), np.nan, self.da).squeeze()
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
            self.domain.longitude.min(),
            self.domain.longitude.max(),
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
        return type(self)(
            self.da.sel({self.domain.time_name: time}, method="nearest")
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
            return type(self)(self.da.isel({self.domain.time_name: time}))
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
        da = self.da.sel(idx)
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
        da = self.da.sel(idx)
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
        [`climatrix.reconstruct.type.ReconstructionType`](/climatrix/api/#climatrix.reconstruct.type.ReconstructionType)
        [`climatrix.reconstruct.idw.IDWReconstructor`](/climatrix/api/#climatrix.reconstruct.idw.IDWReconstructor)
        [`climatrix.reconstruct.kriging.OrdinaryKrigingReconstructor`](/climatrix/api/#climatrix.reconstruct.kriging.OrdinaryKrigingReconstructor)

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
            Path to save the plot. If not provided, the plot will be shown
            on the screen.
        show : bool, optional
            Whether to show the plot. If `target` is provided, this
            parameter is ignored. Default is True.
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
        if self.domain.is_dynamic:
            raise NotImplementedError(
                "Plotting is not yet supported for dynamic datasets."
            )
        figsize = kwargs.pop("figsize", (12, 6))
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        cmap = kwargs.pop("cmap", "seismic")
        ax = kwargs.pop("ax", None)
        cbar_name = kwargs.pop("cbar_name", None)
        title = title or self.da.name or "Climatrix Dataset"

        lat = self.da[self.domain.latitude_name]
        lon = self.da[self.domain.longitude_name]
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
                c=self.da,
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
                self.da,
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
