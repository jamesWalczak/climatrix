from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from collections import OrderedDict
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self

import numpy as np
import xarray as xr

from climatrix.dataset.axis import Axis, AxisType, Time
from climatrix.exceptions import MissingAxisError
from climatrix.types import Latitude, Longitude
from climatrix.warnings import (
    DomainMismatchWarning,
    TooLargeSamplePortionWarning,
)

_DEFAULT_LAT_RESOLUTION = 0.1
_DEFAULT_LON_RESOLUTION = 0.1

log = logging.getLogger(__name__)


def validate_input(da: xr.Dataset | xr.DataArray):
    if not isinstance(da, (xr.Dataset, xr.DataArray)):
        raise NotImplementedError(
            "At the moment, dataset can be created only based on "
            "xarray.DataArray or single-variable xarray.Dataset "
            f"objects, but provided {type(da).__name__}"
        )


def ensure_single_var(da: xr.Dataset | xr.DataArray) -> xr.DataArray:
    if isinstance(da, xr.Dataset):
        if len(da.data_vars) > 1:
            raise ValueError(
                "Dataset can be created only based on "
                "xarray.DataArray or xarray.Dataset with single variable "
                "objects, but provided xarray.Dataset with multiple "
                "data_vars."
            )
        elif len(da.data_vars) == 1:
            return da[list(da.data_vars.keys())[0]]
    return da


def match_axes(da: xr.DataArray) -> dict[AxisType, str]:
    axes = OrderedDict()
    for dim in da.dims:
        for axis in Axis.get_all_axes():
            if axis.matches(dim):
                axes[axis.type] = Axis(
                    dim, np.atleast_1d(da[dim].values).astype(axis.dtype), True
                )
    for coord in da.coords.keys():
        for axis in Axis.get_all_axes():
            if axis.type in axes:
                # If the axis is already matched by dim, skip it
                continue
            if axis.matches(coord):
                axes[axis.type] = Axis(
                    coord,
                    np.atleast_1d(da[coord].values).astype(axis.dtype),
                    False,
                )

    return axes


def validate_spatial_axes(axis_mapping: dict[AxisType, Axis]):
    for axis in [AxisType.LATITUDE, AxisType.LONGITUDE]:
        if axis not in axis_mapping:
            raise ValueError(f"Dataset has no {axis.name} axis")


def check_is_dense(axis_mapping: dict[AxisType, Axis]) -> bool:
    return (
        axis_mapping[AxisType.LATITUDE].is_dimension
        and axis_mapping[AxisType.LONGITUDE].is_dimension
    )


def ensure_all_numpy_arrays(coords: dict) -> None:
    return {k: np.array(v, ndmin=1) for k, v in coords.items()}


def filter_out_single_value_coord(coords: dict):
    return {k: v for k, v in coords.items() if len(v) > 1}


class SamplingNaNPolicy(StrEnum):
    IGNORE = "ignore"
    RESAMPLE = "resample"
    RAISE = "raise"

    def __missing__(self, value):
        raise ValueError(f"Unknown NaN policy: {value}")

    @classmethod
    def get(cls, value: str | Self):
        if isinstance(value, cls):
            return value
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Unknown Nan policy: {value}")


class Domain:
    """
    Base class for domain objects.

    Attributes
    ----------
    is_sparse : ClassVar[bool]
        Indicates if the domain is sparse or dense.
    _axes : dict[AxisType, Axis]
        Mapping of `AxisType` to the corresponding `Axis` object.
    """

    __slots__ = ("_axes",)
    is_sparse: ClassVar[bool]
    _axes: dict[AxisType, Axis]

    def __new__(cls, xarray_obj: xr.Dataset | xr.DataArray):
        validate_input(xarray_obj)
        da = ensure_single_var(xarray_obj)
        axis_mapping = match_axes(da)
        validate_spatial_axes(axis_mapping)

        if cls is not Domain:
            return super().__new__(cls)
        if check_is_dense(axis_mapping):
            domain = DenseDomain(da)
        else:
            domain = SparseDomain(da)
        domain._axes = axis_mapping
        return domain

    @classmethod
    def from_lat_lon(
        cls,
        lat: slice | np.ndarray = slice(-90, 90, _DEFAULT_LAT_RESOLUTION),
        lon: slice | np.ndarray = slice(-180, 180, _DEFAULT_LON_RESOLUTION),
        kind: Literal["sparse", "dense"] = "dense",
    ) -> Self:
        """
        Create a domain from latitude and longitude coordinates.

        Parameters
        ----------
        lat : slice or np.ndarray
            Latitude coordinates. If a slice is provided, it will be
            converted to a numpy array using the specified step.
        lon : slice or np.ndarray
            Longitude coordinates. If a slice is provided, it will be
            converted to a numpy array using the specified step.
        kind : str
            Type of domain to create. Can be either "dense" or "sparse".
            Default is "dense".

        Returns
        -------
        Domain
            An instance of the Domain class with the specified latitude
            and longitude coordinates.
        """
        if isinstance(lat, slice):
            lat = np.arange(
                lat.start,
                lat.stop + lat.step,
                lat.step,
            )
        if isinstance(lon, slice):
            lon = np.arange(
                lon.start,
                lon.stop + lon.step,
                lon.step,
            )
        if kind == "dense":
            return cls(xr.Dataset(coords={"lat": lat, "lon": lon}))
        elif kind == "sparse":
            if len(lat) != len(lon):
                raise ValueError(
                    "For sparse domain, lat and lon must have the same length"
                )
            return cls(
                xr.Dataset(
                    coords={"lat": ("point", lat), "lon": ("point", lon)}
                )
            )
        else:
            raise ValueError(f"Unknown kind: {kind}")

    @property
    def latitude(self) -> Axis:
        """Latitude axis"""
        if AxisType.LATITUDE not in self._axes:
            raise MissingAxisError(
                f"Latitude axis not found in axes ({list(self._axes.keys())})"
            )
        return self._axes[AxisType.LATITUDE]

    @property
    def longitude(self) -> Axis:
        """Longitude axis"""
        if AxisType.LONGITUDE not in self._axes:
            raise MissingAxisError(
                f"Longitude axis not found in axes ({list(self._axes.keys())})"
            )
        return self._axes[AxisType.LONGITUDE]

    @property
    def point(self) -> Axis | None:
        """Point axis"""
        if AxisType.POINT not in self._axes:
            raise MissingAxisError(
                f"Point axis not found in axes ({list(self._axes.keys())})"
            )
        return self._axes.get(AxisType.POINT)

    @property
    def time(self) -> Axis | None:
        """Time axis"""
        if AxisType.TIME not in self._axes:
            raise MissingAxisError(
                f"Time axis not found in axes ({list(self._axes.keys())})"
            )
        return self._axes.get(AxisType.TIME)

    @property
    def vertical(self) -> Axis | None:
        """Vertical axis"""
        if AxisType.VERTICAL not in self._axes:
            raise MissingAxisError(
                f"Vertical axis not found in axes ({list(self._axes.keys())})"
            )
        return self._axes.get(AxisType.VERTICAL)

    @property
    def all_axes_types(self) -> list[AxisType]:
        """All axis types in the domain."""
        return list(self._axes.keys())

    @property
    def dims(self) -> tuple[AxisType, ...]:
        """
        Get the dimensions of the dataset.

        Returns
        -------
        tuple[AxisType, ...]
            A tuple of `AxisType` objects representing the dimensions
            of the dataset.

        Notes
        -----
        The dimensions are determined by the axes that are marked as
        dimensions in the domain. E.g. if underlying dataset has
        shape `(5, 10, 20)`, it means there are 3 dimensional axes.
        """
        return tuple([k for k, v in self._axes.items() if v.is_dimension])

    def get_size(self, axis: AxisType | str) -> int:
        """
        Get the size of the specified axis.

        Parameters
        ----------
        axis : AxisType
            The axis for which to get the size.

        Returns
        -------
        int
            The size of the specified axis.
        """
        axis_type = AxisType.get(axis)
        if axis_type not in self._axes:
            return 1
        return self._axes[axis_type].size

    def has_axis(self, axis: AxisType | str) -> bool:
        """
        Check if the specified axis exists in the domain.

        Parameters
        ----------
        axis : AxisType
            The axis type to check.

        Returns
        -------
        bool
            True if the axis exists, False otherwise.
        """
        return AxisType.get(axis) in self._axes

    def get_axis(self, axis: AxisType | str) -> Axis | None:
        """
        Get the name of the specified axis.

        Parameters
        ----------
        axis : AxisType
            The axis type for which to get the name.

        Returns
        -------
        Axis | None
            The Axis object, or None if not found.
        """
        return self._axes.get(AxisType.get(axis))

    @property
    def size(self) -> int:
        """Domain size."""
        if AxisType.POINT in self._axes:
            lat_lon = self.get_size(AxisType.POINT)
        else:
            lat_lon = self.get_size(AxisType.LATITUDE) * self.get_size(
                AxisType.LONGITUDE
            )
        try:
            if self.time:
                lat_lon *= self.get_size(AxisType.TIME)
        except MissingAxisError:
            pass
        return lat_lon

    @property
    def is_dynamic(self) -> bool:
        """If the domain is dynamic."""
        try:
            return self.time and self.get_size(AxisType.TIME) > 1
        except MissingAxisError:
            return False

    @property
    def shape(self) -> tuple[int, ...]:
        """Domain shape."""
        return tuple(
            [axis.size for axis in self._axes.values() if axis.is_dimension]
        )

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, Domain):
            return False
        self_only = list(set(self._axes.keys()) - set(value._axes.keys()))
        value_only = list(set(value._axes.keys()) - set(self._axes.keys()))
        if len(self_only) > 0 or len(value_only) > 0:
            log.warning(
                f"Domain mismatch: {self_only} in self, {value_only} in value"
            )
            warnings.warn(
                f"Domain mismatch: {self_only} (self), {value_only} (value). "
                "To compare domains, they need to have the same axis. "
                "Remember axis can be removed by using the `drop` method. ",
                DomainMismatchWarning,
            )
            return False
        for k in self._axes.keys():
            if self._axes[k] != value._axes[k]:
                return False

        return True

    @abstractmethod
    def _compute_subset_indexers(
        north: float | None = None,
        south: float | None = None,
        west: float | None = None,
        east: float | None = None,
    ) -> tuple[dict[str, Any], float, float]:
        raise NotImplementedError

    @abstractmethod
    def _compute_sample_uniform_indexers(
        self, portion: float | None = None, number: int | None = None
    ) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def _compute_sample_no_nans_indexers(
        self,
        portion: float | None = None,
        number: int | None = None,
    ) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def _compute_sample_normal_indexers(
        self,
        portion: float | None = None,
        number: int | None = None,
        nan: SamplingNaNPolicy | str = "ignore",
        center_point: tuple[Longitude, Latitude] = None,
        sigma: float = 10.0,
    ) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def get_all_spatial_points(self) -> np.ndarray:
        raise NotImplementedError

    def _get_sampling_points_nbr(
        self, portion: float | None = None, number: int | None = None
    ) -> int:
        if not (portion or number):
            raise ValueError("Either portion or number must be provided")
        if portion and number:
            raise ValueError(
                "Either portion or number must be provided, but not both"
            )
        if (portion and portion > 1.0) or (number and number > self.size):
            warnings.warn(
                "Requesting more than 100% of the data will result in "
                "duplicates and excessive memory usage",
                TooLargeSamplePortionWarning,
            )
        if portion:
            number = int(self.size * portion)
        return number

    @abstractmethod
    def to_xarray(
        self, values: np.ndarray, name: str | None = None
    ) -> xr.DataArray:
        raise NotImplementedError


class SparseDomain(Domain):
    """
    Sparse domain class.

    Supports operations on sparse spatial domain.
    """

    is_sparse: ClassVar[bool] = True

    def get_all_spatial_points(self) -> np.ndarray:
        """
        Get all spatial points in the domain.

        Returns
        -------
        np.ndarray
            An array of shape (n_points, 2) containing the latitude and
            longitude coordinates of all points in the domain.

        Examples
        --------
        >>> points = domain.get_all_spatial_points()
        >>> points
        array([[ 0. , -0.1],
               [ 0. ,  0. ],
               [ 0. ,  0.1],
               ...
        """
        if self.latitude is None or self.longitude is None:
            raise MissingAxisError(
                "Latitude or Longitude axis is not initialized."
            )
        return np.stack((self.latitude.values, self.longitude.values), axis=1)

    def _compute_subset_indexers(
        self,
        north: float | None = None,
        south: float | None = None,
        west: float | None = None,
        east: float | None = None,
    ) -> tuple[dict[str, Any], float, float]:
        if not (north or south or west or east):
            warnings.warn(
                "Subset parameters not provided. Returning the source dataset"
            )
            return type(self)(self.da)
        if north and south and north < south:
            raise ValueError("North must be greater than south")
        if west and east and west > east:
            raise ValueError("East must be greater than west")
        lat_mask = np.logical_and(
            self.latitude.values >= south, self.latitude.values <= north
        )
        lon_mask = np.logical_and(
            self.longitude.values >= west, self.longitude.values <= east
        )
        point_mask = np.logical_and(lat_mask, lon_mask)
        point_idx = np.where(point_mask)[0]
        idx = {self.point.name: point_idx}

        lon_vals = self.longitude.values[lon_mask]
        return idx, lon_vals.min(), lon_vals.max()

    def _compute_sample_uniform_indexers(
        self, portion: float | None = None, number: int | None = None
    ) -> dict[str, int]:
        indices = np.random.choice(
            self.point.size,
            size=self._get_sampling_points_nbr(portion=portion, number=number),
        )
        return {self.point.name: indices}

    def _compute_sample_normal_indexers(
        self,
        portion: float | None = None,
        number: int | None = None,
        center_point: tuple[Longitude, Latitude] = None,
        sigma: float = 10.0,
    ) -> dict[str, int]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        if center_point is None:
            center_point = np.array(
                [
                    np.mean(self.latitude.values),
                    np.mean(self.longitude.values),
                ]
            )
        else:
            center_point = np.array(center_point)

        distances = np.sqrt(
            (self.longitude.values - center_point[0]) ** 2
            + (self.latitude.values - center_point[1]) ** 2
        )
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        weights /= weights.sum()

        indices = np.random.choice(
            self.point.values, size=n, p=weights.flatten()
        )
        return {self.point.name: indices}

    def _compute_sample_no_nans_indexers(
        self,
        da: xr.DataArray,
        portion: float | None = None,
        number: int | None = None,
    ) -> dict[str, int]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        notnan_da = da[da.notnull()]
        selected_points_idx = np.random.choice(
            notnan_da[self.point.name].values.size, n
        )
        return {
            self.point.name: selected_points_idx,
        }

    def to_xarray(
        self, values: np.ndarray, name: str | None = None
    ) -> xr.DataArray:
        """
        Convert domain to sparse xarray.DataArray.

        The method applies `values` and (optionally) `name` to
        create a new xarray.DataArray object based on the domain.

        Parameters
        ----------
        values : np.ndarray
            The values to be assigned to the DataArray variable.
        name : str, optional
            The name of the DataArray variable.

        Returns
        -------
        xr.DataArray
            The xarray.DataArray single variable object.

        Raises
        ------
        ValueError
            If the shape of `values` does not match the expected shape.

        Examples
        --------
        >>> domain = Domain.from_lat_lon()
        >>> values = np.random.rand(5, 5)
        >>> da = domain.to_xarray(values, name="example")
        >>> isinstance(da, xr.DataArray)
        True
        >>> da.name
        'example'
        """
        target_shape = tuple([self.get_size(axis) for axis in self.dims])
        values = values.reshape(target_shape)
        coords = {
            self.latitude.name: (
                self.point.name,
                self.latitude.values,
            ),
            self.longitude.name: (
                self.point.name,
                self.longitude.values,
            ),
        }
        dim_names = tuple([self.get_axis(axis).name for axis in self.dims])
        for axis in self._axes.values():
            if axis.name in coords:
                continue
            if isinstance(axis, Time):
                coords[axis.name] = axis.values
            else:
                coords[axis.name] = (self.point.name, axis.values)

        dset = xr.DataArray(
            values,
            coords=coords,
            dims=dim_names,
            name=name,
        )
        return dset


class DenseDomain(Domain):
    """
    Dense domain class.

    Supports operations on dense spatial domain.
    """

    is_sparse: ClassVar[bool] = False

    def get_all_spatial_points(self) -> np.ndarray:
        """
        Get all spatial points in the domain.

        Returns
        -------
        np.ndarray
            An array of shape (n_points, 2) containing the latitude and
            longitude coordinates of all points in the domain.

        Examples
        --------
        >>> points = domain.get_all_spatial_points()
        >>> points
        array([[ 0. , -0.1],
               [ 0. ,  0. ],
               [ 0. ,  0.1],
               ...
        """
        lat_grid, lon_grid = np.meshgrid(
            self.latitude.values, self.longitude.values, indexing="ij"
        )
        lat_grid = lat_grid.flatten()
        lon_grid = lon_grid.flatten()
        return np.stack((lat_grid, lon_grid), axis=1)

    def _compute_subset_indexers(
        self,
        north: float | None = None,
        south: float | None = None,
        west: float | None = None,
        east: float | None = None,
    ) -> tuple[dict[str, Any], float, float]:
        if not (north or south or west or east):
            warnings.warn(
                "Subset parameters not provided. Returning the source dataset"
            )
            return type(self)(self.da)
        if north and south and north < south:
            raise ValueError("North must be greater than south")
        if west and east and west > east:
            raise ValueError("East must be greater than west")

        lats = self.latitude.values
        lons = self.longitude.values
        idx = {
            self.latitude.name: (
                np.s_[south:north]
                if np.all(np.diff(lats) >= 0)
                else np.s_[north:south]
            ),
            self.longitude.name: (
                np.s_[west:east]
                if np.all(np.diff(lons) >= 0)
                else np.s_[east:west]
            ),
        }
        if self.longitude.name in idx:
            start = idx[self.longitude.name].start
            stop = idx[self.longitude.name].stop
        else:
            start = idx[self.latitude.name].start
            stop = idx[self.latitude.name].stop
        return idx, start, stop

    def _compute_sample_uniform_indexers(
        self, portion: float | None = None, number: int | None = None
    ) -> dict[str, int]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        selected_lats_idx = np.random.choice(self.latitude.values.size, n)
        selected_lons_idx = np.random.choice(self.longitude.values.size, n)
        return {
            self.latitude.name: xr.DataArray(
                selected_lats_idx, dims=[AxisType.POINT]
            ),
            self.longitude.name: xr.DataArray(
                selected_lons_idx, dims=[AxisType.POINT]
            ),
        }

    def _compute_sample_normal_indexers(
        self,
        portion: float | None = None,
        number: int | None = None,
        center_point: tuple[Longitude, Latitude] = None,
        sigma: float = 10.0,
    ) -> dict[str, int]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        if center_point is None:
            center_point = np.array(
                [
                    np.mean(self.latitude.values),
                    np.mean(self.longitude.values),
                ]
            )
        else:
            center_point = np.array(center_point)

        x_grid, y_grid = np.meshgrid(
            self.longitude.values, self.latitude.values
        )
        x_grid_idx, y_grid_idx = np.meshgrid(
            np.arange(self.longitude.values.size),
            np.arange(self.latitude.values.size),
        )
        distances = np.sqrt(
            (x_grid - center_point[0]) ** 2 + (y_grid - center_point[1]) ** 2
        )
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        weights /= weights.sum()

        flat_x = x_grid.flatten()

        indices = np.random.choice(len(flat_x), size=n, p=weights.flatten())
        indices_lats = y_grid_idx.flatten()[indices]
        indices_lons = x_grid_idx.flatten()[indices]

        return {
            self.latitude.name: xr.DataArray(
                indices_lats, dims=[AxisType.POINT]
            ),
            self.longitude.name: xr.DataArray(
                indices_lons, dims=[AxisType.POINT]
            ),
        }

    def _compute_sample_no_nans_indexers(
        self,
        da: xr.DataArray,
        portion: float | None = None,
        number: int | None = None,
    ) -> dict[str, int]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        stacked = da.stack(**{AxisType.POINT: da.dims})
        idx = np.arange(len(stacked))[stacked.notnull()]
        selected_idx = np.random.choice(idx, n)
        selected_lat_idx = selected_idx // self.longitude.values.size
        selected_lon_idx = selected_idx % self.longitude.values.size
        return {
            self.latitude.name: xr.DataArray(
                selected_lat_idx, dims=[AxisType.POINT]
            ),
            self.longitude.name: xr.DataArray(
                selected_lon_idx, dims=[AxisType.POINT]
            ),
        }

    def to_xarray(
        self, values: np.ndarray, name: str | None = None
    ) -> xr.DataArray:
        """
        Convert domain to dense xarray.DataArray.

        The method applies `values` and (optionally) `name` to
        create a new xarray.DataArray object based on the domain.

        Parameters
        ----------
        values : np.ndarray
            The values to be assigned to the DataArray variable.
        name : str, optional
            The name of the DataArray variable.

        Returns
        -------
        xr.DataArray
            The xarray.DataArray single variable object.

        Raises
        ------
        ValueError
            If the shape of `values` does not match the expected shape.

        Examples
        --------
        >>> domain = Domain.from_lat_lon()
        >>> values = np.random.rand(5, 5)
        >>> da = domain.to_xarray(values, name="example")
        >>> isinstance(da, xr.DataArray)
        True
        >>> da.name
        'example'
        """
        target_shape = tuple([self.get_size(axis) for axis in self.dims])
        values = values.reshape(target_shape)
        coords = {
            self.latitude.name: self.latitude.values,
            self.longitude.name: self.longitude.values,
        }
        dim_names = tuple([self.get_axis(axis).name for axis in self.dims])
        for axis in self._axes.values():
            if axis.name in coords:
                continue
            coords[axis.name] = axis.values

        return xr.DataArray(
            values,
            coords=coords,
            dims=dim_names,
            name=name,
        )
