from __future__ import annotations

import re
import warnings
from abc import abstractmethod
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self

import numpy as np
import xarray as xr

from climatrix.dataset.axis import Axis
from climatrix.exceptions import TooLargeSamplePortionWarning
from climatrix.types import Latitude, Longitude

_DEFAULT_LAT_RESOLUTION = 0.1
_DEFAULT_LON_RESOLUTION = 0.1

# Based on MetPy
# (https://github.com/Unidata/MetPy/blob/main/src/metpy/xarray.py)
_coords_name_regex: dict[Axis, str] = {
    Axis.TIME: re.compile(r"^(x?)(valid_?)time(s?)([0-9]*)$"),
    Axis.VERTICAL: re.compile(
        r"^(z|lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|"
        r"isobaric|pres|isotherm)"
        r"[a-z_]*[0-9]*$"
    ),
    Axis.LATITUDE: re.compile(r"^(x?)lat[a-z0-9_]*$"),
    Axis.LONGITUDE: re.compile(r"^(x?)lon[a-z0-9_]*$"),
    Axis.POINT: re.compile(r"^(point.*|values|nstation.*)$"),
}


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


def match_axis_names(da: xr.DataArray) -> dict[Axis, str]:
    axis_names = {}
    coords_and_dims = {*da.dims, *da.coords.keys()}
    for coord in coords_and_dims:
        for axis, regex in _coords_name_regex.items():
            if regex.match(coord):
                axis_names[axis] = coord
                break
    return axis_names


def validate_spatial_axes(axis_mapping: dict[Axis, str]):
    for axis in [Axis.LATITUDE, Axis.LONGITUDE]:
        if axis not in axis_mapping:
            raise ValueError(f"Dataset has no {axis.name} axis")


def check_is_dense(da: xr.DataArray, axis_mapping: dict[Axis, str]) -> bool:
    return (axis_mapping[Axis.LATITUDE] in da.dims) and (
        axis_mapping[Axis.LONGITUDE] in da.dims
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

    __slots__ = ("coords", "_axis_mapping")
    is_sparse: ClassVar[bool]
    coords: dict[str, np.ndarray]
    _axis_mapping: dict[Axis, str]

    def __new__(cls, xarray_obj: xr.Dataset | xr.DataArray):
        validate_input(xarray_obj)
        da = ensure_single_var(xarray_obj)
        axis_mapping = match_axis_names(da)
        validate_spatial_axes(axis_mapping)
        coords = {
            axis: da[axis_name].values
            for axis, axis_name in axis_mapping.items()
        }
        coords = ensure_all_numpy_arrays(coords)
        coords = filter_out_single_value_coord(coords)

        if cls is not Domain:
            return super().__new__(cls)
        if check_is_dense(da, axis_mapping):
            domain = DenseDomain(da)
        else:
            domain = SparseDomain(da)
        domain.coords = coords
        domain._axis_mapping = axis_mapping
        return domain

    @classmethod
    def from_lat_lon(
        cls,
        lat: slice | np.ndarray = slice(-90, 90, _DEFAULT_LAT_RESOLUTION),
        lon: slice | np.ndarray = slice(-180, 180, _DEFAULT_LON_RESOLUTION),
        kind: Literal["sparse", "dense"] = "sparse",
    ) -> Self:
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
    def latitude_name(self) -> str:
        return self._axis_mapping[Axis.LATITUDE]

    @property
    def longitude_name(self) -> str:
        return self._axis_mapping[Axis.LONGITUDE]

    @property
    def point_name(self) -> str | None:
        return self._axis_mapping.get(Axis.POINT)

    @property
    def time_name(self) -> str | None:
        return self._axis_mapping.get(Axis.TIME)

    @property
    def latitude(self) -> np.ndarray:
        if Axis.LATITUDE not in self.coords:
            raise ValueError(
                f"Latitude not found in coordinates {self.coords.keys()}"
            )
        return self.coords[Axis.LATITUDE]

    @property
    def longitude(self) -> np.ndarray:
        if Axis.LONGITUDE not in self.coords:
            raise ValueError(
                f"Longitude not found in coordinates {self.coords.keys()}"
            )
        return self.coords[Axis.LONGITUDE]

    @property
    def time(self) -> np.ndarray:
        if Axis.TIME not in self.coords:
            raise ValueError(
                f"Time not found in coordinates {self.coords.keys()}"
            )
        return self.coords[Axis.TIME]

    @property
    def point(self) -> np.ndarray:
        if Axis.POINT not in self.coords:
            raise ValueError(
                f"Point not found in coordinates {self.coords.keys()}"
            )
        return self.coords[Axis.POINT]

    @property
    def size(self) -> int:
        if Axis.POINT in self.coords:
            lat_lon = self.coords[Axis.POINT].size
        lat_lon = self.latitude.size * self.longitude.size
        if not self.time_name:
            return lat_lon
        return lat_lon * self.time.size

    @property
    def is_dynamic(self) -> bool:
        return self.time_name and self.time.size > 1

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, Domain):
            return False
        same_keys = set(self.coords.keys()) == set(value.coords.keys())
        if not same_keys:
            return False
        for k in self.coords.keys():
            if not np.allclose(
                self.coords[k], value.coords[k], equal_nan=True
            ):
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
    ) -> Self:
        raise NotImplementedError

    @abstractmethod
    def _compute_sample_normal_indexers(
        self,
        portion: float | None = None,
        number: int | None = None,
        nan: SamplingNaNPolicy | str = "ignore",
        center_point: tuple[Longitude, Latitude] = None,
        sigma: float = 10.0,
    ) -> Self:
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
        return number


class SparseDomain(Domain):
    is_sparse: ClassVar[bool] = True

    def get_all_spatial_points(self) -> np.ndarray:
        return np.stack((self.latitude, self.longitude), axis=1)

    def _compute_subset_indexers(
        self,
        north: float | None = None,
        south: float | None = None,
        west: float | None = None,
        east: float | None = None,
    ) -> Self:
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
            self.latitude >= south, self.latitude <= north
        )
        lon_mask = np.logical_and(
            self.longitude >= west, self.longitude <= east
        )
        point_mask = np.logical_and(lat_mask, lon_mask)
        point_idx = np.where(point_mask)[0]
        idx = {self.point_name: point_idx}

        lon_vals = self.longitude[lon_mask]
        return idx, lon_vals.min(), lon_vals.max()

    def _compute_sample_uniform_indexers(
        self, portion: float | None = None, number: int | None = None
    ) -> dict[str, Any]:
        indices = np.random.choice(
            self.point.size,
            size=self._get_sampling_points_nbr(portion=portion, number=number),
        )
        return {self.point_name: indices}

    def _compute_sample_normal_indexers(
        self,
        portion: float | None = None,
        number: int | None = None,
        center_point: tuple[Longitude, Latitude] = None,
        sigma: float = 10.0,
    ) -> dict[str, Any]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        if center_point is None:
            center_point = np.array(
                [
                    np.mean(self.latitude),
                    np.mean(self.longitude),
                ]
            )
        else:
            center_point = np.array(center_point)

        distances = np.sqrt(
            (self.longitude - center_point[0]) ** 2
            + (self.latitude - center_point[1]) ** 2
        )
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        weights /= weights.sum()

        indices = np.random.choice(self.point, size=n, p=weights.flatten())
        return {self.point_name: indices}

    def _compute_sample_no_nans_indexers(
        self,
        da: xr.DataArray,
        portion: float | None = None,
        number: int | None = None,
    ) -> dict[str, Any]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        notnan_da = da[da.notnull()]
        selected_points = np.random.choice(
            notnan_da[self.point_name].values, n
        )
        return {
            self.point_name: selected_points,
        }


class DenseDomain(Domain):
    is_sparse: ClassVar[bool] = False

    def get_all_spatial_points(self) -> np.ndarray:
        lat_grid, lon_grid = np.meshgrid(
            self.latitude, self.longitude, indexing="ij"
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

        lats = self.latitude
        lons = self.longitude
        idx = {
            self.latitude_name: (
                np.s_[south:north]
                if np.all(np.diff(lats) >= 0)
                else np.s_[north:south]
            ),
            self.longitude_name: (
                np.s_[west:east]
                if np.all(np.diff(lons) >= 0)
                else np.s_[east:west]
            ),
        }
        start = idx[self.longitude_name].start
        stop = idx[self.longitude_name].stop
        return idx, start, stop

    def _compute_sample_uniform_indexers(
        self, portion: float | None = None, number: int | None = None
    ) -> dict[str, Any]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        selected_lats = np.random.choice(self.latitude, n)
        selected_lons = np.random.choice(self.longitude, n)
        return {
            self.latitude_name: xr.DataArray(selected_lats, dims=[Axis.POINT]),
            self.longitude_name: xr.DataArray(
                selected_lons, dims=[Axis.POINT]
            ),
        }

    def _compute_sample_normal_indexers(
        self,
        portion: float | None = None,
        number: int | None = None,
        center_point: tuple[Longitude, Latitude] = None,
        sigma: float = 10.0,
    ) -> dict[str, Any]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        if center_point is None:
            center_point = np.array(
                [
                    np.mean(self.latitude),
                    np.mean(self.longitude),
                ]
            )
        else:
            center_point = np.array(center_point)

        x_grid, y_grid = np.meshgrid(self.longitude, self.latitude)
        distances = np.sqrt(
            (x_grid - center_point[0]) ** 2 + (y_grid - center_point[1]) ** 2
        )
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        weights /= weights.sum()

        flat_x = x_grid.flatten()
        flat_y = y_grid.flatten()

        indices = np.random.choice(len(flat_x), size=n, p=weights.flatten())
        selected_lats = flat_y[indices]
        selected_lons = flat_x[indices]
        return {
            self.latitude_name: xr.DataArray(selected_lats, dims=[Axis.POINT]),
            self.longitude_name: xr.DataArray(
                selected_lons, dims=[Axis.POINT]
            ),
        }

    def _compute_sample_no_nans_indexers(
        self,
        da: xr.DataArray,
        portion: float | None = None,
        number: int | None = None,
    ) -> dict[str, Any]:
        n = self._get_sampling_points_nbr(portion=portion, number=number)
        stacked = da.stack(**{Axis.POINT: da.dims})
        notnan_da = stacked[stacked.notnull()]
        selected_lats = np.random.choice(
            notnan_da[self.latitude_name].values, n
        )
        selected_lons = np.random.choice(
            notnan_da[self.longitude_name].values, n
        )
        return {
            self.latitude_name: xr.DataArray(selected_lats, dims=[Axis.POINT]),
            self.longitude_name: xr.DataArray(
                selected_lons, dims=[Axis.POINT]
            ),
        }
