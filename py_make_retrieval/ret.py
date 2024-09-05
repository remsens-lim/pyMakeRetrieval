import locale
from datetime import datetime, timezone

import numpy as np
# from numpy import ma
import netCDF4


from py_make_retrieval import utils
from py_make_retrieval import version

# from ret_meta_nc import get_data_attributes
from py_make_retrieval.ret_meta_nc import MetaData


class RetArray:

    def __init__(
        self,
        variable: np.ndarray | float | int,
        name: str,
        units_from_user: str | None = None,
        dimensions: str | None = None
    ):
        self.variable = variable
        self.name = name
        self.data = self._init_data()
        self.units = self._init_units(units_from_user)
        self.data_type = self._init_data_type()
        self.dimensions = dimensions

    def _init_data(self) -> np.ndarray:

        if isinstance(self.variable, np.ndarray):
            return self.variable
        if isinstance(
                self.variable, (int, float, np.float32, np.int8,
                                np.float64, np.int32, np.uint16, np.int16)):
            return np.array(self.variable)
        raise ValueError(f"Incorrect input array: {self.variable}")

    def set_attributes(self, attributes: MetaData) -> None:
        """Overwrites existing instance attributes."""
        for key in attributes._fields:  # To iterate namedtuple fields.
            data = getattr(attributes, key)
            if data:
                setattr(self, key, data)

    def _init_units(self, units_from_user: str | None) -> str:
        if units_from_user is not None:
            return units_from_user
        return getattr(self.variable, "units", "")

    def fetch_attributes(self) -> list:
        """Returns list of user-defined attributes."""

        attributes = []
        for attr in self.__dict__:
            if attr not in ("name", "data", "data_type", "variable", "dimensions"):
                attributes.append(attr)
        return attributes

    def _init_data_type(self) -> str:
        if self.data.dtype in (np.float32, np.float64):
            return "f4"
        return "i4"

    def __getitem__(self, ind: tuple) -> np.ndarray:
        return self.data[ind]


class Ret:

    def __init__(
        self,
        raw_data: dict
    ):
        self.raw_data = raw_data
        self.data = {}
        self.data = self._init_data()

    def _init_data(self) -> dict:
        data = {}
        for key in self.raw_data:
            data[key] = RetArray(self.raw_data[key], key)
        return data


def save_ret(ret: Ret, output_file: str, att: dict, data_type: str) -> None:
    """Saves the Ret file."""
    if data_type == 'tpb':
        dims = {
            "n_freq": len(ret.data["freq"][:]),
            "n_freq_blb": len(ret.data["freq_bl"][:]),
            "n_height_grid": len(ret.data["height_grid"][:]),
            "n_angles": len(ret.data["elevation_predictor"][:]),
            "n_prr_err": len(ret.data["height_grid"][:]),
            "n_prrs_err": 1,
            "n_coeff": len(ret.data["coefficient_mvr"][:, 0]),
        }
    elif data_type in ('tpt', 'hpt'):
        dims = {
            "n_freq_ret": len(ret.data["freq"][:]),
            "n_height_grid": len(ret.data["height_grid"][:]),
            "n_prr_err": len(ret.data["height_grid"][:]),
            "n_prrs_err": 3,
            "n_coeff": len(ret.data["coefficient_mvr"][:, 0]),
        }
    elif data_type in ("tbx", "iwv", "lwp"):
        dims = {
            "n_freq_ret": len(ret.data["freq"][:]),
            "n_prr_err": len(ret.data["freq"][:].T)+1,
            "n_prrs_err": 3,
            "n_coeff": len(ret.data["coefficient_mvr"][:]),
        }

    else:
        raise RuntimeError(["Data type " + data_type + " not supported for file writing."])
    with init_file(output_file, dims, ret.data, att) as rootgrp:
        # setattr(rootgrp, "date", ret.date)
        setattr(rootgrp, 'site', att['site'])


def init_file(
    file_name: str, dimensions: dict, ret_arrays: dict, att_global: dict
) -> netCDF4.Dataset:
    """Initializes a retrieval file for writing.
    Args:
        file_name: File name to be generated.
        dimensions: Dictionary containing dimension for this file.
        ret_arrays: Dictionary containing :class:`RpgArray` instances.
        att_global: Dictionary containing site specific global attributes
    """

    nc = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    for key, dimension in dimensions.items():
        nc.createDimension(key, dimension)

    _write_vars2nc(nc, ret_arrays)
    _add_standard_global_attributes(nc, att_global)
    return nc


def _get_dimensions(nc: netCDF4.Dataset, data: np.ndarray) -> tuple:
    """Finds correct dimensions for a variable."""
    if utils.isscalar(data):
        return ()
    variable_size = ()
    file_dims = nc.dimensions
    array_dims = data.shape
    for length in array_dims:
        dim = [key for key in file_dims.keys() if file_dims[key].size == length][0]
        variable_size = variable_size + (dim,)
    return variable_size


def _write_vars2nc(nc: netCDF4.Dataset, mwr_variables: dict) -> None:
    """Iterates over retrival instances and write to netCDF file."""

    for obj in mwr_variables.values():
        if obj.data_type == "f4":
            fill_value = -999.0
        else:
            fill_value = -99
        size = obj.dimensions or _get_dimensions(nc, obj.data)
        nc_variable = nc.createVariable(
            obj.name, obj.data_type, size, zlib=True, fill_value=fill_value
        )
        nc_variable[:] = obj.data
        for attr in obj.fetch_attributes():
            setattr(nc_variable, attr, getattr(obj, attr))


def _add_standard_global_attributes(nc: netCDF4.Dataset, att_global) -> None:
    nc.make_tpb_ret_version = version.__version__
    locale.setlocale(locale.LC_TIME, "de_DE.UTF-8")
    nc.processing_date = datetime.now(tz=timezone.utc).strftime("%d %b %Y %H:%M:%S") + " UTC"
    for name, value in att_global.items():
        if value is None:
            value = ""
        setattr(nc, name, value)
