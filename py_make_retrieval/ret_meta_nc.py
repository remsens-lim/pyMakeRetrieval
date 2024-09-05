"""Module for retrieval Metadata"""
from collections import namedtuple
from collections.abc import Callable
from typing import TypeAlias


def get_data_attributes(rpg_variables: dict, data_type: str) -> dict:
    """Adds Metadata for RPG MWR Level 1 variables for NetCDF file writing.
    Args:
        rpg_variables: RpgArray instances.
        data_type: Data type of the netCDF file.

    Returns:
        Dictionary

    Raises:
        RuntimeError: Specified data type is not supported.

    Example:
        from level1.lev1_meta_nc import get_data_attributes
        att = get_data_attributes('data','data_type')
    """

    if data_type in ("tbx", "iwv", "lwp", "tpb", "tpt", "hpt"):
        read_att = att_reader[data_type]
        attributes = dict(ATTRIBUTES_COM, **read_att)

    else:
        raise RuntimeError(["Data type " + data_type + " not supported for file writing."])

    for key in list(rpg_variables):
        if key in attributes:
            rpg_variables[key].set_attributes(attributes[key])
        else:
            del rpg_variables[key]

    index_map = {v: i for i, v in enumerate(attributes)}
    rpg_variables = dict(sorted(rpg_variables.items(), key=lambda pair: index_map[pair[0]]))

    return rpg_variables


FIELDS = ("long_name", "units")

MetaData = namedtuple("MetaData", FIELDS)
MetaData.__new__.__defaults__ = (None,) * len(MetaData._fields)


ATTRIBUTES_COM = {
    "freq": MetaData(
        long_name="frequency",
        units="GHz",
    ),
    "lat": MetaData(
        long_name="latitude",
        units="degrees_north",
    ),
    "lon": MetaData(
        long_name="longitude",
        units="degrees_east",
    ),
    "asl": MetaData(
        long_name="altitude above mean sea level",
        units="m",
    ),
    "elevation_predictand": MetaData(
        long_name="elevation angle of predictand",
        units="degree",
    ),
    "elevation_predictor": MetaData(
        long_name="elevation angle of predictor",
        units="degree",
    ),
    "predictor_err": MetaData(
        long_name="random uncertainty of predictor",
        units="K",
    ),
    "surface_err": MetaData(
        long_name="random uncertainty surface T, p and q measurements",
        units='K, kg m-3, Pa',
    ),
}

ATTRIBUTES_tbx = {
    "prdmx": MetaData(
        long_name="predictand maximum",
        units='K',
    ),
    "prdmn": MetaData(
        long_name="predictand minimum",
        units='K',
    ),
    "prrmx": MetaData(
        long_name="predictor maximum",
        units='K',
    ),
    "prrmn": MetaData(
        long_name="predictor minimum",
        units='K',
    ),
    "predictand_err": MetaData(
        long_name="standard error of predictand",
        units="K",
    ),
    "predictand_err_sys": MetaData(
        long_name="bias error of predictand",
        units="K",
    ),
    "coefficient_mvr": MetaData(
        long_name="multi variate regression coefficients",
        units="K/K",
    ),
    "offset_mvr": MetaData(
        long_name="multi variate regression offset",
        units="K",
    ),
}

ATTRIBUTES_iwv_lwp = {
    "prdmx": MetaData(
        long_name="predictand maximum",
        units='kg m-2',
    ),
    "prdmn": MetaData(
        long_name="predictand minimum",
        units='kg m-2',
    ),
    "prrmx": MetaData(
        long_name="predictor maximum",
        units='K',
    ),
    "prrmn": MetaData(
        long_name="predictor minimum",
        units='K',
    ),
    "predictand_err": MetaData(
        long_name="standard error of predictand",
        units="kgm-2",
    ),
    "predictand_err_sys": MetaData(
        long_name="bias error of predictand",
        units="kgm-2",
    ),
    "coefficient_mvr": MetaData(
        long_name="multi variate regression coefficients",
        units="kgm-2/K",
    ),
    "offset_mvr": MetaData(
        long_name="multi variate regression offset",
        units="kgm-2",
    ),
}

ATTRIBUTES_tpb = {
    "freq_bl": MetaData(
        long_name="frequency",
        units="GHz",
    ),
    "height_grid": MetaData(
        long_name="retrieval height grid",
        units='m',
    ),
    "prdmx": MetaData(
        long_name="predictand maximum",
        units='K',
    ),
    "prdmn": MetaData(
        long_name="predictand minimum",
        units='K',
    ),
    "prrmx": MetaData(
        long_name="predictor maximum",
        units='K',
    ),
    "prrmn": MetaData(
        long_name="predictor minimum",
        units='K',
    ),
    "predictand_err": MetaData(
        long_name="standard error of predictand",
        units="K",
    ),
    "predictand_err_sys": MetaData(
        long_name="bias error of predictand",
        units="K",
    ),
    "predictand_rmse": MetaData(
        long_name="root mean square error of predictand",
        units="K",
    ),
    "predictand_r_2": MetaData(
        long_name="coefficient of determination",
        units="1",
    ),
    "coefficient_mvr": MetaData(
        long_name="multi variate regression coefficients",
        units="K/K",
    ),
    "offset_mvr": MetaData(
        long_name="multi variate regression offset",
        units="K",
    ),
}

ATTRIBUTES_tpt = {
    "height_grid": MetaData(
        long_name="retrieval height grid",
        units='m',
    ),
    "prdmx": MetaData(
        long_name="predictand maximum",
        units='K',
    ),
    "prdmn": MetaData(
        long_name="predictand minimum",
        units='K',
    ),
    "prrmx": MetaData(
        long_name="predictor maximum",
        units='K',
    ),
    "prrmn": MetaData(
        long_name="predictor minimum",
        units='K',
    ),
    "predictand_err": MetaData(
        long_name="standard error of predictand",
        units="K",
    ),
    "predictand_err_sys": MetaData(
        long_name="bias error of predictand",
        units="K",
    ),
    "coefficient_mvr": MetaData(
        long_name="multi variate regression coefficients",
        units="K/K",
    ),
    "offset_mvr": MetaData(
        long_name="multi variate regression offset",
        units="K",
    ),
}

ATTRIBUTES_hpt = {
    "height_grid": MetaData(
        long_name="retrieval height grid",
        units='m',
    ),
    "prdmx": MetaData(
        long_name="predictand maximum",
        units='kg m-3',
    ),
    "prdmn": MetaData(
        long_name="predictand minimum",
        units='kg m-3',
    ),
    "prrmx": MetaData(
        long_name="predictor maximum",
        units='K',
    ),
    "prrmn": MetaData(
        long_name="predictor minimum",
        units='K',
    ),
    "predictand_err": MetaData(
        long_name="standard error of predictand",
        units="kg m-3",
    ),
    "predictand_err_sys": MetaData(
        long_name="bias error of predictand",
        units="kg m-3",
    ),
    "coefficient_mvr": MetaData(
        long_name="multi variate regression coefficients",
        units="K/K",
    ),
    "offset_mvr": MetaData(
        long_name="multi variate regression offset",
        units="K",
    ),
}

FuncType: TypeAlias = Callable[[str], dict]
att_reader: dict[str, dict] = {
    "tbx": ATTRIBUTES_tbx,
    "tpb": ATTRIBUTES_tpb,
    "iwv": ATTRIBUTES_iwv_lwp,
    "lwp": ATTRIBUTES_iwv_lwp,
    "tpt": ATTRIBUTES_tpt,
    "hpt": ATTRIBUTES_hpt,
}
