"""Module for making retrieval."""
import glob
import numpy as np
import xarray as xr

from py_make_retrieval.utils import read_yaml_config
from py_make_retrieval.make_ret_core import MakeRetrieval


def main(args):

    if args.ret == 'all':
        ret_types = ['lwp', 'iwv', 'tpt', 'hpt', 'tpb', 'tbx']
    elif args.ret in ['lwp', 'iwv', 'tpt', 'hpt', 'tpb', 'tbx']:
        ret_types = [args.ret]
    else:
        raise ValueError("no valid retrieval type. Please choose lwp, "
                         "iwv, tpt, hpt, tpb, tbx or all")

    # read general config (paths,...)
    general_config = read_yaml_config("py_make_retrieval/configs/general_config.yaml")['params']

    # list files containing the radiative transfer output
    rt_file_list1 = []
    for i_list in np.arange(len(general_config['rt_paths'])):
        for j_list in np.arange(len(general_config['rt_patterns'])):
            rt_file_list0 = glob.glob(
                general_config['rt_paths'][i_list] + general_config['rt_patterns'][j_list]
            )
            rt_file_list0.sort()
            rt_file_list1.append(rt_file_list0)

    rt_file_list = np.concatenate(rt_file_list1)
    print(rt_file_list)

    # concat radiative transfer files
    rt_files = []
    for file in rt_file_list:
        rt_files.append(xr.open_dataset(file).drop_dims('n_cloud_max'))

    rt_data = xr.concat(rt_files, dim='n_date', data_vars='different')
    rt_data = rt_data.dropna(dim='n_date')
    # set frequency and date as coordinates
    rt_data = rt_data.set_coords(['frequency', 'date', 'elevation_angle'])
    rt_data = rt_data.swap_dims({'n_angle': 'elevation_angle'})

    for ret_type in ret_types:
        print('making ' + ret_type + ' retrievals')
        # import general parameters and retrieval configuration
        # read params
        config = read_yaml_config("py_make_retrieval/configs/config_" +
                                  ret_type + ".yaml")['params']
        config.update(general_config)

        # run the makeRetrieval class
        if ret_type == 'tpb':
            # read ret_specs
            ret_specs = read_yaml_config("py_make_retrieval/configs/ret_specs.yaml")

            results = []
            for ret_spec in ret_specs:
                rt_dat = rt_data.interp(
                    elevation_angle=np.sort(
                        np.append(
                            rt_data.elevation_angle.values,
                            ret_specs[ret_spec]['multi_angles']
                        )
                    ), method='cubic'
                )

                ret_specs[ret_spec].update(config)

                results.append(MakeRetrieval(ret_specs[ret_spec],
                                             rt_dat,
                                             ret_type,
                                             )
                               )
        else:
            if not set(config['angle']) <= set(rt_data.elevation_angle.values):
                rt_dat = rt_data.interp(
                    elevation_angle=np.sort(
                        np.append(
                            rt_data.elevation_angle.values,
                            config['angle']
                        )
                    ), method='cubic'
                )
            else:
                rt_dat = rt_data


            MakeRetrieval(config,
                          rt_dat,
                          ret_type,
                          )
