"""
Module that contains the core functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split
from sklearn import linear_model

from utils import mu_calc

from py_make_retrieval.ret_meta_nc import get_data_attributes
from py_make_retrieval import ret
from py_make_retrieval.plotting import plot_performance_2d


class MakeRetrieval:
    r"""The core of making the retrieval from radiative transfer data', which
    contains all required parameters.

    Parameters
    ----------
    _specs : dict
        retrieval specifications and parameters from config_tpb.yaml and ret_specs.yaml
    _rt_data: xarray dataset
        output dataset from radiative transfer calculations with stp_run (idl)
    _ret_type : str
        three-letter code for the retrieved variable (lwp, iwv, tbx, tpb...)

    Attributes
    ----------
    x_test : np.ndarray
        test data brightness temperatures
    x_train : np.ndarray
        training data brightness temperatures
    bias : list
        bias between predicted and true value
    coeff : list
        coefficients
    const : list
        observed measurement vector y.
    ret_type : str
        here fixed as 'tpb', later development will include iwv, lwp, hpt, tpt and tbx
    frequency : np.ndarray
        frequencies used
    height_grid : xarray data array
        output height grid
    mwr_pro_ret : RetArray object
        ...
    n_test : list
        number of valid data points per height
    output_file : str
        name of the output file
    pred_test : list
        prediction of test data
    r_2 : list
        coefiicient of determination between prediction and true value of test data
    reg : SciKit Learn linear regression object
        ...
    rt_dat : xarray Dataset
        filtered data from radiative transfer calculations
    rt_data : xarray Dataset
        data from radiative transfer calculations
    specs : dict
        retrieval specification from yaml files
    std : list
        standard deviation between prediction and true value of test data
    y_test : np.ndarray
        test true value
    y_train : np.ndarray
        training true value

    Returns
    -------

    make_retrieval object
      returns the make_retrieval object


    References
    ----------
    .

    """

    def __init__(self,
                 _specs,
                 _rt_data,
                 _ret_type
                 ):

        self.specs = _specs
        self.rt_data = _rt_data
        # self.name = _specs['name']
        self.ret_type = _ret_type

        # freq_index = self.get_freq_index()

        # make output directory
        if not os.path.exists(f"output/{self.specs['site']}"):
            os.makedirs(f"output/{self.specs['site']}")

        if self.ret_type in ('tbx', 'iwv', 'lwp', 'tpt', 'hpt'):

            self.freq_index = self.get_freq_index(self.specs['freq'], self.rt_data)
            self.angle_index = self.get_angle_index()

            # update radiative transfer data
            self.rt_dat = self.rt_data.isel(n_frequency=self.freq_index,
                                            elevation_angle=self.angle_index,
                                            n_cloud_model=0,
                                            n_height=slice(0, -4),
                                            ).drop_dims(['n_wet_delay_models'])

            # filter out unreasonable data
            # print(self.rt_dat.sizes)
            self.rt_dat = self.rt_dat.where(
                (self.rt_dat.brightness_temperatures.min(dim='n_frequency') > self.specs['predictor_min']) &
                (self.rt_dat.brightness_temperatures.max(dim='n_frequency') < self.specs['predictor_max']) &
                (self.rt_dat.integrated_water_vapor > 0.01) &
                (self.rt_dat.integrated_water_vapor < 80) &
                # (self.rt_dat.atmosphere_temperature.min(dim='n_height') > self.specs['predictand_min']) &
                # (self.rt_dat.atmosphere_temperature.max(dim='n_height') < self.specs['predictand_max']) &
                (self.rt_dat.liquid_water_path < 1.5),
                drop=True
            )
            # print(self.rt_dat.sizes)

            self.height_grid = self.rt_dat.isel(n_date=0, elevation_angle=0,
                                                n_frequency=0).height_grid

            if self.ret_type == 'tbx':
                jj = 0
                # loop over all angles
                for i_ang in self.rt_dat.elevation_angle.values:
                    ang = self.angle_index
                    # ang = self.rt_data.isel(n_date=0, elevation_angle=i_ang
                    #                         ).elevation_angle.values
                    ii = 0
                    # loop over all freqs
                    for i_freq in self.specs['freq']:
                        freq_wo_x = [x for x in self.specs['freq'] if x != i_freq]
                        freq_wo_x_index = self.get_freq_index(freq_wo_x, self.rt_dat)
                        i_freq_index = \
                            np.argwhere(self.rt_dat.isel(n_date=0).frequency.values == i_freq)[0]

                        # define input and output
                        x = self.rt_dat.brightness_temperatures.isel(n_frequency=freq_wo_x_index,
                                                                     elevation_angle=jj
                                                                     ).squeeze().values

                        y = self.rt_dat.isel(n_frequency=i_freq_index,
                                             elevation_angle=jj
                                             ).brightness_temperatures.squeeze().values

                        # add measurement noise to the brightness temperatures
                        x_noise = self.add_noise(x)

                        # make quadratic
                        x_new = np.concatenate([x_noise, x_noise ** 2], axis=1)

                        # split data into traing and test data
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                            x_new,
                            y,
                            test_size=0.2,
                            random_state=42
                        )
                        # print(self.x_train.shape)
                        # print(self.y_train.shape)
                        # ret_name = 'tbx_freq_'+str(i_freq_index[0])+'_angle_'+str(i_ang)+'_ret'

                        # make linear regression
                        self.reg, self.coeff, self.const, \
                            self.r_2 = self.make_linear_regression()

                        # test model
                        self.n_test, self.y_test_fil, self.pred_test, \
                            self.bias, self.std, self.rmse = self.test_model()

                        # plot retrieval performance
                        self.plot_retrieval_performance_1d(i_ang, ii, self.y_test_fil,
                                                           self.pred_test, self.bias,
                                                           self.std, self.r_2)

                        # make file
                        data_mwr = self.make_mwr_pro_comp_file(self.coeff, self.const, self.bias,
                                                               self.std,
                                                               i_ang=jj,
                                                               freq_wo_x=freq_wo_x)
                        self.mwr_pro_ret = ret.Ret(data_mwr)

                        # get data attributes
                        self.mwr_pro_ret.data = get_data_attributes(self.mwr_pro_ret.data,
                                                                    self.ret_type)

                        # define output file
                        self.output_file = f"output/{self.specs['site']}/{self.ret_type}_" \
                                           f"{self.specs['site']}_" \
                                           f"no_surf_{str(int(i_ang*10))}_" \
                                           f"{str(ii).zfill(2)}.nc"

                        # make global attributes
                        global_attributes = self.make_global_attributes(ii)

                        # save nc file
                        ret.save_ret(self.mwr_pro_ret, self.output_file, global_attributes,
                                     self.ret_type)

                        ii = ii + 1

                    jj = jj + 1

            elif self.ret_type in ('iwv', 'lwp'):
                for i_ang in self.specs['angle']:
                    ii = 0
                    xx = np.argwhere(self.rt_data.isel(n_date=0
                                                       ).elevation_angle.values == i_ang)[0][0]

                    if len(self.specs['angle']) == 1:
                        ang = self.rt_dat.isel(n_date=0).elevation_angle.squeeze().values
                    else:
                        ang = self.rt_dat.isel(n_date=0).elevation_angle.squeeze().values[0]

                    mu = np.sin(np.deg2rad(ang))

                    if ang == 0:
                        mu = 1
                        airmf = 1 / mu
                    elif ang == 90:
                        # mu = 0
                        airmf = 1
                    else:
                        airmf = 1 / mu

                    self.specs['predictand_max'] = self.specs['predictand_max'] * airmf

                    yyy = []
                    for kk in range(self.rt_dat.dims['n_date']):

                        mu, _ = mu_calc(
                            self.height_grid.values,
                            self.rt_dat['atmosphere_temperature'].isel(
                                n_date=kk).squeeze().values,
                            self.rt_dat['atmosphere_pressure'].isel(
                                n_date=kk).squeeze().values,
                            self.rt_dat['atmosphere_humidity'].isel(
                                n_date=kk).squeeze().values,
                            self.specs['freq'],
                            ang,
                            air_corr='sphere',
                            z_site=None
                        )
                        if ang == 0:
                            mu = 1
                            airmf = 1 / mu
                        elif ang == 90:
                            # mu = 0
                            airmf = 1
                        else:
                            airmf = 1 / mu[0, 0]

                        if self.ret_type == 'iwv':
                            yyy.append(self.rt_dat.isel(
                                n_date=kk
                            ).integrated_water_vapor.squeeze().values*airmf)
                        elif self.ret_type == 'lwp':
                            yyy.append(self.rt_dat.isel(
                                n_date=kk
                            ).liquid_water_path.squeeze().values*airmf)
                        else:
                            raise ValueError("no valid ret_type")

                    # define input and output
                    x = self.rt_dat.brightness_temperatures.squeeze(dim='elevation_angle').values
                    y = np.array(yyy)
                    # print(x.shape)
                    # print(y.shape)
                    # add measurement noise to the brightness temperatures
                    x_noise = self.add_noise(x)

                    if self.specs['surf_mode'] == 'surface':
                        t_gr_noise = np.random.normal(0, 1., y.shape[0]
                                                      ) * self.specs['surface_error'][0]
                        t_gr = self.rt_dat.atmosphere_temperature_sfc.isel(
                            elevation_angle=0
                        ).values.reshape(-1, 1) + t_gr_noise.reshape(-1, 1)

                        p_gr_noise = np.random.normal(0, 1., y.shape[0]
                                                      ) * self.specs['surface_error'][1]
                        p_gr = self.rt_dat.atmosphere_pressure_sfc.isel(
                            elevation_angle=0
                        ).values.reshape(-1, 1) + p_gr_noise.reshape(-1, 1)

                        q_gr_noise = np.random.normal(0, 1., y.shape[0]
                                                      ) * self.specs['surface_error'][2]
                        q_gr = self.rt_dat.atmosphere_humidity_sfc.isel(
                            elevation_angle=0
                        ).values.reshape(-1, 1) + q_gr_noise.reshape(-1, 1)

                        x_new = np.concatenate([x_noise, t_gr, p_gr, q_gr], axis=1)
                    else:
                        x_new = x_noise

                    # make quadratic
                    x_new = np.concatenate([x_new, x_new ** 2], axis=1)

                    # split data into traing and test data
                    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                        x_new,
                        y,
                        test_size=0.2,
                        random_state=42)

                    # ret_name = 'tbx_freq_'+str(i_freq_index[0])+'_angle_'+str(i_ang)+'_ret'

                    # make linear regression
                    self.reg, self.coeff, self.const, \
                        self.r_2 = self.make_linear_regression()

                    # test model
                    self.n_test, self.y_test_fil, self.pred_test, \
                        self.bias, self.std, self.rmse = self.test_model()

                    # plot retrieval performance
                    self.plot_retrieval_performance_1d(ang, ii, self.y_test_fil, self.pred_test,
                                                       self.bias, self.std, self.r_2
                                                       )

                    # make file
                    data_mwr = self.make_mwr_pro_comp_file(self.coeff, self.const,
                                                           self.bias, self.std,
                                                           i_ang=xx,
                                                           freq_wo_x=self.specs['freq'],
                                                           )

                    self.mwr_pro_ret = ret.Ret(data_mwr)

                    # get data attributes
                    self.mwr_pro_ret.data = get_data_attributes(self.mwr_pro_ret.data,
                                                                self.ret_type
                                                                )

                    # define output file
                    self.output_file = f"output/{self.specs['site']}/{self.ret_type}_" \
                                       f"{self.specs['site']}_{self.specs['retrieval_version']}_" \
                                       f"{str(int(ang))}.nc"

                    # make global attributes
                    global_attributes = self.make_global_attributes(ii)

                    # save nc file
                    ret.save_ret(self.mwr_pro_ret,
                                 self.output_file,
                                 global_attributes,
                                 self.ret_type
                                 )

            elif self.ret_type in ('tpt', 'hpt'):
                for i_ang in self.specs['angle']:
                    ii = 0
                    # xx = np.argwhere(self.rt_data.isel(n_date=0
                    #                                    ).elevation_angle.values == i_ang)[0][0]
                    # ang = self.rt_data.isel(n_date=0, elevation_angle=i_ang
                    #                         ).elevation_angle.values
                    ang = self.angle_index

                    # define input and output
                    x = self.rt_dat.sel(elevation_angle=i_ang).brightness_temperatures.values
                    if self.ret_type == 'tpt':
                        y = self.rt_dat.sel(n_frequency=0, elevation_angle=i_ang
                                            ).atmosphere_temperature.values
                    elif self.ret_type == 'hpt':
                        y = self.rt_dat.sel(n_frequency=0, elevation_angle=i_ang
                                            ).atmosphere_humidity.values
                    else:
                        raise ValueError("no valid ret_type")

                    # add measurement noise to the brightness temperatures
                    x_noise = self.add_noise(x)

                    if self.specs['surf_mode'] == 'surface':
                        t_gr_noise = np.random.normal(0, 1., y.shape[0]
                                                      ) * self.specs['surface_error'][0]
                        t_gr = self.rt_dat.atmosphere_temperature_sfc.isel(
                            elevation_angle=0
                        ).values.reshape(-1, 1) + t_gr_noise.reshape(-1, 1)

                        p_gr_noise = np.random.normal(0, 1., y.shape[0]
                                                      ) * self.specs['surface_error'][1]
                        p_gr = self.rt_dat.atmosphere_pressure_sfc.isel(
                            elevation_angle=0
                        ).values.reshape(-1, 1) + p_gr_noise.reshape(-1, 1)

                        q_gr_noise = np.random.normal(0, 1., y.shape[0]
                                                      ) * self.specs['surface_error'][2]
                        q_gr = self.rt_dat.atmosphere_humidity_sfc.isel(
                            elevation_angle=0
                        ).values.reshape(-1, 1) + q_gr_noise.reshape(-1, 1)

                        x_new = np.concatenate([x_noise, t_gr, p_gr, q_gr], axis=1)
                    else:
                        x_new = x_noise

                    # make quadratic
                    x_new = np.concatenate([x_new, x_new ** 2], axis=1)

                    # split data into traing and test data
                    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                        x_new,
                        y,
                        test_size=0.2,
                        random_state=42)

                    # ret_name = 'tbx_freq_'+str(i_freq_index[0])+'_angle_'+str(i_ang)+'_ret'

                    # make linear regression
                    self.reg, self.coeff, self.const, \
                        self.r_2 = self.make_linear_regression_profile()

                    # test model
                    self.n_test, self.y_test_fil, self.pred_test, \
                        self.bias, self.std, self.rmse = self.test_model_profile()

                    # plot retrieval performance
                    # self.plot_tbx_retrieval_performance(ang, ii, self.y_test, self.pred_test,
                    #                                     self.bias, self.std, self.r2
                    #                                     )

                    # make file
                    data_mwr = self.make_mwr_pro_comp_file(self.coeff, self.const,
                                                           self.bias, self.std,
                                                           i_ang=ang,
                                                           freq_wo_x=self.specs['freq'],
                                                           )

                    self.mwr_pro_ret = ret.Ret(data_mwr)

                    # get data attributes
                    self.mwr_pro_ret.data = get_data_attributes(self.mwr_pro_ret.data,
                                                                self.ret_type
                                                                )

                    # define output file
                    self.output_file = f"output/{self.specs['site']}/{self.ret_type}_" \
                                       f"{self.specs['site']}_" \
                                       f"{self.specs['retrieval_version']}_{str(int(i_ang))}.nc"

                    # make global attributes
                    global_attributes = self.make_global_attributes(ii)

                    # save nc file
                    ret.save_ret(self.mwr_pro_ret,
                                 self.output_file,
                                 global_attributes,
                                 self.ret_type
                                 )

                    # plot model performance
                    plot_performance_2d([self.output_file])

        elif self.ret_type == 'tpb':
            _, freq_z_index, _ = np.intersect1d(
                (self.rt_data.isel(n_date=0, elevation_angle=0
                                   ).frequency.values * 1000).astype(int),
                (np.array(self.specs['freq_z']) * 1000).astype(int),
                assume_unique=False,
                return_indices=True)

            _, freq_bl_index, _ = np.intersect1d(
                (self.rt_data.isel(n_date=0, elevation_angle=0
                                   ).frequency.values * 1000).astype(int),
                (np.array(self.specs['freq_bl']) * 1000).astype(int),
                assume_unique=False,
                return_indices=True)

            freq_index = np.concatenate([freq_z_index, freq_bl_index], axis=0)

            # angle_index = self.get_angle_index()
            _, angle_index, _ = np.intersect1d(
                (self.rt_data.isel(n_date=0, n_frequency=0).elevation_angle.values * 1000).astype(
                    int),
                (np.array(self.specs['multi_angles']) * 1000).astype(int),
                assume_unique=False,
                return_indices=True)

            # update radiative transfer data
            self.rt_dat = self.rt_data.isel(n_frequency=freq_index,
                                            elevation_angle=angle_index,
                                            n_cloud_model=0,
                                            n_height=slice(0, -4),
                                            ).drop_dims(['n_wet_delay_models'])
            self.frequency = self.rt_dat.isel(n_date=0).frequency.values

            # filter out unreasonable data
            self.rt_dat = self.rt_dat.where(
                (self.rt_dat.brightness_temperatures > self.specs['predictor_min']) &
                (self.rt_dat.brightness_temperatures < self.specs['predictor_max']) &
                (self.rt_dat.atmosphere_temperature > self.specs['predictand_min']) &
                (self.rt_dat.atmosphere_temperature < self.specs['predictand_max']),  # &
                # (self.rt_dat.liquid_water_path < 1.5),
                drop=True
            )

            self.height_grid = self.rt_dat.isel(n_date=0, elevation_angle=0, n_frequency=0
                                                ).height_grid

            _, freq_bl_index_1, _ = np.intersect1d(
                (self.rt_dat.isel(n_date=0, elevation_angle=0, n_height=0
                                  ).frequency.values * 1000).astype(int),
                (np.array(self.specs['freq_bl']) * 1000).astype(int),
                assume_unique=False,
                return_indices=True)

            _, angle_bl_index_1, _ = np.intersect1d(
                (self.rt_dat.isel(n_date=0, n_frequency=0, n_height=0).elevation_angle.values * 1000
                 ).astype(int),
                (np.array(self.specs['multi_angles']) * 1000).astype(int),
                assume_unique=False,
                return_indices=True)

            _, angle_z_index_1, _ = np.intersect1d(
                (self.rt_dat.isel(n_date=0, n_frequency=0, n_height=0
                                  ).elevation_angle.values * 1000).astype(int),
                (np.array([90]) * 1000).astype(int),
                assume_unique=False,
                return_indices=True)

            if self.specs['freq_z']:

                _, freq_z_index_1, _ = np.intersect1d(
                    (self.rt_dat.isel(n_date=0, elevation_angle=0, n_height=0
                                      ).frequency.values * 1000).astype(int),
                    (np.array(self.specs['freq_z']) * 1000).astype(int),
                    assume_unique=False,
                    return_indices=True)
                xx_z = np.squeeze(self.rt_dat.isel(elevation_angle=angle_z_index_1,
                                                   n_frequency=freq_z_index_1,
                                                   n_height=0).brightness_temperatures.values)

                xx_bl1 = xx_z
            else:
                xx_bl1 = np.asarray([])

            xx_bl = self.rt_dat.isel(elevation_angle=angle_bl_index_1,
                                     n_frequency=freq_bl_index_1,
                                     n_height=0
                                     ).brightness_temperatures.values

            for i_bl in np.arange(len(freq_bl_index_1)):
                if not xx_bl1.any():
                    xx_bl1 = np.flip(np.squeeze(xx_bl[:, :, i_bl]), axis=1)
                else:
                    xx_bl1 = np.append(xx_bl1, np.flip(np.squeeze(xx_bl[:, :, i_bl]), axis=1
                                                       ), axis=1)

            # define input and output
            x = xx_bl1
            y = self.rt_dat.isel(n_frequency=0, elevation_angle=0).atmosphere_temperature.values

            # add measurement noise to the brightness temperatures
            x_noise = self.add_noise(x)

            if self.specs['surf_mode'] == 'surface':
                t_gr_noise = np.random.normal(
                    0, 1., y.shape[0]) * self.specs['surface_error'][0]
                t_gr = self.rt_dat.atmosphere_temperature_sfc.isel(
                    n_frequency=0, elevation_angle=0, n_height=0
                ).values.reshape(-1, 1) + t_gr_noise.reshape(-1, 1)
                x_new = np.concatenate([x_noise, t_gr], axis=1)
            else:
                x_new = x_noise

            # split data into traing and test data
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                x_new,
                y,
                test_size=0.2,
                random_state=42)

            # make linear regression
            self.reg, self.coeff, self.const, self.r_2 = self.make_linear_regression_profile()

            # test model
            self.n_test, self.y_test_fil, self.pred_test, \
                self.bias, self.std, self.rmse = self.test_model_profile()

            # make file
            data_mwr = self.make_mwr_pro_comp_file(
                self.coeff, self.const, self.bias, self.std,
                freq_wo_x=self.frequency,
                freq_bl_index_1=freq_bl_index_1
            )
            self.mwr_pro_ret = ret.Ret(data_mwr)

            # get data attributes
            self.mwr_pro_ret.data = get_data_attributes(self.mwr_pro_ret.data,
                                                        self.ret_type
                                                        )

            # define output file
            self.output_file = f"output/{self.specs['site']}/{self.ret_type}_"\
                               f"{self.specs['site']}_" \
                               f"{self.specs['retrieval_version']}_{self.specs['handle']}.nc"

            # make global attributes
            global_attributes = self.make_global_attributes()

            # save nc file
            ret.save_ret(self.mwr_pro_ret, self.output_file, global_attributes, self.ret_type)

            # plot model performance
            plot_performance_2d([self.output_file])

    def add_noise(self, x):
        noise = np.random.normal(0, 1., x.shape) * self.specs['tb_noise']
        x_noise = x + noise
        return x_noise

    def test_model(self):
        i_test = np.where(
            (self.y_test > self.specs['predictand_min']) &
            (self.y_test < self.specs['predictand_max'])
        )

        n_test = i_test[0].shape[0]
        y_test_fil = self.y_test[i_test[0]]
        pred_test = self.const + np.sum(
            np.array(self.x_test[i_test[0], :]) * np.asarray(self.coeff), axis=1)
        bias = np.sum(np.subtract(pred_test, y_test_fil)) / n_test
        std = np.std(np.subtract(pred_test, y_test_fil))
        rmse = np.sqrt(np.mean(np.subtract(pred_test, y_test_fil)**2))

        return n_test, y_test_fil, pred_test, bias, std, rmse

    def test_model_profile(self):
        n_test = []
        y_test_fil = []
        pred_test = []
        bias = []
        std = []
        rmse = []

        for hh in range(len(self.height_grid)):
            if self.ret_type == 'hpt':
                i_test = np.where(
                    (self.y_test[:, hh] < self.specs['predictand_max'])
                )
            else:
                i_test = np.where(
                    (self.y_test[:, hh] > self.specs['predictand_min']) &
                    (self.y_test[:, hh] < self.specs['predictand_max'])
                )

            n_test.append(i_test[0].shape)
            y_test_fil.append(self.y_test[i_test[0], hh])

            pred_test.append(self.const[hh] + np.sum(
                np.array(self.x_test[i_test[0], :]) * np.asarray(self.coeff)[hh, :], axis=1))
            bias.append(np.sum(np.subtract(pred_test, y_test_fil)) / i_test[0].shape[0])
            std.append(np.std(np.subtract(pred_test, y_test_fil)))

            rmse.append(np.sqrt(np.mean(np.subtract(pred_test, y_test_fil)**2)))

        return n_test, y_test_fil, pred_test, bias, std, rmse

    def make_linear_regression(self):
        i_train = np.where(
            (self.y_train > self.specs['predictand_min']) &
            (self.y_train < self.specs['predictand_max'])
        )

        reg = linear_model.LinearRegression()
        reg.fit(self.x_train[i_train[0], :], self.y_train[i_train[0]])
        coeff = reg.coef_
        const = reg.intercept_
        r2 = reg.score(self.x_train[i_train[0], :], self.y_train[i_train[0]])

        return reg, coeff, const, r2

    def make_linear_regression_profile(self):
        coeff = []
        const = []
        r2 = []

        reg = linear_model.LinearRegression()

        for hh in range(len(self.height_grid)):
            i_train = np.where(
                (self.y_train[:, hh] > self.specs['predictand_min']) &
                (self.y_train[:, hh] < self.specs['predictand_max'])
            )
            reg.fit(self.x_train[i_train[0], :], self.y_train[i_train[0], hh])
            coeff.append(reg.coef_)
            const.append(reg.intercept_)
            r2.append(reg.score(self.x_train[i_train[0], :], self.y_train[i_train[0], hh]))

        return reg, coeff, const, r2

    def make_mwr_pro_comp_file(self, coeff, const, bias, std,
                               freq_wo_x=None, freq_bl_index_1=None, i_ang=None):
        """make ret files which is compatible with mwr-pro"""
        if self.ret_type in ('tbx', 'iwv', 'lwp'):
            data_mwr_pro_add = {
                'elevation_predictand': self.rt_data.isel(
                    n_date=0, elevation_angle=i_ang).elevation_angle.values,
                'elevation_predictor': self.rt_data.isel(
                    n_date=0, elevation_angle=i_ang).elevation_angle.values,
            }

        elif self.ret_type == 'tpb':
            data_mwr_pro_add = {
                'freq_bl': np.asarray(
                    self.rt_dat.isel(
                        n_date=0, elevation_angle=0, n_frequency=freq_bl_index_1, n_height=0
                    ).frequency.values
                ).astype('float64'),
                'elevation_predictand': np.int32(90),
                'elevation_predictor': np.asarray(self.specs['multi_angles']),
                'height_grid': np.asarray(self.height_grid.values),
            }
        elif self.ret_type in ('tpt', 'hpt'):
            data_mwr_pro_add = {
                'elevation_predictand': self.rt_data.isel(
                    n_date=0, elevation_angle=i_ang).elevation_angle.values,
                'elevation_predictor': self.rt_data.isel(
                    n_date=0, elevation_angle=i_ang).elevation_angle.values,
                'height_grid': np.asarray(self.height_grid.values),
            }
        else:
            data_mwr_pro_add = {}

        data_mwr_pro = {
            'freq': np.asarray(freq_wo_x).astype('float64'),
            'lat': self.rt_data.latitude,
            'lon': self.rt_data.longitude,
            'prdmx': self.specs['predictand_max'],
            'prdmn': self.specs['predictand_min'],
            'prrmx': self.specs['predictor_max'],
            'prrmn': self.specs['predictor_min'],
            'asl': self.rt_data.attrs['altitude MSL'],
            'surface_err': np.asarray(self.specs['surface_error']),
            'predictor_err': np.asarray(self.specs['predictor_error']),
            'predictand_err': np.asarray(std),
            'predictand_err_sys': np.asarray(bias),
            'predictand_rmse': np.asarray(self.rmse),
            'predictand_r_2': np.asarray(self.r_2),
            'coefficient_mvr': np.asarray(coeff).T,
            'offset_mvr': np.asarray(const),
        }
        data_mwr_pro.update(data_mwr_pro_add)

        return data_mwr_pro

    def make_global_attributes(self, ii=None):
        global_attributes = {
            'rt_data_path': str(self.specs['rt_paths']),
            'site': self.specs['site_location'],
            'number_of_profiles_used': self.rt_data.sizes['n_date'],
            'date_start': self.rt_data.date.values[0],
            'date_end': self.rt_data.date.values[-1],
            'rt_calc_cut_off_height_in_m': self.rt_data[
                'cap_height_above_sea_level'].values,
            'gas_absorption_model': self.rt_data['gas_absorption_model'].values,
            'cloud_absorption_model': self.rt_data[
                'cloud_absorption_model'].values,
            'wv_linewidth': self.rt_data['linewidth'].values,
            'wv_continuum_correction': self.rt_data['cont_corr'].values,
            'air_mass_correction': self.rt_data['air_mass_correction'].values
        }

        if self.ret_type == 'tpb':
            global_attributes['predictand'] = 'tpb'
            global_attributes['predictand_unit'] = 'K'
            global_attributes['predictor'] = self.specs['predictor']
            global_attributes['predictor_unit'] = 'K'
        elif self.ret_type == 'tpt':
            global_attributes['predictand'] = 'tpt'
            global_attributes['predictand_unit'] = 'K'
            global_attributes['predictor'] = self.specs['predictor']
            global_attributes['predictor_unit'] = 'K'
        elif self.ret_type == 'hpt':
            global_attributes['predictand'] = 'hpt'
            global_attributes['predictand_unit'] = 'kg m-3'
            global_attributes['predictor'] = self.specs['predictor']
            global_attributes['predictor_unit'] = 'kg m-3'
        elif self.ret_type == 'tbx':
            global_attributes['predictand'] = 'tbx_' + str(ii)
            global_attributes['predictand_unit'] = 'K'
            global_attributes['predictor'] = self.specs['predictor']
            global_attributes['predictor_unit'] = 'K'
        elif self.ret_type == 'iwv':
            global_attributes['predictand'] = 'iwv'
            global_attributes['predictand_unit'] = 'k'
            global_attributes['predictor'] = self.specs['predictor']
            global_attributes['predictor_unit'] = 'kg m-2'
        elif self.ret_type == 'lwp':
            global_attributes['predictand'] = 'tbx'
            global_attributes['predictand_unit'] = 'K'
            global_attributes['predictor'] = self.specs['predictor']
            global_attributes['predictor_unit'] = 'kg m-2'

        global_attributes['retrieval_version'] = self.specs['retrieval_version']
        global_attributes['surface_mode'] = self.specs['surf_mode']
        global_attributes['regression_type'] = self.specs['regression_type']

        global_attributes['cloudy_clear'] = 'all'
        global_attributes['cloud_diagnosis'] = 'Karstens_1994'
        global_attributes['cloud_diagnosis_rh_threshold'] = self.rt_data['rh_thres'].values

        global_attributes['bandwidth_correction'] = 'no'
        global_attributes['beamwidth_correction'] = 'no'

        return global_attributes

    @staticmethod
    def get_freq_index(_freqs, _rt_dat):

        freq_index = []
        for item in _freqs:
            freq_index.extend(np.argwhere(_rt_dat.isel(n_date=0).frequency.values == item)[0])
        return freq_index

    def get_angle_index(self):

        angle_index = []
        for item in self.specs['angle']:
            angle_index.extend(
                np.argwhere(self.rt_data.isel(n_date=0).elevation_angle.values == item)[0])
        return angle_index

    def plot_retrieval_performance_1d(self, ang, ii, y_test, pred_test, bias, std, r2):

        if self.ret_type == 'tbx':
            x_min = 10 * np.floor(np.min(y_test) / 10)
            x_max = 10 * np.ceil(np.max(y_test) / 10)
            y_min = 10 * np.floor(np.min(pred_test) / 10)
            y_max = 10 * np.ceil(np.max(pred_test) / 10)
            bins = [np.arange(x_min, x_max, .5), np.arange(y_min, y_max, .5)]
        elif self.ret_type == 'iwv':
            x_min = 0
            x_max = 60
            y_min = 0
            y_max = 60
            bins = [np.arange(x_min, x_max, .5), np.arange(y_min, y_max, .5)]
        else:
            x_min = 0
            x_max = 1.2
            y_min = 0
            y_max = 1.2
            bins = [np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01)]

        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')

        dummy = ax.hist2d(y_test,
                          pred_test,
                          bins=bins,
                          norm=mcolors.LogNorm(),
                          cmin=1e0,
                          cmax=1e3,
                          cmap="cividis",
                          )
        cbar = plt.colorbar(dummy[3], ax=ax)
        cbar.set_label('Frequency of occurrence')

        ax.set_xlabel('Reference brightness temperature (K)')
        ax.set_ylabel('Retrieved brightness temperature (K)')
        if self.ret_type == 'tbx':
            ax.set_title(str(self.specs['freq'][ii]) + ' GHz')
            ax.set_xlabel('Reference brightness temperature (K)')
            ax.set_ylabel('Retrieved brightness temperature (K)')
        elif self.ret_type == 'iwv':
            ax.set_xlabel('Reference integrated water vapor (Kg m-2)')
            ax.set_ylabel('Retrieved integrated water vapor (Kg m-2)')
        elif self.ret_type == 'lwp':
            ax.set_xlabel('Reference liquid water path (kg m-2)')
            ax.set_ylabel('Retrieved liquid water path (kg m-2)')
        else:
            pass

        rrr = Patch(fill=False, edgecolor='none', visible=False)
        ax.legend([rrr, rrr, rrr],
                  ['std = ' + str(np.round(std, 2)),
                   'bias = ' + str(np.round(bias, 4)),
                   'r2 = ' + str(np.round(r2, 4))],
                  loc='upper left',
                  frameon=False)

        if self.ret_type == 'tbx':
            png_file = f"output/{self.specs['site']}/" \
                       f"{self.ret_type}_{self.specs['site']}" \
                       f"_no_surf_{str(int(ang*10))}_{str(ii).zfill(2)}.png"
        else:
            png_file = f"output/{self.specs['site']}/" \
                       f"{self.ret_type}_{self.specs['site']}_" \
                       f"{self.specs['retrieval_version']}_{str(int(ang))}.png"

        plt.tight_layout()
        plt.savefig(png_file)
        plt.close(fig)
