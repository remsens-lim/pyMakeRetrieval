"""contains helper functions"""
import yaml
import numpy as np
from numpy import ma
from yaml.loader import SafeLoader


def read_yaml_config(_file) -> dict:
    """Reads config yaml files."""
    with open(_file, encoding="utf-8") as _f:
        config = yaml.load(_f, Loader=SafeLoader)

    return config


def isscalar(array: any) -> bool:
    """Tests if input is scalar.
    By "scalar" we mean that array has a single value.
    Examples:
        >>> isscalar(1)
            True
        >>> isscalar([1])
            True
        >>> isscalar(np.array(1))
            True
        >>> isscalar(np.array([1]))
            True
    """

    arr = ma.array(array)
    if not hasattr(arr, "__len__") or arr.shape == () or len(arr) == 1:
        return True
    return False


def mu_calc(z, temp, p, q, f, theta0, air_corr=None, z_site=None):

    zz = z_site is not None
    if not zz:
        re = 6370950.
    else:
        re = 6370950. + z_site

    alt = 90. - theta0
    theta0 = np.deg2rad(theta0)

    n_l = len(z)
    n_f = len(f)

    mu = np.zeros((n_f, n_l - 1))
    deltas = np.zeros(n_l - 1)

    coeff = refractive_index_coeff(air_corr)

    if air_corr == 'no':
        mu[:, :] = np.cos(theta0)
    elif air_corr == 'rozenberg_66':
        mu[:, :] = np.cos(theta0) + 0.025 * np.exp(-11 * np.cos(theta0))
    elif air_corr == 'young_94':
        mu[:, :] = (np.cos(theta0) ** 3 + 0.149864 * np.cos(theta0) ** 2 +
                    0.0102963 * np.cos(theta0) + 0.000303978) / \
                   (1.002432 * np.cos(theta0) ** 2 + 0.148386 * np.cos(
                    theta0) + 0.0096467)
    elif air_corr == 'pickering_02':
        mu[:, :] = np.sin(np.deg2rad(alt + 244 / (165 + 47 * (alt ** 1.1))))
    elif air_corr == '43':
        re = (4. / 3.) * re

    for k in range(n_f):
        theta_bot = theta0
        r_bot = re

        for i in range(1, n_l):

            # if air_corr == 'liebe_93':
            #     n_top = 1. + ref[i - 1, k] * 1e-6
            #     n_bot = 1. + ref[i - 2, k] * 1e-6 if i > 1 else n_top
            # else:
            temp_top = 0.5 * (temp[i] + temp[i - 1])
            p_top = 0.5 * (p[i] + p[i - 1])
            q_top = 0.5 * (q[i] + q[i - 1])

            n_top = calculate_atm_params(temp_top, p_top, q_top, coeff)

            if i > 1:
                temp_bot = 0.5 * (temp[i - 1] + temp[i - 2])
                p_bot = 0.5 * (p[i - 1] + p[i - 2])
                q_bot = 0.5 * (q[i - 1] + q[i - 2])
                n_bot = calculate_atm_params(temp_bot, p_bot, q_bot, coeff)
            else:
                n_bot = n_top
            deltaz = z[i] - z[i - 1]
            r_top = r_bot + deltaz

            theta_top = np.arcsin(((n_bot * r_bot) / (n_top * r_top)) * np.sin(theta_bot))
            alpha = np.pi - theta_bot
            deltas[i - 1] = r_bot * np.cos(alpha) + np.sqrt(
                r_top ** 2 + r_bot ** 2 * ((np.cos(alpha)) ** 2 - 1)
            )

            mu[k, i - 1] = deltaz / deltas[i - 1]
            theta_bot = theta_top
            r_bot = r_top

    return mu, deltas


def abshum2mixture(temp, p, rho):
    _Rl = 287.
    _Rv = 462.
    fak = _Rl / _Rv

    e = rho * _Rv * temp

    # es = ESAT(temp)  # Assuming ESAT function is defined elsewhere
    m = fak * e / (p - e)

    return m


def sat_vap_press(temp):
    e0 = 610.78
    _Rv = 462.
    _L = lat_heat_vap(temp)
    temp0 = 273.15
    fak = _L / (_Rv * temp0)

    x = e0 * np.exp(fak * (temp - temp0) / temp)

    return x


def lat_heat_vap(temp):
    x = 2.501e6 - 2372. * (temp - 273.15)
    return x


def refractive_index_coeff(air_corr):
    coeff_mapping = {
        'thayer_74': [77.604, 64.79, 3.776],
        'liebe_77': [77.676, 71.631, 3.74656],
        'hill_80': [0., 98., 3.58300],
        'bevis_94': [77.6, 70.4, 3.739],
        'rueeger_avai_02': [77.695, 71.97, 3.75406],
        'rueeger_aver_02': [77.689, 71.2952, 3.75463],
        'sphere': [0., 0., 0.],
        '43': [0., 0., 0.]
    }

    return coeff_mapping.get(air_corr)


def calculate_atm_params(temp, p, q, coeff):
    mixr = abshum2mixture(temp, p, q)
    _e = w2e(mixr * 1000, p / 100)
    n = 1 + (coeff[0] * (((p / 100) - _e) / temp) + coeff[1] *
             (_e / temp) + coeff[2] * (_e / (temp ** 2))) * 1e-6
    return n


def w2e(_w, p):
    """
    Converts water vapor mixing ratio to the partial pressure of water vapor.

    :param _w: Water vapor mixing ratio in g/kg.
    :param p: Barometric pressure in mb.
    :return: Vapor pressure in mb.
    """
    ww = _w / 1000.0  # Convert g/kg to g/g
    e = p * ww / (0.622 + ww)
    return e
