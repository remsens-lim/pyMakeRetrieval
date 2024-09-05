"""Module for plotting"""
import xarray as xr
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def plot_performance_2d(
        files: list,
        labels=None
):
    """Generates performance (bias, std.) plot of the selected retrievals
    Args:
    files: list of input retrieval nc files, maximum is 3
    labels: list of labels to use in the legend, default is None

    Returns:
        fig and ax object of the plot

    Examples:
        >>> from py_make_retrieval.plotting import plot_performance_2d
        >>> plot_performance_2d(
        ['tpb_rao_rt00_7-10.nc', 'tpb_rao_rt00_4-10.nc', 'tpb_rao_rt00_4-9.nc'],
        labels = ['7 freqs, 10 angles\nwith_zenith', '4 freqs, 10 angles\nwith zenith',
        '4 freqs 9 angles\nwithout zenith']
        )
    """
    if not labels:
        legend = False
        labels = ['a', 'b', 'c']
    else:
        legend = True
    lstyles = ['--', '-.', ':']

    # ret_type = str(files[0])[-3:1]
    if 'hpt' in files[0]:
        unit = 'g m-3'
        xlimits_bias = (-1, 1 )
        xlimits_std = (0, 3)
        fac = 1e3
    else:
        unit = 'K'
        xlimits_bias = (-1, 1)
        xlimits_std = (0, 3)
        fac = 1.

    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(ncols=2, figsize=(8, 4), sharey='row')

    # i = 0
    for file, i in zip(files, np.arange(len(files))):
        data = xr.open_dataset(file)
        ax[0].plot(data.predictand_err_sys * fac,
                   data.height_grid / 1000.,
                   label=labels[i],
                   ls=lstyles[i]
                   )

        ax[1].plot(data.predictand_err * fac,
                   data.height_grid / 1000.,
                   label=labels[i],
                   ls=lstyles[i]
                   )

        # ax[2].plot(data.r2, data.height_grid / 1000., label=label, ls=lstyles[i])
        # i = i + 1

    ax[0].set_xlabel('Bias ('+unit+')')
    ax[1].set_xlabel('St. dev. ('+unit+')')
    # ax[2].set_xlabel('Coeff. of det. R$^2$ (K)')

    ax[0].set_ylabel('Height (km)')
    if 'tpb' in files[0]:
        ax[0].set_ylim(0, 3)
    else:
        ax[0].set_ylim(0, 10)

    ax[0].set_xlim(xlimits_bias[0], xlimits_bias[1])
    ax[1].set_xlim(xlimits_std[0], xlimits_std[1])
    # ax[2].set_xlim(0.85, 1)

    ax[0].text(.05, .9, 'a)', fontsize=16, transform=ax[0].transAxes)
    ax[1].text(.05, .9, 'b)', fontsize=16, transform=ax[1].transAxes)
    # ax[2].text(.05, .9, 'c)', fontsize=16, transform=ax[2].transAxes)

    for ax_i in ax:
        ax_i.tick_params(axis='x', which='both', bottom=True, top=True, labeltop=False)
        ax_i.tick_params(axis='y', which='both', left=True, right=True)
        ax_i.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_i.yaxis.set_minor_locator(AutoMinorLocator(2))

    if legend:
        ax[1].legend(loc=4, frameon=False, fontsize=9)

    plt.tight_layout()

    if len(files) == 1:
        outfile_name = str(files[0])[:-3]
    else:
        outfile_name = 'retrieval_performance'

    plt.savefig(outfile_name + '.png')
    plt.close(fig)

    # return fig, ax
