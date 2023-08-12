import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from chainconsumer import ChainConsumer
# from natsort import natsorted
# import imageio
import numpy as np
from sklearn.preprocessing import PowerTransformer
# import torch
import os
# import healpy as hp
import pandas as pd
"""import warnings

warnings.filterwarnings("error")"""


# def plot_chain(data_frame, plot_name, max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12, columns=None,
#                parameter=None, extends=None):
#     """
#
#     :param extends: extents={
#                 "mag r": (17.5, 26),
#                 "mag i": (17.5, 26),
#                 "mag z": (17.5, 26),
#                 "snr": (-11, 55),
#                 "size ratio": (-1.5, 4),
#                 "T": (-1, 2.5)
#             }
#     :param label_font_size:
#     :param tick_font_size:
#     :param shade_alpha:
#     :param max_ticks:
#     :param plot_name: "generated observed properties: chat*"
#     :param data_frame:
#     :param columns: Mutable list, default values are columns = [
#             "unsheared/mag_r",
#             "unsheared/mag_i",
#             "unsheared/mag_z",
#             "unsheared/snr",
#             "unsheared/size_ratio",
#             "unsheared/T"
#         ]
#     :param parameter: Mutable list, default values are parameter = [
#                 "mag r",
#                 "mag i",
#                 "mag z",
#                 "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
#                 "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
#                 "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
#             ]
#     :return:
#     """
#     df_plot = pd.DataFrame({})
#
#     if columns is None:
#         columns = [
#             "unsheared/mag_r",
#             "unsheared/mag_i",
#             "unsheared/mag_z",
#             "unsheared/snr",
#             "unsheared/size_ratio",
#             "unsheared/T"
#         ]
#
#     if parameter is None:
#         parameter = [
#                 "mag r",
#                 "mag i",
#                 "mag z",
#                 "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
#                 "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
#                 "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
#             ]
#
#     for col in columns:
#         df_plot[col] = np.array(data_frame[col])
#
#     chain = ChainConsumer()
#     chain.add_chain(df_plot.to_numpy(), parameters=parameter, name=plot_name)
#     chain.configure(
#         max_ticks=max_ticks,
#         shade_alpha=shade_alpha,
#         tick_font_size=tick_font_size,
#         label_font_size=label_font_size
#     )
#     # if extends is not None:
#     chain.plotter.plot(
#         figsize="page",
#         extents=extends
#     )
#     plt.show()
#     plt.clf()


def load_healpix(path2file, hp_show=False, nest=True, partial=False, field=None):
    """
    # Function to load fits datasets
    # Returns:

    """
    """if field is None:
        hp_map = hp.read_map(path2file, nest=nest, partial=partial)
    else:
        hp_map = hp.read_map(path2file, nest=nest, partial=partial, field=field)
    if hp_show is True:
        hp_map_show = hp_map
        if field is not None:
            hp_map_show = hp_map[1]
        hp.mollview(
            hp_map_show,
            norm="hist",
            nest=nest
        )
        hp.graticule()
        plt.show()
        
    return hp_map"""


def match_skybrite_2_footprint(path2footprint, path2skybrite, hp_show=False, nest_footprint=True, nest_skybrite=True,
                               partial_footprint=False, partial_skybrite=True, field_footprint=None,
                               field_skybrite=None):
    """
    Main function to run
    Returns:

    """
    """hp_map_footprint = load_healpix(
        path2file=path2footprint,
        hp_show=hp_show,
        nest=nest_footprint,
        partial=partial_footprint,
        field=field_footprint
    )

    hp_map_skybrite = load_healpix(
        path2file=path2skybrite,
        hp_show=hp_show,
        nest=nest_skybrite,
        partial=partial_skybrite,
        field=field_skybrite
    )
    sky_in_footprint = hp_map_skybrite[:, hp_map_footprint != hp.UNSEEN]
    good_indices = sky_in_footprint[0, :].astype(int)
    return np.column_stack((good_indices, sky_in_footprint[1]))"""


# def generate_normal_distribution(size, mu, sigma, num=1, as_tensor=True):
#     """
#     Generate uniform distributed random data for discriminator.
#
#     Args:
#         size: size of the tensor
#
#     Returns:
#         random data as torch tensor
#     """
#     # random_data = torch.randn(size)
#
#     if as_tensor is False:
#         return np.random.normal(mu, sigma, size=(size, num))
#     return torch.FloatTensor([np.random.normal(mu, sigma, size=(size, num))[0][0]])
#
#
# def generate_uniform_distribution(size, low, high, num=1, as_tensor=True):
#     """
#     Generate normal distributed random data for generator.
#
#     Args:
#         size: size of the tensor
#
#     Returns:
#         random data as torch tensor
#     """
#     # random_data = torch.rand(size)
#
#     if not as_tensor:
#         return np.random.uniform(low, high, size=(num, size))
#     return torch.FloatTensor(np.random.uniform(low, high, size=(num, size)))


def luptize(flux, var, s, zp):
    # s: measurement error (variance) of the flux (with zero pt zp) of an object at the limiting magnitude of the survey
    # a: Pogson's ratio
    # b: softening parameter that sets the scale of transition between linear and log behavior of the luptitudes
    a = 2.5 * np.log10(np.exp(1))
    b = a**(1./2) * s
    mu0 = zp -2.5 * np.log10(b)

    # turn into luptitudes and their errors
    lupt = mu0 - a * np.arcsinh(flux / (2 * b))
    lupt_var = a ** 2 * var / ((2 * b) ** 2 + flux ** 2)
    return lupt, lupt_var


def luptize_deep(flux, bins, var=0, zp=22.5):
    """
    The flux must be in the same dimension as the bins.
    The bins must be given as list like ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]
    the ordering of the softening parameter b
    """
    dict_mags = {
        "i": 24.66,
        "g": 25.57,
        "r": 25.27,
        "z": 24.06,
        "u": 24.64,
        "Y": 24.6,  # y band value is copied from array above because Y band is not in the up to date catalog
        "J": 24.02,
        "H": 23.69,
        "K": 23.58
    }
    lst_mags = []
    for b in bins:
        if b in ["I", "G", "R", "Z", "U"]:
            actual_b = b.lower()
        elif b in ["y", "j", "h", "k"]:
            actual_b = b.upper()
        elif b in ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]:
            actual_b = b
        else:
            raise IOError("bin not defined")
        lst_mags.append(dict_mags[actual_b])
    arr_mags = np.array(lst_mags)
    s = (10**((zp-arr_mags)/2.5)) / 10
    return luptize(flux, var, s, zp)


def luptize_fluxes(data_frame, flux_col, lupt_col, bins):
    """

    :param bins: ["I", "R", "Z", "J", "H", "K"]
    :param data_frame:
    :param flux_col: ("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB")
    :param lupt_col: ("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB")
    :return:
    """
    lst_flux = []
    lst_var = []
    for bin in bins:
        lst_flux.append(data_frame[f"{flux_col[0]}_{bin}"])
        lst_var.append(data_frame[f"{flux_col[1]}_{bin}"])
    arr_flux = np.array(lst_flux).T
    arr_var = np.array(lst_var).T
    lupt_mag, lupt_var = luptize_deep(flux=arr_flux, bins=bins, var=arr_var)
    lupt_mag = lupt_mag.T
    lupt_var = lupt_var.T
    for idx_bin, bin in enumerate(bins):
        data_frame[f"{lupt_col[0]}_{bin}"] = lupt_mag[idx_bin]
        data_frame[f"{lupt_col[1]}_{bin}"] = lupt_var[idx_bin]
    return data_frame


def luptize_inverse(lupt, lupt_var, s, zp):
    """"""
    # s: measurement error (variance) of the flux (with zero pt zp) of an object at the limiting magnitude of the survey
    # a: Pogson's ratio
    # b: softening parameter that sets the scale of transition between linear and log behavior of the luptitudes
    a = 2.5 * np.log10(np.exp(1))
    b = a**(1./2) * s
    mu0 = zp -2.5 * np.log10(b)

    # turn into luptitudes and their errors
    # lupt = mu0 - a * np.arcsinh(flux / (2 * b))
    flux = 2 * b * np.sinh((mu0 - lupt) / a)
    var = (lupt_var * ((2 * b)**2 + flux**2)) / (a**2)
    # lupt_var = a ** 2 * var / ((2 * b) ** 2 + flux ** 2)
    return flux, var


def luptize_inverse_deep(lupt, bins, lupt_var=0, zp=22.5):
    """
        The flux must be in the same dimension as the bins.
        The bins must be given as list like ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]
        the ordering of the softening parameter b
    """
    dict_mags = {
        "i": 24.66,
        "g": 25.57,
        "r": 25.27,
        "z": 24.06,
        "u": 24.64,
        "Y": 24.6,  # y band value is copied from array above because Y band is not in the up to date catalog
        "J": 24.02,
        "H": 23.69,
        "K": 23.58
    }
    lst_mags = []
    for b in bins:
        if b in ["I", "G", "R", "Z", "U"]:
            actual_b = b.lower()
        elif b in ["y", "j", "h", "k"]:
            actual_b = b.upper()
        elif b in ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]:
            actual_b = b
        else:
            raise IOError("bin not defined")
        lst_mags.append(dict_mags[actual_b])
    arr_mags = np.array(lst_mags)
    s = (10 ** ((zp - arr_mags) / 2.5)) / 10
    return luptize_inverse(lupt, lupt_var, s, zp)


def luptize_inverse_fluxes(data_frame, flux_col, lupt_col, bins):
    """

    :param bins: ["I", "R", "Z", "J", "H", "K"]
    :param data_frame:
    :param flux_col: ("BDF_FLUX_DERED_CALIB_I", "BDF_FLUX_ERR_DERED_CALIB_I")
    :param lupt_col: ("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB")
    :return:
    """
    lst_lupt = []
    lst_lupt_var = []
    for bin in bins:
        lst_lupt.append(data_frame[f"{lupt_col[0]}_{bin}"])
        lst_lupt_var.append(data_frame[f"{lupt_col[1]}_{bin}"])
    arr_lupt = np.array(lst_lupt).T
    arr_lupt_var = np.array(lst_lupt_var).T
    arr_flux, arr_var = luptize_inverse_deep(lupt=arr_lupt, bins=bins, lupt_var=arr_lupt_var)
    arr_flux = arr_flux.T
    arr_var = arr_var.T
    for idx_bin, bin in enumerate(bins):
        data_frame[f"{flux_col[0]}_{bin}"] = arr_flux[idx_bin]
        data_frame[f"{flux_col[1]}_{bin}"] = arr_var[idx_bin]
    return data_frame


def calc_mag(data_frame, flux_col, mag_col, bins):
    """"""
    for b in bins:
        if isinstance(mag_col, tuple):
            data_frame[f"{mag_col[0]}_{b}"] = flux2mag(data_frame[f"{flux_col[0]}_{b}"])
            data_frame[f"{mag_col[1]}_{b}"] = flux2mag(data_frame[f"{flux_col[1]}_{b}"])
        else:
            data_frame[f"{mag_col}_{b}"] = flux2mag(data_frame[f"{flux_col}_{b}"])
    return data_frame


def calc_color(data_frame, mag_type, flux_col, mag_col, bins, plot_data=False):
    """

    :param data_frame:
    :param mag_type:
    :param flux_col:
    :param mag_col:
    :param bins:
    :return:
    """

    if isinstance(mag_col, tuple):
        mag = mag_col[0]
    else:
        mag = mag_col

    if mag not in data_frame.keys():
        if mag_type[0] == "MAG":
            data_frame = calc_mag(
                data_frame=data_frame,
                flux_col=flux_col,
                mag_col=mag_col,
                bins=bins
            )
        elif mag_type[0] == "LUPT":
            data_frame = luptize_fluxes(
                data_frame=data_frame,
                flux_col=flux_col,
                lupt_col=mag_col,
                bins=bins
            )
    lst_color_cols = []
    lst_mag_cols = []
    lst_mag_parameter = []
    lst_color_parameter = []
    for idx_b, b in enumerate(bins):
        next_b = bins[idx_b+1]
        lst_color_cols.append(f"Color {mag_type[1]} {mag_type[0]} {b}-{next_b}")
        lst_mag_cols.append(f"{mag}_{b}")
        lst_mag_parameter.append(f"{mag_type[0]} {b}")
        data_frame[f"Color {mag_type[1]} {mag_type[0]} {b}-{next_b}"] = data_frame[f"{mag}_{b}"] - data_frame[f"{mag}_{next_b}"]
        lst_color_parameter.append(f"{mag_type[0]} {b}-{next_b}")
        if idx_b+2 >= len(bins):
            lst_mag_cols.append(f"{mag}_{next_b}")
            lst_mag_parameter.append(f"{mag_type[0]} {next_b}")
            break
    return data_frame, lst_color_cols, lst_color_parameter, lst_mag_cols, lst_mag_parameter


def replace_nan_with_gaussian(val, loc, scale):
    if pd.isna(val):
        while True:
            random_val = np.random.normal(loc=loc, scale=scale, size=1)[0]
            if loc-scale <= random_val <= loc+scale:
                return random_val
    else:
        return val


def replace_values(data_frame, replace_value):
    for col in replace_value.keys():
        replace_value_index = None if replace_value[col] == "None" else replace_value[col]
        if replace_value_index is not None:
            replace_value_tuple = replace_value_index
            if not isinstance(replace_value_tuple, tuple):
                replace_value_tuple = eval(replace_value_index)
            data_frame[col] = data_frame[col].replace(replace_value_tuple[0], replace_value_tuple[1])
    return data_frame


def replace_values_with_gaussian(data_frame, replace_value):
    for col in replace_value.keys():
        replace_value_index = None if replace_value[col] == "None" else replace_value[col]
        if replace_value_index is not None:
            replace_value_tuple = replace_value_index
            if not isinstance(replace_value_tuple, tuple):
                replace_value_tuple = eval(replace_value_index)
            while True:
                loc = replace_value_tuple[1]
                scale = replace_value_tuple[2]
                random_val = np.random.normal(loc=loc, scale=scale, size=1)[0]
                if loc - scale <= random_val <= loc + scale:
                    data_frame[col] = data_frame[col].replace(replace_value_tuple[0], random_val)
    return data_frame


def replace_and_transform_data(data_frame, columns):
    """"""
    dict_pt = {}
    for col in columns:
        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(np.array(data_frame[col]).reshape(-1, 1))
        data_frame[col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
        dict_pt[f"{col} pt"] = pt
    return data_frame, dict_pt


def mag2flux(magnitude, zero_pt=30):
    # convert flux to magnitude
    try:
        flux = 10**((zero_pt-magnitude)/2.5)
        return flux
    except RuntimeWarning:
        print("Warning")


def flux2mag(flux, zero_pt=30, clip=0.001):
    # convert flux to magnitude
    """lst_mag = []
    for f in flux:
        try:
            magnitude = zero_pt - 2.5 * np.log10(f)
            lst_mag.append(magnitude)
        except RuntimeWarning:
            print("Warning")
            # lst_mag.append(-100)"""
    if clip is None:
        return zero_pt - 2.5 * np.log10(flux)
    return zero_pt - 2.5 * np.log10(flux.clip(clip))
    # return np.array(lst_mag)


# def unsheared_mag_cut(data_frame):
#     """"""
#     mag_cuts = (
#             (18 < data_frame["unsheared/mag_i"]) &
#             (data_frame["unsheared/mag_i"] < 23.5) &
#             (15 < data_frame["unsheared/mag_r"]) &
#             (data_frame["unsheared/mag_r"] < 26) &
#             (15< data_frame["unsheared/mag_z"]) &
#             (data_frame["unsheared/mag_z"] < 26) &
#             (-1.5 < data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"]) &
#             (data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"] < 4) &
#             (-4 < data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"]) &
#             (data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"] < 1.5)
#     )
#     data_frame = data_frame[mag_cuts]
#     shear_cuts = (
#             (10 < data_frame["unsheared/snr"]) &
#             (data_frame["unsheared/snr"] < 1000) &
#             (0.5 < data_frame["unsheared/size_ratio"]) &
#             (data_frame["unsheared/T"] < 10)
#     )
#     data_frame = data_frame[shear_cuts]
#     data_frame = data_frame[~((2 < data_frame["unsheared/T"]) & (data_frame["unsheared/snr"] < 30))]
#     return data_frame
#
#
# def bdf_mag_cuts(data_frame):
#     """"""
#     bdf_cuts = (
#         (data_frame["BDF_MAG_ERR_DERED_CALIB_J"] < 37.5) &
#         (10 < data_frame["BDF_MAG_ERR_DERED_CALIB_J"]) &
#         (data_frame["BDF_MAG_ERR_DERED_CALIB_H"] < 37.5) &
#         (10 < data_frame["BDF_MAG_ERR_DERED_CALIB_H"]) &
#         (data_frame["BDF_MAG_ERR_DERED_CALIB_K"] < 37.5) &
#         (10 < data_frame["BDF_MAG_ERR_DERED_CALIB_K"])
#     )
#     data_frame = data_frame[bdf_cuts]
#     return data_frame
#
#
# def metacal_cuts(data_frame):
#     """"""
#     print("Apply mcal cuts")
#     mcal_cuts = (data_frame["unsheared/extended_class_sof"] >= 0) & (data_frame["unsheared/flags_gold"] < 2)
#     data_frame = data_frame[mcal_cuts]
#     print('Length of mcal catalog after applying cuts: {}'.format(len(data_frame)))
#     return data_frame
#
#
# def detection_cuts(data_frame):
#     """"""
#     print("Apply detection cuts")
#     detect_cuts = (data_frame["match_flag_1.5_asec"] < 2) & \
#                   (data_frame["flags_foreground"] == 0) & \
#                   (data_frame["flags_badregions"] < 2) & \
#                   (data_frame["flags_footprint"] == 1)
#     data_frame = data_frame[detect_cuts]
#     print('Length of detection catalog after applying cuts: {}'.format(len(data_frame)))
#     return data_frame
#
#
# def airmass_cut(data_frame):
#     """"""
#     print('Cut mcal_detect_df_survey catalog so that AIRMASS_WMEAN_R is not null')
#     data_frame = data_frame[pd.notnull(data_frame["AIRMASS_WMEAN_R"])]
#     return data_frame

