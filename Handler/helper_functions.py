import pickle
import numpy as np
from sklearn.preprocessing import PowerTransformer
import pandas as pd
from Handler.plot_functions import *
"""import warnings

warnings.filterwarnings("error")"""


def load_healpix(path2file, hp_show=False, nest=True, partial=False, field=None):
    """
    # Function to load fits datasets
    # Returns:

    """
    """
    import healpy as hp
    if field is None:
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
        
    return hp_map
    """


def match_skybrite_2_footprint(path2footprint, path2skybrite, hp_show=False, nest_footprint=True, nest_skybrite=True,
                               partial_footprint=False, partial_skybrite=True, field_footprint=None,
                               field_skybrite=None):
    """
    Main function to run
    Returns:

    """
    """
    import healpy as hp
    hp_map_footprint = load_healpix(
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
    return np.column_stack((good_indices, sky_in_footprint[1]))
    """


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


def calc_color(data_frame, mag_type, flux_col, mag_col, bins, plot_name, plot_data=False):
    """"""

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
    if plot_data is True:
        plot_chain(
            data_frame=data_frame,
            columns=lst_color_cols,
            parameter=lst_color_parameter,
            plot_name=f"color_{plot_name}"
        )
        plot_chain(
            data_frame=data_frame,
            columns=lst_mag_cols,
            parameter=lst_mag_parameter,
            plot_name=f"mag_{plot_name}"
        )
    return data_frame


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


# def replace_values(data_frame, replace_value):
#     for col in replace_value.keys():
#         replace_value_index = None if replace_value[col] == "None" else replace_value[col]
#         if replace_value_index is not None:
#             replace_value_tuple = eval(replace_value_index)
#             data_frame[col] = data_frame[col].replace(replace_value_tuple[0], replace_value_tuple[1])
#     return data_frame


def unreplace_values(data_frame, replace_value):
    for col in replace_value.keys():
        if replace_value[col] is not None:
            data_frame[col] = data_frame[col].replace(replace_value[col][1], replace_value[col][0])
    return data_frame


def yj_transform_data(data_frame, columns):
    """"""
    dict_pt = {}
    for col in columns:
        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(np.array(data_frame[col]).reshape(-1, 1))
        data_frame[col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
        dict_pt[f"{col} pt"] = pt
    return data_frame, dict_pt


def yj_inverse_transform_data(data_frame, dict_pt, columns):
    """"""
    for col in columns:
        pt = dict_pt[f"{col} pt"]
        data_frame[col] = pt.inverse_transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
    return data_frame


def mag2flux(magnitude, zero_pt=30):
    # convert magnitude to flux
    try:
        flux = 10**((zero_pt-magnitude)/2.5)
        return flux
    except RuntimeWarning:
        print("Warning")


def flux2mag(flux, zero_pt=30, clip=0.001):
    # convert flux to magnitude
    if clip is None:
        return zero_pt - 2.5 * np.log10(flux)
    return zero_pt - 2.5 * np.log10(flux.clip(clip))


def open_all_balrog_dataset(path_all_balrog_data):
    """"""
    infile = open(path_all_balrog_data, 'rb')
    # load pickle as pandas dataframe
    df_balrog = pd.DataFrame(pickle.load(infile, encoding='latin1'))
    # close file
    infile.close()
    return df_balrog


def save_balrog_subset(data_frame, path_balrog_subset, protocol):
    """"""
    if protocol == 2:
        with open(path_balrog_subset, "wb") as f:
            pickle.dump(data_frame.to_dict(), f, protocol=2)
    else:
        data_frame.to_pickle(path_balrog_subset)


def assign_loggrid( x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
    # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps
    x = np.maximum(x, xmin)
    x = np.minimum(x, xmax)
    y = np.maximum(y, ymin)
    y = np.minimum(y, ymax)
    logstepx = np.log10(xmax/xmin)/xsteps
    logstepy = np.log10(ymax/ymin)/ysteps
    indexx = (np.log10(x/xmin)/logstepx).astype(int)
    indexy = (np.log10(y/ymin)/logstepy).astype(int)
    indexx = np.minimum(indexx, xsteps-1)
    indexy = np.minimum(indexy, ysteps-1)
    return indexx,indexy


def apply_loggrid(x, y, grid, xmin=10, xmax=300, xsteps=20, ymin=0.5, ymax=5, ysteps=20):
    # step 2 - assign weight to each galaxy
    indexx, indexy = assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps)
    res = np.zeros(len(x))
    res = grid[indexx, indexy]
    return res
