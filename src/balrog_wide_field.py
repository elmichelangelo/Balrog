import pickle
import pandas as pd
from sklearn.preprocessing import PowerTransformer
# from sklearn.model_selection import train_test_split
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from Handler.helper_functions import *
from Handler.cut_functions import *
from Handler.plot_functions import *
import seaborn as sns


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


def create_balrog_subset(path_all_balrog_data, path_save, name_save_file, number_of_samples, only_detected,
                         apply_unsheared_cuts, apply_bdf_cuts, apply_fill_na, bdf_bins,
                         unsheared_bins, protocol=None, plot_data=False):
    """"""
    df_balrog = open_all_balrog_dataset(path_all_balrog_data)
    df_balrog.rename(
        columns={
            'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
            'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
        inplace=True)
    df_balrog["BDF_G"] = np.sqrt(df_balrog["BDF_G_0"] ** 2 + df_balrog["BDF_G_1"] ** 2)

    dict_fill_na = {
        # 'unsheared/snr': (-10, 2.0),
        'unsheared/T': (-10, 2.0),
        'unsheared/size_ratio': (-1, 2.0),
        # 'AIRMASS_WMEAN_R': (-10, 2.0),
        # 'AIRMASS_WMEAN_I': (-10, 2.0),
        # 'AIRMASS_WMEAN_Z': (-10, 2.0),
        # 'FWHM_WMEAN_R': (-10, 2.0),
        # 'FWHM_WMEAN_I': (-10, 2.0),
        # 'FWHM_WMEAN_Z': (-10, 2.0),
        # 'MAGLIM_R': (-10, 2.0),
        # 'MAGLIM_I': (-10, 2.0),
        # 'MAGLIM_Z': (-10, 2.0),
        # 'EBV_SFD98': (-10, 2.0),
        # 'unsheared/flags': (-10, 2.0),
        # 'unsheared/flux_r': (-20000, 2000),
        # 'unsheared/flux_i': (-20000, 2000),
        # 'unsheared/flux_z': (-20000, 2000),
        # 'unsheared/flux_err_r': (-10, 2.0),
        # 'unsheared/flux_err_i': (-10, 2.0),
        # 'unsheared/flux_err_z': (-10, 2.0),
        # 'unsheared/extended_class_sof': (-20, 2.0),
        # 'unsheared/flags_gold': (-10, 2.0),
        # 'unsheared/weight': (-10, 2.0)
    }

    dict_replace_defaults = {
        # 'unsheared/snr': (-7070.360705084288, -2, 2.0),
        'unsheared/T': (-9999, -2, 2.0),
        # 'AIRMASS_WMEAN_R': (-9999, -2, 2.0),
        # 'AIRMASS_WMEAN_I': (-9999, -2, 2.0),
        # 'AIRMASS_WMEAN_Z': (-9999, -2, 2.0),
        # 'FWHM_WMEAN_R': (-9999, -2, 2.0),
        # 'FWHM_WMEAN_I': (-9999, -2, 2.0),
        # 'FWHM_WMEAN_Z': (-9999, -2, 2.0),
        # 'MAGLIM_R': (-9999, -2, 2.0),
        # 'MAGLIM_I': (-9999, -2, 2.0),
        # 'MAGLIM_Z': (-9999, -2, 2.0)
    }
    print(df_balrog.isna().sum())
    if apply_fill_na == "Gauss":
        for col in dict_fill_na.keys():
            df_balrog[col] = df_balrog[col].apply(
                replace_nan_with_gaussian, args=(dict_fill_na[col][0], dict_fill_na[col][1]))
        df_balrog = replace_values_with_gaussian(df_balrog, dict_replace_defaults)
    elif apply_fill_na == "Default":
        for col in dict_fill_na.keys():
            df_balrog[col].fillna(dict_fill_na[col][0])
        df_balrog = replace_values(df_balrog, dict_replace_defaults)

    print(df_balrog.isna().sum())

    df_balrog, lst_bdf_color_cols_lupt, lst_bdf_color_parameter_lupt, lst_bdf_mag_cols_lupt, lst_bdf_mag_parameter_lupt = calc_color(
        data_frame=df_balrog,
        mag_type=("LUPT", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
        bins=bdf_bins,
        plot_data=plot_data
    )
    df_balrog, lst_unsheared_color_cols_lupt, lst_unsheared_color_parameter_lupt, lst_unsheared_mag_cols_lupt,\
        lst_unsheared_mag_parameter_lupt = calc_color(
        data_frame=df_balrog,
        mag_type=("LUPT", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/lupt", "unsheared/lupt_err"),
        bins=unsheared_bins,
        plot_data=plot_data
    )
    df_balrog, lst_bdf_color_cols_mag, lst_bdf_color_parameter_mag, lst_bdf_mag_cols_mag, lst_bdf_mag_parameter_mag = calc_color(
        data_frame=df_balrog,
        mag_type=("MAG", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
        bins=bdf_bins,
        plot_data=plot_data
    )
    df_balrog, lst_unsheared_color_cols_mag, lst_unsheared_mag_cols_mag, lst_unsheared_color_parameter_mag, \
        lst_unsheared_mag_parameter_mag = calc_color(
        data_frame=df_balrog,
        mag_type=("MAG", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/mag", "unsheared/mag_err"),
        bins=unsheared_bins,
        plot_data=plot_data
    )

    print(f"length of all balrog objects {len(df_balrog)}")
    if only_detected is True:
        df_balrog = df_balrog[df_balrog["detected"] == 1]
        print(f"length of only detected balrog objects {len(df_balrog)}")
    plot_histo(
        data_frame=df_balrog,
        cols=["unsheared/size_ratio"]
    )
    if apply_unsheared_cuts is True:
        df_balrog_cut = unsheared_object_cuts(df_balrog)
        df_balrog_cut = flag_cuts(df_balrog_cut)
        df_balrog_cut = unsheared_mag_cut(df_balrog_cut)
        df_balrog_cut = unsheared_shear_cuts(df_balrog_cut)
        df_balrog_cut = airmass_cut(df_balrog_cut)
        print(f"length of catalog after applying unsheared cuts {len(df_balrog_cut)}")
    plot_histo(
        data_frame=df_balrog_cut,
        cols=["unsheared/size_ratio"]
    )
    df_balrog, dict_pt = replace_and_transform_data(
        data_frame=df_balrog,
        columns=["unsheared/size_ratio"]
    )
    df_balrog_cut, dict_pt = replace_and_transform_data(
        data_frame=df_balrog_cut,
        columns=["unsheared/size_ratio"]
    )
    df_balrog_subset = df_balrog.sample(number_of_samples, ignore_index=True, replace=False)
    df_balrog_cut_subset = df_balrog_cut.sample(number_of_samples, ignore_index=True, replace=False)
    plot_histo(
        data_frame=df_balrog_subset,
        cols=["unsheared/size_ratio"]
    )
    plot_histo(
        data_frame=df_balrog_cut_subset,
        cols=["unsheared/size_ratio"]
    )
    exit()
    if plot_data is True:
        plot_chain(
            data_frame=df_balrog,
            columns=lst_bdf_mag_cols_lupt,
            parameter=lst_bdf_mag_parameter_lupt,
            plot_name=f"bdf_mag_lupt",
        )
        plot_chain(
            data_frame=df_balrog,
            columns=lst_bdf_color_cols_lupt,
            parameter=lst_bdf_color_parameter_lupt,
            plot_name="bdf_color_plot_lupt",
        )
        plot_chain(
            data_frame=df_balrog,
            columns=lst_unsheared_mag_cols_lupt,
            parameter=lst_unsheared_mag_parameter_lupt,
            plot_name=f"unsheared_mag_lupt",
        )
        plot_chain(
            data_frame=df_balrog,
            columns=lst_unsheared_color_cols_lupt,
            parameter=lst_unsheared_color_parameter_lupt,
            plot_name="unsheared_color_plot_lupt",
        )
        plot_chain(
            data_frame=df_balrog,
            columns=lst_bdf_mag_cols_mag,
            parameter=lst_bdf_mag_parameter_mag,
            plot_name=f"bdf_mag_mag",
        )
        plot_chain(
            data_frame=df_balrog,
            columns=lst_bdf_color_cols_mag,
            parameter=lst_bdf_color_parameter_mag,
            plot_name="bdf_color_plot_mag",
        )
        plot_chain(
            data_frame=df_balrog,
            columns=lst_unsheared_mag_cols_mag,
            parameter=lst_unsheared_mag_parameter_mag,
            plot_name=f"unsheared_mag_mag",
        )
        plot_chain(
            data_frame=df_balrog,
            columns=lst_unsheared_color_cols_mag,
            parameter=lst_unsheared_color_parameter_mag,
            plot_name="unsheared_color_plot_mag",
        )

    if number_of_samples is None:
        number_of_samples = len(df_balrog)
    df_balrog_subset = df_balrog.sample(number_of_samples, ignore_index=True, replace=False)
    if plot_data is True:
        plot_histo(data_frame=df_balrog_subset)

    path_balrog_subset = f"{path_save}/{name_save_file}_{number_of_samples}.pkl"

    save_balrog_subset(
        data_frame=df_balrog_subset,
        path_balrog_subset=path_balrog_subset,
        protocol=protocol
    )


if __name__ == '__main__':
    path = os.path.abspath(sys.path[0])
    no_samples = None  # int(250000)  # int(250000) # 20000 # int(8E6)
    create_balrog_subset(
        path_all_balrog_data=f"{path}/../Data/balrog_catalog_mcal_detect_deepfield_weights_21558485.pkl",
        path_save = f"{path}/../Output",
        name_save_file="balrog_all_w_undetected",
        number_of_samples=no_samples,
        only_detected=False,
        protocol=None
    )
