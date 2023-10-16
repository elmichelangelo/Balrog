import pandas as pd
from Handler.cut_functions import *
from Handler.plot_functions import *
from Handler.helper_functions import *
from Handler import *
import seaborn as sns
import os
import sys


def create_balrog_subset(path_all_balrog_data, path_save, path_log, name_save_file, number_of_samples, only_detected,
                         apply_fill_na, apply_replace_defaults, apply_object_cut, apply_flag_cut,
                         apply_unsheared_mag_cut, apply_unsheared_shear_cut, apply_airmass_cut, apply_binary_cut,
                         apply_mask_cut, bdf_bins, unsheared_bins, fill_na, replace_defaults,
                         protocol=None, plot_color=False, path_master=None):
    """"""
    # Initialize the logger
    start_window_logger = LoggerHandler(
        logger_dict={"logger_name": "start window",
                     "info_logger": INFO_LOGGER,
                     "error_logger": ERROR_LOGGER,
                     "debug_logger": DEBUG_LOGGER,
                     "stream_logger": STREAM_LOGGER,
                     "stream_logging_level": LOGGING_LEVEL},
        log_folder_path=path_log
    )

    # Get the list of loggers
    lst_of_loggers = start_window_logger.lst_of_loggers

    # Write status to logger
    for log in lst_of_loggers:
        log.info("Start Create balrog subset")
        log.info(f"open balrog data {path_all_balrog_data}")
    df_balrog = open_all_balrog_dataset(path_all_balrog_data)
    for log in lst_of_loggers:
        log.info("Rename col's: BDF_FLUX_DERED_CALIB_KS: BDF_FLUX_DERED_CALIB_ and "
                 "BDF_FLUX_ERR_DERED_CALIB_KS: BDF_FLUX_ERR_DERED_CALIB_K")
    df_balrog.rename(
        columns={
            'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
            'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
        inplace=True)
    for log in lst_of_loggers:
        log.info("Calc BDF_G=np.sqrt(BDF_G_0** 2 + BDF_G_1 ** 2)")
    df_balrog["BDF_G"] = np.sqrt(df_balrog["BDF_G_0"] ** 2 + df_balrog["BDF_G_1"] ** 2)

    print(df_balrog.isna().sum())
    print(df_balrog.isna().sum().sum())
    for log in lst_of_loggers:
        log.info(df_balrog.isna().sum())
        log.info(df_balrog.isna().sum().sum())

    if apply_fill_na == "Gauss":
        print(f"start fill na gauss")
        for log in lst_of_loggers:
            log.info(f"start fill na gauss")
        for col in fill_na.keys():
            df_balrog[col] = df_balrog[col].apply(
                replace_nan_with_gaussian, args=(fill_na[col][0], fill_na[col][1]))
    elif apply_fill_na == "Default":
        print(f"start fill na default")
        for log in lst_of_loggers:
            log.info(f"start fill na default")
        for col in fill_na.keys():
            df_balrog[col].fillna(fill_na[col][0], inplace=True)
            print(f"fill na default: col={col} val={fill_na[col][0]}")
            for log in lst_of_loggers:
                log.info(f"fill na default: col={col} val={fill_na[col][0]}")
    elif apply_fill_na is None:
        print("No fill na")
        for log in lst_of_loggers:
            log.info("No fill na")
    print(df_balrog.isna().sum())
    print(df_balrog.isna().sum().sum())
    for log in lst_of_loggers:
        log.info(df_balrog.isna().sum())
        log.info(df_balrog.isna().sum().sum())
    print(len(df_balrog))
    if apply_replace_defaults == "Gauss":
        print(f"start replace default gauss")
        for log in lst_of_loggers:
            log.info(f"start replace default gauss")
        df_balrog = replace_values_with_gaussian(df_balrog, replace_defaults)
    elif apply_replace_defaults == "Default":
        print(f"start replace default default")
        for log in lst_of_loggers:
            log.info(f"start replace default default")
        for col in replace_defaults.keys():
            print(f"replace defaults default: col={col} val=({replace_defaults[col][0]}, {replace_defaults[col][1]})")
            for log in lst_of_loggers:
                log.info(f"fill na default: col={col} val=({replace_defaults[col][0]}, {replace_defaults[col][1]})")
        df_balrog = replace_values(df_balrog, replace_defaults)
    elif apply_replace_defaults == "Drop":
        print("Drop defaults")
        for log in lst_of_loggers:
            log.info("Drop defaults")
        for col in replace_defaults.keys():
            print(f"replace defaults drop: col={col} val={replace_defaults[col][0]}")
            for log in lst_of_loggers:
                log.info(f"replace defaults drop: col={col} val={replace_defaults[col][0]}")
            indices_to_drop = df_balrog[df_balrog[col] == replace_defaults[col][0]].index
            df_balrog.drop(indices_to_drop, inplace=True)
    elif apply_replace_defaults is None:
        print("No default replace")
        for log in lst_of_loggers:
            log.info("No default replace")

    for log in lst_of_loggers:
        log.info("Calc color")
    df_balrog = calc_color(
        data_frame=df_balrog,
        mag_type=("LUPT", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
        bins=bdf_bins,
        plot_data=plot_color,
        plot_name=f"bdf_lupt"
    )
    df_balrog = calc_color(
        data_frame=df_balrog,
        mag_type=("LUPT", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/lupt", "unsheared/lupt_err"),
        bins=unsheared_bins,
        plot_data=plot_color,
        plot_name=f"unsheared/lupt"
    )
    df_balrog = calc_color(
        data_frame=df_balrog,
        mag_type=("MAG", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
        bins=bdf_bins,
        plot_data=plot_color,
        plot_name=f"bdf_mag"
    )
    df_balrog = calc_color(
        data_frame=df_balrog,
        mag_type=("MAG", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/mag", "unsheared/mag_err"),
        bins=unsheared_bins,
        plot_data=plot_color,
        plot_name=f"unsheared/mag"
    )

    for log in lst_of_loggers:
        log.info(f"length of all balrog objects {len(df_balrog)}")
    print(f"length of all balrog objects {len(df_balrog)}")
    if only_detected is True:
        df_balrog = df_balrog[df_balrog["detected"] == 1]
        print(f"length of only detected balrog objects {len(df_balrog)}")
        for log in lst_of_loggers:
            log.info(f"length of only detected balrog objects {len(df_balrog)}")
    if apply_object_cut is True:
        print(f"Only detected objects")
        for log in lst_of_loggers:
            log.info(f"Only detected objects")
        df_balrog = unsheared_object_cuts(data_frame=df_balrog)
    if apply_flag_cut is True:
        print(f"Flag cuts")
        for log in lst_of_loggers:
            log.info(f"Flag cuts")
        df_balrog = flag_cuts(data_frame=df_balrog)
    if apply_unsheared_mag_cut is True:
        print(f"mag cuts")
        for log in lst_of_loggers:
            log.info(f"mag cuts")
        df_balrog = unsheared_mag_cut(data_frame=df_balrog)
    if apply_unsheared_shear_cut is True:
        print(f"shear cuts")
        for log in lst_of_loggers:
            log.info(f"shear cuts")
        df_balrog = unsheared_shear_cuts(data_frame=df_balrog)
    if apply_airmass_cut is True:
        print(f"airmass cuts")
        for log in lst_of_loggers:
            log.info(f"airmass cuts")
        df_balrog = airmass_cut(data_frame=df_balrog)
    if apply_binary_cut is True:
        print(f"binary cuts")
        for log in lst_of_loggers:
            log.info(f"binary cuts")
        df_balrog = binary_cut(data_frame=df_balrog)
    if apply_mask_cut is True:
        print(f"mask cuts")
        for log in lst_of_loggers:
            log.info(f"mask cuts")
        df_balrog = mask_cut(data_frame=df_balrog, master=path_master)
    print(f"length of catalog after applying unsheared cuts {len(df_balrog)}")
    for log in lst_of_loggers:
        log.info(f"length of catalog after applying unsheared cuts {len(df_balrog)}")

    if number_of_samples is None:
        number_of_samples = len(df_balrog)
    for log in lst_of_loggers:
        log.info(f"Number of samples {number_of_samples}")
    df_balrog_subset = df_balrog.sample(number_of_samples, ignore_index=True, replace=False)

    for k in df_balrog_subset.keys():
        print(k)

    save_balrog_subset(
        data_frame=df_balrog_subset,
        path_balrog_subset=f"{path_save}/Catalogs/{name_save_file}_{number_of_samples}.pkl",
        protocol=protocol,
        lst_of_loggers=lst_of_loggers
    )


if __name__ == '__main__':
    path = os.path.abspath(sys.path[0])
    path_data = "/Users/P.Gebhardt/Development/PhD/data"
    path_output = "/Users/P.Gebhardt/Development/PhD/output/Balrog"
    no_samples = int(3E6)  # int(3E6)  # int(3E6)  # int(8E6)

    dict_fill_na = {
        'unsheared/snr': (-10, 2.0),
        'unsheared/T': (-10, 2.0),
        'unsheared/size_ratio': (-10, 5.0),
        "unsheared/e_1": (-10, 2.0),
        "unsheared/e_2": (-10, 2.0),
        'unsheared/flags': (-10, 2.0),
        'unsheared/flux_r': (-20000, 2000),
        'unsheared/flux_i': (-20000, 2000),
        'unsheared/flux_z': (-20000, 2000),
        'unsheared/flux_err_r': (-10, 2.0),
        'unsheared/flux_err_i': (-10, 2.0),
        'unsheared/flux_err_z': (-10, 2.0),
        'unsheared/extended_class_sof': (-20, 2.0),
        'unsheared/flags_gold': (-10, 2.0),
        'unsheared/weight': (-10, 2.0)
    }

    dict_replace_defaults = {
        'unsheared/snr': (-7070.360705084288, -2, 2.0),
        'unsheared/T': (-9999, -2, 2.0),
        "unsheared/e_1": (-9999, -5, 2.0),
        "unsheared/e_2": (-9999, -5, 2.0),
        'AIRMASS_WMEAN_R': (-9999, -2, 2.0),
        'AIRMASS_WMEAN_I': (-9999, -2, 2.0),
        'AIRMASS_WMEAN_Z': (-9999, -2, 2.0),
        'FWHM_WMEAN_R': (-9999, -2, 2.0),
        'FWHM_WMEAN_I': (-9999, -2, 2.0),
        'FWHM_WMEAN_Z': (-9999, -2, 2.0),
        'MAGLIM_R': (-9999, -2, 2.0),
        'MAGLIM_I': (-9999, -2, 2.0),
        'MAGLIM_Z': (-9999, -2, 2.0)
    }

    create_balrog_subset(
        path_all_balrog_data=f"{path_data}/balrog_training_no_cuts_20208363.pkl",
        path_save=path_output,
        path_log=f"{path_output}/Logs/",
        name_save_file="gandalf_training_data_odet_ncuts_ndef_rnan",
        number_of_samples=no_samples,
        only_detected=True,
        apply_fill_na="Default",  # "Default"
        apply_replace_defaults=None,  # "Default"
        apply_object_cut=False,
        apply_flag_cut=False,
        apply_unsheared_mag_cut=False,
        apply_unsheared_shear_cut=False,
        apply_airmass_cut=False,
        apply_binary_cut=False,
        apply_mask_cut=False,
        bdf_bins=["U", "G", "R", "I", "Z", "J", "H", "K"],
        unsheared_bins=["r", "i", "z"],
        fill_na=dict_fill_na,
        replace_defaults=dict_replace_defaults,
        protocol=2,
        plot_color=False,
        path_master=f"{path_data}/Y3_mastercat_03_31_20.h5"
    )
