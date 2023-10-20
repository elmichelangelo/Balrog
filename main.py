import logging

import pandas as pd
from Handler.cut_functions import *
from Handler.plot_functions import *
from Handler.helper_functions import *
from Handler import *
import seaborn as sns
import os
import sys
import yaml
import argparse


def create_balrog_subset(cfg):
    """"""
    # Initialize the logger

    log_lvl = logging.INFO
    if cfg["LOGGING_LEVEL"] == "DEBUG":
        log_lvl = logging.DEBUG
    elif cfg["LOGGING_LEVEL"] == "ERROR":
        log_lvl = logging.ERROR

    # Initialize the logger
    start_window_logger = LoggerHandler(
        logger_dict={"logger_name": "create balrog subset",
                     "info_logger": cfg['INFO_LOGGER'],
                     "error_logger": cfg['ERROR_LOGGER'],
                     "debug_logger": cfg['DEBUG_LOGGER'],
                     "stream_logger": cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{cfg['PATH_OUTPUT']}Logs/"
    )

    # Get the list of loggers
    lst_of_loggers = start_window_logger.lst_of_loggers
    # Write status to logger
    for log in lst_of_loggers:
        log.info("Start create balrog subset")

    # Write status to logger
    for log in lst_of_loggers:
        log.info("Start Create balrog subset")
        log.info(f"open balrog data {cfg['PATH_OUTPUT']}Catalogs/{cfg['FILENAME_MERGED_CAT']}")
    df_balrog = open_all_balrog_dataset(f"{cfg['PATH_OUTPUT']}Catalogs/{cfg['FILENAME_MERGED_CAT']}")
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

    if cfg['ONLY_DETECT'] is True:
        df_balrog = df_balrog[df_balrog["detected"] == 1]
        print(f"length of only detected balrog objects {len(df_balrog)}")
        for log in lst_of_loggers:
            log.info(f"length of only detected balrog objects {len(df_balrog)}")

    if cfg['APPLY_FILL_NA'] is True:
        print(f"start fill na default")
        for log in lst_of_loggers:
            log.info(f"start fill na default")
        for col in cfg['FILL_NA'].keys():
            df_balrog[col].fillna(cfg['FILL_NA'][col], inplace=True)
            print(f"fill na default: col={col} val={cfg['FILL_NA'][col]}")
            for log in lst_of_loggers:
                log.info(f"fill na default: col={col} val={cfg['FILL_NA'][col]}")
    else:
        print("No fill na")
        for log in lst_of_loggers:
            log.info("No fill na")

    print(df_balrog.isna().sum())
    print(df_balrog.isna().sum().sum())
    for log in lst_of_loggers:
        log.info(df_balrog.isna().sum())
        log.info(df_balrog.isna().sum().sum())
    print(len(df_balrog))

    for log in lst_of_loggers:
        log.info("Calc color")

    df_balrog = calc_color(
        cfg=cfg,
        data_frame=df_balrog,
        mag_type=("LUPT", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
        bins=cfg['BDF_BINS'],
        save_name=f"bdf_lupt"
    )
    df_balrog = calc_color(
        cfg=cfg,
        data_frame=df_balrog,
        mag_type=("LUPT", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/lupt", "unsheared/lupt_err"),
        bins=cfg['UNSHEARED_BINS'],
        save_name=f"unsheared/lupt"
    )
    df_balrog = calc_color(
        cfg=cfg,
        data_frame=df_balrog,
        mag_type=("MAG", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
        bins=cfg['BDF_BINS'],
        save_name=f"bdf_mag"
    )
    df_balrog = calc_color(
        cfg=cfg,
        data_frame=df_balrog,
        mag_type=("MAG", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/mag", "unsheared/mag_err"),
        bins=cfg['UNSHEARED_BINS'],
        save_name=f"unsheared/mag"
    )

    for log in lst_of_loggers:
        log.info(f"length of all balrog objects {len(df_balrog)}")
    print(f"length of all balrog objects {len(df_balrog)}")

    if cfg['CUT_OBJECT'] is True:
        print(f"Only detected objects")
        for log in lst_of_loggers:
            log.info(f"Only detected objects")
        df_balrog = unsheared_object_cuts(data_frame=df_balrog)
    if cfg['CUT_FLAG'] is True:
        print(f"Flag cuts")
        for log in lst_of_loggers:
            log.info(f"Flag cuts")
        df_balrog = flag_cuts(data_frame=df_balrog)
    if cfg['CUT_MAG'] is True:
        print(f"mag cuts")
        for log in lst_of_loggers:
            log.info(f"mag cuts")
        df_balrog = unsheared_mag_cut(data_frame=df_balrog)
    if cfg['CUT_SHEAR'] is True:
        print(f"shear cuts")
        for log in lst_of_loggers:
            log.info(f"shear cuts")
        df_balrog = unsheared_shear_cuts(data_frame=df_balrog)
    if cfg['CUT_AIRMASS'] is True:
        print(f"airmass cuts")
        for log in lst_of_loggers:
            log.info(f"airmass cuts")
        df_balrog = airmass_cut(data_frame=df_balrog)
    if cfg['CUT_BINARY'] is True:
        print(f"binary cuts")
        for log in lst_of_loggers:
            log.info(f"binary cuts")
        df_balrog = binary_cut(data_frame=df_balrog)
    if cfg['CUT_MASK'] is True:
        print(f"mask cuts")
        for log in lst_of_loggers:
            log.info(f"mask cuts")
        df_balrog = mask_cut(data_frame=df_balrog, master=cfg['PATH_DATA']+cfg['FILENAME_MASTER_CAT'])
    print(f"length of catalog after cut section {len(df_balrog)}")
    for log in lst_of_loggers:
        log.info(f"length of catalog after cut section {len(df_balrog)}")

    number_of_samples = cfg['NSAMPLES']
    if number_of_samples is None:
        number_of_samples = len(df_balrog)
    for log in lst_of_loggers:
        log.info(f"Number of samples {number_of_samples}")
    df_balrog_subset = df_balrog.sample(number_of_samples, ignore_index=True, replace=False)

    for k in df_balrog_subset.keys():
        print(k)

    save_balrog_subset(
        data_frame=df_balrog_subset,
        path_balrog_subset=f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['FILENAME_GANDALF_TRAIN_CAT']}_{number_of_samples}.pkl",
        protocol=cfg['PROTOCOL'],
        lst_of_loggers=lst_of_loggers
    )


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__))
    path = os.path.abspath(sys.path[-1])
    if get_os() == "Mac":
        config_file_name = "mac.cfg"
    elif get_os() == "Windows":
        config_file_name = "windows.cfg"
    elif get_os() == "Linux":
        config_file_name = "LMU.cfg"
    else:
        print(f"OS Error: {get_os()}")

    parser = argparse.ArgumentParser(description='Start create balrog training dataset')
    parser.add_argument(
        '--config_filename',
        "-cf",
        type=str,
        nargs=1,
        required=False,
        default=config_file_name,
        help='Name of config file'
    )
    args = parser.parse_args()

    if isinstance(args.config_filename, list):
        args.config_filename = args.config_filename[0]

    with open(f"{path}/config/{args.config_filename}", 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    create_balrog_subset(cfg=cfg)
