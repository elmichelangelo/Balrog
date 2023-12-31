import logging
import joblib
import numpy as np
import pandas as pd
from Handler.cut_functions import *
from Handler.plot_functions import *
from Handler.helper_functions import *
from Handler import *
from sklearn.preprocessing import PowerTransformer, MaxAbsScaler, MinMaxScaler, StandardScaler
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

    mock_prefix = ""
    if cfg['USE_MOCK'] is True:
        for log in lst_of_loggers:
            log.info("Start Create balrog subset")
            log.info(f"open balrog data {cfg['PATH_OUTPUT']}Catalogs/{cfg['FILENAME_MOCK_CAT']}")
        df_balrog = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MOCK_CAT']}")
        mock_prefix = "_mock"
    else:
        for log in lst_of_loggers:
            log.info("Start Create balrog subset")
            log.info(f"open balrog data {cfg['PATH_OUTPUT']}Catalogs/{cfg['FILENAME_MERGED_CAT']}")
        df_balrog = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MERGED_CAT']}")
    for log in lst_of_loggers:
        log.info("Rename col's: BDF_FLUX_DERED_CALIB_KS: BDF_FLUX_DERED_CALIB_ and "
                 "BDF_FLUX_ERR_DERED_CALIB_KS: BDF_FLUX_ERR_DERED_CALIB_K")

    df_balrog.rename(
        columns={
            'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
            'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
        inplace=True)

    if cfg['USE_MOCK'] is not True:
        for log in lst_of_loggers:
            log.info("Calc BDF_G=np.sqrt(BDF_G_0** 2 + BDF_G_1 ** 2)")
        df_balrog["BDF_G"] = np.sqrt(df_balrog["BDF_G_0"] ** 2 + df_balrog["BDF_G_1"] ** 2)

    print(df_balrog.isna().sum())
    print(df_balrog.isna().sum().sum())
    for log in lst_of_loggers:
        log.info(df_balrog.isna().sum())
        log.info(df_balrog.isna().sum().sum())

    df_balrog_only_detected = df_balrog.copy()

    if cfg['APPLY_FILL_NA'] is True and cfg['USE_MOCK'] is not True:
        print(f"start fill na default")
        for log in lst_of_loggers:
            log.info(f"start fill na default")
        for col in cfg['FILL_NA'].keys():
            df_balrog[col].fillna(cfg['FILL_NA'][col], inplace=True)
            df_balrog_only_detected[col].fillna(cfg['FILL_NA'][col], inplace=True)
            print(f"fill na default: col={col} val={cfg['FILL_NA'][col]}")
            for log in lst_of_loggers:
                log.info(f"fill na default: col={col} val={cfg['FILL_NA'][col]}")
    else:
        print("No fill na")
        for log in lst_of_loggers:
            log.info("No fill na")
    if cfg["USE_MOCK"] is not True:
        df_balrog_only_detected = df_balrog_only_detected[df_balrog_only_detected["detected"] == 1]
        print(f"length of only detected balrog objects {len(df_balrog_only_detected)}")
    for log in lst_of_loggers:
        log.info(f"length of only detected balrog objects {len(df_balrog)}")

    print(df_balrog.isna().sum())
    print(df_balrog.isna().sum().sum())
    if cfg["USE_MOCK"] is not True:
        print(df_balrog_only_detected.isna().sum())
        print(df_balrog_only_detected.isna().sum().sum())
    for log in lst_of_loggers:
        log.info(df_balrog.isna().sum())
        log.info(df_balrog.isna().sum().sum())
    print(len(df_balrog))

    if cfg["USE_MOCK"] is True:
        pass
    else:
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

        df_balrog_only_detected = calc_color(
            cfg=cfg,
            data_frame=df_balrog_only_detected,
            mag_type=("LUPT", "BDF"),
            flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
            mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
            bins=cfg['BDF_BINS'],
            save_name=f"bdf_lupt"
        )
        df_balrog_only_detected = calc_color(
            cfg=cfg,
            data_frame=df_balrog_only_detected,
            mag_type=("LUPT", "unsheared"),
            flux_col=("unsheared/flux", "unsheared/flux_err"),
            mag_col=("unsheared/lupt", "unsheared/lupt_err"),
            bins=cfg['UNSHEARED_BINS'],
            save_name=f"unsheared/lupt"
        )
        df_balrog_only_detected = calc_color(
            cfg=cfg,
            data_frame=df_balrog_only_detected,
            mag_type=("MAG", "BDF"),
            flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
            mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
            bins=cfg['BDF_BINS'],
            save_name=f"bdf_mag"
        )
        df_balrog_only_detected = calc_color(
            cfg=cfg,
            data_frame=df_balrog_only_detected,
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
        if cfg["USE_MOCK"] is not True:
            df_balrog_only_detected = unsheared_object_cuts(data_frame=df_balrog_only_detected)
    if cfg['CUT_FLAG'] is True:
        print(f"Flag cuts")
        for log in lst_of_loggers:
            log.info(f"Flag cuts")
        df_balrog = flag_cuts(data_frame=df_balrog)
        if cfg["USE_MOCK"] is not True:
            df_balrog_only_detected = flag_cuts(data_frame=df_balrog_only_detected)
    if cfg['CUT_MAG'] is True:
        print(f"mag cuts")
        for log in lst_of_loggers:
            log.info(f"mag cuts")
        df_balrog = unsheared_mag_cut(data_frame=df_balrog)
        if cfg["USE_MOCK"] is not True:
            df_balrog_only_detected = unsheared_mag_cut(data_frame=df_balrog_only_detected)
    if cfg['CUT_SHEAR'] is True:
        print(f"shear cuts")
        for log in lst_of_loggers:
            log.info(f"shear cuts")
        df_balrog = unsheared_shear_cuts(data_frame=df_balrog)
        if cfg["USE_MOCK"] is not True:
            df_balrog_only_detected = unsheared_shear_cuts(data_frame=df_balrog_only_detected)
    if cfg['CUT_AIRMASS'] is True:
        print(f"airmass cuts")
        for log in lst_of_loggers:
            log.info(f"airmass cuts")
        df_balrog = airmass_cut(data_frame=df_balrog)
        if cfg["USE_MOCK"] is not True:
            df_balrog_only_detected = airmass_cut(data_frame=df_balrog_only_detected)
    if cfg['CUT_BINARY'] is True:
        print(f"binary cuts")
        for log in lst_of_loggers:
            log.info(f"binary cuts")
        df_balrog = binary_cut(data_frame=df_balrog)
        if cfg["USE_MOCK"] is not True:
            df_balrog_only_detected = binary_cut(data_frame=df_balrog_only_detected)
    if cfg['CUT_MASK'] is True:
        print(f"mask cuts")
        for log in lst_of_loggers:
            log.info(f"mask cuts")
        df_balrog = mask_cut(data_frame=df_balrog, master=cfg['PATH_DATA']+cfg['FILENAME_MASTER_CAT'])
        if cfg["USE_MOCK"] is not True:
            df_balrog_only_detected = mask_cut(data_frame=df_balrog_only_detected, master=cfg['PATH_DATA']+cfg['FILENAME_MASTER_CAT'])
    print(f"length of catalog after cut section {len(df_balrog)}")
    if cfg["USE_MOCK"] is not True:
        print(f"length of only detected catalog after cut section {len(df_balrog_only_detected)}")
    for log in lst_of_loggers:
        log.info(f"length of catalog after cut section {len(df_balrog)}")
        if cfg["USE_MOCK"] is not True:
            log.info(f"length of only detected catalog after cut section {len(df_balrog_only_detected)}")

    def get_yj_transformer(data_frame, columns):
        """"""
        dict_pt = {}
        for col in columns:
            pt = PowerTransformer(method="yeo-johnson")
            pt.fit(np.array(data_frame[col]).reshape(-1, 1))
            data_frame[col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
            dict_pt[f"{col} pt"] = pt
        return data_frame, dict_pt

    df_balrog_yj = df_balrog.copy()
    if cfg["USE_MOCK"] is not True:
        df_balrog_only_detected_yj = df_balrog_only_detected.copy()
        df_balrog_yj, dict_balrog_yj_transformer = get_yj_transformer(
            data_frame=df_balrog_yj,
            columns=cfg['YJ_TRANSFORM_COLS']
        )
        df_balrog_only_detected_yj, dict_balrog_only_detected_yj_transformer = get_yj_transformer(
            data_frame=df_balrog_only_detected_yj,
            columns=cfg['YJ_TRANSFORM_COLS']
        )
        joblib.dump(
            dict_balrog_yj_transformer,
            f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['FILENAME_YJ_TRANSFORMER']}.joblib"
        )
        joblib.dump(
            dict_balrog_only_detected_yj_transformer,
            f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['FILENAME_YJ_TRANSFORMER_ONLY_DETECTED']}.joblib"
        )
    else:
        df_balrog_yj, dict_balrog_yj_transformer = get_yj_transformer(
            data_frame=df_balrog_yj,
            columns=cfg['YJ_TRANSFORM_COLS_MOCK']
        )
        joblib.dump(
            dict_balrog_yj_transformer,
            f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['FILENAME_YJ_TRANSFORMER']}{mock_prefix}.joblib"
        )

    def get_scaler(data_frame):
        """"""
        if cfg[f"SCALER"] == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif cfg[f"SCALER"] == "MaxAbsScaler":
            scaler = MaxAbsScaler()
        elif cfg[f"SCALER"] == "StandardScaler":
            scaler = StandardScaler()
        else:
            raise TypeError(f'{cfg[f"SCALER"]} is no valid scaler')
        if scaler is not None:
            scaler.fit(data_frame)
        return scaler

    if cfg['USE_MOCK'] is True:
        dict_data_frames = {
            "scaler_balrog_mag_yj": (
                df_balrog_yj[cfg['SCALER_COLS_MAG']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_mag_yj{mock_prefix}.joblib"
            ),
        }
    else:
        dict_data_frames = {
            "scaler_balrog_flux_yj": (
                df_balrog_yj[cfg['SCALER_COLS_FLUX']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_flux_yj.joblib"
            ),
            "scaler_balrog_only_detected_flux_yj": (
                df_balrog_only_detected_yj[cfg['SCALER_COLS_FLUX']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_flux_yj.joblib"
            ),
            "scaler_balrog_mag_yj": (
                df_balrog_yj[cfg['SCALER_COLS_MAG']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_mag_yj.joblib"
            ),
            "scaler_balrog_only_detected_mag_yj": (
                df_balrog_only_detected_yj[cfg['SCALER_COLS_MAG']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_mag_yj.joblib"
            ),
            "scaler_balrog_lupt_yj": (
                df_balrog_yj[cfg['SCALER_COLS_LUPT']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_lupt_yj.joblib"
            ),
            "scaler_balrog_only_detected_lup_yjt": (
                df_balrog_only_detected_yj[cfg['SCALER_COLS_LUPT']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_lupt_yj.joblib"
            ),
        }

    for key in dict_data_frames.keys():
        scaler = get_scaler(
            data_frame=dict_data_frames[key][0]
        )
        joblib.dump(
            scaler,
            dict_data_frames[key][1]
        )

    del df_balrog_yj
    if cfg["USE_MOCK"] is not True:
        del df_balrog_only_detected_yj

    if cfg['USE_MOCK'] is True:
        dict_data_frames = {
            "scaler_balrog_mag": (
                df_balrog[cfg['SCALER_COLS_MAG']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_mag{mock_prefix}.joblib"
            ),
        }
    else:
        dict_data_frames = {
            "scaler_balrog_flux": (
                df_balrog[cfg['SCALER_COLS_FLUX']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_flux.joblib"
            ),
            "scaler_balrog_only_detected_flux": (
                df_balrog_only_detected[cfg['SCALER_COLS_FLUX']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_flux.joblib"
            ),
            "scaler_balrog_mag": (
                df_balrog[cfg['SCALER_COLS_MAG']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_mag.joblib"
            ),
            "scaler_balrog_only_detected_mag": (
                df_balrog_only_detected[cfg['SCALER_COLS_MAG']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_mag.joblib"
            ),
            "scaler_balrog_lupt": (
                df_balrog[cfg['SCALER_COLS_LUPT']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_lupt.joblib"
            ),
            "scaler_balrog_only_detected_lupt": (
                df_balrog_only_detected[cfg['SCALER_COLS_LUPT']],
                f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_lupt.joblib"
            ),
        }
    for key in dict_data_frames.keys():
        scaler = get_scaler(
            data_frame=dict_data_frames[key][0]  # df_balrog[cfg['SCALER_COLS_FLUX']]
        )
        joblib.dump(
            scaler,
            dict_data_frames[key][1]
        )

    number_of_samples = cfg['NSAMPLES']
    if number_of_samples is None:
        number_of_samples = len(df_balrog)
    for log in lst_of_loggers:
        log.info(f"Number of samples {number_of_samples}")
    df_balrog_subset = df_balrog.sample(
        number_of_samples,
        ignore_index=True,
        replace=False
    )
    save_balrog_subset(
        cfg=cfg,
        data_frame=df_balrog_subset,
        save_name_train=cfg['FILENAME_GANDALF_TRAIN_CAT'],
        save_name_valid=cfg['FILENAME_GANDALF_VALIDATION_CAT'],
        save_name_test=cfg['FILENAME_GANDALF_TEST_CAT'],
        lst_of_loggers=lst_of_loggers,
        mock_prefix=mock_prefix
    )

    if cfg["USE_MOCK"] is not True:
        number_of_samples_only_detected = cfg['NSAMPLES_ONLY_DETECTED']
        if number_of_samples_only_detected is None:
            number_of_samples_only_detected = len(df_balrog_only_detected)
        for log in lst_of_loggers:
            log.info(f"Number of samples only detected {number_of_samples_only_detected}")
        df_balrog_only_detected_subset = df_balrog_only_detected.sample(
            number_of_samples_only_detected,
            ignore_index=True,
            replace=False
        )
        save_balrog_subset(
            cfg=cfg,
            data_frame=df_balrog_only_detected_subset,
            save_name_train=cfg['FILENAME_GANDALF_TRAIN_CAT_ONLY_DETECTED'],
            save_name_valid=cfg['FILENAME_GANDALF_VALIDATION_CAT_ONLY_DETECTED'],
            save_name_test=cfg['FILENAME_GANDALF_TEST_CAT_ONLY_DETECTED'],
            lst_of_loggers=lst_of_loggers,
            mock_prefix=mock_prefix
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
