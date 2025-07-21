import logging
import joblib
import matplotlib.pyplot as plt
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

    # Write status to logger
    start_window_logger.log_info_stream("Start create balrog subset")

    mock_prefix = ""
    if cfg['USE_MOCK'] is True:
        start_window_logger.log_info_stream("Start Create balrog subset")
        start_window_logger.log_info_stream(f"open balrog data {cfg['PATH_OUTPUT']}Catalogs/{cfg['FILENAME_MOCK_CAT']}")
        df_balrog = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MOCK_CAT']}")
        mock_prefix = "_mock"
    else:
        start_window_logger.log_info_stream("Start Create balrog subset")
        start_window_logger.log_info_stream(f"open balrog data {cfg['PATH_OUTPUT']}Catalogs/{cfg['FILENAME_MERGED_CAT']}")
        df_balrog = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MERGED_CAT']}")
    start_window_logger.log_info_stream("Rename col's: BDF_FLUX_DERED_CALIB_KS: BDF_FLUX_DERED_CALIB_ and "
                 "BDF_FLUX_ERR_DERED_CALIB_KS: BDF_FLUX_ERR_DERED_CALIB_K")

    df_balrog.rename(
        columns={
            'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
            'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
        inplace=True)

    if cfg['USE_MOCK'] is not True:
        start_window_logger.log_info_stream("Calc BDF_G=np.sqrt(BDF_G_0** 2 + BDF_G_1 ** 2)")
        df_balrog["BDF_G"] = np.sqrt(df_balrog["BDF_G_0"] ** 2 + df_balrog["BDF_G_1"] ** 2)

    start_window_logger.log_info_stream(df_balrog.isna().sum())
    start_window_logger.log_info_stream(df_balrog.isna().sum().sum())

    if cfg['APPLY_FILL_NA'] is True and cfg['USE_MOCK'] is not True:
        start_window_logger.log_info_stream(f"start fill na default")
        df_balrog.loc[df_balrog['unsheared/e_1'] == -9999.0, 'unsheared/flux_err_r'] = df_balrog.loc[df_balrog['unsheared/e_1'] == -9999.0, 'unsheared/flux_err_r'].fillna(.5)
        df_balrog.loc[df_balrog['unsheared/e_1'] == -9999.0, 'unsheared/flux_err_i'] = df_balrog.loc[df_balrog['unsheared/e_1'] == -9999.0, 'unsheared/flux_err_i'].fillna(.5)
        df_balrog.loc[df_balrog['unsheared/e_1'] == -9999.0, 'unsheared/flux_err_z'] = df_balrog.loc[df_balrog['unsheared/e_1'] == -9999.0, 'unsheared/flux_err_z'].fillna(.5)
        # for col in cfg['FILL_NA'].keys():
        #     df_balrog[col].fillna(cfg['FILL_NA'][col], inplace=True)
        #     # df_balrog_only_detected[col].fillna(cfg['FILL_NA'][col], inplace=True)
        #     start_window_logger.log_info_stream(f"fill na default: col={col} val={cfg['FILL_NA'][col]}")
    else:
        start_window_logger.log_info_stream("No fill na")

    # df_balrog_only_detected = df_balrog[df_balrog['detected'] == 1].copy()

    # if cfg["USE_MOCK"] is not True:
    #     df_balrog_only_detected = df_balrog_only_detected[df_balrog_only_detected["detected"] == 1]
    #     start_window_logger.log_info_stream(f"length of only detected balrog objects {len(df_balrog_only_detected)}")
    start_window_logger.log_info_stream(f"length of Balrog objects {len(df_balrog)}")

    start_window_logger.log_info_stream(df_balrog.isna().sum())
    start_window_logger.log_info_stream(df_balrog.isna().sum().sum())
    # if cfg["USE_MOCK"] is not True:
    #     start_window_logger.log_info_stream(df_balrog_only_detected.isna().sum())
    #     start_window_logger.log_info_stream(df_balrog_only_detected.isna().sum().sum())
    start_window_logger.log_info_stream(df_balrog.isna().sum())
    start_window_logger.log_info_stream(df_balrog.isna().sum().sum())
    start_window_logger.log_info_stream(f"length of Balrog objects {len(df_balrog)}")
    if cfg["USE_MOCK"] is True:
        pass
    else:
        start_window_logger.log_info_stream("Calc color")

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


        if cfg["CHECK_DATA"] is True:
            plt.close('all')
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)

            for col in cfg["DEFAULT_COLUMNS"].keys():
                df_balrog[f"{col}_default_shifted"] = df_balrog[col].replace(cfg["DEFAULT_COLUMNS"][col][0], cfg["DEFAULT_COLUMNS"][col][1])

            for col in cfg["SHIFT_COLUMNS"]:
                col_min = df_balrog[col].min()
                df_balrog[f"{col}_{col_min:.4f}_shifted"] = df_balrog[col] + np.abs(col_min)

            for col in cfg["CLASSIFIER_COLUMNS_OF_INTEREST"]:
                save_name = col.replace("unsheared/", "")
                minmaxscaler = MinMaxScaler(feature_range=(-30, 30))

                if col in cfg["DEFAULT_COLUMNS"].keys():
                    value = df_balrog[f"{col}_default_shifted"].values
                elif col in cfg["SHIFT_COLUMNS"]:
                    col_min = df_balrog[col].min()
                    value = df_balrog[f"{col}_{col_min:.4f}_shifted"].values
                else:
                    value = df_balrog[col].values

                if col in cfg["COLUMNS_LOG1P"]:
                    log_value = np.log1p(value)
                    log_value_scaled = minmaxscaler.fit_transform(log_value.reshape(-1, 1)).flatten()

                    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
                    fig.suptitle(f"Histogramme für {col}", fontsize=14)

                    if col in list(cfg["DEFAULT_COLUMNS"].keys()) + cfg["SHIFT_COLUMNS"]:
                        origin_title = f"Shifted Value"
                    else:
                        origin_title = f"Original Value"
                    sns.histplot(value, stat="count", bins=100, ax=axes[0])
                    axes[0].set_title(origin_title)
                    axes[0].set_yscale("log")

                    sns.histplot(log_value, stat="count", bins=100, ax=axes[1])
                    axes[1].set_title("log1p(Value)")
                    axes[1].set_yscale("log")

                    sns.histplot(log_value_scaled, stat="count", bins=100, ax=axes[2])
                    axes[2].set_title("log1p(Value) Scaled")
                    axes[2].set_yscale("log")

                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data_new/classifier_{save_name}.pdf", dpi=300,
                                bbox_inches='tight')
                    plt.close(fig)
                else:
                    value_scaled = minmaxscaler.fit_transform(value.reshape(-1, 1)).flatten()

                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                    fig.suptitle(f"Histogramme für {col}", fontsize=14)

                    sns.histplot(value, stat="count", bins=100, ax=axes[0])
                    axes[0].set_title("Original Value")
                    axes[0].set_yscale("log")

                    sns.histplot(value_scaled, stat="count", bins=100, ax=axes[1])
                    axes[1].set_title("Value Scaled")
                    axes[1].set_yscale("log")

                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data_new/nf_{save_name}.pdf", dpi=300,
                                bbox_inches='tight')
                    plt.close(fig)

            df_balrog_only_detected = df_balrog[df_balrog['detected']==1].copy()

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

            for col in cfg["NF_COLUMNS_OF_INTEREST"]:
                save_name = col.replace("unsheared/", "")
                minmaxscaler = MinMaxScaler(feature_range=(-30, 30))

                if col in cfg["DEFAULT_COLUMNS"].keys():
                    value = df_balrog[f"{col}_default_shifted"].values
                elif col in cfg["SHIFT_COLUMNS"]:
                    col_min = df_balrog[col].min()
                    value = df_balrog[f"{col}_{col_min:.4f}_shifted"].values
                else:
                    value = df_balrog[col].values

                if col in cfg["COLUMNS_LOG1P"]:
                    log_value = np.log1p(value)
                    log_value_scaled = minmaxscaler.fit_transform(log_value.reshape(-1, 1)).flatten()

                    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
                    fig.suptitle(f"Histogramme für {col}", fontsize=14)

                    if col in list(cfg["DEFAULT_COLUMNS"].keys()) + cfg["SHIFT_COLUMNS"]:
                        origin_title = f"Shifted Value"
                    else:
                        origin_title = f"Original Value"
                    sns.histplot(value, stat="count", bins=100, ax=axes[0])
                    axes[0].set_title(origin_title)
                    axes[0].set_yscale("log")

                    sns.histplot(log_value, stat="count", bins=100, ax=axes[1])
                    axes[1].set_title("log1p(Value)")
                    axes[1].set_yscale("log")

                    sns.histplot(log_value_scaled, stat="count", bins=100, ax=axes[2])
                    axes[2].set_title("log1p(Value) Scaled")
                    axes[2].set_yscale("log")

                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    # plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data/{save_name}_all.pdf", dpi=300,
                    #             bbox_inches='tight')
                    plt.show()
                    plt.close()
                else:
                    value_scaled = minmaxscaler.fit_transform(value.reshape(-1, 1)).flatten()

                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                    fig.suptitle(f"Histogramme für {col}", fontsize=14)

                    sns.histplot(value, stat="count", bins=100, ax=axes[0])
                    axes[0].set_title("Original Value")
                    axes[0].set_yscale("log")

                    sns.histplot(value_scaled, stat="count", bins=100, ax=axes[1])
                    axes[1].set_title("Value Scaled")
                    axes[1].set_yscale("log")

                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    # plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data/{save_name}_all.pdf", dpi=300,
                    #             bbox_inches='tight')
                    plt.show()
                    plt.close()

            exit()

            for col in cfg["COLUMNS_OF_INTEREST"]:
                minmaxscaler = MinMaxScaler(feature_range=(-30, 30))
                value = df_balrog_only_detected[col].values
                save_name = col.replace("unsheared/", "")

                value_scaled = minmaxscaler.fit_transform(value.reshape(-1, 1)).flatten()

                if col in cfg["COLUMNS_LOG1P"]:
                    log_value = np.log1p(value)
                    log_value_scaled = minmaxscaler.fit_transform(log_value.reshape(-1, 1)).flatten()

                    # 4 Subplots (log1p-Fall)
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    fig.suptitle(f"Histogramme für {col}", fontsize=14)

                    sns.histplot(value, stat="density", bins=100, ax=axes[0, 0])
                    axes[0, 0].set_title("Original Value")
                    axes[0, 0].set_yscale("log")

                    sns.histplot(value_scaled, stat="density", bins=100, ax=axes[0, 1])
                    axes[0, 1].set_title("Value Scaled")
                    axes[0, 1].set_yscale("log")

                    sns.histplot(log_value, stat="density", bins=100, ax=axes[1, 0])
                    axes[1, 0].set_title("log1p(Value)")
                    axes[1, 0].set_yscale("log")

                    sns.histplot(log_value_scaled, stat="density", bins=100, ax=axes[1, 1])
                    axes[1, 1].set_title("log1p(Value) Scaled")
                    axes[1, 1].set_yscale("log")

                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data/{save_name}_all.pdf", dpi=300,
                                bbox_inches='tight')
                    plt.close()

                else:
                    # 2 Subplots (kein log1p)
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    fig.suptitle(f"Histogramme für {col}", fontsize=14)

                    sns.histplot(value, stat="density", bins=100, ax=axes[0])
                    axes[0].set_title("Original Value")
                    axes[0].set_yscale("log")

                    sns.histplot(value_scaled, stat="density", bins=100, ax=axes[1])
                    axes[1].set_title("Value Scaled")
                    axes[1].set_yscale("log")

                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data/{save_name}_all.pdf", dpi=300,
                                bbox_inches='tight')
                    plt.close()

            exit()

        if cfg['SAVE_SCALER'] is True:
            df_balrog_only_galaxies = df_balrog[df_balrog['unsheared/extended_class_sof'] > 1]
            df_balrog_only_detected_only_galaxies = df_balrog_only_galaxies[df_balrog_only_galaxies['detected'] == 1]
            scalers_odet = {}
            scalers_all = {}
            scalers_odet_galaxies = {}
            scalers_all_galaxies = {}

            for col in cfg["COLUMNS_OF_INTEREST"]:
                minmaxscaler_odet = MinMaxScaler(feature_range=(-30, 30))
                value_odet = df_balrog_only_detected[col].values
                if col in cfg["COLUMNS_LOG1P"]:
                    value_odet = np.log1p(value_odet)
                minmaxscaler_odet.fit(value_odet.reshape(-1, 1))
                scalers_odet[col] = minmaxscaler_odet

                minmaxscaler_all = MinMaxScaler(feature_range=(-30, 30))
                value_all = df_balrog[col].values
                if col in cfg["COLUMNS_LOG1P"]:
                    value_all = np.log1p(value_all)
                minmaxscaler_all.fit(value_all.reshape(-1, 1))
                scalers_all[col] = minmaxscaler_all

                minmaxscaler_odet_galaxies = MinMaxScaler(feature_range=(-30, 30))
                value_odet_galaxies = df_balrog_only_detected_only_galaxies[col].values
                if col in cfg["COLUMNS_LOG1P"]:
                    value_odet_galaxies = np.log1p(value_odet_galaxies)
                minmaxscaler_odet_galaxies.fit(value_odet_galaxies.reshape(-1, 1))
                scalers_odet_galaxies[col] = minmaxscaler_odet_galaxies

                minmaxscaler_all_galaxies = MinMaxScaler(feature_range=(-30, 30))
                value_all_galaxies = df_balrog_only_galaxies[col].values
                if col in cfg["COLUMNS_LOG1P"]:
                    value_all_galaxies = np.log1p(value_all_galaxies)
                minmaxscaler_all_galaxies.fit(value_all_galaxies.reshape(-1, 1))
                scalers_all_galaxies[col] = minmaxscaler_all_galaxies

            joblib.dump(scalers_odet, f"{cfg['PATH_OUTPUT']}MinMaxScalers_odet.pkl")
            joblib.dump(scalers_all, f"{cfg['PATH_OUTPUT']}MinMaxScalers_all.pkl")
            joblib.dump(scalers_odet_galaxies, f"{cfg['PATH_OUTPUT']}MinMaxScalers_odet_galaxies.pkl")
            joblib.dump(scalers_all_galaxies, f"{cfg['PATH_OUTPUT']}MinMaxScalers_all_galaxies.pkl")
            exit()

    start_window_logger.log_info_stream(f"length of all balrog objects {len(df_balrog)}")

    # if cfg['CUT_OBJECT'] is True:
    #     start_window_logger.log_info_stream(f"Only detected objects")
    #     df_balrog = unsheared_object_cuts(data_frame=df_balrog)
    #     if cfg["USE_MOCK"] is not True:
    #         df_balrog_only_detected = unsheared_object_cuts(data_frame=df_balrog_only_detected)
    # if cfg['CUT_FLAG'] is True:
    #     start_window_logger.log_info_stream(f"Flag Cuts")
    #     df_balrog = flag_cuts(data_frame=df_balrog)
    #     if cfg["USE_MOCK"] is not True:
    #         df_balrog_only_detected = flag_cuts(data_frame=df_balrog_only_detected)
    # if cfg['CUT_MAG'] is True:
    #     start_window_logger.log_info_stream(f"mag cuts")
    #     df_balrog = unsheared_mag_cut(data_frame=df_balrog)
    #     if cfg["USE_MOCK"] is not True:
    #         df_balrog_only_detected = unsheared_mag_cut(data_frame=df_balrog_only_detected)
    # if cfg['CUT_SHEAR'] is True:
    #     start_window_logger.log_info_stream(f"shear cuts")
    #     df_balrog = unsheared_shear_cuts(data_frame=df_balrog)
    #     if cfg["USE_MOCK"] is not True:
    #         df_balrog_only_detected = unsheared_shear_cuts(data_frame=df_balrog_only_detected)
    # if cfg['CUT_AIRMASS'] is True:
    #     start_window_logger.log_info_stream(f"airmass cuts")
    #     df_balrog = airmass_cut(data_frame=df_balrog)
    #     if cfg["USE_MOCK"] is not True:
    #         df_balrog_only_detected = airmass_cut(data_frame=df_balrog_only_detected)
    # if cfg['CUT_BINARY'] is True:
    #     start_window_logger.log_info_stream(f"binary cuts")
    #     df_balrog = binary_cut(data_frame=df_balrog)
    #     if cfg["USE_MOCK"] is not True:
    #         df_balrog_only_detected = binary_cut(data_frame=df_balrog_only_detected)
    # if cfg['CUT_MASK'] is True:
    #     start_window_logger.log_info_stream(f"mask cuts")
    #     df_balrog = mask_cut(data_frame=df_balrog, master=cfg['PATH_DATA']+cfg['FILENAME_MASTER_CAT'])
    #     if cfg["USE_MOCK"] is not True:
    #         df_balrog_only_detected = mask_cut(data_frame=df_balrog_only_detected, master=cfg['PATH_DATA']+cfg['FILENAME_MASTER_CAT'])

    start_window_logger.log_info_stream(f"length of catalog after cut section {len(df_balrog)}")
    if cfg["USE_MOCK"] is not True:
        start_window_logger.log_info_stream(f"length of only detected catalog after cut section {len(df_balrog_only_detected)}")

    # def get_yj_transformer(data_frame, columns):
    #     """"""
    #     dict_pt = {}
    #     for col in columns:
    #         pt = PowerTransformer(method="yeo-johnson")
    #         pt.fit(np.array(data_frame[col]).reshape(-1, 1))
    #         data_frame[col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
    #         dict_pt[f"{col} pt"] = pt
    #     return data_frame, dict_pt

    # df_balrog_yj = df_balrog.copy()
    # if cfg["USE_MOCK"] is not True:
    #     df_balrog_only_detected_yj = df_balrog_only_detected.copy()
    #     df_balrog_yj, dict_balrog_yj_transformer = get_yj_transformer(
    #         data_frame=df_balrog_yj,
    #         columns=cfg['YJ_TRANSFORM_COLS']
    #     )
    #     df_balrog_only_detected_yj, dict_balrog_only_detected_yj_transformer = get_yj_transformer(
    #         data_frame=df_balrog_only_detected_yj,
    #         columns=cfg['YJ_TRANSFORM_COLS']
    #     )
    #     joblib.dump(
    #         dict_balrog_yj_transformer,
    #         f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['FILENAME_YJ_TRANSFORMER']}.joblib"
    #     )
    #     joblib.dump(
    #         dict_balrog_only_detected_yj_transformer,
    #         f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['FILENAME_YJ_TRANSFORMER_ONLY_DETECTED']}.joblib"
    #     )
    # else:
    #     df_balrog_yj, dict_balrog_yj_transformer = get_yj_transformer(
    #         data_frame=df_balrog_yj,
    #         columns=cfg['YJ_TRANSFORM_COLS_MOCK']
    #     )
    #     joblib.dump(
    #         dict_balrog_yj_transformer,
    #         f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['FILENAME_YJ_TRANSFORMER']}{mock_prefix}.joblib"
    #     )
    #
    # def get_scaler(data_frame):
    #     """"""
    #     if cfg[f"SCALER"] == "MinMaxScaler":
    #         scaler = MinMaxScaler()
    #     elif cfg[f"SCALER"] == "MaxAbsScaler":
    #         scaler = MaxAbsScaler()
    #     elif cfg[f"SCALER"] == "StandardScaler":
    #         scaler = StandardScaler()
    #     else:
    #         raise TypeError(f'{cfg[f"SCALER"]} is no valid scaler')
    #     if scaler is not None:
    #         scaler.fit(data_frame)
    #     return scaler

    # if cfg['USE_MOCK'] is True:
    #     dict_data_frames = {
    #         "scaler_balrog_mag_yj": (
    #             df_balrog_yj[cfg['SCALER_COLS_MAG']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_mag_yj{mock_prefix}.joblib"
    #         ),
    #     }
    # else:
    #     dict_data_frames = {
    #         "scaler_balrog_flux_yj": (
    #             df_balrog_yj[cfg['SCALER_COLS_FLUX']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_flux_yj.joblib"
    #         ),
    #         "scaler_balrog_only_detected_flux_yj": (
    #             df_balrog_only_detected_yj[cfg['SCALER_COLS_FLUX']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_flux_yj.joblib"
    #         ),
    #         "scaler_balrog_mag_yj": (
    #             df_balrog_yj[cfg['SCALER_COLS_MAG']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_mag_yj.joblib"
    #         ),
    #         "scaler_balrog_only_detected_mag_yj": (
    #             df_balrog_only_detected_yj[cfg['SCALER_COLS_MAG']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_mag_yj.joblib"
    #         ),
    #         "scaler_balrog_lupt_yj": (
    #             df_balrog_yj[cfg['SCALER_COLS_LUPT']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_lupt_yj.joblib"
    #         ),
    #         "scaler_balrog_only_detected_lup_yjt": (
    #             df_balrog_only_detected_yj[cfg['SCALER_COLS_LUPT']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_lupt_yj.joblib"
    #         ),
    #     }
    #
    # for key in dict_data_frames.keys():
    #     scaler = get_scaler(
    #         data_frame=dict_data_frames[key][0]
    #     )
    #     joblib.dump(
    #         scaler,
    #         dict_data_frames[key][1]
    #     )
    #
    # del df_balrog_yj
    # if cfg["USE_MOCK"] is not True:
    #     del df_balrog_only_detected_yj
    #
    # if cfg['USE_MOCK'] is True:
    #     dict_data_frames = {
    #         "scaler_balrog_mag": (
    #             df_balrog[cfg['SCALER_COLS_MAG']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_mag{mock_prefix}.joblib"
    #         ),
    #     }
    # else:
    #     dict_data_frames = {
    #         "scaler_balrog_flux": (
    #             df_balrog[cfg['SCALER_COLS_FLUX']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_flux.joblib"
    #         ),
    #         "scaler_balrog_only_detected_flux": (
    #             df_balrog_only_detected[cfg['SCALER_COLS_FLUX']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_flux.joblib"
    #         ),
    #         "scaler_balrog_mag": (
    #             df_balrog[cfg['SCALER_COLS_MAG']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_mag.joblib"
    #         ),
    #         "scaler_balrog_only_detected_mag": (
    #             df_balrog_only_detected[cfg['SCALER_COLS_MAG']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_mag.joblib"
    #         ),
    #         "scaler_balrog_lupt": (
    #             df_balrog[cfg['SCALER_COLS_LUPT']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_lupt.joblib"
    #         ),
    #         "scaler_balrog_only_detected_lupt": (
    #             df_balrog_only_detected[cfg['SCALER_COLS_LUPT']],
    #             f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['SCALER'].lower()}_odet_lupt.joblib"
    #         ),
    #     }
    # for key in dict_data_frames.keys():
    #     scaler = get_scaler(
    #         data_frame=dict_data_frames[key][0]  # df_balrog[cfg['SCALER_COLS_FLUX']]
    #     )
    #     joblib.dump(
    #         scaler,
    #         dict_data_frames[key][1]
    #     )

    number_of_samples = cfg['NSAMPLES']
    if number_of_samples is None:
        number_of_samples = len(df_balrog)

    start_window_logger.log_info_stream(f"Number of samples {number_of_samples}")
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
        loggers=start_window_logger,
        mock_prefix=mock_prefix
    )

    if cfg["USE_MOCK"] is not True:
        number_of_samples_only_detected = cfg['NSAMPLES_ONLY_DETECTED']
        if number_of_samples_only_detected is None:
            number_of_samples_only_detected = len(df_balrog_only_detected)
        start_window_logger.log_info_stream(f"Number of samples only detected {number_of_samples_only_detected}")
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
            loggers=start_window_logger,
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
