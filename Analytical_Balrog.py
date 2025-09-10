import matplotlib.pyplot as plt
import numpy as np
from Handler.cut_functions import *
import sys
import yaml
import argparse
import pandas as pd


def data_preprocessing(cfg):
    """"""
    df_balrog = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MERGED_CAT']}")

    df_balrog = df_balrog[df_balrog["detected"] == 1]

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

    # df_balrog = apply_cuts(cfg=cfg, data_frame=df_balrog)
    df_balrog["BDF_G"] = np.sqrt(df_balrog["BDF_G_0"] ** 2 + df_balrog["BDF_G_1"] ** 2)

    dict_offset = get_offset(cfg=cfg, data_frame=df_balrog)

    for bin in cfg["UNSHEARED_BINS"]:
        df_balrog[f"flux_diff_{bin}"] = df_balrog[f"unsheared/flux_{bin}"].values - df_balrog[f"BDF_FLUX_DERED_CALIB_{bin.upper()}"].values

    sns.histplot(
        x=df_balrog["unsheared/flux_r"].values,
        element="step",
        fill=False,
        color="red",
        binwidth=2500,
        log_scale=(False, True),
        stat="probability",
        label=f"unsheared"
    )
    sns.histplot(
        x=df_balrog["flux_diff_r"].values,
        element="step",
        fill=False,
        color="green",
        binwidth=2500,
        log_scale=(False, True),
        stat="probability",
        label=f"diff"
    )
    sns.histplot(
        x=df_balrog["BDF_FLUX_DERED_CALIB_R"].values,
        element="step",
        fill=False,
        color="blue",
        binwidth=2500,
        log_scale=(False, True),
        stat="probability",
        label=f"BDF"
    )
    plt.xlim(-60000, 60000)
    plt.legend()
    plt.show()

    df_balrog_cut = apply_cuts(cfg=cfg, data_frame=df_balrog.copy())

    sns.histplot(
        x=df_balrog_cut["unsheared/flux_r"].values,
        element="step",
        fill=False,
        color="red",
        binwidth=2500,
        log_scale=(False, True),
        stat="probability",
        label=f"unsheared"
    )
    sns.histplot(
        x=df_balrog_cut["flux_diff_r"].values,
        element="step",
        fill=False,
        color="green",
        binwidth=2500,
        log_scale=(False, True),
        stat="probability",
        label=f"diff"
    )
    sns.histplot(
        x=df_balrog_cut["BDF_FLUX_DERED_CALIB_R"].values,
        element="step",
        fill=False,
        color="blue",
        binwidth=2500,
        log_scale=(False, True),
        stat="probability",
        label=f"BDF"
    )
    plt.xlim(-60000, 60000)
    plt.legend()
    plt.show()

    if cfg["PLOT_MOCK_NOISE"] is True:
        plot_histo(
            data_frame=df_balrog,
            columns=cfg["COVARIANCE_COLUMNS"],
            colors=None,
            bin_size=50,
            log_scale=(False, True),
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            xlabel="mag",
            save_name=f"{cfg['PATH_OUTPUT']}/noise_true_hist.png",
            title=f"Noise Histogram"
        )
    df_balrog.rename(columns={"ID": "true_id"}, inplace=True)
    df_balrog = df_balrog[cfg["SOMPZ_COLS"]+cfg["COVARIANCE_COLUMNS"]]

    print(df_balrog)
    print(df_balrog.isna().sum())
    print(df_balrog.isna().sum().sum())

    return df_balrog, dict_offset


def apply_cuts(cfg, data_frame):
    """"""
    data_frame = unsheared_object_cuts(data_frame=data_frame)
    data_frame = flag_cuts(data_frame=data_frame)
    data_frame = unsheared_shear_cuts(data_frame=data_frame)
    data_frame = binary_cut(data_frame=data_frame)
    data_frame = mask_cut(data_frame=data_frame, master=f"{cfg['PATH_DATA']}/{cfg['FILENAME_MASTER_CAT']}")
    # if cfg["MOCK_APPLY_MAG_CUTS_BEFORE"] is True:
    data_frame = unsheared_mag_cut(data_frame=data_frame)
    return data_frame


# def compute_injection_counts(det_catalog):
#     unique, ucounts = np.unique(det_catalog['true_id'], return_counts=True)
#     freq = pd.DataFrame()
#     freq['true_id'] = unique
#     freq['injection_counts'] = ucounts
#     return det_catalog.merge(freq, on='true_id', how='left')


def get_covariance_matrix(cfg, data_frame):
    """"""
    df_cov_difference = data_frame[cfg["COVARIANCE_COLUMNS"]]

    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)

    # Calculate the IQR for each column
    iqr_values = df_cov_difference.apply(iqr)

    # Create a DataFrame where each cell is the product of the IQRs of its row and column
    cov_matrix_difference = df_cov_difference.cov().values
    sns.heatmap(cov_matrix_difference, annot=True, fmt='g')
    plt.show()

    cov_matrix_difference = pd.DataFrame(np.outer(iqr_values, iqr_values), columns=df_cov_difference.columns, index=df_cov_difference.columns)
    sns.heatmap(cov_matrix_difference, annot=True, fmt='g')
    plt.show()

    print(f"covariance matrix deep distribution: {cov_matrix_difference}")

    df_temp = data_frame.copy()
    for idx_bin, bin in enumerate(cfg["UNSHEARED_BINS"]):
        df_temp.loc[:, f"Color unsheared mag {bin}-{cfg['UNSHEARED_BINS'][idx_bin+1]}"] = (
                df_temp[f"unsheared/mag_{bin}"].values - df_temp[f"unsheared/mag_{cfg['UNSHEARED_BINS'][idx_bin+1]}"].values
        )
        if idx_bin + 1 == len(cfg["UNSHEARED_BINS"]) - 1:
            break

    if cfg["PLOT_TRUE_MEAS_MAG"] is True:
        plot_corner(
            df_temp,
            columns=[
                "unsheared/mag_r",
                "unsheared/mag_i",
                "unsheared/mag_z",
            ],
            labels=["r", "i", "z"],
            title="True Measured Magnitude",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/true_meas_mag.png"
        )

    if cfg["PLOT_TRUE_MEAS_COLOR"] is True:
        plot_corner(
            df_temp,
            columns=[
                "Color unsheared mag r-i",
                "Color unsheared mag i-z"
            ],
            labels=["r-i", "i-z"],
            title="True Measured Color",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/true_meas_color.png"
        )

    return df_temp, cov_matrix_difference


def get_offset(cfg, data_frame):
    bin_stats_r = plot_bin_offset(
        data_frame=data_frame,
        true_column="BDF_MAG_DERED_CALIB_R",
        measured_column="unsheared/mag_r",
        color="#fc810a",
        save_name=f"{cfg['PATH_OUTPUT']}/bin_offset_r.png",
        show_plot=cfg["SHOW_PLOT_MOCK"],
        save_plot=cfg["SAVE_PLOT_MOCK"]
    )
    bin_stats_i = plot_bin_offset(
        data_frame=data_frame,
        true_column="BDF_MAG_DERED_CALIB_I",
        measured_column="unsheared/mag_i",
        color="#1b78bc",
        save_name=f"{cfg['PATH_OUTPUT']}/bin_offset_i.png",
        show_plot=cfg["SHOW_PLOT_MOCK"],
        save_plot=cfg["SAVE_PLOT_MOCK"]
    )
    bin_stats_z = plot_bin_offset(
        data_frame=data_frame,
        true_column="BDF_MAG_DERED_CALIB_Z",
        measured_column="unsheared/mag_z",
        color="#9463be",
        save_name=f"{cfg['PATH_OUTPUT']}/bin_offset_z.png",
        show_plot=cfg["SHOW_PLOT_MOCK"],
        save_plot=cfg["SAVE_PLOT_MOCK"]
    )

    dict_stats = {
        "weighted_mean_mag_r": bin_stats_r['weighted_mean'],
        "weighted_mean_mag_i": bin_stats_i['weighted_mean'],
        "weighted_mean_mag_z": bin_stats_z['weighted_mean'],
        "weighted_median_mag_r": bin_stats_r['weighted_median'],
        "weighted_median_mag_i": bin_stats_i['weighted_median'],
        "weighted_median_mag_z": bin_stats_z['weighted_median'],
        "percent_deviation_mean_r": bin_stats_r['percent_deviation_mean'],
        "percent_deviation_mean_i": bin_stats_i['percent_deviation_mean'],
        "percent_deviation_mean_z": bin_stats_z['percent_deviation_mean'],
        "percent_deviation_median_r": bin_stats_r['percent_deviation_median'],
        "percent_deviation_median_i": bin_stats_i['percent_deviation_median'],
        "percent_deviation_median_z": bin_stats_z['percent_deviation_median']
    }

    return dict_stats


def generate_mock(cfg, dict_offset, cov_diff, df_true):
    """"""
    size = cfg["SIZE_MOCK"]

    if size is None:
        size = len(df_true)

    df_diff = df_true[cfg["COVARIANCE_COLUMNS"]]
    arr_mean_diff = np.zeros(len(cfg["COVARIANCE_COLUMNS"]))  # df_diff.mean().values

    arr_multi_normal_diff = np.random.multivariate_normal(
        arr_mean_diff,
        cov_diff,
        df_diff.shape[0]
    )

    df_mock = pd.DataFrame(
        arr_multi_normal_diff,
        columns=cfg["COVARIANCE_COLUMNS"]
    )
    if cfg["PLOT_MOCK_NOISE"] is True:
        plot_histo(
            data_frame=df_mock,
            columns=cfg["COVARIANCE_COLUMNS"],
            colors=None,
            bin_size=50,
            log_scale=(False, True),
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            xlabel="mag",
            save_name=f"{cfg['PATH_OUTPUT']}/noise_hist.png",
            title=f"Noise Histogram"
        )



    # df_mock["unsheared/flux_err_r"] = df_true["unsheared/flux_err_r"]
    # df_mock["unsheared/flux_err_i"] = df_true["unsheared/flux_err_i"]
    # df_mock["unsheared/flux_err_z"] = df_true["unsheared/flux_err_z"]

    for idx_bin, bin in enumerate(cfg["UNSHEARED_BINS"]):
        new_val = df_true[f"BDF_FLUX_DERED_CALIB_{bin.upper()}"].values * 10 ** (-0.4 * dict_offset[f"weighted_mean_mag_{bin}"]) + df_mock[f"flux_diff_{bin}"].values

        # Check if new_val is smaller than min_val, if so, set to min_val
        df_mock[f"unsheared/flux_{bin}"] = new_val  # np.where(new_val < min_val, min_val, new_val)

        df_mock[f"unsheared/mag_{bin}"] = flux2mag(df_mock[f"unsheared/flux_{bin}"].values)
        df_mock[f"BDF_FLUX_DERED_CALIB_{bin.upper()}"] = df_true[f"BDF_FLUX_DERED_CALIB_{bin.upper()}"].values
        df_mock[f"BDF_MAG_DERED_CALIB_{bin.upper()}"] = flux2mag(df_true[f"BDF_FLUX_DERED_CALIB_{bin.upper()}"].values)

        df_true[f"unsheared/mag_{bin}"] = flux2mag(df_true[f"unsheared/flux_{bin}"].values)

    for idx_bin, bin in enumerate(cfg["UNSHEARED_BINS"]):
        df_mock.loc[:, f"Color unsheared mag {bin}-{cfg['UNSHEARED_BINS'][idx_bin+1]}"] = (
                df_mock[f"unsheared/mag_{bin}"].values - df_mock[f"unsheared/mag_{cfg['UNSHEARED_BINS'][idx_bin+1]}"].values
        )
        if idx_bin + 1 == len(cfg["UNSHEARED_BINS"]) - 1:
            break

    print(f"len generated data: {len(df_mock)}")
    print(f"len true data: {len(df_true)}")
    df_mock = df_mock.reset_index(drop=True)
    df_true = df_true.reset_index(drop=True)

    df_mock = add_needed_columns(
        data_frame_mock=df_mock,
        df_true=df_true
    )

    sns.histplot(
        x=df_mock["unsheared/flux_r"].values,
        element="step",
        fill=False,
        color="red",
        binwidth=2500,
        log_scale=(False, False),
        stat="probability",
        label=f"mock"
    )
    sns.histplot(
        x=df_true["unsheared/flux_r"].values,
        element="step",
        fill=False,
        color="blue",
        binwidth=2500,
        log_scale=(False, False),
        stat="probability",
        label=f"true"
    )
    plt.xlim(-60000, 60000)
    plt.legend()
    plt.show()

    df_mock = apply_cuts(cfg=cfg, data_frame=df_mock)
    df_true = apply_cuts(cfg=cfg, data_frame=df_true)
    # if cfg["MOCK_APPLY_MAG_CUTS_BEFORE"] is False:
    #     print("Apply mag cuts")
    #     df_mock = unsheared_mag_cut(data_frame=df_mock)
    #     df_true = unsheared_mag_cut(data_frame=df_true)

    sns.histplot(
        x=df_mock["unsheared/flux_r"].values,
        element="step",
        fill=False,
        color="red",
        binwidth=2500,
        log_scale=(False, False),
        stat="probability",
        label=f"mock"
    )
    sns.histplot(
        x=df_true["unsheared/flux_r"].values,
        element="step",
        fill=False,
        color="blue",
        binwidth=2500,
        log_scale=(False, False),
        stat="probability",
        label=f"true"
    )
    plt.xlim(-60000, 60000)
    plt.legend()
    plt.show()

    if cfg["PLOT_MOCK_OFFSET"] is True:
        plot_bin_offset(
            data_frame=df_mock,
            true_column="BDF_MAG_DERED_CALIB_R",
            measured_column="unsheared/mag_r",
            color="#fc810a",
            save_name=f"{cfg['PATH_OUTPUT']}/offset_mock_after_mag_cuts_r.png",
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"]
        )
        plot_bin_offset(
            data_frame=df_true,
            true_column="BDF_MAG_DERED_CALIB_R",
            measured_column="unsheared/mag_r",
            color="#fc810a",
            save_name=f"{cfg['PATH_OUTPUT']}/offset_true_after_mag_cuts_r.png",
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"]
        )
        # plot_bin_offset(
        #     data_frame=df_mock,
        #     true_column="BDF_MAG_DERED_CALIB_R",
        #     measured_column="unsheared/mag_r_wo_corr",
        #     color="#fc810a",
        #     save_name=f"{cfg['PATH_OUTPUT']}/offset_mock_after_mag_cuts_wo_corr_r.png",
        #     show_plot=cfg["SHOW_PLOT_MOCK"],
        #     save_plot=cfg["SAVE_PLOT_MOCK"]
        # )
        # plot_bin_offset(
        #     data_frame=df_true,
        #     true_column="BDF_MAG_DERED_CALIB_R",
        #     measured_column="unsheared/mag_r_wo_corr",
        #     color="#fc810a",
        #     save_name=f"{cfg['PATH_OUTPUT']}/offset_true_after_mag_cuts_wo_corr_r.png",
        #     show_plot=cfg["SHOW_PLOT_MOCK"],
        #     save_plot=cfg["SAVE_PLOT_MOCK"]
        # )

        plot_bin_offset(
            data_frame=df_mock,
            true_column="BDF_MAG_DERED_CALIB_I",
            measured_column="unsheared/mag_i",
            color="#1b78bc",
            save_name=f"{cfg['PATH_OUTPUT']}/offset_mock_after_mag_cuts_i.png",
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"]
        )
        plot_bin_offset(
            data_frame=df_true,
            true_column="BDF_MAG_DERED_CALIB_I",
            measured_column="unsheared/mag_i",
            color="#1b78bc",
            save_name=f"{cfg['PATH_OUTPUT']}/offset_true_after_mag_cuts_i.png",
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"]
        )
        # plot_bin_offset(
        #     data_frame=df_mock,
        #     true_column="BDF_MAG_DERED_CALIB_I",
        #     measured_column="unsheared/mag_i_wo_corr",
        #     color="#1b78bc",
        #     save_name=f"{cfg['PATH_OUTPUT']}/offset_mock_after_mag_cuts_wo_corr_i.png",
        #     show_plot=cfg["SHOW_PLOT_MOCK"],
        #     save_plot=cfg["SAVE_PLOT_MOCK"]
        # )
        # plot_bin_offset(
        #     data_frame=df_true,
        #     true_column="BDF_MAG_DERED_CALIB_I",
        #     measured_column="unsheared/mag_i_wo_corr",
        #     color="#1b78bc",
        #     save_name=f"{cfg['PATH_OUTPUT']}/offset_true_after_mag_cuts_wo_corr_z.png",
        #     show_plot=cfg["SHOW_PLOT_MOCK"],
        #     save_plot=cfg["SAVE_PLOT_MOCK"]
        # )

        plot_bin_offset(
            data_frame=df_mock,
            true_column="BDF_MAG_DERED_CALIB_Z",
            measured_column="unsheared/mag_z",
            color="#9463be",
            save_name=f"{cfg['PATH_OUTPUT']}/offset_mock_after_mag_cuts_z.png",
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"]
        )
        plot_bin_offset(
            data_frame=df_true,
            true_column="BDF_MAG_DERED_CALIB_Z",
            measured_column="unsheared/mag_z",
            color="#9463be",
            save_name=f"{cfg['PATH_OUTPUT']}/offset_true_after_mag_cuts_z.png",
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"]
        )
        # plot_bin_offset(
        #     data_frame=df_mock,
        #     true_column="BDF_MAG_DERED_CALIB_Z",
        #     measured_column="unsheared/mag_z_wo_corr",
        #     color="#9463be",
        #     save_name=f"{cfg['PATH_OUTPUT']}/offset_mock_after_mag_cuts_wo_corr_z.png",
        #     show_plot=cfg["SHOW_PLOT_MOCK"],
        #     save_plot=cfg["SAVE_PLOT_MOCK"]
        # )
        # plot_bin_offset(
        #     data_frame=df_true,
        #     true_column="BDF_MAG_DERED_CALIB_Z",
        #     measured_column="unsheared/mag_z_wo_corr",
        #     color="#9463be",
        #     save_name=f"{cfg['PATH_OUTPUT']}/offset_true_after_mag_cuts_wo_corr_z.png",
        #     show_plot=cfg["SHOW_PLOT_MOCK"],
        #     save_plot=cfg["SAVE_PLOT_MOCK"]
        # )

    if cfg["PLOT_COMPARE_MEAS_MAG"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock,
            data_frame_true=df_true,
            columns=[
                "unsheared/mag_r",
                "unsheared/mag_i",
                "unsheared/mag_z",
            ],
            labels=["r", "i", "z"],
            title="True Measured Magnitude",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_meas_mag.png"
        )

    if cfg["PLOT_COMPARE_MEAS_FLUX"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock,
            data_frame_true=df_true,
            columns=[
                "unsheared/flux_r",
                "unsheared/flux_i",
                "unsheared/flux_z",
            ],
            labels=["r", "i", "z"],
            title="True Measured Flux",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_meas_flux.png"
        )

    if cfg["PLOT_MOCK_MEAS_MAG"] is True:
        plot_corner(
            df_mock,
            columns=[
                "unsheared/mag_r",
                "unsheared/mag_i",
                "unsheared/mag_z",
            ],
            labels=["r", "i", "z"],
            title="True Measured Magnitude",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_meas_mag.png"
        )

    if cfg["PLOT_COMPARE_MEAS_COLOR"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock,
            data_frame_true=df_true,
            columns=[
                "Color unsheared mag r-i",
                "Color unsheared mag i-z"
            ],
            labels=["r-i", "i-z"],
            title="Compare Measured Color",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_meas_color.png"
        )

    if cfg["PLOT_MOCK_MEAS_COLOR"] is True:
        plot_corner(
            df_mock,
            columns=[
                "Color unsheared mag r-i",
                "Color unsheared mag i-z"
            ],
            labels=["r-i", "i-z"],
            title="Mock Measured Color",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_meas_color.png"
        )

    if cfg["PLOT_COMPARE_HISTOGRAM_MAG"] is True:
        columns = [
            ["unsheared/mag_r", "BDF_MAG_DERED_CALIB_R"],
            ["unsheared/mag_i", "BDF_MAG_DERED_CALIB_I"],
            ["unsheared/mag_z", "BDF_MAG_DERED_CALIB_Z"]
        ]

        for idx, column in enumerate(columns):
            plot_histo_compare(
                data_frame_generated=df_mock,
                data_frame_true=df_true,
                columns=column,
                colors=[["blue", "green"], ["red", "purple"]],
                bin_size=50,
                log_scale=(False, True),
                show_plot=cfg["SHOW_PLOT_MOCK"],
                save_plot=cfg["SAVE_PLOT_MOCK"],
                xlabel="mag",
                save_name=f"{cfg['PATH_OUTPUT']}/compare_probability_hist_mag_{cfg['UNSHEARED_BINS'][idx]}.png",
                title=f"probability histogram of DES wide field mag {cfg['UNSHEARED_BINS'][idx]}"
            )

    if cfg["PLOT_COMPARE_HISTOGRAM_MAG_ERR"] is True:
        columns = [["unsheared/mag_err_r"], ["unsheared/mag_err_i"], ["unsheared/mag_err_z"]]
        for idx, column in enumerate(columns):
            plot_histo_compare(
                data_frame_generated=df_mock,
                data_frame_true=df_true,
                columns=column,
                bin_size=50,
                log_scale=(False, True),
                show_plot=cfg["SHOW_PLOT_MOCK"],
                save_plot=cfg["SAVE_PLOT_MOCK"],
                xlabel="mag",
                save_name=f"{cfg['PATH_OUTPUT']}/compare_probability_hist_mag_err_{cfg['UNSHEARED_BINS'][idx]}.png",
                title=f"probability histogram of DES wide field mag err {cfg['UNSHEARED_BINS'][idx]}"
            )
    if cfg["PLOT_COMPARE_HISTOGRAM_FLUX"] is True:
        columns = [["unsheared/flux_r"], ["unsheared/flux_i"], ["unsheared/flux_z"]]
        for idx, column in enumerate(columns):
            plot_histo_compare(
                data_frame_generated=df_mock,
                data_frame_true=df_true,
                columns=column,
                bin_size=50,
                log_scale=(False, False),
                show_plot=cfg["SHOW_PLOT_MOCK"],
                save_plot=cfg["SAVE_PLOT_MOCK"],
                xlabel="flux",
                save_name=f"{cfg['PATH_OUTPUT']}/compare_probability_hist_flux_{cfg['UNSHEARED_BINS'][idx]}.png",
                title=f"probability histogram of DES wide field flux {cfg['UNSHEARED_BINS'][idx]}"
            )

    if cfg["PLOT_COMPARE_HISTOGRAM_FLUX_ERR"] is True:
        columns = [["unsheared/flux_err_r"], ["unsheared/flux_err_i"], ["unsheared/flux_err_z"]]
        for idx, column in enumerate(columns):
            plot_histo_compare(
                data_frame_generated=df_mock,
                data_frame_true=df_true,
                columns=column,
                bin_size=50,
                log_scale=(False, False),
                show_plot=cfg["SHOW_PLOT_MOCK"],
                save_plot=cfg["SAVE_PLOT_MOCK"],
                xlabel="flux",
                save_name=f"{cfg['PATH_OUTPUT']}/compare_probability_hist_flux_err_{cfg['UNSHEARED_BINS'][idx]}.png",
                title=f"probability histogram of DES wide field flux err {cfg['UNSHEARED_BINS'][idx]}"
            )

    return df_mock, df_true


def add_needed_columns(data_frame_mock, df_true):
    """"""
    for column in cfg["SOMPZ_COLS"]:
        if column not in data_frame_mock.columns:
            data_frame_mock[column] = df_true.loc[data_frame_mock.index, column]
    # data_frame_mock = compute_injection_counts(data_frame_mock)
    return data_frame_mock


def save_data(cfg, data_frame, filename, protocol):
    """"""
    if protocol == 'hdf5':
        print("Saving file as HDF5")
        data_frame.to_hdf(f"{cfg['PATH_OUTPUT']}/Catalogs/{filename}.h5", key='df', mode='w')
    elif protocol == 2:
        print("save file with protocol 2")
        with open(f"{cfg['PATH_OUTPUT']}/Catalogs/{filename}.pkl", 'wb') as f:
            pickle.dump(data_frame, f, protocol=2)
    else:
        print("save file with protocol 5")
        data_frame.to_pickle(f"{cfg['PATH_OUTPUT']}/Catalogs/{filename}.pkl")


def main(cfg):
    """"""

    df_true, dict_offset = data_preprocessing(cfg=cfg)

    df_true, cov_difference = get_covariance_matrix(
        cfg=cfg,
        data_frame=df_true
    )

    df_mock, df_true_cut = generate_mock(
        cfg=cfg,
        dict_offset=dict_offset,
        cov_diff=cov_difference,
        df_true=df_true
    )

    save_data(
        cfg=cfg,
        data_frame=df_mock,
        filename=f"{cfg['FILENAME_MOCK_CAT']}_{len(df_mock)}",
        protocol='hdf5'
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

    main(cfg=cfg)