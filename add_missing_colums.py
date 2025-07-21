import matplotlib.pyplot as plt
import pandas as pd

from Handler import *
from Handler.helper_functions import *
from Handler.cut_functions import apply_cuts
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy.stats import mstats
from datetime import datetime
import sys
import os
import yaml
import argparse


def plot_feature_histograms(cfg, df, columns, title, save_name, bins=100):
    n = len(columns)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.histplot(df[col], bins=bins, ax=ax)
        ax.set_title(col)
        ax.set_yscale("log")
        ax.set_xlabel(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title)
    plt.savefig(f"{cfg['PATH_PLOTS']}/{save_name}.pdf", bbox_inches='tight', dpi=300)


def main(cfg):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    os.makedirs(cfg['PATH_OUTPUT'], exist_ok=True)
    os.makedirs(os.path.join(cfg['PATH_OUTPUT'], "logs"), exist_ok=True)

    log_lvl = logging.INFO
    if cfg["LOGGING_LEVEL"] == "DEBUG":
        log_lvl = logging.DEBUG
    elif cfg["LOGGING_LEVEL"] == "ERROR":
        log_lvl = logging.ERROR

    start_window_logger = LoggerHandler(
        logger_dict={"logger_name": "add missing columns",
                     "info_logger": cfg['INFO_LOGGER'],
                     "error_logger": cfg['ERROR_LOGGER'],
                     "debug_logger": cfg['DEBUG_LOGGER'],
                     "stream_logger": cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{cfg['PATH_OUTPUT']}Logs/"
    )

    start_window_logger.log_info_stream("Start 'add missing columns'")

    df_balrog = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MERGED_CAT']}")

    df_balrog.rename(
        columns={
            'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
            'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
        inplace=True)

    start_window_logger.log_info_stream("Calc BDF_G=np.sqrt(BDF_G_0** 2 + BDF_G_1 ** 2)")
    df_balrog["BDF_G"] = np.sqrt(df_balrog["BDF_G_0"] ** 2 + df_balrog["BDF_G_1"] ** 2)

    for k in df_balrog.keys():
        start_window_logger.log_info_stream(f"Number of NaNs in {k}: {df_balrog[k].isna().sum()}")

    start_window_logger.log_info_stream(f"Total number of NaNs: {df_balrog.isna().sum().sum()}")
    start_window_logger.log_info_stream(f"length of Balrog objects {len(df_balrog)}")

    start_window_logger.log_info_stream("Calc bdf lupt and color")
    df_balrog = calc_color(
        cfg=cfg,
        data_frame=df_balrog,
        mag_type=("LUPT", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
        bins=cfg['BDF_BINS'],
        save_name=f"bdf_lupt"
    )
    start_window_logger.log_info_stream("Calc measured lupt and color")
    df_balrog = calc_color(
        cfg=cfg,
        data_frame=df_balrog,
        mag_type=("LUPT", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/lupt", "unsheared/lupt_err"),
        bins=cfg['UNSHEARED_BINS'],
        save_name=f"unsheared/lupt"
    )

    # start_window_logger.log_info_stream("Calc bdf mag and color")
    # df_balrog = calc_color(
    #     cfg=cfg,
    #     data_frame=df_balrog,
    #     mag_type=("MAG", "BDF"),
    #     flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
    #     mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
    #     bins=cfg['BDF_BINS'],
    #     save_name=f"bdf_mag"
    # )
    # start_window_logger.log_info_stream("Calc measured mag and color")
    # df_balrog = calc_color(
    #     cfg=cfg,
    #     data_frame=df_balrog,
    #     mag_type=("MAG", "unsheared"),
    #     flux_col=("unsheared/flux", "unsheared/flux_err"),
    #     mag_col=("unsheared/mag", "unsheared/mag_err"),
    #     bins=cfg['UNSHEARED_BINS'],
    #     save_name=f"unsheared/mag"
    # )

    # plot_feature_histograms(df_balrog, cfg["INPUT_COLUMNS"], title="Input Features before Default Replacement")
    # plot_feature_histograms(df_balrog, cfg["OUTPUT_COLUMNS"], title="Output Features before Default Replacement")

    df_balrog_detected = df_balrog[df_balrog["detected"] == 1].copy()
    df_default_nan = pd.DataFrame()
    print("Start")
    # for k in df_balrog_detected.keys():
    #     print(k, df_balrog_detected[k].isna().sum())
    # if cfg["PLOT_DATA"] is True:
    #     plot_feature_histograms(
    #         cfg=cfg,
    #         df=df_balrog_detected,
    #         columns=cfg["INPUT_COLUMNS"],
    #         title="Input Features Start",
    #         save_name=f"input_features_start"
    #     )
    #
    #     plot_feature_histograms(
    #         cfg=cfg,
    #         df=df_balrog_detected,
    #         columns=cfg["OUTPUT_COLUMNS"],
    #         title="Output Features Start",
    #         save_name=f"output_features_start"
    #     )
    # for col in cfg["NF_COLUMNS_OF_INTEREST"]:
    #     print(f"{col}: min={df_balrog_detected[col].min()}, max={df_balrog_detected[col].max()}")

    # for col in cfg["DEFAULT_COLUMNS"]:
    #     df = df_balrog[df_balrog[col] != cfg["DEFAULT_COLUMNS"][col][0]].copy()
    #     print(f"{col}: min={df[col].min()}, max={df[col].max()}")

    for col, _ in cfg.get("REPLACE_NAN", {}).items():
        mask = df_balrog_detected[col].isna()
        n = mask.sum()
        if n > 0:
            start_window_logger.log_info_stream(f"Replace NaNs in {col} to {-9999.0}")

        df_balrog_detected.loc[mask, col] = -9999.0

        df_default_nan[f"{col}_was_nan"] = mask.astype(int)
        df_default_nan[f"{col}_original_nan_val"] = [np.nan for _ in range(len(df_balrog_detected))]
        df_default_nan[f"{col}_nan_range"] = [-9999.0 for _ in range(len(df_balrog_detected))]

    if cfg["TEST_TRANSFORMER"] is True:
        df_balrog_detected_trans = df_balrog_detected[df_balrog_detected["AIRMASS_WMEAN_R"]>-9999].copy()
        pt = PowerTransformer(method="yeo-johnson")
        arr_yj = pt.fit_transform(df_balrog_detected_trans[cfg["TRANSFORM_COLUMNS"]])
        for i, col in enumerate(cfg["TRANSFORM_COLUMNS"]):
            start_window_logger.log_info_stream(f"YJ lambda for {col}: {pt.lambdas_[i]:.6f}")
        df_balrog_detected_trans[cfg["TRANSFORM_COLUMNS"]] = arr_yj

        scaler = StandardScaler()
        arr_scaled = scaler.fit_transform(df_balrog_detected_trans[cfg["SCALE_COLUMNS"]])
        df_balrog_detected_trans[cfg["SCALE_COLUMNS"]] = arr_scaled

        if cfg["PLOT_DATA"] is True:
            plot_feature_histograms(
                cfg=cfg,
                df=df_balrog_detected_trans,
                columns=cfg["INPUT_COLUMNS"],
                title="Input Features After YJ Transform w/o defaults",
                save_name=f"input_features_yj_wo_defaults"
            )

            plot_feature_histograms(
                cfg=cfg,
                df=df_balrog_detected_trans,
                columns=cfg["OUTPUT_COLUMNS"],
                title="Output Features After YJ Transform w/o defaults",
                save_name=f"output_features_yj_wo_defaults"
            )

    print("After replace Nan")
    # if cfg["PLOT_DATA"] is True:
    #     plot_feature_histograms(
    #         cfg=cfg,
    #         df=df_balrog_detected,
    #         columns=cfg["INPUT_COLUMNS"],
    #         title="Input Features After NaN",
    #         save_name=f"input_features_nan"
    #     )
    #
    #     plot_feature_histograms(
    #         cfg=cfg,
    #         df=df_balrog_detected,
    #         columns=cfg["OUTPUT_COLUMNS"],
    #         title="Output Features After NaN",
    #         save_name=f"output_features_nan"
    #     )
    # for col in cfg["NF_COLUMNS_OF_INTEREST"]:
    #     print(f"{col}: min={df_balrog_detected[col].min()}, max={df_balrog_detected[col].max()}")

    for col, mode_dict in cfg["CLIP_COLUMNS"].items():
        if col not in df_balrog_detected.columns:
            continue

        if not isinstance(mode_dict, dict):
            start_window_logger.log_info_stream(
                f"SKIP {col}: CLIP_COLUMNS-Entry is not a dict, it is {type(mode_dict)} ({mode_dict})"
            )
            continue

        real_vals = df_balrog_detected[col].values

        if "clip" in mode_dict:
            min_val, max_val = mode_dict["clip"]
            clipped = np.clip(real_vals, min_val if min_val is not None else -np.inf,
                              max_val if max_val is not None else np.inf)
            df_balrog_detected[col] = clipped
            start_window_logger.log_info_stream(f"Clip {col} to min={min_val}, max={max_val}")

        elif "winsorize" in mode_dict:
            limits = mode_dict["winsorize"]
            wins = winsorize(real_vals, limits=limits)
            df_balrog_detected[col] = wins
            start_window_logger.log_info_stream(f"Winsorize {col} with {limits}")

        else:
            start_window_logger.log_info_stream(f"Warning: No clipping/winsorize for {col} defined!")

    print("After Clipping")
    # if cfg["PLOT_DATA"] is True:
    #     plot_feature_histograms(
    #         cfg=cfg,
    #         df=df_balrog_detected,
    #         columns=cfg["INPUT_COLUMNS"],
    #         title="Input Features After Clipping",
    #         save_name=f"input_features_clipping"
    #     )
    #
    #     plot_feature_histograms(
    #         cfg=cfg,
    #         df=df_balrog_detected,
    #         columns=cfg["OUTPUT_COLUMNS"],
    #         title="Output Features After Clipping",
    #         save_name=f"output_features_clipping"
    #     )
    # for col in cfg["NF_COLUMNS_OF_INTEREST"]:
    #     print(f"{col}: min={df_balrog_detected[col].min()}, max={df_balrog_detected[col].max()}")

    # for col, (default_val, _) in cfg["DEFAULT_COLUMNS"].items():
    #     real_vals = df_balrog_detected.loc[df_balrog_detected[col] != default_val, col]
    #     std = real_vals.std()
    #     min_val = real_vals.min()
    #     max_val = real_vals.max()
    #     epsilon = 0.001 * std
    #
    #     mask = df_balrog_detected[col] == default_val
    #     n = mask.sum()
    #
    #     if default_val < min_val:
    #         low = min_val - epsilon
    #         high = min_val
    #     elif default_val > max_val:
    #         low = max_val
    #         high = max_val + epsilon
    #     else:
    #         low = default_val - epsilon
    #         high = default_val + epsilon
    #
    #     df_default_nan["bal_id"] = df_balrog_detected["bal_id"]
    #     df_default_nan[f"{col}_is_default"] = mask.astype(int)
    #     df_default_nan[f"{col}_original_default_val"] = [default_val for _ in range(len(df_balrog_detected))]
    #     df_default_nan[f"{col}_default_range"] = [[low, high] for _ in range(len(df_balrog_detected))]
    #     start_window_logger.log_info_stream(f"Default range {col} to min={low}, max={high}")
    #     df_balrog_detected.loc[mask, col] = np.random.uniform(low, high, size=n)

    for col, (default_val, _) in cfg["DEFAULT_COLUMNS"].items():
        real_vals = df_balrog_detected.loc[df_balrog_detected[col] != default_val, col]
        std = real_vals.std()
        min_val = real_vals.min()
        max_val = real_vals.max()
        value_range = max_val - min_val
        epsilon = max(0.01 * value_range, 1e-6)  # mind. ein kleiner Wert, falls alles constant

        mask = df_balrog_detected[col] == default_val
        n = mask.sum()

        if default_val < min_val:
            low = min_val - 2 * epsilon
            high = min_val - epsilon
        elif default_val > max_val:
            low = max_val + epsilon
            high = max_val + 2 * epsilon
        else:
            # Edge-Case: default liegt innerhalb der echten Werte – aber das willst du meist nicht!
            # Notlösung: auch außerhalb schieben
            low = min_val - 2 * epsilon
            high = min_val - epsilon

        df_balrog_detected.loc[mask, col] = np.random.uniform(low, high, size=n)

        df_default_nan["bal_id"] = df_balrog_detected["bal_id"]
        df_default_nan[f"{col}_is_default"] = mask.astype(int)
        df_default_nan[f"{col}_original_default_val"] = [default_val for _ in range(len(df_balrog_detected))]
        df_default_nan[f"{col}_default_range"] = [[low, high] for _ in range(len(df_balrog_detected))]

        start_window_logger.log_info_stream(f"{col}: default_range=({low},{high})")

    print("After replace Defaults")

    # if cfg["PLOT_DATA"] is True:
    #     plot_feature_histograms(
    #         cfg=cfg,
    #         df=df_balrog_detected,
    #         columns=cfg["INPUT_COLUMNS"],
    #         title="Input Features After Default",
    #         save_name=f"input_features_default"
    #     )
    #
    #     plot_feature_histograms(
    #         cfg=cfg,
    #         df=df_balrog_detected,
    #         columns=cfg["OUTPUT_COLUMNS"],
    #         title="Output Features After Default",
    #         save_name=f"output_features_default"
    #     )
    # for col in cfg["NF_COLUMNS_OF_INTEREST"]:
    #     print(f"{col}: min={df_balrog_detected[col].min()}, max={df_balrog_detected[col].max()}")

    if cfg["TEST_TRANSFORMER"] is True:
        df_balrog_detected_trans = df_balrog_detected.copy()
        df_balrog_detected_inv = df_balrog_detected.copy()
        pt = PowerTransformer(method="yeo-johnson")
        arr_yj = pt.fit_transform(df_balrog_detected_trans[cfg["TRANSFORM_COLUMNS"]])
        for i, col in enumerate(cfg["TRANSFORM_COLUMNS"]):
            start_window_logger.log_info_stream(f"YJ lambda for {col}: {pt.lambdas_[i]:.6f}")
        df_balrog_detected_trans[cfg["TRANSFORM_COLUMNS"]] = arr_yj

        scaler = StandardScaler()
        arr_scaled = scaler.fit_transform(df_balrog_detected_trans[cfg["SCALE_COLUMNS"]])
        df_balrog_detected_trans[cfg["SCALE_COLUMNS"]] = arr_scaled

        if cfg["PLOT_DATA"] is True:
            plot_feature_histograms(
                cfg=cfg,
                df=df_balrog_detected_trans,
                columns=cfg["INPUT_COLUMNS"],
                title="Input Features After YJ Transform",
                save_name=f"input_features_yj"
            )

            plot_feature_histograms(
                cfg=cfg,
                df=df_balrog_detected_trans,
                columns=cfg["OUTPUT_COLUMNS"],
                title="Output Features After YJ Transform",
                save_name=f"output_features_yj"
            )

        df_balrog_detected_inv[cfg["TRANSFORM_COLUMNS"]] = pt.inverse_transform(df_balrog_detected_trans[cfg["TRANSFORM_COLUMNS"]])
        df_balrog_detected_inv[cfg["SCALE_COLUMNS"]] = scaler.inverse_transform(df_balrog_detected_trans[cfg["SCALE_COLUMNS"]])

        if cfg["PLOT_DATA"] is True:
            plot_feature_histograms(
                cfg=cfg,
                df=df_balrog_detected_inv,
                columns=cfg["INPUT_COLUMNS"],
                title="Input Features After YJ Inverse",
                save_name=f"input_features_inv"
            )

            plot_feature_histograms(
                cfg=cfg,
                df=df_balrog_detected_inv,
                columns=cfg["OUTPUT_COLUMNS"],
                title="Output Features After YJ Inverse",
                save_name=f"output_features_inv"
            )

    start_window_logger.log_info_stream("Get bdf color after clipping")

    # calc_color_only(
    #     data_frame=df_balrog_detected,
    #     mag_type=("LUPT", "BDF"),
    #     mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
    #     bins=cfg['BDF_BINS_COLOR']
    # )
    # start_window_logger.log_info_stream("Get measured color after clipping")
    # calc_color_only(
    #     data_frame=df_balrog_detected,
    #     mag_type=("LUPT", "unsheared"),
    #     mag_col=("unsheared/lupt", "unsheared/lupt_err"),
    #     bins=cfg['UNSHEARED_BINS']
    # )

    df_training_balrog = df_balrog_detected[cfg["TRAINING_COLUMNS"]].copy()
    df_flag_balrog = df_balrog_detected[cfg["FLAG_COLUMNS"]].copy()

    today = datetime.now().strftime("%Y%m%d")

    start_window_logger.log_info_stream(f"Save training dataset as {today}_balrog_training_{len(df_training_balrog)}_nf.pkl")
    df_training_balrog.to_pickle(f"{cfg['PATH_OUTPUT']}/{today}_balrog_training_{len(df_training_balrog)}_nf.pkl")

    start_window_logger.log_info_stream(f"Save flag dataset as {today}_balrog_flag_{len(df_flag_balrog)}_nf.pkl")
    df_flag_balrog.to_pickle(f"{cfg['PATH_OUTPUT']}/{today}_balrog_flag_{len(df_flag_balrog)}_nf.pkl")

    start_window_logger.log_info_stream(f"Save default dataset as {today}_balrog_default_nan_nf.pkl")
    df_default_nan.to_pickle(f"{cfg['PATH_OUTPUT']}/{today}_balrog_default_nan_nf.pkl")

if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    sys.path.append(os.path.dirname(__file__))
    path = os.path.abspath(sys.path[-1])
    config_file_name = "mac_add_missing_columns.cfg"

    parser = argparse.ArgumentParser(description="'Start 'add missing columns'")
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

    with open(f"{path}/config/{args.config_filename}", 'r') as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    main(cfg=config)