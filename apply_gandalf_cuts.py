import matplotlib.pyplot as plt
import pandas as pd

from Handler import *
from Handler.helper_functions import *
from Handler.cut_functions import apply_gandalf_cuts, apply_cuts, apply_gatti_cuts
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy.stats import mstats
from datetime import datetime
import sys
import os
import yaml
import argparse


def plot_feature_histograms(cfg, df, columns, title, save_name, bins=100):
    os.makedirs(cfg['PATH_PLOTS'], exist_ok=True)
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
    # plt.savefig(f"{cfg['PATH_PLOTS']}/{save_name}.pdf", bbox_inches='tight', dpi=300)
    plt.show()

def main(cfg):
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
    df_balrog = df_balrog[df_balrog["detected"] == 1]

    print(f"Length Balrog w/o cuts:", len(df_balrog))
    start_window_logger.log_info_stream("Calc measured mag and color")
    df_balrog = calc_color(
        cfg=cfg,
        data_frame=df_balrog,
        mag_type=("MAG", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/mag", "unsheared/mag_err"),
        bins=cfg['UNSHEARED_BINS'],
        save_name=f"unsheared/mag"
    )

    df_balrog = apply_gandalf_cuts(
        cfg=cfg,
        data_frame=df_balrog
    )

    if cfg["DO_SOME_TESTS"] is True:
        for k in cfg["OUTPUT_COLUMNS"]:
            print(k, df_balrog[k].isna().sum())
        plot_feature_histograms(
            cfg=cfg,
            df=df_balrog,
            columns=cfg["OUTPUT_COLUMNS"],
            title="Applying the gandalf et al. Cuts",
            save_name=f"after_applying_gandalf_cuts"
        )

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
        mag_type=("MAG", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
        bins=cfg['BDF_BINS'],
        save_name=f"bdf_mag"
    )

    if cfg["DO_SOME_TESTS"] is True:
        for k in cfg["INPUT_COLUMNS"]:
            print(k, df_balrog[k].isna().sum())
        plot_feature_histograms(
            cfg=cfg,
            df=df_balrog,
            columns=cfg["INPUT_COLUMNS"],
            title="Applying the WL Cuts",
            save_name=f"after_applying_cuts"
        )
        len(df_balrog)

    df_training_balrog = df_balrog[cfg["TRAINING_COLUMNS"]].copy()
    df_flag_balrog = df_balrog[cfg["FLAG_COLUMNS"]].copy()

    today = datetime.now().strftime("%Y%m%d")

    start_window_logger.log_info_stream(f"Save training dataset as {today}_balrog_training_{len(df_training_balrog)}_nf.pkl")
    df_training_balrog.to_pickle(f"{cfg['PATH_OUTPUT']}/{today}_balrog_training_{len(df_training_balrog)}_nf.pkl")

    start_window_logger.log_info_stream(f"Save flag dataset as {today}_balrog_flag_{len(df_flag_balrog)}_nf.pkl")
    df_flag_balrog.to_pickle(f"{cfg['PATH_OUTPUT']}/{today}_balrog_flag_{len(df_flag_balrog)}_nf.pkl")

if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    sys.path.append(os.path.dirname(__file__))
    path = os.path.abspath(sys.path[-1])
    config_file_name = "mac_apply_gandalf_cuts.cfg"

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