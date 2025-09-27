from Handler import *
from Handler.helper_functions import *
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
    if cfg['SAVE_PLOT'] is True:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
    if cfg['SHOW_PLOT'] is True:
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

    start_window_logger.log_info_stream("Calc bdf mag and color")
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

    today = datetime.now().strftime("%Y%m%d")

    if cfg["PLOT_DATA"] is True:
        os.makedirs(cfg["PATH_PLOTS"], exist_ok=True)
        plot_feature_histograms(
            cfg=cfg,
            df=df_balrog,
            columns=cfg["CF_INPUT_COLUMNS"],
            title=f"Classifier input features w/o cuts",
            save_name=f"{cfg['PATH_PLOTS']}/{today}_cf_input_features_w-o_cuts.pdf",
            bins=100
        )
        plot_feature_histograms(
            cfg=cfg,
            df=df_balrog,
            columns=cfg["NF_OUTPUT_COLUMNS"],
            title=f"Normalizing FLow input features w/o cuts",
            save_name=f"{cfg['PATH_PLOTS']}/{today}_nf_input_features_w-o_cuts.pdf",
            bins=100
        )

    os.makedirs(cfg["PATH_OUTPUT"], exist_ok=True)

    start_window_logger.log_info_stream(f"Save complete dataset as {today}_balrog_complete_{len(df_balrog)}.pkl")
    df_balrog.to_pickle(f"{cfg['PATH_OUTPUT']}/{today}_balrog_complete_{len(df_balrog)}.pkl")

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