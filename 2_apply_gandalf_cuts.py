from Handler import *
from Handler.helper_functions import *
from Handler.cut_functions import apply_galaxy_cuts, apply_photometric_cuts
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

    start_window_logger.log_info_stream("Start 'apply gandalf cuts'")
    df_balrog = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_COMPLETE_CAT']}")

    start_window_logger.log_info_stream(f"Length Balrog w/o cuts: {len(df_balrog)}")

    df_balrog = apply_galaxy_cuts(
        data_frame=df_balrog
    )

    start_window_logger.log_info_stream(f"Length Balrog after applying galaxy cuts: {len(df_balrog)}")

    df_balrog_nf = df_balrog[df_balrog["detected"] == 1].copy()

    start_window_logger.log_info_stream(f"Length Balrog catalog only detected: {len(df_balrog_nf)}")

    df_balrog_nf = apply_photometric_cuts(
        cfg=cfg,
        data_frame=df_balrog_nf
    )

    today = datetime.now().strftime("%Y%m%d")

    if cfg["PLOT_DATA"] is True:
        os.makedirs(cfg["PATH_PLOTS"], exist_ok=True)
        plot_feature_histograms(
            cfg=cfg,
            df=df_balrog,
            columns=cfg["CF_INPUT_COLUMNS"],
            title=f"Classifier input features w/ cuts",
            save_name=f"{cfg['PATH_PLOTS']}/{today}_cf_input_features_w_cuts.pdf",
            bins=100
        )

        plot_feature_histograms(
            cfg=cfg,
            df=df_balrog_nf,
            columns=cfg["NF_OUTPUT_COLUMNS"],
            title=f"NF output features w/ cuts",
            save_name=f"{cfg['PATH_PLOTS']}/{today}_nf_output_features_w_cuts.pdf",
            bins=100
        )

    for k in df_balrog.keys():
        start_window_logger.log_info_stream(f"Number of NaNs in cf data {k}: {df_balrog[k].isna().sum()}")
        start_window_logger.log_info_stream(f"Number of NaNs in nf data {k}: {df_balrog_nf[k].isna().sum()}")

    start_window_logger.log_info_stream(f"Total number of NaNs in cf data: {df_balrog.isna().sum().sum()}")
    start_window_logger.log_info_stream(f"Total number of NaNs in nf data: {df_balrog_nf.isna().sum().sum()}")

    df_balrog = df_balrog[cfg['COLUMNS_OF_INTEREST']]
    df_balrog_nf = df_balrog_nf[cfg['COLUMNS_OF_INTEREST']]

    if cfg["SAVE_CATALOGS"] is True:
        start_window_logger.log_info_stream(f"Save training dataset as {today}_balrog_w_cuts_{len(df_balrog)}_cf.pkl")
        df_balrog.to_pickle(f"{cfg['PATH_OUTPUT']}/{today}_balrog_w_cuts_{len(df_balrog)}_cf.pkl")

        start_window_logger.log_info_stream(f"Save training dataset as {today}_balrog_w_cuts_{len(df_balrog_nf)}_nf.pkl")
        df_balrog_nf.to_pickle(f"{cfg['PATH_OUTPUT']}/{today}_balrog_w_cuts_{len(df_balrog_nf)}_nf.pkl")

if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    sys.path.append(os.path.dirname(__file__))
    path = os.path.abspath(sys.path[-1])
    config_file_name = "mac_apply_gandalf_cuts.cfg"

    parser = argparse.ArgumentParser(description="'Start 'apply gandalf cuts'")
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