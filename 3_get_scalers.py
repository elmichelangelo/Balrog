from Handler import *
from Handler.helper_functions import *
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import sys
import os
import yaml
import argparse
import joblib


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
        if col in ["unsheared/mag_err_r", "unsheared/mag_err_i", "unsheared/mag_err_z"]:
            ax.set_xlabel(f"log10({col})")
        else:
            ax.set_xlabel(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title)
    if cfg['SAVE_PLOT'] is True:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
    if cfg['SHOW_PLOT'] is True:
        plt.show()


def get_scaler(lst_columns_of_interest, log_scaler, data_frame):
    dict_sscaler = {}
    log_scaler.log_info_stream("Get scaler")
    data_frame_scaled = data_frame.copy()

    for col in lst_columns_of_interest:
        log_scaler.log_info_stream(f"Get scaler for {col}")
        sscaler = StandardScaler()
        value = data_frame[col].values
        data_frame_scaled[col]= sscaler.fit_transform(value.reshape(-1, 1))
        dict_sscaler[col] = sscaler
    return dict_sscaler, data_frame_scaled

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

    # Initialize the logger
    start_window_logger = LoggerHandler(
        logger_dict={"logger_name": "create save scaler",
                     "info_logger": cfg['INFO_LOGGER'],
                     "error_logger": cfg['ERROR_LOGGER'],
                     "debug_logger": cfg['DEBUG_LOGGER'],
                     "stream_logger": cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{cfg['PATH_OUTPUT']}Logs/"
    )

    # Write status to logger
    start_window_logger.log_info_stream("Start save scaler")
    today = datetime.now().strftime("%Y%m%d")

    if cfg["GET_FLOW_SCALER"] is True:
        start_window_logger.log_info_stream("Load Flow Catalog")
        df_balrog_flow = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_NF_CATALOG']}")

        for band in ["r", "i", "z"]:
            df_balrog_flow[f"unsheared/mag_err_{band}"] = np.log10(df_balrog_flow[f"unsheared/mag_err_{band}"])

        start_window_logger.log_info_stream("Get Scaler FLow")
        dict_scalers_standard_nf, data_frame_nf_scaled = get_scaler(
            lst_columns_of_interest=cfg["NF_COLUMNS_OF_INTEREST"],
            log_scaler=start_window_logger,
            data_frame=df_balrog_flow
        )

        if cfg["PLOT_DATA"] is True:
            os.makedirs(cfg['PATH_PLOTS'], exist_ok=True)
            plot_feature_histograms(
                cfg=cfg,
                df=data_frame_nf_scaled,
                columns=cfg["NF_INPUT_COLUMNS"],
                title=f"Standard Scaled Input Features",
                save_name=f"{cfg['PATH_PLOTS']}/{today}_scaled_input_features_nf.pdf",
                bins=100
            )
            plot_feature_histograms(
                cfg=cfg,
                df=data_frame_nf_scaled,
                columns=cfg["NF_OUTPUT_COLUMNS"],
                title=f"Standard Scaled Output Features",
                save_name=f"{cfg['PATH_PLOTS']}/{today}_scaled_output_features_nf.pdf",
                bins=100
            )

        if cfg["SAVE_SCALER"] is True:
            os.makedirs(cfg['PATH_OUTPUT'], exist_ok=True)
            start_window_logger.log_info_stream(f"Save nf standard scaler as {cfg['PATH_OUTPUT']}{today}_StandardScalers_nf.pkl")
            joblib.dump(dict_scalers_standard_nf, f"{cfg['PATH_OUTPUT']}{today}_StandardScalers_nf.pkl")


    if cfg["GET_CLASSIFIER_SCALER"] is True:
        start_window_logger.log_info_stream("Load Classifier Catalog")
        df_balrog_classifier = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_CF_CATALOG']}")

        start_window_logger.log_info_stream("Get Scaler Classifier")
        dict_scalers_standard_classifier, data_frame_classifier_scaled = get_scaler(
            lst_columns_of_interest=cfg["CF_COLUMNS_OF_INTEREST"],
            log_scaler=start_window_logger,
            data_frame=df_balrog_classifier

        )

        if cfg["PLOT_DATA"] is True:
            os.makedirs(cfg['PATH_PLOTS'], exist_ok=True)
            plot_feature_histograms(
                cfg=cfg,
                df=data_frame_classifier_scaled,
                columns=cfg["CF_INPUT_COLUMNS"],
                title=f"Standard Scaled Input Features",
                save_name=f"{cfg['PATH_PLOTS']}/{today}_scaled_input_features_cf.pdf",
                bins=100
            )

        if cfg["SAVE_SCALER"] is True:
            os.makedirs(cfg['PATH_OUTPUT'], exist_ok=True)
            start_window_logger.log_info_stream(f"Save classifier standard scaler as {cfg['PATH_OUTPUT']}/{today}_StandardScalers_cf.pkl")
            joblib.dump(dict_scalers_standard_classifier, f"{cfg['PATH_OUTPUT']}/{today}_StandardScalers_cf.pkl")


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__))
    path = os.path.abspath(sys.path[-1])
    config_file_name = "mac_get_scaler.cfg"

    parser = argparse.ArgumentParser(description='Start get scaler minmax')
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