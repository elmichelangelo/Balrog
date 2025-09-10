import matplotlib.pyplot as plt

from Handler import *
from Handler.helper_functions import *
from datetime import datetime
import sys
import os
import yaml
import argparse


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
        logger_dict={"logger_name": "create training catalogs",
                     "info_logger": cfg['INFO_LOGGER'],
                     "error_logger": cfg['ERROR_LOGGER'],
                     "debug_logger": cfg['DEBUG_LOGGER'],
                     "stream_logger": cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{cfg['PATH_OUTPUT']}Logs/"
    )

    # Write status to logger
    start_window_logger.log_info_stream("Start training catalogs")

    if cfg["MAKE_FLOW_CAT"] is True:
        df_balrog_flow = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_FLOW_TRAINING_CATALOG']}")

        start_window_logger.log_info_stream(f"Split flow data train {cfg['SIZE_TRAINING_SET']} validation {cfg['SIZE_VALIDATION_SET']} test {cfg['SIZE_TEST_SET']}")
        assert cfg['SIZE_TRAINING_SET'] + cfg['SIZE_VALIDATION_SET'] + cfg['SIZE_TEST_SET'] == 1
        valid_test_ratio = cfg['SIZE_VALIDATION_SET'] / (cfg['SIZE_VALIDATION_SET'] + cfg['SIZE_TEST_SET'])

        df_train_nf, df_temp_nf = train_test_split(df_balrog_flow, train_size=cfg['SIZE_TRAINING_SET'])
        df_valid_nf, df_test_nf = train_test_split(df_temp_nf, train_size=valid_test_ratio)

        today = datetime.now().strftime("%Y%m%d")
        start_window_logger.log_info_stream(f"Save flow training data as {cfg['PATH_OUTPUT']}{today}_balrog_train_{len(df_train_nf)}_nf_drop.pkl")
        df_train_nf.to_pickle(f"{cfg['PATH_OUTPUT']}{today}_balrog_train_{len(df_train_nf)}_nf_drop.pkl")

        start_window_logger.log_info_stream(f"Save flow validation data as {cfg['PATH_OUTPUT']}{today}_balrog_valid_{len(df_valid_nf)}_nf_drop.pkl")
        df_valid_nf.to_pickle(f"{cfg['PATH_OUTPUT']}{today}_balrog_valid_{len(df_valid_nf)}_nf_drop.pkl")

        start_window_logger.log_info_stream(f"Save flow test data as {cfg['PATH_OUTPUT']}{today}_balrog_test_{len(df_test_nf)}_nf_drop.pkl")
        df_test_nf.to_pickle(f"{cfg['PATH_OUTPUT']}{today}_balrog_test_{len(df_test_nf)}_nf_drop.pkl")

    if cfg["MAKE_CLASSIFIER_CAT"] is True:
        df_balrog_classifier = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_CLASSIFIER_TRAINING_CATALOG']}")

        start_window_logger.log_info_stream(f"Split classifier data train {cfg['SIZE_TRAINING_SET']} validation {cfg['SIZE_VALIDATION_SET']} test {cfg['SIZE_TEST_SET']}")
        assert cfg['SIZE_TRAINING_SET'] + cfg['SIZE_VALIDATION_SET'] + cfg['SIZE_TEST_SET'] == 1
        valid_test_ratio = cfg['SIZE_VALIDATION_SET'] / (cfg['SIZE_VALIDATION_SET'] + cfg['SIZE_TEST_SET'])

        df_train_cf, df_temp_cf = train_test_split(df_balrog_classifier, train_size=cfg['SIZE_TRAINING_SET'])
        df_valid_cf, df_test_cf = train_test_split(df_temp_cf, train_size=valid_test_ratio)

        today = datetime.now().strftime("%Y%m%d")
        start_window_logger.log_info_stream(f"Save classifier training data as {cfg['PATH_OUTPUT']}{today}_balrog_train_{len(df_train_cf)}_classifier.pkl")
        df_train_cf.to_pickle(f"{cfg['PATH_OUTPUT']}{today}_balrog_train_{len(df_train_cf)}_classifier.pkl")

        start_window_logger.log_info_stream(f"Save classifier validation data as {cfg['PATH_OUTPUT']}{today}_balrog_valid_{len(df_valid_cf)}_classifier.pkl")
        df_valid_cf.to_pickle(f"{cfg['PATH_OUTPUT']}{today}_balrog_valid_{len(df_valid_cf)}_classifier.pkl")

        start_window_logger.log_info_stream(f"Save classifier test data as {cfg['PATH_OUTPUT']}{today}_balrog_test_{len(df_test_cf)}_classifier.pkl")
        df_test_cf.to_pickle(f"{cfg['PATH_OUTPUT']}{today}_balrog_test_{len(df_test_cf)}_classifier.pkl")


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__))
    path = os.path.abspath(sys.path[-1])
    config_file_name = "mac_create_training_catalogs.cfg"

    parser = argparse.ArgumentParser(description='Start create training catalogs')
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