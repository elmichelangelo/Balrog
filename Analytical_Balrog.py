from Handler.helper_functions import *
import sys
import yaml
import argparse
import pandas as pd


def data_preprocessing(cfg):
    """"""
    df_balrog = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MERGED_CAT']}")
    df_balrog = df_balrog[df_balrog["detected"] == 1]

    df_balrog.rename(
        columns={
            'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
            'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
        inplace=True)
    df_balrog["BDF_G"] = np.sqrt(df_balrog["BDF_G_0"] ** 2 + df_balrog["BDF_G_1"] ** 2)

    print(df_balrog)
    print(df_balrog.isna().sum())

    if cfg['APPLY_FILL_NA'] is True:
        print(f"start fill na default")
        for col in cfg['FILL_NA'].keys():
            df_balrog[col].fillna(cfg['FILL_NA'][col], inplace=True)
            print(f"fill na default: col={col} val={cfg['FILL_NA'][col]}")
    else:
        print("No fill na")

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

    df_add_columns = df_balrog[cfg["ADD_COLUMNS"]]
    df_balrog = df_balrog[cfg["COVARIANCE_COLUMNS"]]

    print(df_balrog)
    print(df_balrog.isna().sum())
    return df_balrog, df_add_columns


def get_covariance_matrix(cfg, data_frame):
    """"""

    data_frame = data_frame[cfg["COVARIANCE_COLUMNS"]]
    arr_data = data_frame.to_numpy()

    # Now calculate the covariance matrix
    cov_matrix = np.cov(arr_data, rowvar=False)
    print(f"covariance matrix deep distribution: {cov_matrix}")

    # Creating the covariance matrix dictionary
    dictionary_covariance = {}

    for column in cfg["COVARIANCE_COLUMNS"]:
        dictionary_covariance[f"mean {column}"] = data_frame[column].mean()

    data_frame_cov_mean = pd.DataFrame(dictionary_covariance, index=[0])

    for idx_bin, bin in enumerate(cfg["BDF_BINS"]):
        data_frame.loc[:, f"Color BDF MAG {bin}-{cfg['BDF_BINS'][idx_bin+1]}"] = (
                data_frame[f"BDF_MAG_DERED_CALIB_{bin}"].values - data_frame[f"BDF_MAG_DERED_CALIB_{cfg['BDF_BINS'][idx_bin+1]}"].values
        )
        if idx_bin + 1 == len(cfg["BDF_BINS"]) - 1:
            break

    for idx_bin, bin in enumerate(cfg["UNSHEARED_BINS"]):
        data_frame.loc[:, f"Color unsheared mag {bin}-{cfg['UNSHEARED_BINS'][idx_bin+1]}"] = (
                data_frame[f"unsheared/mag_{bin}"].values - data_frame[f"unsheared/mag_{cfg['UNSHEARED_BINS'][idx_bin+1]}"].values
        )
        if idx_bin + 1 == len(cfg["UNSHEARED_BINS"]) - 1:
            break

    if cfg["PLOT_TRUE_DEEP_COLOR"] is True:
        plot_corner(
            data_frame,
            columns=[
                "Color BDF MAG U-G",
                "Color BDF MAG G-R",
                "Color BDF MAG R-I",
                "Color BDF MAG I-Z",
                "Color BDF MAG Z-J",
                "Color BDF MAG J-H",
                "Color BDF MAG H-K"
            ],
            labels=[
                "U-G",
                "G-R",
                "R-I",
                "I-Z",
                "Z-J",
                "J-H",
                "H-K"
            ],
            title="True Deep Color",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/true_deep_color.png"
        )

    if cfg["PLOT_TRUE_DEEP_MAG"] is True:
        plot_corner(
            data_frame,
            columns=[
                "BDF_MAG_DERED_CALIB_R",
                "BDF_MAG_DERED_CALIB_I",
                "BDF_MAG_DERED_CALIB_Z"
            ],
            labels=["r", "i", "z"],
            title="True Deep Magnitude",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/true_deep_mag.png"
        )

    if cfg["PLOT_TRUE_OBS_CONDITIONS"] is True:
        plot_corner(
            data_frame,
            columns=[
                "BDF_T",
                "BDF_G",
                "FWHM_WMEAN_R",
                "FWHM_WMEAN_I",
                "FWHM_WMEAN_Z",
                "AIRMASS_WMEAN_R",
                "AIRMASS_WMEAN_I",
                "AIRMASS_WMEAN_Z",
                "MAGLIM_R",
                "MAGLIM_I",
                "MAGLIM_Z",
                "EBV_SFD98",
            ],
            labels=[
                "bdf t",
                "bdf g",
                "fwhm r",
                "fwhm i",
                "fwhm z",
                "airmass r",
                "airmass i",
                "airmass z",
                "maglim r",
                "maglim i",
                "maglim z",
                "ebv"
            ],
            title="True Observing Conditions",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/true_obs_cond.png"
        )

    if cfg["PLOT_TRUE_MEAS_PROPERTIES"] is True:
        plot_corner(
            data_frame,
            columns=[
                "unsheared/snr",
                "unsheared/size_ratio",
                "unsheared/weight",
                "unsheared/T"
            ],
            labels=["snr", "size ratio", "weight", "t"],
            title="True Measured Properties",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/true_meas_properties.png"
        )

    if cfg["PLOT_TRUE_MEAS_MAG"] is True:
        plot_corner(
            data_frame,
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
            data_frame,
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

    return data_frame, data_frame_cov_mean, cov_matrix


def generate_mock(cfg, df_cov_mean, cov_matrix, df_true):
    """"""
    size = cfg["SIZE_MOCK"]

    if size is None:
        size = len(df_true)

    arr_mean = np.concatenate([df_cov_mean[f"mean {column}"].values for column in cfg["COVARIANCE_COLUMNS"]])
    arr_multi_normal = np.random.multivariate_normal(
        arr_mean,
        cov_matrix,
        size
    )

    df_mock = pd.DataFrame(
        arr_multi_normal,
        columns=cfg["COVARIANCE_COLUMNS"]
    )

    for idx_bin, bin in enumerate(cfg["BDF_BINS"]):
        df_mock.loc[:, f"Color BDF MAG {bin}-{cfg['BDF_BINS'][idx_bin+1]}"] = (
                df_mock[f"BDF_MAG_DERED_CALIB_{bin}"].values - df_mock[f"BDF_MAG_DERED_CALIB_{cfg['BDF_BINS'][idx_bin+1]}"].values
        )
        if idx_bin + 1 == len(cfg["BDF_BINS"]) - 1:
            break

    for idx_bin, bin in enumerate(cfg["UNSHEARED_BINS"]):
        df_mock.loc[:, f"Color unsheared mag {bin}-{cfg['UNSHEARED_BINS'][idx_bin+1]}"] = (
                df_mock[f"unsheared/mag_{bin}"].values - df_mock[f"unsheared/mag_{cfg['UNSHEARED_BINS'][idx_bin+1]}"].values
        )
        if idx_bin + 1 == len(cfg["UNSHEARED_BINS"]) - 1:
            break

    print(f"len generated data: {len(df_mock)}")
    print(f"len true data: {len(df_true)}")

    if cfg["PLOT_COMPARE_DEEP_COLOR"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock,
            data_frame_true=df_true,
            columns=[
                "Color BDF MAG U-G",
                "Color BDF MAG G-R",
                "Color BDF MAG R-I",
                "Color BDF MAG I-Z",
                "Color BDF MAG Z-J",
                "Color BDF MAG J-H",
                "Color BDF MAG H-K"
            ],
            labels=[
                "U-G",
                "G-R",
                "R-I",
                "I-Z",
                "Z-J",
                "J-H",
                "H-K"
            ],
            title="Compare Deep Color",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_deep_color.png"
        )

    if cfg["PLOT_MOCK_DEEP_COLOR"] is True:
        plot_corner(
            df_mock,
            columns=[
                "Color BDF MAG U-G",
                "Color BDF MAG G-R",
                "Color BDF MAG R-I",
                "Color BDF MAG I-Z",
                "Color BDF MAG Z-J",
                "Color BDF MAG J-H",
                "Color BDF MAG H-K"
            ],
            labels=[
                "U-G",
                "G-R",
                "R-I",
                "I-Z",
                "Z-J",
                "J-H",
                "H-K"
            ],
            title="Mock Deep Color",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_deep_color.png"
        )

    if cfg["PLOT_COMPARE_DEEP_MAG"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock,
            data_frame_true=df_true,
            columns=[
                "BDF_MAG_DERED_CALIB_R",
                "BDF_MAG_DERED_CALIB_I",
                "BDF_MAG_DERED_CALIB_Z"
            ],
            labels=[
                "r",
                "i",
                "z"
            ],
            title="Compare Deep Magnitude",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_deep_mag.png"
        )

    if cfg["PLOT_MOCK_DEEP_MAG"] is True:
        plot_corner(
            df_mock,
            columns=[
                "BDF_MAG_DERED_CALIB_R",
                "BDF_MAG_DERED_CALIB_I",
                "BDF_MAG_DERED_CALIB_Z"
            ],
            labels=[
                "r",
                "i",
                "z"
            ],
            title="Mock Deep Magnitude",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_deep_mag.png"
        )

    if cfg["PLOT_COMPARE_OBS_CONDITIONS"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock,
            data_frame_true=df_true,
            columns=[
                "BDF_T",
                "BDF_G",
                "FWHM_WMEAN_R",
                "FWHM_WMEAN_I",
                "FWHM_WMEAN_Z",
                "AIRMASS_WMEAN_R",
                "AIRMASS_WMEAN_I",
                "AIRMASS_WMEAN_Z",
                "MAGLIM_R",
                "MAGLIM_I",
                "MAGLIM_Z",
                "EBV_SFD98"
            ],
            labels=[
                "bdf t",
                "bdf g",
                "fwhm r",
                "fwhm i",
                "fwhm z",
                "airmass r",
                "airmass i",
                "airmass z",
                "maglim r",
                "maglim i",
                "maglim z",
                "ebv"
            ],
            title="Compare Observing Conditions",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_obs_cond.png"
        )

    if cfg["PLOT_MOCK_OBS_CONDITIONS"] is True:
        plot_corner(
            df_mock,
            columns=[
                "BDF_T",
                "BDF_G",
                "FWHM_WMEAN_R",
                "FWHM_WMEAN_I",
                "FWHM_WMEAN_Z",
                "AIRMASS_WMEAN_R",
                "AIRMASS_WMEAN_I",
                "AIRMASS_WMEAN_Z",
                "MAGLIM_R",
                "MAGLIM_I",
                "MAGLIM_Z",
                "EBV_SFD98"
            ],
            labels=[
                "bdf t",
                "bdf g",
                "fwhm r",
                "fwhm i",
                "fwhm z",
                "airmass r",
                "airmass i",
                "airmass z",
                "maglim r",
                "maglim i",
                "maglim z",
                "ebv"
            ],
            title="Mock Observing Conditions",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_obs_cond.png"
        )

    if cfg["PLOT_COMPARE_MEAS_PROPERTIES"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock,
            data_frame_true=df_true,
            columns=[
                "unsheared/snr",
                "unsheared/size_ratio",
                "unsheared/weight",
                "unsheared/T"
            ],
            labels=["snr", "size ratio", "weight", "t"],
            title="True Measured Properties",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_meas_properties.png"
        )

    if cfg["PLOT_MOCK_MEAS_PROPERTIES"] is True:
        plot_corner(
            df_mock,
            columns=[
                "unsheared/snr",
                "unsheared/size_ratio",
                "unsheared/weight",
                "unsheared/T"
            ],
            labels=["snr", "size ratio", "weight", "t"],
            title="Mock Measured Properties",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_meas_properties.png"
        )

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

    return df_mock

def add_additional_columns(data_frame_mock, data_frame_add_columns):
    """"""
    for column in data_frame_add_columns.columns:
        data_frame_mock[column] = data_frame_add_columns[column].values

    return data_frame_mock


def save_mock(cfg, data_frame):
    """"""
    data_frame.to_pickle(f"{cfg['PATH_OUTPUT']}/Catalogs/{cfg['FILENAME_MOCK_CAT']}")


def main(cfg):
    """"""
    df_true, df_add_columns = data_preprocessing(cfg=cfg)

    df_true, df_covariance, cov_matrix = get_covariance_matrix(
        cfg=cfg,
        data_frame=df_true
    )

    df_mock = generate_mock(
        cfg=cfg,
        df_cov_mean=df_covariance,
        cov_matrix=cov_matrix,
        df_true=df_true
    )

    df_mock = add_additional_columns(
        data_frame_mock=df_mock,
        data_frame_add_columns=df_add_columns
    )

    save_mock(
        cfg=cfg,
        data_frame=df_mock
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