import matplotlib.pyplot as plt

from Handler.helper_functions import *
import sys
import yaml
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from chainconsumer import ChainConsumer


# def set_plot_settings():
#     # Set figsize and figure layout
#     plt.rcParams["figure.figsize"] = [16, 9]
#     plt.rcParams["figure.autolayout"] = True
#     sns.set_theme()
#     sns.set_context("paper")
#     sns.set(font_scale=2)
#     # plt.rc('font', size=10)  # controls default text size
#     plt.rc('axes', titlesize=18)  # fontsize of the title
#     plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=10)  # fontsize of the x tick labels
#     plt.rc('ytick', labelsize=10)  # fontsize of the y tick labels
#     plt.rc('legend', fontsize=12)  # fontsize of the legend


def data_preprocessing(cfg):
    """"""
    df_balrog = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MERGED_CAT']}")
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

    arr_detected = df_balrog["detected"].values
    df_balrog = df_balrog[cfg["ANALYTICAL_MODEL_COLUMNS"]]

    print(df_balrog)
    print(df_balrog.isna().sum())
    return df_balrog, arr_detected


# def exploratory_data_analysis(cfg, data_frame):
#     """"""
#
#     # Call get_covariance_matrix to get the covariance matrix
#
#
#     # calc_fit_parameters(
#     #     distribution=dictionary_generated_data["generated deep i mag"],
#     #     lower_bound=18.5,
#     #     upper_bound=23,
#     #     bin_size=40,
#     #     function=exp_func,
#     #     column="generated deep i mag",
#     #     plot_data=True,
#     #     save_plot=True,
#     #     save_name="_generated",
#     #     save_path_plots="Output"
#     # )


# def calc_fit_parameters(distribution, lower_bound, upper_bound, bin_size, function, column=None, plot_data=None,
#                         save_plot=None, save_path_plots="", save_name="", show_plot=False):
#     """"""
#     # Create bins for fit range
#     bins = np.linspace(lower_bound, upper_bound, bin_size)
#
#     # Calculate bin width for hist plot
#     bin_width = bins[1] - bins[0]
#
#     # Calculate the center of the bins
#     bins_centers = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
#
#     if plot_data is True:
#         sns.histplot(
#             x=distribution,
#             element="step",
#             fill=False,
#             color="darkred",
#             binwidth=bin_width,
#             log_scale=(False, True),
#             stat="probability",
#             label=f"probability of DES deep field bin: {column}"
#         )
#
#         # Set some plot parameters
#         plt.title(f"probability histogram of DES deep field '{column}'")
#         plt.xlabel("magnitude deep field i band")
#         plt.legend()
#         if save_plot is True:
#             plt.savefig(f"{save_path_plots}/Power_law_DES-{column}_log_wo_bounds{save_name}.png")
#         if show_plot is True:
#             plt.show()
#         plt.clf()
#
#     distribution = distribution[distribution[:] <= upper_bound]
#     distribution = distribution[lower_bound <= distribution[:]]
#
#     # Calculate the histogram fit values
#     data_entries = np.histogram(distribution, bins=bins)[0]
#
#     # Get the probability
#     data_probability_entries = data_entries / np.sum(data_entries)
#
#     if plot_data is True:
#         sns.histplot(
#             x=distribution,
#             element="step",
#             fill=False,
#             color="darkred",
#             binwidth=bin_width,
#             log_scale=(False, True),
#             stat="probability",
#             label=f"probability of DES deep field bin: {column}"
#         )
#
#         plt.plot(bins_centers, data_probability_entries, '.', color="darkred")
#
#         # Set some plot parameters
#         plt.xlim((lower_bound, upper_bound))
#         plt.title(f"probability histogram of DES deep field '{column}'")
#         plt.xlabel("magnitude deep field i band")
#         plt.legend()
#         if save_plot is True:
#             plt.savefig(f"{save_path_plots}/Probability_DES-{column}{save_name}.png")
#         if show_plot is True:
#             plt.show()
#         plt.clf()
#
#     # fit exponential function to calculated histogram values from before
#     popt, pcov = curve_fit(function, xdata=bins_centers, ydata=data_probability_entries)
#
#     if plot_data is True:
#         sns.histplot(
#             x=distribution,
#             element="step",
#             fill=False,
#             color="darkred",
#             binwidth=bin_width,
#             log_scale=(False, True),
#             stat="probability",
#             label=f"probability of DES deep field bin: {column}"
#         )
#
#         plt.plot(bins_centers, data_probability_entries, '.', color="darkred")
#
#         # Plot the fit function
#         plt.plot(
#             bins_centers,
#             function(bins_centers, *popt),
#             color='darkblue',
#             linewidth=2.5,
#             label=f'power law - f(x)=Exp[a+b*x]\n bin width {bin_width:2.4f} ; a={popt[0]:2.4f} ; b={popt[1]:2.4f}'
#         )
#
#         # Set some plot parameters
#         plt.xlim((lower_bound, upper_bound))
#         plt.title(f"probability histogram and power law of DES deep field '{column}'")
#         plt.xlabel("magnitude deep field i band")
#         plt.legend()
#         if save_plot is True:
#             plt.savefig(f"{save_path_plots}/Curve_fit_DES-{column}{save_name}.png")
#         if show_plot is True:
#             plt.show()
#         plt.clf()
#
#     dictionary_fit_parameter = {
#         f"{column} parameter optimal": popt,
#         f"{column} parameter covariance": pcov,
#         f"{column}": np.array(distribution),
#         f"{column} probability": data_probability_entries,
#         f"{column} counts": data_entries,
#         "bin width": bin_width,
#         "bin centers": bins_centers,
#         "bins": bins
#     }
#
#     return dictionary_fit_parameter


# def exp_func(x, a, b):
#     """
#     Exponential fit function: f(x)=exp[a+b*x]
#     Args:
#         x (np.array): array of x values
#         a (int): intersection with y-axis
#         b (int): slope
#
#     Returns:
#         y (numpy array): array with calculated f(x)
#     """
#     y = np.exp(a) * np.exp(b*x)
#     return y


# def generate_distribution(dict_fit_params, column, lower_bound, upper_bound, plot_data, save_plot, save_path_plots,
#                           size, show_plot=False):
#     """"""
#
#     # Create numpy array for x-axis
#     xspace = np.linspace(lower_bound, upper_bound, num=size, endpoint=True)
#
#     # Init new dictionary
#     dictionary_generated_data = {}
#
#     # Calculating the probability with defined fit function and calculated fit parameters
#     prob = exp_func(
#         xspace,
#         *dict_fit_params[f"{column} parameter optimal"]
#     )
#
#     # Normalize the probability
#     norm_prob = prob / np.sum(prob)
#
#     # Use probability to select magnitudes
#     arr_generated_data = np.random.choice(xspace, size=size, p=norm_prob)
#
#     if plot_data is True:
#         sns.histplot(
#             x=arr_generated_data,
#             element="step",
#             fill=False,
#             color="darkgreen",
#             binwidth=dict_fit_params["bin width"],
#             log_scale=(False, True),
#             stat="probability",
#             label=f"generated probability"
#         )
#
#         # plt.xlim((lower_bound, upper_bound))
#         plt.title(f"generated probability histogram")
#         plt.xlabel("magnitude deep field i band")
#         plt.legend()
#         if save_plot is True:
#             plt.savefig(f"{save_path_plots}/Generated_distribution.png")
#         if show_plot is True:
#             plt.show()
#
#     # Write generated Data to dictionary
#     dictionary_generated_data[f"generated {column}"] = arr_generated_data
#
#     # Calculate the histogram fit values
#     data_entries = np.histogram(arr_generated_data, bins=dict_fit_params["bins"])[0]
#
#     # fit exponential function to calculated histogram values from before
#     # Get the probability
#     data_probability_entries = data_entries / np.sum(data_entries)
#     popt, pcov = curve_fit(exp_func, xdata=dict_fit_params["bin centers"], ydata=data_probability_entries)
#
#     print(f"generated {column} slope", popt[1])
#     print(f"generated {column} offset", popt[0])
#     print(f"{column} slope", dict_fit_params[f"{column} parameter optimal"][1])
#     print(f"{column} offet", dict_fit_params[f"{column} parameter optimal"][0])
#
#     if plot_data is True:
#         sns.histplot(
#             x=dict_fit_params[f"{column}"],
#             element="step",
#             fill=False,
#             color="darkred",
#             binwidth=dict_fit_params["bin width"],
#             log_scale=(False, True),
#             stat="probability",
#             label=f"probability of DES deep field mag in {column}"
#         )
#
#         plt.plot(dict_fit_params["bin centers"], dict_fit_params[f"{column} probability"], '.', color="darkred")
#
#         sns.histplot(
#             x=arr_generated_data,
#             element="step",
#             fill=False,
#             color="darkgreen",
#             binwidth=dict_fit_params["bin width"],
#             log_scale=(False, True),
#             stat="probability",
#             label=f"probability of generated mag in I"
#         )
#
#         plt.plot(dict_fit_params["bin centers"], data_probability_entries, '.', color="darkgreen")
#
#         # Plot the fit function
#         plt.plot(
#             dict_fit_params["bin centers"],
#             exp_func(dict_fit_params["bin centers"], *dict_fit_params[f"{column} parameter optimal"]),
#             color='darkblue',
#             linewidth=2.5,
#             label=f'power law of BDF_FLUX_DERED_CALIB_I - f(x)=Exp[a+b*x]\nbin width {dict_fit_params["bin width"]:2.4f} ; a={dict_fit_params[f"{column} parameter optimal"][0]:2.4f} ; b={dict_fit_params[f"{column} parameter optimal"][1]:2.4f}'
#         )
#
#         # Plot the fit function
#         plt.plot(
#             xspace,
#             exp_func(xspace, *popt),
#             color='darkorange',
#             linewidth=2.5,
#             label=f'power law generated magnitude - f(x)=Exp[a+b*x]\nbin width {dict_fit_params["bin width"]:2.4f} ; a={popt[0]:2.4f} ; b={popt[1]:2.4f}'
#         )
#
#         # Set some plot parameters
#         plt.xlim((lower_bound, upper_bound))
#         plt.title(f"Calculate {column} fit from DES deep field catalog")
#         plt.xlabel("magnitude deep field i band")
#         plt.legend()
#         if save_plot is True:
#             plt.savefig(f"{save_path_plots}/Curve_fit_DES-{column}_and_generated_data.png")
#         if show_plot is True:
#             plt.show()
#
#     return dictionary_generated_data


def get_covariance_matrix(cfg, data_frame):
    """"""
    data_frame = data_frame[cfg["COVARIANCE_COLUMNS"]]
    # sampled_df = data_frame.sample(n=len(data_frame), random_state=42)
    arr_data = data_frame.to_numpy()

    # Now calculate the covariance matrix
    cov_matrix = np.cov(arr_data, rowvar=False)
    print(f"covariance matrix deep distribution ugrizjhk: {cov_matrix}")

    # Creating the covariance matrix dictionary
    dictionary_covariance = {}

    for column in cfg["COVARIANCE_COLUMNS"]:
        dictionary_covariance[f"mean {column}"] = data_frame[column].mean()

    data_frame_cov_mean = pd.DataFrame(dictionary_covariance, index=[0])
    dictionary_covariance["covariance matrix deep ugrizjhk"] = cov_matrix

    if cfg["PLOT_DEEP_COLOR"] is True:
        plot_corner(
            data_frame_cov_mean,
            ["deep u-g", "deep g-r", "deep r-i", "deep i-z", "deep z-j", "deep j-h", "deep h-ks"],
            ["u-g", "g-r", "r-i", "i-z", "z-j", "j-h", "h-ks"],
            "title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/deep_color.png"
        )

    if cfg["PLOT_DEEP_MAG"] is True:
        plot_corner(
            data_frame_cov_mean,
            ["deep r", "deep i", "deep z"],
            ["r", "i", "z"],
            "title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/deep_magnitude.png"
        )

    if cfg["PLOT_DEEP_CONDITIONS"] is True:
        plot_corner(
            data_frame_cov_mean,
            ["bdf t", "bdf g", "fwhm r", "fwhm i", "fwhm z", "airmass r", "airmass i", "airmass z", "maglim r", "maglim i", "maglim z", "ebv"],
            ["bdf t", "bdf g", "fwhm r", "fwhm i", "fwhm z", "airmass r", "airmass i", "airmass z", "maglim r", "maglim i", "maglim z", "ebv"],
            "title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/deep_cond.png"
        )

    if cfg["PLOT_MEAS_PROPERTIES"] is True:
        plot_corner(
            data_frame_cov_mean,
            ["snr", "size ratio", "weight", "t"],
            ["snr", "size ratio", "weight", "t"],
            "title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/meas_prob.png"
        )
    return data_frame_cov_mean, cov_matrix


def generate_deep_mock(cfg, df_cov_mean, cov_matrix, df_true_deep):
    """"""
    size = cfg["SIZE_MOCK"]

    if size is None:
        size = len(df_true_deep)

    arr_mean = np.concatenate([df_cov_mean[f"mean {column}"].values for column in cfg["COVARIANCE_COLUMNS"]])
    arr_multi_normal = np.random.multivariate_normal(
        arr_mean,
        cov_matrix,
        size
    )

    df_mock_deep = pd.DataFrame(
        arr_multi_normal,
        columns=cfg["COVARIANCE_COLUMNS"]
    )
    df_mock_deep.loc[:, "deep r-i"] = (
            df_mock_deep["BDF_MAG_DERED_CALIB_R"].values - df_mock_deep["BDF_MAG_DERED_CALIB_I"].values
    )
    df_mock_deep.loc[:, "deep i-z"] = (
            df_mock_deep["BDF_MAG_DERED_CALIB_I"].values - df_mock_deep["BDF_MAG_DERED_CALIB_Z"].values
    )
    df_true_deep.loc[:, "deep r-i"] = (
            df_true_deep["BDF_MAG_DERED_CALIB_R"].values - df_true_deep["BDF_MAG_DERED_CALIB_I"].values
    )
    df_true_deep.loc[:, "deep i-z"] = (
            df_true_deep["BDF_MAG_DERED_CALIB_I"].values - df_true_deep["BDF_MAG_DERED_CALIB_Z"].values
    )

    print(f"len generated data: {len(df_mock_deep)}")
    print(f"len true data: {len(df_true_deep)}")

    if cfg["PLOT_COMPARE_COLOR"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock_deep,
            data_frame_true=df_true_deep,
            columns=[
                "deep r-i",
                "deep i-z"
            ],
            labels=[
                "r-i",
                "i-z"
            ],
            title="title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_color.png"
        )

    if cfg["PLOT_MOCK_COLOR"] is True:
        plot_corner(
            df_mock_deep,
            columns=[
                "deep r-i",
                "deep i-z"
            ],
            labels=[
                "r-i",
                "i-z"
            ],
            title="title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_color.png"
        )

    if cfg["PLOT_COMPARE_MAG"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock_deep,
            data_frame_true=df_true_deep,
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
            title="title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_mag.png"
        )

    if cfg["PLOT_MOCK_MAG"] is True:
        plot_corner(
            df_mock_deep,
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
            title="title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_mag.png"
        )

    if cfg["PLOT_COMPARE_CONDITIONS"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock_deep,
            data_frame_true=df_true_deep,
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
            title="title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_cond.png"
        )

    if cfg["PLOT_MOCK_CONDITIONS"] is True:
        plot_corner(
            df_mock_deep,
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
            title="title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_cond.png"
        )

    if cfg["PLOT_COMPARE_PROPERTIES"] is True:
        plot_compare_corner(
            data_frame_generated=df_mock_deep,
            data_frame_true=df_true_deep,
            columns=[
                "unsheared/snr",
                "unsheared/size_ratio",
                "unsheared/weight",
                "unsheared/T"
            ],
            labels=["snr", "size ratio", "weight", "t"],
            title="title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/compare_meas_prob.png"
        )

    if cfg["PLOT_MOCK_PROPERTIES"] is True:
        plot_corner(
            df_mock_deep,
            columns=[
                "unsheared/snr",
                "unsheared/size_ratio",
                "unsheared/weight",
                "unsheared/T"
            ],
            labels=["snr", "size ratio", "weight", "t"],
            title="title",
            ranges=None,
            show_plot=cfg["SHOW_PLOT_MOCK"],
            save_plot=cfg["SAVE_PLOT_MOCK"],
            save_name=f"{cfg['PATH_OUTPUT']}/mock_meas_prob.png"
        )

    return df_mock_deep


def generate_wide_mock(cfg, df_mock_deep, df_true_deep):
    """"""
    df_mock = df_mock_deep.copy()

    cov_r_mag = np.cov(np.array([
        df_true_deep["BDF_MAG_DERED_CALIB_R"].values,
        df_true_deep["unsheared/mag_r"].values
    ]))

    cov_i_mag = np.cov(np.array([
        df_true_deep["BDF_MAG_DERED_CALIB_I"].values,
        df_true_deep["unsheared/mag_i"].values
    ]))

    cov_z_mag = np.cov(np.array([
        df_true_deep["BDF_MAG_DERED_CALIB_Z"].values,
        df_true_deep["unsheared/mag_z"].values
    ]))

    print(f"covariance matrix r-band between deep and wide field mag: {cov_r_mag}")
    print(f"covariance matrix i-band between deep and wide field mag: {cov_i_mag}")
    print(f"covariance matrix z-band between deep and wide field mag: {cov_z_mag}")

    print(f"var r: sqrt({cov_r_mag[0, 1]})={np.sqrt(cov_r_mag[0, 1])}")
    print(f"var i: sqrt({cov_i_mag[0, 1]})={np.sqrt(cov_i_mag[0, 1])}")
    print(f"var z: sqrt({cov_z_mag[0, 1]})={np.sqrt(cov_z_mag[0, 1])}")

    _alpha = 1.15
    _beta = 100
    arr_normal_r_mag = np.random.normal(
        loc=0,
        scale=np.sqrt(cov_r_mag[0, 1]),
        size=cfg["SIZE_MOCK"]
    )
    df_mock.loc[:, f"unsheared/mag_r"] = \
        _alpha * df_mock[f"BDF_MAG_DERED_CALIB_R"].values + (_beta * (1 - arr_normal_r_mag) / arr_normal_r_mag)

    arr_normal_i_mag = np.random.normal(
        loc=0,
        scale=np.sqrt(cov_i_mag[0, 1]),
        size=cfg["SIZE_MOCK"]
    )
    df_mock.loc[:, f"unsheared/mag_i"] = \
        _alpha * df_mock[f"BDF_MAG_DERED_CALIB_I"].values + (_beta * (1 - arr_normal_i_mag) / arr_normal_i_mag)

    arr_normal_z_mag = np.random.normal(
        loc=0,
        scale=np.sqrt(cov_z_mag[0, 1]),
        size=cfg["SIZE_MOCK"]
    )
    df_mock.loc[:, f"unsheared/mag_z"] = \
        _alpha * df_mock[f"BDF_MAG_DERED_CALIB_Z"].values + (_beta * (1 - arr_normal_z_mag) / arr_normal_z_mag)

    cov_r_mag_mock = np.cov(np.array(
        [df_mock["BDF_MAG_DERED_CALIB_R"],
         df_mock["unsheared/mag_r"]]))

    cov_i_mag_mock = np.cov(np.array(
        [df_mock["BDF_MAG_DERED_CALIB_I"],
         df_mock["unsheared/mag_i"]]))

    cov_z_mag_mock = np.cov(np.array(
        [df_mock["BDF_MAG_DERED_CALIB_Z"],
         df_mock["unsheared/mag_z"]]))

    print(f"covariance matrix r-band between deep and wide field mag mock: {cov_r_mag_mock}")
    print(f"covariance matrix i-band between deep and wide field mag mock: {cov_i_mag_mock}")
    print(f"covariance matrix z-band between deep and wide field mag mock: {cov_z_mag_mock}")

    print(f"var r mock: sqrt({cov_r_mag_mock[0, 1]})={np.sqrt(cov_r_mag_mock[0, 1])}")
    print(f"var i mock: sqrt({cov_i_mag_mock[0, 1]})={np.sqrt(cov_i_mag_mock[0, 1])}")
    print(f"var z mock: sqrt({cov_z_mag_mock[0, 1]})={np.sqrt(cov_z_mag_mock[0, 1])}")

    plot_compare_corner(
        data_frame_generated=df_mock,
        data_frame_true=df_true_deep,
        columns=[
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_i"
        ],
        labels=["mag_r", "mag_i", "mag_z"],
        title="title",
        ranges=None,
        show_plot=cfg["SHOW_PLOT_MOCK"],
        save_plot=cfg["SAVE_PLOT_MOCK"],
        save_name=f"{cfg['PATH_OUTPUT']}/compare_meas_mag.png"
    )

    # dict_wide_field[f"mock mag r wide field"] = flux2mag(dict_wide_field[f"generated flux r wide field"])
    # dict_wide_field[f"mock mag i wide field"] = flux2mag(dict_wide_field[f"generated flux i wide field"])
    # dict_wide_field[f"mock mag z wide field"] = flux2mag(dict_wide_field[f"generated flux z wide field"])
    #
    # arr_gen_ri_df = dict_wide_field[f"generated mag r deep field"] - dict_wide_field[f"generated mag i deep field"]
    # arr_gen_iz_df = dict_wide_field[f"generated mag i deep field"] - dict_wide_field[f"generated mag z deep field"]
    # arr_gen_ri_wf = dict_wide_field[f"generated mag r wide field"] - dict_wide_field[f"generated mag i wide field"]
    # arr_gen_iz_wf = dict_wide_field[f"generated mag i wide field"] - dict_wide_field[f"generated mag z wide field"]
    #
    # cov_matrix_gen_df = np.cov(np.array([arr_gen_ri_df, arr_gen_iz_df]))
    # cov_matrix_gen_wf = np.cov(np.array([arr_gen_ri_wf, arr_gen_iz_wf]))
    # cov_matrix_mcal_df = np.cov(np.array([arr_mcal_df_ri_mag, arr_mcal_df_iz_mag]))
    # cov_matrix_mcal_wf = np.cov(np.array([arr_mcal_wf_ri_mag, arr_mcal_wf_iz_mag]))
    #
    # print("generated covariance matrix r-i, i-z deep field", cov_matrix_gen_df)
    # print("generated covariance matrix r-i, i-z wide field", cov_matrix_gen_wf)
    # print("metacal covariance matrix r-i, i-z deep field", cov_matrix_mcal_df)
    # print("metacal covariance matrix r-i, i-z wide field", cov_matrix_mcal_wf)
    #
    # if plot_data is True:
    #     chaincon = ChainConsumer()
    #     df_gen_deep_field = pd.DataFrame({
    #         "generated mag r deep field": np.array(dict_wide_field["generated mag r deep field"]),
    #         "generated mag i deep field": np.array(dict_wide_field["generated mag i deep field"]),
    #         "generated mag z deep field": np.array(dict_wide_field["generated mag z deep field"]),
    #     })
    #     arr_gen_deep_field = df_gen_deep_field.to_numpy()
    #     df_deep_field = pd.DataFrame({
    #         "deep r mag": np.array(dict_wide_field["metacal mag r deep field"]),
    #         "deep i mag": np.array(dict_wide_field["metacal mag i deep field"]),
    #         "deep z mag": np.array(dict_wide_field["metacal mag z deep field"])
    #     })
    #     arr_deep_field = df_deep_field.to_numpy()
    #     df_gen_wide_field = pd.DataFrame({
    #         "generated mag r wide field": np.array(dict_wide_field["generated mag r wide field"]),
    #         "generated mag i wide field": np.array(dict_wide_field["generated mag i wide field"]),
    #         "generated mag z wide field": np.array(dict_wide_field["generated mag z wide field"]),
    #     })
    #     arr_gen_wide_field = df_gen_wide_field.to_numpy()
    #     df_wide_field = pd.DataFrame({
    #         "wide r mag": np.array(dict_wide_field["metacal mag r wide field"]),
    #         "wide i mag": np.array(dict_wide_field["metacal mag i wide field"]),
    #         "wide z mag": np.array(dict_wide_field["metacal mag z wide field"])
    #     })
    #     arr_wide_field = df_wide_field.to_numpy()
    #     parameter = [
    #         "r",
    #         "i",
    #         "z"
    #     ]
    #     chaincon.add_chain(arr_deep_field, parameters=parameter, name="BDF_MAG_DERED_CALIB_{R, I, Z}")
    #     chaincon.add_chain(arr_gen_deep_field, parameters=parameter, name="generated deep field mag {R, I, Z}")
    #     chaincon.add_chain(arr_wide_field, parameters=parameter, name="unsheared/flux_{r, i, z}")
    #     chaincon.add_chain(arr_gen_wide_field, parameters=parameter, name="generated deep field mag {r, i, z}")
    #     chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
    #     chaincon.plotter.plot(
    #         # filename=f"{dictionary_plot_paths['chain plot']}/chain_plot_epoch_{epoch}.png",
    #         figsize="page",
    #         display=True,
    #         # truth=[
    #         #     df_test_data["unsheared/mag_r"].mean(),
    #         #     df_test_data["unsheared/mag_i"].mean(),
    #         #     df_test_data["unsheared/mag_z"].mean(),
    #         #     df_test_data["unsheared/snr"].mean(),
    #         #     df_test_data["unsheared/size_ratio"].mean(),
    #         #     df_test_data["unsheared/T"].mean(),
    #         # ]
    #     )
    #     plt.clf()
    #
    #
    # df_mock.loc[:, "unsheared/mag_r"] = df_mock["BDF_MAG_DERED_CALIB_R"].values

    return df_mock


def main(cfg):
    """"""
    df_true_deep, arr_detected = data_preprocessing(cfg=cfg)
    # exploratory_data_analysis(cfg=cfg, data_frame=df_true_deep)

    df_covariance, cov_matrix = get_covariance_matrix(
        cfg=cfg,
        data_frame=df_true_deep
    )

    df_mock_deep = generate_deep_mock(
        cfg=cfg,
        df_cov_mean=df_covariance,
        cov_matrix=cov_matrix,
        df_true_deep=df_true_deep[cfg["COVARIANCE_COLUMNS"]]
    )

    df_mock = generate_wide_mock(
        cfg=cfg,
        df_mock_deep=df_mock_deep,
        df_true_deep=df_true_deep[cfg["WIDE_COLUMNS"]]
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