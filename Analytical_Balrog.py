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


def exploratory_data_analysis(cfg, data_frame):
    """"""
    # df_analytical = pd.DataFrame()
    #
    # for col in data_frame.keys():
    #     quantile = np.quantile(data_frame[col], q=[0.16, 0.84], axis=0)
    #     df_analytical[f"{col}: mean, median, std, q16, q84"] = np.array([
    #         data_frame[col].mean(),
    #         data_frame[col].median(),
    #         data_frame[col].std(),
    #         quantile[0],
    #         quantile[1]
    #     ])
    # dict_fit_dist = calc_fit_parameters(
    #     distribution=data_frame["BDF_MAG_DERED_CALIB_I"],
    #     lower_bound=18.5,
    #     upper_bound=23,
    #     bin_size=40,
    #     function=exp_func,
    #     column="BDF_MAG_DERED_CALIB_I",
    #     plot_data=True,
    #     save_plot=True,
    #     save_path_plots="Output"
    # )
    #
    # bin_width = dict_fit_dist["bin width"]
    #
    # # Generate distribution
    # dict_generated_dist = generate_distribution(
    #     dict_fit_params=dict_fit_dist,
    #     column="BDF_MAG_DERED_CALIB_I",
    #     lower_bound=16,
    #     upper_bound=26,
    #     plot_data=False,
    #     save_plot=False,
    #     show_plot=False,
    #     save_path_plots="",
    #     size=100000  # len(data_frame)
    # )

    # Call get_covariance_matrix to get the covariance matrix
    df_covariance, cov_matrix = get_covariance_matrix(
        data_frame=data_frame,
        columns=cfg["COVARIANCE_COLUMNS"],
        plot_data=False,
        save_plot=False,
        show_plot=False,
        save_path_plots="Output/"
    )

    dictionary_generated_data = generate_mag_distribution(
        data_frame_cov_mean=df_covariance,
        cov_matrix=cov_matrix,
        data_frame_true=data_frame[cfg["COVARIANCE_COLUMNS"]],
        # arr_i_mag=dict_generated_dist["generated BDF_MAG_DERED_CALIB_I"],
        columns=cfg["COVARIANCE_COLUMNS"],
        size=len(data_frame),
        plot_data=True,
        save_plot=True,
        show_plot=False,
        save_path_plots="Output/"
    )

    # calc_fit_parameters(
    #     distribution=dictionary_generated_data["generated deep i mag"],
    #     lower_bound=18.5,
    #     upper_bound=23,
    #     bin_size=40,
    #     function=exp_func,
    #     column="generated deep i mag",
    #     plot_data=True,
    #     save_plot=True,
    #     save_name="_generated",
    #     save_path_plots="Output"
    # )


def calc_fit_parameters(distribution, lower_bound, upper_bound, bin_size, function, column=None, plot_data=None,
                        save_plot=None, save_path_plots="", save_name="", show_plot=False):
    """"""
    # Create bins for fit range
    bins = np.linspace(lower_bound, upper_bound, bin_size)

    # Calculate bin width for hist plot
    bin_width = bins[1] - bins[0]

    # Calculate the center of the bins
    bins_centers = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])

    if plot_data is True:
        sns.histplot(
            x=distribution,
            element="step",
            fill=False,
            color="darkred",
            binwidth=bin_width,
            log_scale=(False, True),
            stat="probability",
            label=f"probability of DES deep field bin: {column}"
        )

        # Set some plot parameters
        plt.title(f"probability histogram of DES deep field '{column}'")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Power_law_DES-{column}_log_wo_bounds{save_name}.png")
        if show_plot is True:
            plt.show()
        plt.clf()

    distribution = distribution[distribution[:] <= upper_bound]
    distribution = distribution[lower_bound <= distribution[:]]

    # Calculate the histogram fit values
    data_entries = np.histogram(distribution, bins=bins)[0]

    # Get the probability
    data_probability_entries = data_entries / np.sum(data_entries)

    if plot_data is True:
        sns.histplot(
            x=distribution,
            element="step",
            fill=False,
            color="darkred",
            binwidth=bin_width,
            log_scale=(False, True),
            stat="probability",
            label=f"probability of DES deep field bin: {column}"
        )

        plt.plot(bins_centers, data_probability_entries, '.', color="darkred")

        # Set some plot parameters
        plt.xlim((lower_bound, upper_bound))
        plt.title(f"probability histogram of DES deep field '{column}'")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Probability_DES-{column}{save_name}.png")
        if show_plot is True:
            plt.show()
        plt.clf()

    # fit exponential function to calculated histogram values from before
    popt, pcov = curve_fit(function, xdata=bins_centers, ydata=data_probability_entries)

    if plot_data is True:
        sns.histplot(
            x=distribution,
            element="step",
            fill=False,
            color="darkred",
            binwidth=bin_width,
            log_scale=(False, True),
            stat="probability",
            label=f"probability of DES deep field bin: {column}"
        )

        plt.plot(bins_centers, data_probability_entries, '.', color="darkred")

        # Plot the fit function
        plt.plot(
            bins_centers,
            function(bins_centers, *popt),
            color='darkblue',
            linewidth=2.5,
            label=f'power law - f(x)=Exp[a+b*x]\n bin width {bin_width:2.4f} ; a={popt[0]:2.4f} ; b={popt[1]:2.4f}'
        )

        # Set some plot parameters
        plt.xlim((lower_bound, upper_bound))
        plt.title(f"probability histogram and power law of DES deep field '{column}'")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Curve_fit_DES-{column}{save_name}.png")
        if show_plot is True:
            plt.show()
        plt.clf()

    dictionary_fit_parameter = {
        f"{column} parameter optimal": popt,
        f"{column} parameter covariance": pcov,
        f"{column}": np.array(distribution),
        f"{column} probability": data_probability_entries,
        f"{column} counts": data_entries,
        "bin width": bin_width,
        "bin centers": bins_centers,
        "bins": bins
    }

    return dictionary_fit_parameter


def exp_func(x, a, b):
    """
    Exponential fit function: f(x)=exp[a+b*x]
    Args:
        x (np.array): array of x values
        a (int): intersection with y-axis
        b (int): slope

    Returns:
        y (numpy array): array with calculated f(x)
    """
    y = np.exp(a) * np.exp(b*x)
    return y


def generate_distribution(dict_fit_params, column, lower_bound, upper_bound, plot_data, save_plot, save_path_plots,
                          size, show_plot=False):
    """
    Generate a distribution with given fit parameters
    Args:
        dict_fit_params (dict): dictionary with given fit parameters and column names (same names as get_imag_distribution column names)
        column (list): column names (same names as get_imag_distribution column names)
        lower_bound (int): lower bound of distribution
        upper_bound (int): upper bound of distribution
        bin_size (int): number of bins
        size (int): How many values do you want to generate?

    Returns:
        dictionary_generated_data (dict): dictionary with generated bands with calculated distribution
    """

    # Create numpy array for x-axis
    xspace = np.linspace(lower_bound, upper_bound, num=size, endpoint=True)

    # Init new dictionary
    dictionary_generated_data = {}

    # Calculating the probability with defined fit function and calculated fit parameters
    prob = exp_func(
        xspace,
        *dict_fit_params[f"{column} parameter optimal"]
    )

    # Normalize the probability
    norm_prob = prob / np.sum(prob)

    # Use probability to select magnitudes
    arr_generated_data = np.random.choice(xspace, size=size, p=norm_prob)

    if plot_data is True:
        sns.histplot(
            x=arr_generated_data,
            element="step",
            fill=False,
            color="darkgreen",
            binwidth=dict_fit_params["bin width"],
            log_scale=(False, True),
            stat="probability",
            label=f"generated probability"
        )

        # plt.xlim((lower_bound, upper_bound))
        plt.title(f"generated probability histogram")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Generated_distribution.png")
        if show_plot is True:
            plt.show()

    # Write generated Data to dictionary
    dictionary_generated_data[f"generated {column}"] = arr_generated_data

    # Calculate the histogram fit values
    data_entries = np.histogram(arr_generated_data, bins=dict_fit_params["bins"])[0]

    # fit exponential function to calculated histogram values from before
    # Get the probability
    data_probability_entries = data_entries / np.sum(data_entries)
    popt, pcov = curve_fit(exp_func, xdata=dict_fit_params["bin centers"], ydata=data_probability_entries)

    print(f"generated {column} slope", popt[1])
    print(f"generated {column} offset", popt[0])
    print(f"{column} slope", dict_fit_params[f"{column} parameter optimal"][1])
    print(f"{column} offet", dict_fit_params[f"{column} parameter optimal"][0])

    if plot_data is True:
        sns.histplot(
            x=dict_fit_params[f"{column}"],
            element="step",
            fill=False,
            color="darkred",
            binwidth=dict_fit_params["bin width"],
            log_scale=(False, True),
            stat="probability",
            label=f"probability of DES deep field mag in {column}"
        )

        plt.plot(dict_fit_params["bin centers"], dict_fit_params[f"{column} probability"], '.', color="darkred")

        sns.histplot(
            x=arr_generated_data,
            element="step",
            fill=False,
            color="darkgreen",
            binwidth=dict_fit_params["bin width"],
            log_scale=(False, True),
            stat="probability",
            label=f"probability of generated mag in I"
        )

        plt.plot(dict_fit_params["bin centers"], data_probability_entries, '.', color="darkgreen")

        # Plot the fit function
        plt.plot(
            dict_fit_params["bin centers"],
            exp_func(dict_fit_params["bin centers"], *dict_fit_params[f"{column} parameter optimal"]),
            color='darkblue',
            linewidth=2.5,
            label=f'power law of BDF_FLUX_DERED_CALIB_I - f(x)=Exp[a+b*x]\nbin width {dict_fit_params["bin width"]:2.4f} ; a={dict_fit_params[f"{column} parameter optimal"][0]:2.4f} ; b={dict_fit_params[f"{column} parameter optimal"][1]:2.4f}'
        )

        # Plot the fit function
        plt.plot(
            xspace,
            exp_func(xspace, *popt),
            color='darkorange',
            linewidth=2.5,
            label=f'power law generated magnitude - f(x)=Exp[a+b*x]\nbin width {dict_fit_params["bin width"]:2.4f} ; a={popt[0]:2.4f} ; b={popt[1]:2.4f}'
        )

        # Set some plot parameters
        plt.xlim((lower_bound, upper_bound))
        plt.title(f"Calculate {column} fit from DES deep field catalog")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Curve_fit_DES-{column}_and_generated_data.png")
        if show_plot is True:
            plt.show()

    return dictionary_generated_data


def get_covariance_matrix(data_frame, columns, plot_data, save_plot, save_path_plots, show_plot=False):
    """
    Calculate the covariance matrix of given pandas DataFrame. The columns must be a list of the column name of the
    different bins in i,r, and z. The order of the column names has to be i-band, r-band, z-band.
    e.g ["BDF_MAG_DERED_CALIB_I", "BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_Z"].

    Args:
        data_frame (pandas DataFrame): DataFrame with magnitudes in different bands (i,r and z)
        columns (list): List of column names in the order i,r and z. e.g.:
                                        ["BDF_MAG_DERED_CALIB_I", "BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_Z"]

    Returns:
        dict_cov_matrix (dict): dictionary of the calculated covariance matrix and all usefull informations (covariance
        matrix, original data in all three bans, magnitude difference in r-i and i-z, magnitude difference shifted to
        zero as mean in r-i and i-z, mean value of r-i and i-z, standard deviation in r-i and i-z)
    """

    # arr_r_mag_df = data_frame[columns[0]]
    # arr_i_mag_df = data_frame[columns[1]]
    # arr_z_mag_df = data_frame[columns[2]]
    # arr_u_mag_df = data_frame[columns[3]]
    # arr_g_mag_df = data_frame[columns[4]]
    # arr_j_mag_df = data_frame[columns[5]]
    # arr_h_mag_df = data_frame[columns[6]]
    # arr_k_mag_df = data_frame[columns[7]]
    # arr_bdf_t_df = data_frame[columns[8]]
    # arr_bdf_g_df = data_frame[columns[9]]
    # arr_fwhm_r_df = data_frame[columns[10]]
    # arr_fwhm_i_df = data_frame[columns[11]]
    # arr_fwhm_z_df = data_frame[columns[12]]
    # arr_airmass_r_df = data_frame[columns[13]]
    # arr_airmass_i_df = data_frame[columns[14]]
    # arr_airmass_z_df = data_frame[columns[15]]
    # arr_maglim_r_df = data_frame[columns[16]]
    # arr_maglim_i_df = data_frame[columns[17]]
    # arr_maglim_z_df = data_frame[columns[18]]
    # arr_ebv_df = data_frame[columns[19]]
    # arr_unsheared_snr_df = data_frame[columns[20]]
    # arr_unsheared_size_ratio_df = data_frame[columns[21]]
    # arr_unsheared_weight_df = data_frame[columns[22]]
    # arr_unsheared_t_df = data_frame[columns[23]]
    #
    # # Create a matrix r-i and i-z and calculating the covariance matrix
    # arr_mag_ugrizjhk_2 = np.array([
    #     arr_u_mag_df,
    #     arr_g_mag_df,
    #     arr_r_mag_df,
    #     arr_i_mag_df,
    #     arr_z_mag_df,
    #     arr_j_mag_df,
    #     arr_h_mag_df,
    #     arr_k_mag_df,
    #     arr_bdf_t_df,
    #     arr_bdf_g_df,
    #     arr_fwhm_r_df,
    #     arr_fwhm_i_df,
    #     arr_fwhm_z_df,
    #     arr_airmass_r_df,
    #     arr_airmass_i_df,
    #     arr_airmass_z_df,
    #     arr_maglim_r_df,
    #     arr_maglim_i_df,
    #     arr_maglim_z_df,
    #     arr_ebv_df,
    #     arr_unsheared_snr_df,
    #     arr_unsheared_size_ratio_df,
    #     arr_unsheared_weight_df,
    #     arr_unsheared_t_df
    # ])
    data_frame = data_frame[columns]
    sampled_df = data_frame.sample(n=len(data_frame), random_state=42)
    arr_mag_ugrizjhk = data_frame.to_numpy()

    # Now calculate the covariance matrix
    covariance_matrix_ugrizjhk = np.cov(arr_mag_ugrizjhk, rowvar=False)
    print(f"covariance matrix deep distribution ugrizjhk: {covariance_matrix_ugrizjhk}")

    # Creating the covariance matrix dictionary
    # data_frame_cov_mean = {
    #     "covariance matrix deep ugrizjhk": covariance_matrix_ugrizjhk,
    #     "mean deep u": arr_u_mag_df.mean(),
    #     "mean deep g": arr_g_mag_df.mean(),
    #     "mean deep r": arr_r_mag_df.mean(),
    #     "mean deep i": arr_i_mag_df.mean(),
    #     "mean deep z": arr_z_mag_df.mean(),
    #     "mean deep j": arr_j_mag_df.mean(),
    #     "mean deep h": arr_h_mag_df.mean(),
    #     "mean deep k": arr_k_mag_df.mean(),
    #     "mean snr": arr_unsheared_snr_df.mean(),
    #     "mean size ratio": arr_unsheared_size_ratio_df.mean(),
    #     "mean weight": arr_unsheared_weight_df.mean(),
    #     "mean t": arr_unsheared_t_df.mean(),
    #     "mean bdf t": arr_bdf_t_df.mean(),
    #     "mean bdf g": arr_bdf_g_df.mean(),
    #     "mean fwhm r": arr_fwhm_r_df.mean(),
    #     "mean fwhm i": arr_fwhm_i_df.mean(),
    #     "mean fwhm z": arr_fwhm_z_df.mean(),
    #     "mean airmass r": arr_airmass_r_df.mean(),
    #     "mean airmass i": arr_airmass_i_df.mean(),
    #     "mean airmass z": arr_airmass_z_df.mean(),
    #     "mean maglim r": arr_maglim_r_df.mean(),
    #     "mean maglim i": arr_maglim_i_df.mean(),
    #     "mean maglim z": arr_maglim_z_df.mean(),
    #     "mean ebv": arr_ebv_df.mean(),
    #     # "mean deep u-g": arr_u_g_mag_df.mean(),
    #     # "mean deep g-r": arr_g_r_mag_df.mean(),
    #     # "mean deep r-i": arr_r_i_mag_df.mean(),
    #     # "mean deep i-z": arr_i_z_mag_df.mean(),
    #     # "mean deep z-j": arr_z_j_mag_df.mean(),
    #     # "mean deep j-h": arr_j_h_mag_df.mean(),
    #     # "mean deep h-ks": arr_h_ks_mag_df.mean(),
    #     "array deep u": arr_u_mag_df,
    #     "array deep g": arr_g_mag_df,
    #     "array deep r": arr_r_mag_df,
    #     "array deep i": arr_i_mag_df,
    #     "array deep z": arr_z_mag_df,
    #     "array deep j": arr_j_mag_df,
    #     "array deep h": arr_h_mag_df,
    #     "array deep k": arr_k_mag_df,
    #     "array bdf t": arr_bdf_t_df,
    #     "array bdf g": arr_bdf_g_df,
    #     "array fwhm r": arr_fwhm_r_df,
    #     "array fwhm i": arr_fwhm_i_df,
    #     "array fwhm z": arr_fwhm_z_df,
    #     "array airmass r": arr_airmass_r_df,
    #     "array airmass i": arr_airmass_i_df,
    #     "array airmass z": arr_airmass_z_df,
    #     "array maglim r": arr_maglim_r_df,
    #     "array maglim i": arr_maglim_i_df,
    #     "array maglim z": arr_maglim_z_df,
    #     "array ebv": arr_ebv_df,
    #     "array snr": arr_unsheared_snr_df,
    #     "array size ratio": arr_unsheared_size_ratio_df,
    #     "array weight": arr_unsheared_weight_df,
    #     "array t": arr_unsheared_t_df,
    #     # "array deep u-g": arr_u_g_mag_df,
    #     # "array deep g-r": arr_g_r_mag_df,
    #     # "array deep r-i": arr_r_i_mag_df,
    #     # "array deep i-z": arr_i_z_mag_df,
    #     # "array deep z-j": arr_z_j_mag_df,
    #     # "array deep j-h": arr_j_h_mag_df,
    #     # "array deep h-ks": arr_h_ks_mag_df
    # }
    dictionary_covariance = {}

    for column in columns:
        dictionary_covariance[f"mean {column}"] = data_frame[column].mean()

    data_frame_cov_mean = pd.DataFrame(dictionary_covariance, index=[0])
    dictionary_covariance["covariance matrix deep ugrizjhk"] = covariance_matrix_ugrizjhk

    # data_frame_cov_mean = pd.DataFrame({
    #     "covariance matrix deep ugrizjhk": covariance_matrix_ugrizjhk,
    #     "mean deep u": arr_u_mag_df.mean(),
    #     "mean deep g": arr_g_mag_df.mean(),
    #     "mean deep r": arr_r_mag_df.mean(),
    #     "mean deep i": arr_i_mag_df.mean(),
    #     "mean deep z": arr_z_mag_df.mean(),
    #     "mean deep j": arr_j_mag_df.mean(),
    #     "mean deep h": arr_h_mag_df.mean(),
    #     "mean deep k": arr_k_mag_df.mean(),
    #     "mean snr": arr_unsheared_snr_df.mean(),
    #     "mean size ratio": arr_unsheared_size_ratio_df.mean(),
    #     "mean weight": arr_unsheared_weight_df.mean(),
    #     "mean t": arr_unsheared_t_df.mean(),
    #     "mean bdf t": arr_bdf_t_df.mean(),
    #     "mean bdf g": arr_bdf_g_df.mean(),
    #     "mean fwhm r": arr_fwhm_r_df.mean(),
    #     "mean fwhm i": arr_fwhm_i_df.mean(),
    #     "mean fwhm z": arr_fwhm_z_df.mean(),
    #     "mean airmass r": arr_airmass_r_df.mean(),
    #     "mean airmass i": arr_airmass_i_df.mean(),
    #     "mean airmass z": arr_airmass_z_df.mean(),
    #     "mean maglim r": arr_maglim_r_df.mean(),
    #     "mean maglim i": arr_maglim_i_df.mean(),
    #     "mean maglim z": arr_maglim_z_df.mean(),
    #     "mean ebv": arr_ebv_df.mean(),
    #     "deep u": arr_u_mag_df,
    #     "deep g": arr_g_mag_df,
    #     "deep r": arr_r_mag_df,
    #     "deep i": arr_i_mag_df,
    #     "deep z": arr_z_mag_df,
    #     "deep j": arr_j_mag_df,
    #     "deep h": arr_h_mag_df,
    #     "deep k": arr_k_mag_df,
    #     "bdf t": arr_bdf_t_df,
    #     "bdf g": arr_bdf_g_df,
    #     "fwhm r": arr_fwhm_r_df,
    #     "fwhm i": arr_fwhm_i_df,
    #     "fwhm z": arr_fwhm_z_df,
    #     "airmass r": arr_airmass_r_df,
    #     "airmass i": arr_airmass_i_df,
    #     "airmass z": arr_airmass_z_df,
    #     "maglim r": arr_maglim_r_df,
    #     "maglim i": arr_maglim_i_df,
    #     "maglim z": arr_maglim_z_df,
    #     "ebv": arr_ebv_df,
    #     "snr": arr_unsheared_snr_df,
    #     "size ratio": arr_unsheared_size_ratio_df,
    #     "weight": arr_unsheared_weight_df,
    #     "t": arr_unsheared_t_df,
    # })

    if plot_data is True:
        plot_corner(
            data_frame_cov_mean,
            ["deep u-g", "deep g-r", "deep r-i", "deep i-z", "deep z-j", "deep j-h", "deep h-ks"],
            ["u-g", "g-r", "r-i", "i-z", "z-j", "j-h", "h-ks"],
            "title",
            ranges=None,
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/deep_color.png"
        )
        plot_corner(
            data_frame_cov_mean,
            ["deep r", "deep i", "deep z"],
            ["r", "i", "z"],
            "title",
            ranges=None,
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/deep_magnitude.png"
        )
        plot_corner(
            data_frame_cov_mean,
            ["bdf t", "bdf g", "fwhm r", "fwhm i", "fwhm z", "airmass r", "airmass i", "airmass z", "maglim r", "maglim i", "maglim z", "ebv"],
            ["bdf t", "bdf g", "fwhm r", "fwhm i", "fwhm z", "airmass r", "airmass i", "airmass z", "maglim r", "maglim i", "maglim z", "ebv"],
            "title",
            ranges=None,
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/obs_cond.png"
        )
        plot_corner(
            data_frame_cov_mean,
            ["snr", "size ratio", "weight", "t"],
            ["snr", "size ratio", "weight", "t"],
            "title",
            ranges=None,
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/meas_prob.png"
        )
    return data_frame_cov_mean, covariance_matrix_ugrizjhk


def generate_mag_distribution(data_frame_cov_mean, cov_matrix, columns, data_frame_true, size, plot_data, save_plot,
                              save_path_plots, show_plot=False):
    """
    Generate the values in the r and z band depend on the covariance matrix and the distribution of generated i-band
    values (e.g. from generate_distribution in i-band)

    Args:
        data_frame_cov_mean (dataframe): dictionary of the covariance matrix. Calculated with the get_covariance_matrix
                                        function
        arr_mag (numpy array): array of generated magnitudes in one band e.g. i-band

    Returns:
        dict_generated_data (dict): dictionary of generated data in all three bands (i,r and z)
    """



    # arr_mean_ugrizjhk = np.array([
    #     data_frame_cov_mean["mean deep u"],
    #     data_frame_cov_mean["mean deep g"],
    #     data_frame_cov_mean["mean deep r"],
    #     data_frame_cov_mean["mean deep i"],
    #     data_frame_cov_mean["mean deep z"],
    #     data_frame_cov_mean["mean deep j"],
    #     data_frame_cov_mean["mean deep h"],
    #     data_frame_cov_mean["mean deep k"],
    #     data_frame_cov_mean["mean bdf t"],
    #     data_frame_cov_mean["mean bdf g"],
    #     data_frame_cov_mean["mean fwhm r"],
    #     data_frame_cov_mean["mean fwhm i"],
    #     data_frame_cov_mean["mean fwhm z"],
    #     data_frame_cov_mean["mean airmass r"],
    #     data_frame_cov_mean["mean airmass i"],
    #     data_frame_cov_mean["mean airmass z"],
    #     data_frame_cov_mean["mean maglim r"],
    #     data_frame_cov_mean["mean maglim i"],
    #     data_frame_cov_mean["mean maglim z"],
    #     data_frame_cov_mean["mean ebv"],
    #     data_frame_cov_mean["mean snr"],
    #     data_frame_cov_mean["mean size ratio"],
    #     data_frame_cov_mean["mean weight"],
    #     data_frame_cov_mean["mean t"],
    # ])

    arr_mean_ugrizjhk_1d = np.concatenate([data_frame_cov_mean[f"mean {column}"].values for column in columns])
    arr_multi_normal_ugrizjhk_df = np.random.multivariate_normal(
        arr_mean_ugrizjhk_1d,
        cov_matrix,
        size
    )

    # arr_mean_ug_gr_df = np.array([data_frame_cov_mean["mean deep u-g"], data_frame_cov_mean["mean deep g-r"]])
    # arr_multi_normal_ug_gr_df = np.random.multivariate_normal(
    #     arr_mean_ug_gr_df, data_frame_cov_mean["covariance matrix deep u-g, g-r"], size)
    #
    # arr_mean_gr_ri_df = np.array([data_frame_cov_mean["mean deep g-r"], data_frame_cov_mean["mean deep r-i"]])
    # arr_multi_normal_gr_ri_df = np.random.multivariate_normal(
    #     arr_mean_gr_ri_df, data_frame_cov_mean["covariance matrix deep g-r, r-i"], size)
    #
    # arr_mean_ri_iz_df = np.array([data_frame_cov_mean["mean deep r-i"], data_frame_cov_mean["mean deep i-z"]])
    # arr_multi_normal_ri_iz_df = np.random.multivariate_normal(
    #     arr_mean_ri_iz_df, data_frame_cov_mean["covariance matrix deep r-i, i-z"], size)
    #
    # arr_mean_iz_zj_df = np.array([data_frame_cov_mean["mean deep i-z"], data_frame_cov_mean["mean deep z-j"]])
    # arr_multi_normal_iz_zj_df = np.random.multivariate_normal(
    #     arr_mean_iz_zj_df, data_frame_cov_mean["covariance matrix deep i-z, z-j"], size)
    #
    # arr_mean_zj_jh_df = np.array([data_frame_cov_mean["mean deep z-j"], data_frame_cov_mean["mean deep j-h"]])
    # arr_multi_normal_zj_jh_df = np.random.multivariate_normal(
    #     arr_mean_zj_jh_df, data_frame_cov_mean["covariance matrix deep z-j, j-h"], size)
    #
    # arr_mean_jh_hk_df = np.array([data_frame_cov_mean["mean deep j-h"], data_frame_cov_mean["mean deep h-ks"]])
    # arr_multi_normal_jh_hks_df = np.random.multivariate_normal(
    #     arr_mean_jh_hk_df, data_frame_cov_mean["covariance matrix deep j-h, h-ks"], size)
    #
    # lst_u_df = []
    # lst_g_df = []
    # lst_r_df = []
    # lst_z_df = []
    # lst_j_df = []
    # lst_h_df = []
    # lst_k_df = []
    df_ugrizjhk = pd.DataFrame(
        arr_multi_normal_ugrizjhk_df,
        columns=columns
    )

    # for idx, value in enumerate(arr_multi_normal_ri_iz_df):
    #     lst_r_df.append(value[0] + arr_i_mag[idx])
    #     lst_z_df.append(arr_i_mag[idx] - value[1])
    # arr_r_mag_df = np.array(lst_r_df)
    # arr_z_mag_df = np.array(lst_z_df)
    # print(arr_r_mag_df.shape)
    # arr_r_mag_df = arr_multi_normal_riz_df[0]
    # arr_i_mag = arr_multi_normal_riz_df[1]
    # arr_z_mag_df = arr_multi_normal_riz_df[2]
    # print(arr_r_mag_df.shape)

    # arr_u_mag_df = df_ugrizjhk["u"].values
    # arr_g_mag_df = df_ugrizjhk["g"].values
    # arr_r_mag_df = df_ugrizjhk["r"].values
    # arr_i_mag = df_ugrizjhk["i"].values
    # arr_z_mag_df = df_ugrizjhk["z"].values
    # arr_j_mag_df = df_ugrizjhk["j"].values
    # arr_h_mag_df = df_ugrizjhk["h"].values
    # arr_ks_mag_df = df_ugrizjhk["k"].values
    #
    # arr_bdf_t_df = df_ugrizjhk["bdf t"].values
    # arr_bdf_g_df = df_ugrizjhk["bdf g"].values
    # arr_fwhm_r_df = df_ugrizjhk["fwhm r"].values
    # arr_fwhm_i_df = df_ugrizjhk["fwhm i"].values
    # arr_fwhm_z_df = df_ugrizjhk["fwhm z"].values
    # arr_airmass_r_df = df_ugrizjhk["airmass r"].values
    # arr_airmass_i_df = df_ugrizjhk["airmass i"].values
    # arr_airmass_z_df = df_ugrizjhk["airmass z"].values
    # arr_maglim_r_df = df_ugrizjhk["maglim r"].values
    # arr_maglim_i_df = df_ugrizjhk["maglim i"].values
    # arr_maglim_z_df = df_ugrizjhk["maglim z"].values
    # arr_ebv_df = df_ugrizjhk["ebv"].values
    #
    # arr_unsheared_snr_df = df_ugrizjhk["snr"].values
    # arr_unsheared_size_ratio_df = df_ugrizjhk["size ratio"].values
    # arr_unsheared_weight_df = df_ugrizjhk["weight"].values
    # arr_unsheared_t_df = df_ugrizjhk["t"].values

    # for idx, value in enumerate(arr_multi_normal_gr_ri_df):
    #     lst_g_df.append(value[0] + arr_r_mag_df[idx])
    # arr_g_mag_df = np.array(lst_g_df)
    #
    # for idx, value in enumerate(arr_multi_normal_ug_gr_df):
    #     lst_u_df.append(value[0] + arr_g_mag_df[idx])
    # arr_u_mag_df = np.array(lst_u_df)
    #
    # for idx, value in enumerate(arr_multi_normal_iz_zj_df):
    #     lst_j_df.append(arr_z_mag_df[idx] - value[1])
    # arr_j_mag_df = np.array(lst_j_df)
    #
    # for idx, value in enumerate(arr_multi_normal_zj_jh_df):
    #     lst_h_df.append(arr_j_mag_df[idx] - value[1])
    # arr_h_mag_df = np.array(lst_h_df)
    #
    # for idx, value in enumerate(arr_multi_normal_jh_hks_df):
    #     lst_k_df.append(arr_h_mag_df[idx] - value[1])
    # arr_ks_mag_df = np.array(lst_k_df)

    # arr_u_g_mag_df = arr_u_mag_df - arr_g_mag_df
    # arr_g_r_mag_df = arr_g_mag_df - arr_r_mag_df
    # arr_r_i_mag_df = arr_r_mag_df - arr_i_mag
    # arr_i_z_mag_df = arr_i_mag - arr_z_mag_df
    # arr_z_j_mag_df = arr_z_mag_df - arr_j_mag_df
    # arr_j_h_mag_df = arr_j_mag_df - arr_h_mag_df
    # arr_h_ks_mag_df = arr_h_mag_df - arr_ks_mag_df

    # cov_matrix_gen_ug_gr_df = np.cov(np.array([arr_u_g_mag_df, arr_g_r_mag_df]))
    # cov_matrix_gen_gr_ri_df = np.cov(np.array([arr_g_r_mag_df, arr_r_i_mag_df]))
    # cov_matrix_gen_ri_iz_df = np.cov(np.array([arr_r_i_mag_df, arr_i_z_mag_df]))
    # cov_matrix_gen_iz_zj_df = np.cov(np.array([arr_i_z_mag_df, arr_z_j_mag_df]))
    # cov_matrix_gen_zj_jh_df = np.cov(np.array([arr_z_j_mag_df, arr_j_h_mag_df]))
    # cov_matrix_gen_jh_hks_df = np.cov(np.array([arr_j_h_mag_df, arr_h_ks_mag_df]))
    # dictionary_generated_data = {
    #     "generated deep u mag": arr_u_mag_df,
    #     "generated deep g mag": arr_g_mag_df,
    #     "generated deep r mag": arr_r_mag_df,
    #     "generated deep i mag": arr_i_mag,
    #     "generated deep z mag": arr_z_mag_df,
    #     "generated deep j mag": arr_j_mag_df,
    #     "generated deep h mag": arr_h_mag_df,
    #     "generated deep ks mag": arr_ks_mag_df,
    #     "generated deep i flux": mag2flux(arr_i_mag),
    #     "generated deep r flux": mag2flux(arr_r_mag_df),
    #     "generated deep z flux": mag2flux(arr_z_mag_df),
    #     "generated deep u-g mag": arr_u_g_mag_df,
    #     "generated deep g-r mag": arr_g_r_mag_df,
    #     "generated deep r-i mag": arr_r_i_mag_df,
    #     "generated deep i-z mag": arr_i_z_mag_df,
    #     "generated deep z-j mag": arr_z_j_mag_df,
    #     "generated deep j-h mag": arr_j_h_mag_df,
    #     "generated deep h-ks mag": arr_h_ks_mag_df,
    #     "generated array bdf t": arr_bdf_t_df,
    #     "generated array bdf g": arr_bdf_g_df,
    #     "generated array fwhm r": arr_fwhm_r_df,
    #     "generated array fwhm i": arr_fwhm_i_df,
    #     "generated array fwhm z": arr_fwhm_z_df,
    #     "generated array airmass r": arr_airmass_r_df,
    #     "generated array airmass i": arr_airmass_i_df,
    #     "generated array airmass z": arr_airmass_z_df,
    #     "generated array maglim r": arr_maglim_r_df,
    #     "generated array maglim i": arr_maglim_i_df,
    #     "generated array maglim z": arr_maglim_z_df,
    #     "generated array ebv": arr_ebv_df,
    #     "generated array snr": arr_unsheared_snr_df,
    #     "generated array size ratio": arr_unsheared_size_ratio_df,
    #     "generated array weight": arr_unsheared_weight_df,
    #     "generated array t": arr_unsheared_t_df,
    #     # "cov matrix generated deep u-g, g-r": cov_matrix_gen_ug_gr_df,
    #     # "cov matrix generated deep g-r, r-i": cov_matrix_gen_gr_ri_df,
    #     # "cov matrix generated deep r-i, i-z": cov_matrix_gen_ri_iz_df,
    #     # "cov matrix generated deep i-z, z-j": cov_matrix_gen_iz_zj_df,
    #     # "cov matrix generated deep z-j, j-h": cov_matrix_gen_zj_jh_df,
    #     # "cov matrix generated deep j-h, h-ks": cov_matrix_gen_jh_hks_df
    # }

    # print(f"compare cov matrix deep u-g, g-r:"
    #       f"\t generated {cov_matrix_gen_ug_gr_df}"
    #       f"\t original {data_frame_cov_mean['covariance matrix deep u-g, g-r']}")
    # print(f"compare cov matrix deep g-r, r-i:"
    #       f"\t generated {cov_matrix_gen_gr_ri_df}"
    #       f"\t original {data_frame_cov_mean['covariance matrix deep g-r, r-i']}")
    # print(f"compare cov matrix deep r-i, i-z:"
    #       f"\t generated {cov_matrix_gen_ri_iz_df}"
    #       f"\t original {data_frame_cov_mean['covariance matrix deep r-i, i-z']}")
    # print(f"compare cov matrix deep i-z, z-j:"
    #       f"\t generated {cov_matrix_gen_iz_zj_df}"
    #       f"\t original {data_frame_cov_mean['covariance matrix deep i-z, z-j']}")
    # print(f"compare cov matrix deep z-j, j-h:"
    #       f"\t generated {cov_matrix_gen_zj_jh_df}"
    #       f"\t original {data_frame_cov_mean['covariance matrix deep z-j, j-h']}")
    # print(f"compare cov matrix deep j-h, h-ks:"
    #       f"\t generated {cov_matrix_gen_jh_hks_df}"
    #       f"\t original {data_frame_cov_mean['covariance matrix deep j-h, h-ks']}")

    if plot_data is True:
        # df_deep_field = pd.DataFrame({
        #     "deep r-i": np.array(data_frame_cov_mean["array deep r-i"]),
        #     "deep i-z": np.array(data_frame_cov_mean["array deep i-z"]),
        #     "deep r": np.array(data_frame_cov_mean["array deep r"]),
        #     "deep i": np.array(data_frame_cov_mean["array deep i"]),
        #     "deep z": np.array(data_frame_cov_mean["array deep z"]),
        #     "bdf t": np.array(data_frame_cov_mean["array bdf t"]),
        #     "bdf g": np.array(data_frame_cov_mean["array bdf g"]),
        #     "fwhm r": np.array(data_frame_cov_mean["array fwhm r"]),
        #     "fwhm i": np.array(data_frame_cov_mean["array fwhm i"]),
        #     "fwhm z": np.array(data_frame_cov_mean["array fwhm z"]),
        #     "airmass r": np.array(data_frame_cov_mean["array airmass r"]),
        #     "airmass i": np.array(data_frame_cov_mean["array airmass i"]),
        #     "airmass z": np.array(data_frame_cov_mean["array airmass z"]),
        #     "maglim r": np.array(data_frame_cov_mean["array maglim r"]),
        #     "maglim i": np.array(data_frame_cov_mean["array maglim i"]),
        #     "maglim z": np.array(data_frame_cov_mean["array maglim z"]),
        #     "ebv": np.array(data_frame_cov_mean["array ebv"]),
        #     "snr": np.array(data_frame_cov_mean["array snr"]),
        #     "size ratio": np.array(data_frame_cov_mean["array size ratio"]),
        #     "weight": np.array(data_frame_cov_mean["array weight"]),
        #     "t": np.array(data_frame_cov_mean["array t"]),
        # })
        #
        # df_generated = pd.DataFrame({
        #     "deep r-i": np.array(dictionary_generated_data["generated deep r-i mag"]),
        #     "deep i-z": np.array(dictionary_generated_data["generated deep i-z mag"]),
        #     "deep r": np.array(dictionary_generated_data["generated deep r mag"]),
        #     "deep i": np.array(dictionary_generated_data["generated deep i mag"]),
        #     "deep z": np.array(dictionary_generated_data["generated deep z mag"]),
        #     "bdf t": np.array(dictionary_generated_data["generated array bdf t"]),
        #     "bdf g": np.array(dictionary_generated_data["generated array bdf g"]),
        #     "fwhm r": np.array(dictionary_generated_data["generated array fwhm r"]),
        #     "fwhm i": np.array(dictionary_generated_data["generated array fwhm i"]),
        #     "fwhm z": np.array(dictionary_generated_data["generated array fwhm z"]),
        #     "airmass r": np.array(dictionary_generated_data["generated array airmass r"]),
        #     "airmass i": np.array(dictionary_generated_data["generated array airmass i"]),
        #     "airmass z": np.array(dictionary_generated_data["generated array airmass z"]),
        #     "maglim r": np.array(dictionary_generated_data["generated array maglim r"]),
        #     "maglim i": np.array(dictionary_generated_data["generated array maglim i"]),
        #     "maglim z": np.array(dictionary_generated_data["generated array maglim z"]),
        #     "ebv": np.array(dictionary_generated_data["generated array ebv"]),
        #     "snr": np.array(dictionary_generated_data["generated array snr"]),
        #     "size ratio": np.array(dictionary_generated_data["generated array size ratio"]),
        #     "weight": np.array(dictionary_generated_data["generated array weight"]),
        #     "t": np.array(dictionary_generated_data["generated array t"]),
        # })
        # df_ugrizjhk["deep r-i"] = df_ugrizjhk["BDF_MAG_DERED_CALIB_R"].values - df_ugrizjhk["BDF_MAG_DERED_CALIB_I"].values
        # df_ugrizjhk["deep i-z"] = df_ugrizjhk["BDF_MAG_DERED_CALIB_I"].values - df_ugrizjhk["BDF_MAG_DERED_CALIB_Z"].values
        # data_frame_true["deep r-i"] = data_frame_true["BDF_MAG_DERED_CALIB_R"].values - data_frame_true["BDF_MAG_DERED_CALIB_I"].values
        # data_frame_true["deep i-z"] = data_frame_true["BDF_MAG_DERED_CALIB_I"].values - data_frame_true["BDF_MAG_DERED_CALIB_Z"].values

        df_ugrizjhk.loc[:, "deep r-i"] = (
                df_ugrizjhk["BDF_MAG_DERED_CALIB_R"].values - df_ugrizjhk["BDF_MAG_DERED_CALIB_I"].values
        )
        df_ugrizjhk.loc[:, "deep i-z"] = (
                df_ugrizjhk["BDF_MAG_DERED_CALIB_I"].values - df_ugrizjhk["BDF_MAG_DERED_CALIB_Z"].values
        )
        data_frame_true.loc[:, "deep r-i"] = (
                data_frame_true["BDF_MAG_DERED_CALIB_R"].values - data_frame_true["BDF_MAG_DERED_CALIB_I"].values
        )
        data_frame_true.loc[:, "deep i-z"] = (
                data_frame_true["BDF_MAG_DERED_CALIB_I"].values - data_frame_true["BDF_MAG_DERED_CALIB_Z"].values
        )
        
        print(f"len generated data: {len(df_ugrizjhk)}")
        print(f"len true data: {len(data_frame_true)}")

        plot_compare_corner(
            data_frame_generated=df_ugrizjhk,
            data_frame_true=data_frame_true,
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
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/compare_generated_color.png"
        )

        plot_corner(
            df_ugrizjhk,
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
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/generated_color.png"
        )

        plot_compare_corner(
            data_frame_generated=df_ugrizjhk,
            data_frame_true=data_frame_true,
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
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/compare_generated_mag.png"
        )
        plot_corner(
            df_ugrizjhk,
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
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/generated_mag.png"
        )

        plot_compare_corner(
            data_frame_generated=df_ugrizjhk,
            data_frame_true=data_frame_true,
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
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/compare_generated_galaxy_cond.png"
        )
        plot_corner(
            df_ugrizjhk,
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
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/generated_galaxy_cond.png"
        )

        plot_compare_corner(
            data_frame_generated=df_ugrizjhk,
            data_frame_true=data_frame_true,
            columns=[
                "unsheared/snr",
                "unsheared/size_ratio",
                "unsheared/weight",
                "unsheared/T"
            ],
            labels=["snr", "size ratio", "weight", "t"],
            title="title",
            ranges=None,
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/compare_generated_meas_prob.png"
        )
        plot_corner(
            df_ugrizjhk,
            columns=[
                "unsheared/snr",
                "unsheared/size_ratio",
                "unsheared/weight",
                "unsheared/T"
            ],
            labels=["snr", "size ratio", "weight", "t"],
            title="title",
            ranges=None,
            show_plot=show_plot,
            save_plot=save_plot,
            save_name=f"{save_path_plots}/generated_meas_prob.png"
        )

    return df_ugrizjhk


def main(cfg):
    """"""
    # set_plot_settings()
    df_balrog, arr_detected = data_preprocessing(cfg=cfg)
    exploratory_data_analysis(cfg=cfg, data_frame=df_balrog)


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