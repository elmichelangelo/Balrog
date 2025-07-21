import matplotlib.pyplot as plt

from Handler import *
from Handler.helper_functions import *
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import yaml
import argparse
import healpy as hp
from astropy.table import Table
import fitsio


def check(cfg, data_frame, save_path):
    lst_cols = ["BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_I", "BDF_MAG_DERED_CALIB_Z",
                "BDF_MAG_ERR_DERED_CALIB_R",
                "BDF_MAG_ERR_DERED_CALIB_I", "BDF_MAG_ERR_DERED_CALIB_Z", "BDF_DET_MAG",
                "unsheared/extended_class_sof", "detected", "unsheared/mag_r", "unsheared/mag_i",
                "unsheared/mag_z",
                "unsheared/mag_err_r", "unsheared/mag_err_i", "unsheared/mag_err_z"]

    data_frame = calc_color(
        cfg=cfg,
        data_frame=data_frame,
        mag_type=("MAG", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
        bins=cfg['BDF_BINS'],
        save_name=f"bdf_mag"
    )
    data_frame = calc_color(
        cfg=cfg,
        data_frame=data_frame,
        mag_type=("MAG", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/mag", "unsheared/mag_err"),
        bins=cfg['UNSHEARED_BINS'],
        save_name=f"unsheared/mag"
    )

    data_frame["BDF_DET_MAG"] = data_frame[
        ["BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_I", "BDF_MAG_DERED_CALIB_Z"]
    ].mean(axis=1)

    data_frame_r = data_frame[data_frame["BDF_MAG_DERED_CALIB_R"] > 30]
    data_frame_i = data_frame[data_frame["BDF_MAG_DERED_CALIB_I"] > 30]
    data_frame_z = data_frame[data_frame["BDF_MAG_DERED_CALIB_Z"] > 30]
    data_frame_ri = data_frame[(data_frame["BDF_MAG_DERED_CALIB_R"] > 30) & (data_frame["BDF_MAG_DERED_CALIB_I"] > 30)]
    data_frame_rz = data_frame[(data_frame["BDF_MAG_DERED_CALIB_R"] > 30) & (data_frame["BDF_MAG_DERED_CALIB_Z"] > 30)]
    data_frame_iz = data_frame[(data_frame["BDF_MAG_DERED_CALIB_I"] > 30) & (data_frame["BDF_MAG_DERED_CALIB_Z"] > 30)]
    data_frame_riz = data_frame[(data_frame["BDF_MAG_DERED_CALIB_R"] > 30) & (data_frame["BDF_MAG_DERED_CALIB_I"] > 30) & (
                data_frame["BDF_MAG_DERED_CALIB_Z"] > 30)]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("data_frame_r")
    print(data_frame_r[lst_cols].head())
    print("data_frame_i")
    print(data_frame_i[lst_cols].head())
    print("data_frame_z")
    print(data_frame_z[lst_cols].head())
    print("data_frame_ri")
    print(data_frame_ri[lst_cols].head())
    print("data_frame_rz")
    print(data_frame_rz[lst_cols].head())
    print("data_frame_iz")
    print(data_frame_iz[lst_cols].head())
    print("data_frame_riz")
    print(data_frame_riz[lst_cols].head())

    data_frame_r_detected = data_frame_r[data_frame_r["detected"] == 1]
    data_frame_i_detected = data_frame_i[data_frame_i["detected"] == 1]
    data_frame_z_detected = data_frame_z[data_frame_z["detected"] == 1]

    print("data_frame_r_detected")
    print(data_frame_r_detected[lst_cols].head())
    print("data_frame_i_detected")
    print(data_frame_i_detected[lst_cols].head())
    print("data_frame_z_detected")
    print(data_frame_z_detected[lst_cols].head())

    print("data_frame_r_detected max")
    print(data_frame_r_detected['BDF_MAG_DERED_CALIB_R'].max())
    print("data_frame_i_detected max")
    print(data_frame_i_detected['BDF_MAG_DERED_CALIB_I'].max())
    print("data_frame_z_detected max")
    print(data_frame_z_detected['BDF_MAG_DERED_CALIB_Z'].max())

    data_frame_r_detected_galaxies = data_frame_r_detected[data_frame_r_detected['unsheared/extended_class_sof'] > 1]
    data_frame_i_detected_galaxies = data_frame_i_detected[data_frame_i_detected['unsheared/extended_class_sof'] > 1]
    data_frame_z_detected_galaxies = data_frame_z_detected[data_frame_z_detected['unsheared/extended_class_sof'] > 1]
    data_frame_r_detected_stars = data_frame_r_detected[data_frame_r_detected['unsheared/extended_class_sof'] < 2]
    data_frame_i_detected_stars = data_frame_i_detected[data_frame_i_detected['unsheared/extended_class_sof'] < 2]
    data_frame_z_detected_stars = data_frame_z_detected[data_frame_z_detected['unsheared/extended_class_sof'] < 2]

    print("data_frame_r_detected_stars")
    print(data_frame_r_detected_stars[lst_cols].head())
    print("data_frame_r_detected_galaxies")
    print(data_frame_r_detected_galaxies[lst_cols].head())

    print("data_frame_r_detected_galaxies isna")
    print(data_frame_r_detected_galaxies[lst_cols].isna().sum())
    print("data_frame_i_detected_galaxies isna")
    print(data_frame_i_detected_galaxies[lst_cols].isna().sum())
    print("data_frame_z_detected_galaxies isna")
    print(data_frame_z_detected_galaxies[lst_cols].isna().sum())
    # print(data_frame_i[["BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_I", "BDF_MAG_DERED_CALIB_Z", "BDF_MAG_ERR_DERED_CALIB_R", "BDF_DET_MAG"]].head())

    data_frame_r_detected_match = data_frame_r_detected[data_frame_r_detected["match_flag_1.5_asec"] < 2]
    data_frame_i_detected_match = data_frame_i_detected[data_frame_i_detected["match_flag_1.5_asec"] < 2]
    data_frame_z_detected_match = data_frame_z_detected[data_frame_z_detected["match_flag_1.5_asec"] < 2]

    print("data_frame_r_detected_match")
    print(data_frame_r_detected_match[lst_cols].head())
    print("data_frame_i_detected_match")
    print(data_frame_i_detected_match[lst_cols].head())
    print("data_frame_z_detected_match")
    print(data_frame_z_detected_match[lst_cols].head())
    data_frame_only_detected = data_frame[data_frame['detected']==1]
    for col in cfg["COLUMNS_OF_INTEREST"]:
        minmaxscaler = MinMaxScaler(feature_range=(-30, 30))
        value = data_frame_only_detected[col].values
        save_name = col.replace("unsheared/", "")

        value_scaled = minmaxscaler.fit_transform(value.reshape(-1, 1)).flatten()

        if col in cfg["COLUMNS_LOG1P"]:
            log_value = np.log1p(value)
            log_value_scaled = minmaxscaler.fit_transform(log_value.reshape(-1, 1)).flatten()

            # 4 Subplots (log1p-Fall)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Histogramme für {col}", fontsize=14)

            sns.histplot(value, stat="density", bins=100, ax=axes[0, 0])
            axes[0, 0].set_title("Original Value")
            axes[0, 0].set_yscale("log")

            sns.histplot(value_scaled, stat="density", bins=100, ax=axes[0, 1])
            axes[0, 1].set_title("Value Scaled")
            axes[0, 1].set_yscale("log")

            sns.histplot(log_value, stat="density", bins=100, ax=axes[1, 0])
            axes[1, 0].set_title("log1p(Value)")
            axes[1, 0].set_yscale("log")

            sns.histplot(log_value_scaled, stat="density", bins=100, ax=axes[1, 1])
            axes[1, 1].set_title("log1p(Value) Scaled")
            axes[1, 1].set_yscale("log")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"{save_path}{save_name}_all.pdf", dpi=300,
                        bbox_inches='tight')
            plt.close()

        else:
            # 2 Subplots (kein log1p)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f"Histogramme für {col}", fontsize=14)

            sns.histplot(value, stat="density", bins=100, ax=axes[0])
            axes[0].set_title("Original Value")
            axes[0].set_yscale("log")

            sns.histplot(value_scaled, stat="density", bins=100, ax=axes[1])
            axes[1].set_title("Value Scaled")
            axes[1].set_yscale("log")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"{save_path}{save_name}_all.pdf", dpi=300,
                        bbox_inches='tight')
            plt.close()


def plot_luptitudes_logscale(zp=30, bands=None, band_maglims=None, save_path=None):
    """
    Plottet Luptitudes in logarithmischer Darstellung für angegebene Bänder mit Softening-Marker.

    :param zp: Zero Point (default: 30)
    :param bands: Liste der Bänder (z.B. ["i", "g", "r", "z", "u", "Y", "J", "H", "K"])
    :param band_maglims: Dict mit Limiting Magnitudes je Band (default: predefined)
    :param save_path: Optionaler Pfad zum Speichern des Plots (png)
    """
    # Standardwerte, falls nichts übergeben
    if bands is None:
        bands = ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]
    if band_maglims is None:
        band_maglims = {
            "i": 24.66,
            "g": 25.57,
            "r": 25.27,
            "z": 24.06,
            "u": 24.64,
            "Y": 24.6,
            "J": 24.02,
            "H": 23.69,
            "K": 23.58
        }

    def luptize(flux, var, s, zp):
        a = 2.5 * np.log10(np.exp(1))
        b = a ** 0.5 * s
        mu0 = zp - 2.5 * np.log10(b)
        lupt = mu0 - a * np.arcsinh(flux / (2 * b))
        lupt_var = a ** 2 * var / ((2 * b) ** 2 + flux ** 2)
        return lupt, lupt_var, b

    # Flux-Werte (log-spaced positiv)
    # flux_vals = np.logspace(-9999000000.0, 1000000, 5000)
    flux_vals = np.linspace(-9999000000.0, 1e6, 5000)
    var_vals = np.zeros_like(flux_vals)

    # Subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharex=False, sharey=True)
    axes = axes.flatten()

    for i, band in enumerate(bands):
        mag_lim = band_maglims[band]
        s = (10 ** ((zp - mag_lim) / 2.5)) / 10
        lupt_vals, _, b = luptize(flux_vals, var_vals, s, zp)

        ax = axes[i]
        ax.plot(flux_vals, lupt_vals, label=f"{band}-band", color="blue")
        ax.axvline(b, color="red", linestyle=":", label=f"b ≈ {b:.2e}")
        # ax.set_xscale("log")
        ax.set_title(f"{band}-Band (lim_mag={mag_lim})")
        ax.invert_yaxis()
        ax.grid(True, which="both", ls="--")
        if i % 3 == 0:
            ax.set_ylabel("Luptitude")
        if i >= 6:
            ax.set_xlabel("Flux")
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def create_scatter_plot(data_frame, lst_col, column, title, save_name, save_plot):
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(lst_col):
        if col == column:
            continue
        ax = axes[i]
        sns.scatterplot(y=data_frame[column], x=data_frame[col], ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel(column)

    plt.suptitle(title)
    plt.tight_layout()
    if save_plot is True:
        plt.savefig(save_name, dpi=300)
    else:
        plt.show()


def plot_hist(data_frame, lst_col, column, title, save_name, save_plot):
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(lst_col):
        if col == column:
            continue
        sns.histplot(data=data_frame[column], ax=axes[i], bins=30)
        plt.show()

def plot_data(data_frame, lst_input_cols):
    df_cut_err_r = data_frame[data_frame['BDF_LUPT_ERR_DERED_CALIB_R'] > 10]
    df_cut_err_i = data_frame[data_frame['BDF_LUPT_ERR_DERED_CALIB_I'] > 10]
    df_cut_err_z = data_frame[data_frame['BDF_LUPT_ERR_DERED_CALIB_Z'] > 10]
    df_cut_err_r = df_cut_err_r[lst_input_cols]
    df_cut_err_i = df_cut_err_i[lst_input_cols]
    df_cut_err_z = df_cut_err_z[lst_input_cols]

    df_cut_err_r_drop = df_cut_err_r.drop_duplicates()
    df_cut_err_i_drop = df_cut_err_i.drop_duplicates()
    df_cut_err_z_drop = df_cut_err_z.drop_duplicates()

    print(f"length values > 10 in BDF_LUPT_ERR_DERED_CALIB_R: {len(df_cut_err_r)}")
    print(f"length values > 10 in BDF_LUPT_ERR_DERED_CALIB_I: {len(df_cut_err_i)}")
    print(f"length values > 10 in BDF_LUPT_ERR_DERED_CALIB_Z: {len(df_cut_err_z)}")

    print(f"length values > 10 in BDF_LUPT_ERR_DERED_CALIB_R drop duplicates: {len(df_cut_err_r_drop)}")
    print(f"length values > 10 in BDF_LUPT_ERR_DERED_CALIB_I drop duplicates: {len(df_cut_err_i_drop)}")
    print(f"length values > 10 in BDF_LUPT_ERR_DERED_CALIB_Z drop duplicates: {len(df_cut_err_z_drop)}")

    create_scatter_plot(
        data_frame=df_cut_err_r,
        lst_col=[col for col in lst_input_cols if col != "BDF_LUPT_ERR_DERED_CALIB_R"],
        column="BDF_LUPT_ERR_DERED_CALIB_R",
        title=f"len BDF_LUPT_ERR_DERED_CALIB_R > 10: {len(df_cut_err_r)}, drop duplicates: {len(df_cut_err_r_drop)}",
        save_name="/Users/P.Gebhardt/Desktop/histograms_of_my_data/Scatter_err_r.pdf",
        save_plot=False
    )
    create_scatter_plot(
        data_frame=df_cut_err_i,
        lst_col=[col for col in lst_input_cols if col != "BDF_LUPT_ERR_DERED_CALIB_I"],
        column="BDF_LUPT_ERR_DERED_CALIB_I",
        title=f"len BDF_LUPT_ERR_DERED_CALIB_I > 10: {len(df_cut_err_i)}, drop duplicates: {len(df_cut_err_i_drop)}",
        save_name="/Users/P.Gebhardt/Desktop/histograms_of_my_data/Scatter_err_i.pdf",
        save_plot=False
    )
    create_scatter_plot(
        data_frame=df_cut_err_z,
        lst_col=[col for col in lst_input_cols if col != "BDF_LUPT_ERR_DERED_CALIB_Z"],
        column="BDF_LUPT_ERR_DERED_CALIB_Z",
        title=f"len BDF_LUPT_ERR_DERED_CALIB_Z > 10: {len(df_cut_err_z)}, drop duplicates: {len(df_cut_err_z_drop)}",
        save_name="/Users/P.Gebhardt/Desktop/histograms_of_my_data/Scatter_err_z.pdf",
        save_plot=False
    )


def replace_defaults(cfg, data_frame):
    len_before = len(data_frame)
    if cfg["REPLACE_DEFAULTS"] is True:
        for col in cfg["DEFAULTS"].keys():
            # print(col, (data_frame[col] == cfg['DEFAULTS'][col]).sum())
            print(f"replace defaults drop: col={col} val={cfg['DEFAULTS'][col]}")
            indices_to_drop = data_frame[data_frame[col] == cfg['DEFAULTS'][col]].index
            data_frame.drop(indices_to_drop, inplace=True)
        len_after = len(data_frame)
        # for k in data_frame.keys():
        #     print(k, data_frame[k].min(), data_frame[k].max())
        print("Dropped {} rows".format(len_before - len_after))
        return data_frame


def check_det_mag_limit(data_frame, save_plot, save_name):
    data_frame["BDF_DET_FLUX"] = data_frame[
        ["BDF_FLUX_DERED_CALIB_R", "BDF_FLUX_DERED_CALIB_I", "BDF_FLUX_DERED_CALIB_Z"]
    ].mean(axis=1)
    data_frame["BDF_DET_MAG"] = flux2mag(data_frame["BDF_DET_FLUX"])

    sns.scatterplot(x=data_frame["ID"], y=data_frame["BDF_DET_MAG"])
    y_max = data_frame["BDF_DET_MAG"].max()
    plt.axhline(y_max, color='red', linestyle='--', label=f'Max: {y_max:.2f}')
    plt.text(
        x=data_frame["ID"].max(),
        y=y_max,
        s=f'Max: {y_max:.2f}',
        va='bottom',
        ha='right',
        backgroundcolor='white',
        fontsize=10
    )

    plt.xlabel("ID")
    plt.ylabel("BDF_DET_MAG")
    plt.title("Injection Limit")
    if save_plot is True:
        plt.savefig(save_name, dpi=300)
    else:
        plt.show()
    return data_frame

def check_snr(data_frame, save_plot=False, save_name="snr_plot.png"):
    # SNR berechnen
    data_frame["SNR_r"] = data_frame["BDF_FLUX_DERED_CALIB_R"] / data_frame["BDF_FLUX_ERR_DERED_CALIB_R"]
    data_frame["SNR_i"] = data_frame["BDF_FLUX_DERED_CALIB_I"] / data_frame["BDF_FLUX_ERR_DERED_CALIB_I"]
    data_frame["SNR_z"] = data_frame["BDF_FLUX_DERED_CALIB_Z"] / data_frame["BDF_FLUX_ERR_DERED_CALIB_Z"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Band r
    sns.scatterplot(x=data_frame["ID"], y=data_frame["SNR_r"], ax=axes[0], color="red", s=10)
    axes[0].set_title("r-Band")
    axes[0].axhline(data_frame["SNR_r"].min(), color='black', linestyle='--', label=f"min: {data_frame['SNR_r'].min():.2f}")
    axes[0].legend()
    axes[0].set_ylabel("SNR_r")

    # Band i
    sns.scatterplot(x=data_frame["ID"], y=data_frame["SNR_i"], ax=axes[1], color="green", s=10)
    axes[1].set_title("i-Band")
    axes[1].axhline(data_frame["SNR_i"].min(), color='black', linestyle='--', label=f"min: {data_frame['SNR_i'].min():.2f}")
    axes[1].legend()
    axes[1].set_ylabel("SNR_i")

    # Band z
    sns.scatterplot(x=data_frame["ID"], y=data_frame["SNR_z"], ax=axes[2], color="blue", s=10)
    axes[2].set_title("z-Band")
    axes[2].axhline(data_frame["SNR_z"].min(), color='black', linestyle='--', label=f"min: {data_frame['SNR_z'].min():.2f}")
    axes[2].legend()
    axes[2].set_ylabel("SNR_z")
    axes[2].set_xlabel("Objekt-ID")

    plt.tight_layout()

    if save_plot:
        plt.savefig(save_name, dpi=300)
    else:
        plt.show()

def plot_histograms(cfg, data_frame):
    plt.close('all')
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    for col in cfg["DEFAULT_COLUMNS"].keys():
        data_frame[f"{col}_default_shifted"] = data_frame[col].replace(cfg["DEFAULT_COLUMNS"][col][0],
                                                                     cfg["DEFAULT_COLUMNS"][col][1])

    for col in cfg["SHIFT_COLUMNS"]:
        col_min = data_frame[col].min()
        data_frame[f"{col}_{col_min:.4f}_shifted"] = data_frame[col] + np.abs(col_min)

    for col in cfg["CLASSIFIER_COLUMNS_OF_INTEREST"]:
        save_name = col.replace("unsheared/", "")
        minmaxscaler = MinMaxScaler(feature_range=(-30, 30))

        if col in cfg["DEFAULT_COLUMNS"].keys():
            value = data_frame[f"{col}_default_shifted"].values
        elif col in cfg["SHIFT_COLUMNS"]:
            col_min = data_frame[col].min()
            value = data_frame[f"{col}_{col_min:.4f}_shifted"].values
        else:
            value = data_frame[col].values

        if col in cfg["COLUMNS_LOG1P"]:
            log_value = np.log1p(value)
            log_value_scaled = minmaxscaler.fit_transform(log_value.reshape(-1, 1)).flatten()

            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            fig.suptitle(f"Histogramme für {col}", fontsize=14)

            if col in list(cfg["DEFAULT_COLUMNS"].keys()) + cfg["SHIFT_COLUMNS"]:
                origin_title = f"Shifted Value"
            else:
                origin_title = f"Original Value"
            sns.histplot(value, stat="count", bins=100, ax=axes[0])
            axes[0].set_title(origin_title)
            axes[0].set_yscale("log")

            sns.histplot(log_value, stat="count", bins=100, ax=axes[1])
            axes[1].set_title("log1p(Value)")
            axes[1].set_yscale("log")

            sns.histplot(log_value_scaled, stat="count", bins=100, ax=axes[2])
            axes[2].set_title("log1p(Value) Scaled")
            axes[2].set_yscale("log")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data_new/classifier_{save_name}.pdf", dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
        else:
            value_scaled = minmaxscaler.fit_transform(value.reshape(-1, 1)).flatten()

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f"Histogramme für {col}", fontsize=14)

            sns.histplot(value, stat="count", bins=100, ax=axes[0])
            axes[0].set_title("Original Value")
            axes[0].set_yscale("log")

            sns.histplot(value_scaled, stat="count", bins=100, ax=axes[1])
            axes[1].set_title("Value Scaled")
            axes[1].set_yscale("log")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data_new/nf_{save_name}.pdf", dpi=300,
                        bbox_inches='tight')
            plt.close(fig)

    data_frame_only_detected = data_frame[data_frame['detected'] == 1].copy()

    data_frame_only_detected = calc_color(
        cfg=cfg,
        data_frame=data_frame_only_detected,
        mag_type=("LUPT", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
        bins=cfg['BDF_BINS'],
        save_name=f"bdf_lupt"
    )
    data_frame_only_detected = calc_color(
        cfg=cfg,
        data_frame=data_frame_only_detected,
        mag_type=("LUPT", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/lupt", "unsheared/lupt_err"),
        bins=cfg['UNSHEARED_BINS'],
        save_name=f"unsheared/lupt"
    )

    for col in cfg["NF_COLUMNS_OF_INTEREST"]:
        save_name = col.replace("unsheared/", "")
        minmaxscaler = MinMaxScaler(feature_range=(-30, 30))

        if col in cfg["DEFAULT_COLUMNS"].keys():
            value = data_frame[f"{col}_default_shifted"].values
        elif col in cfg["SHIFT_COLUMNS"]:
            col_min = data_frame[col].min()
            value = data_frame[f"{col}_{col_min:.4f}_shifted"].values
        else:
            value = data_frame[col].values

        if col in cfg["COLUMNS_LOG1P"]:
            log_value = np.log1p(value)
            log_value_scaled = minmaxscaler.fit_transform(log_value.reshape(-1, 1)).flatten()

            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            fig.suptitle(f"Histogramme für {col}", fontsize=14)

            if col in list(cfg["DEFAULT_COLUMNS"].keys()) + cfg["SHIFT_COLUMNS"]:
                origin_title = f"Shifted Value"
            else:
                origin_title = f"Original Value"
            sns.histplot(value, stat="count", bins=100, ax=axes[0])
            axes[0].set_title(origin_title)
            axes[0].set_yscale("log")

            sns.histplot(log_value, stat="count", bins=100, ax=axes[1])
            axes[1].set_title("log1p(Value)")
            axes[1].set_yscale("log")

            sns.histplot(log_value_scaled, stat="count", bins=100, ax=axes[2])
            axes[2].set_title("log1p(Value) Scaled")
            axes[2].set_yscale("log")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            # plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data/{save_name}_all.pdf", dpi=300,
            #             bbox_inches='tight')
            plt.show()
            plt.close()
        else:
            value_scaled = minmaxscaler.fit_transform(value.reshape(-1, 1)).flatten()

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f"Histogramme für {col}", fontsize=14)

            sns.histplot(value, stat="count", bins=100, ax=axes[0])
            axes[0].set_title("Original Value")
            axes[0].set_yscale("log")

            sns.histplot(value_scaled, stat="count", bins=100, ax=axes[1])
            axes[1].set_title("Value Scaled")
            axes[1].set_yscale("log")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            # plt.savefig(f"/Users/P.Gebhardt/Desktop/histograms_of_my_data/{save_name}_all.pdf", dpi=300,
            #             bbox_inches='tight')
            plt.show()
            plt.close()

def plot_hpix(data_frame, column):
    nside = 4096
    npix = hp.nside2npix(nside)
    hp_map = np.full(npix, np.nan)
    grouped = data_frame.groupby("HPIX_4096")[column].mean()
    for pix, val in grouped.items():
        hp_map[int(pix)] = val
    hp.mollview(hp_map, title=column, unit=column, nest=True, cmap="viridis")
    plt.show()

def plot_hpix_detected(data_frame, column):
    nside = 4096
    npix = hp.nside2npix(nside)
    hp_map = np.full(npix, np.nan)
    grouped = data_frame.groupby("HPIX_4096")[column].any().astype(int)
    for pix, val in grouped.items():
        hp_map[int(pix)] = val
    hp.mollview(hp_map, title=column, unit=column, nest=True, cmap="viridis")
    plt.show()

    nside = 4096
    npix = hp.nside2npix(nside)
    hp_map = np.full(npix, np.nan)
    grouped = data_frame.groupby("HPIX_4096")[column].sum()
    for pix, val in grouped.items():
        hp_map[int(pix)] = val
    hp.mollview(hp_map, title=column, unit=column, nest=True, cmap="viridis")
    plt.show()

def check_hpix_assignemnt(cfg):
    detection_data = Table(fitsio.read(cfg["PATH_DETECT"]).byteswap().newbyteorder())

    df_detect = pd.DataFrame({
        "detected": detection_data["detected"],
        "true_ra": detection_data["true_ra"],
        "true_dec": detection_data["true_dec"],
    })

    NSIDE = 4096
    arr_dec = np.array(df_detect["true_dec"])
    arr_ra = np.array(df_detect["true_ra"])

    df_detect[f"HPIX_{NSIDE}"] = hp.pixelfunc.ang2pix(NSIDE, np.radians(-arr_dec + 90.), np.radians(360. + arr_ra), nest=True).astype(int)

    print(df_detect.head())

def main(cfg):
    df_merged_cat = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MERGED_CAT']}")

    lst_obs_cond = [
        "AIRMASS_WMEAN_R",
        "AIRMASS_WMEAN_I",
        "AIRMASS_WMEAN_Z",
        "FWHM_WMEAN_R",
        "FWHM_WMEAN_I",
        "FWHM_WMEAN_Z",
        "MAGLIM_R",
        "MAGLIM_I",
        "MAGLIM_Z",
        "EBV_SFD98"
    ]

    # for con_col in lst_obs_cond:
    #     data_frame_wo_def = df_merged_cat[df_merged_cat[con_col]!=-9999.0].copy()
    #     data_frame_o_def = df_merged_cat[df_merged_cat[con_col]==-9999.0].copy()
    #     plot_hpix(data_frame=data_frame_wo_def, column=con_col)
    #     plot_hpix(data_frame=data_frame_o_def, column=con_col)

    plot_hpix_detected(data_frame=df_merged_cat, column="detected")
    exit()
    # check_hpix_assignemnt(cfg)
    df_merged_cat_true_deep = open_all_balrog_dataset(f"{cfg['PATH_DATA']}/{cfg['FILENAME_MERGED_CAT_TRUE_DEEP']}")
    # check_det_mag_limit(
    #     data_frame=df_merged_cat,
    #     save_plot=True,
    #     save_name="/Users/P.Gebhardt/Desktop/histograms_of_my_data/My_deep_mag_det_limit.pdf"
    # )
    #
    # check_det_mag_limit(
    #     data_frame=df_merged_cat_true_deep,
    #     save_plot=True,
    #     save_name="/Users/P.Gebhardt/Desktop/histograms_of_my_data/True_deep_mag_det_limit.pdf"
    # )

    check_snr(
        data_frame=df_merged_cat,
        save_plot=False,
        save_name="/Users/P.Gebhardt/Desktop/histograms_of_my_data/snr_my_deep.pdf"
    )

    check_snr(
        data_frame=df_merged_cat_true_deep,
        save_plot=False,
        save_name="/Users/P.Gebhardt/Desktop/histograms_of_my_data/snr_true_deep.pdf"
    )

    df_merged_cat.rename(
        columns={
            'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
            'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
        inplace=True)

    df_merged_cat_true_deep.rename(
        columns={
            'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
            'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
        inplace=True)

    df_merged_cat["BDF_G"] = np.sqrt(df_merged_cat["BDF_G_0"] ** 2 + df_merged_cat["BDF_G_1"] ** 2)

    df_merged_cat = calc_color(
        cfg=cfg,
        data_frame=df_merged_cat,
        mag_type=("LUPT", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
        bins=cfg['BDF_BINS'],
        save_name=f"bdf_lupt"
    )
    df_merged_cat = calc_color(
        cfg=cfg,
        data_frame=df_merged_cat,
        mag_type=("LUPT", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/lupt", "unsheared/lupt_err"),
        bins=cfg['UNSHEARED_BINS'],
        save_name=f"unsheared/lupt"
    )

    df_merged_cat_true_deep = calc_color(
        cfg=cfg,
        data_frame=df_merged_cat_true_deep,
        mag_type=("LUPT", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
        bins=cfg['BDF_BINS'],
        save_name=f"bdf_lupt"
    )
    df_merged_cat_true_deep = calc_color(
        cfg=cfg,
        data_frame=df_merged_cat_true_deep,
        mag_type=("LUPT", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/lupt", "unsheared/lupt_err"),
        bins=cfg['UNSHEARED_BINS'],
        save_name=f"unsheared/lupt"
    )

    # check(
    #     cfg=cfg,
    #     data_frame=df_merged_cat,
    #     save_path=f"/Users/P.Gebhardt/Desktop/histograms_of_my_data_my_deep/"
    # )

    check(
        cfg=cfg,
        data_frame=df_merged_cat_true_deep,
        save_path=f"/Users/P.Gebhardt/Desktop/histograms_of_my_data_true_deep/"
    )

    plot_data(
        data_frame=df_merged_cat,
        lst_input_cols=[
            "BDF_LUPT_DERED_CALIB_R",
            "BDF_LUPT_DERED_CALIB_I",
            "BDF_LUPT_DERED_CALIB_Z",
            "BDF_LUPT_ERR_DERED_CALIB_R",
            "BDF_LUPT_ERR_DERED_CALIB_I",
            "BDF_LUPT_ERR_DERED_CALIB_Z",
            "BDF_T",
            "BDF_G",
            'Color BDF LUPT U-G',
            'Color BDF LUPT G-R',
            'Color BDF LUPT R-I',
            'Color BDF LUPT I-Z',
            'Color BDF LUPT Z-J',
            'Color BDF LUPT J-H',
            'Color BDF LUPT H-K',
            'ID'
        ]
    )
    plot_data(
        data_frame=df_merged_cat_true_deep,
        lst_input_cols=[
            "BDF_LUPT_DERED_CALIB_R",
            "BDF_LUPT_DERED_CALIB_I",
            "BDF_LUPT_DERED_CALIB_Z",
            "BDF_LUPT_ERR_DERED_CALIB_R",
            "BDF_LUPT_ERR_DERED_CALIB_I",
            "BDF_LUPT_ERR_DERED_CALIB_Z",
            'Color BDF LUPT U-G',
            'Color BDF LUPT G-R',
            'Color BDF LUPT R-I',
            'Color BDF LUPT I-Z',
            'Color BDF LUPT Z-J',
            'Color BDF LUPT J-H',
            'Color BDF LUPT H-K',
            'ID'
        ]
    )



if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    path = os.path.abspath(sys.path[-1])
    config_file_name = "mac_check_data.cfg"

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

    with open(f"{path}/config/{args.config_filename}", 'r') as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    main(cfg=config)