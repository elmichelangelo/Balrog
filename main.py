import pandas as pd

from Handler.cut_functions import *
from Handler.plot_functions import *
from Handler.helper_functions import *
import seaborn as sns
import os
import sys


def create_balrog_subset(path_all_balrog_data, path_save, name_save_file, number_of_samples, only_detected,
                         apply_fill_na, apply_replace_defaults, apply_object_cut, apply_flag_cut,
                         apply_unsheared_mag_cut, apply_unsheared_shear_cut, apply_airmass_cut, apply_binary_cut,
                         apply_mask_cut, bdf_bins, unsheared_bins, fill_na, replace_defaults, protocol=None,
                         plot_color=False, path_master=None):
    """"""
    df_balrog = open_all_balrog_dataset(path_all_balrog_data)
    df_balrog.rename(
        columns={
            'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
            'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
        inplace=True)
    # df_balrog["BDF_G"] = np.sqrt(df_balrog["BDF_G_0"] ** 2 + df_balrog["BDF_G_1"] ** 2)

    print(df_balrog.isna().sum())
    if apply_fill_na == "Gauss":
        print(f"start fill na gauss")
        for col in fill_na.keys():
            df_balrog[col] = df_balrog[col].apply(
                replace_nan_with_gaussian, args=(fill_na[col][0], fill_na[col][1]))
    elif apply_fill_na == "Default":
        print(f"start fill na default")
        for col in fill_na.keys():
            df_balrog[col].fillna(fill_na[col][0])
    elif apply_fill_na is None:
        print("No fill na")
    print(df_balrog.isna().sum())

    if apply_replace_defaults == "Gauss":
        print(f"start replace default gauss")
        df_balrog = replace_values_with_gaussian(df_balrog, replace_defaults)
    elif apply_replace_defaults == "Default":
        print(f"start replace default default")
        df_balrog = replace_values(df_balrog, replace_defaults)
    elif apply_replace_defaults is None:
        print("No default replace")

    df_balrog = calc_color(
        data_frame=df_balrog,
        mag_type=("LUPT", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB"),
        bins=bdf_bins,
        plot_data=plot_color,
        plot_name=f"bdf_lupt"
    )
    df_balrog = calc_color(
        data_frame=df_balrog,
        mag_type=("LUPT", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/lupt", "unsheared/lupt_err"),
        bins=unsheared_bins,
        plot_data=plot_color,
        plot_name=f"unsheared/lupt"
    )
    df_balrog = calc_color(
        data_frame=df_balrog,
        mag_type=("MAG", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
        bins=bdf_bins,
        plot_data=plot_color,
        plot_name=f"bdf_mag"
    )
    df_balrog = calc_color(
        data_frame=df_balrog,
        mag_type=("MAG", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/mag", "unsheared/mag_err"),
        bins=unsheared_bins,
        plot_data=plot_color,
        plot_name=f"unsheared/mag"
    )

    print(f"length of all balrog objects {len(df_balrog)}")
    if only_detected is True:
        df_balrog = df_balrog[df_balrog["detected"] == 1]
        print(f"length of only detected balrog objects {len(df_balrog)}")
    if apply_object_cut is True:
        df_balrog = unsheared_object_cuts(data_frame=df_balrog)
    if apply_flag_cut is True:
        df_balrog = flag_cuts(data_frame=df_balrog)
    if apply_unsheared_mag_cut is True:
        df_balrog = unsheared_mag_cut(data_frame=df_balrog)
    if apply_unsheared_shear_cut is True:
        df_balrog = unsheared_shear_cuts(data_frame=df_balrog)
    if apply_airmass_cut is True:
        df_balrog = airmass_cut(data_frame=df_balrog)
    if apply_binary_cut is True:
        df_balrog = binary_cut(data_frame=df_balrog)
    if apply_mask_cut is True:
        df_balrog = mask_cut(data_frame=df_balrog, master=path_master)
    print(f"length of catalog after applying unsheared cuts {len(df_balrog)}")

    if number_of_samples is None:
        number_of_samples = len(df_balrog)

    df_balrog_subset = df_balrog.sample(number_of_samples, ignore_index=True, replace=False)
    path_balrog_subset = f"{path_save}/{name_save_file}_{number_of_samples}.pkl"

    ####################################################################################################################
    # Todo size_ratio macht schwierigkeiten beim lernen. Was kann ich machen, damit das besser wird?
    # plot_histo(
    #     data_frame=df_balrog,
    #     cols=[
    #         "unsheared/size_ratio"
    #     ],
    # )
    # plot_chain(
    #     data_frame=df_balrog,
    #     plot_name="hist_t",
    #     columns=[
    #         "unsheared/T",
    #         "unsheared/size_ratio"
    #     ],
    #     parameter=[
    #         "T",
    #         "size_ratio"
    #     ]
    # )
    # yj_transform_data(
    #     data_frame=df_balrog,
    #     columns=[
    #         "unsheared/T"
    #     ]
    # )
    #
    # plot_histo(
    #     data_frame=df_balrog,
    #     cols=[
    #         "unsheared/size_ratio"
    #     ],
    # )
    # plot_chain(
    #     data_frame=df_balrog,
    #     plot_name="hist_t",
    #     columns=[
    #         "unsheared/T",
    #         "unsheared/size_ratio"
    #     ],
    #     parameter=[
    #         "T",
    #         "size_ratio"
    #     ]
    # )
    # exit()
    ####################################################################################################################

    save_balrog_subset(
        data_frame=df_balrog_subset,
        path_balrog_subset=path_balrog_subset,
        protocol=protocol
    )


if __name__ == '__main__':
    path = os.path.abspath(sys.path[0])
    no_samples = None  # int(3E6)  # int(250000)  # int(250000) # 20000 # int(8E6)

    dict_fill_na = {
        'unsheared/snr': (-10, 2.0),
        'unsheared/T': (-10, 2.0),
        'unsheared/size_ratio': (-10, 5.0),
        'AIRMASS_WMEAN_R': (-10, 2.0),
        'AIRMASS_WMEAN_I': (-10, 2.0),
        'AIRMASS_WMEAN_Z': (-10, 2.0),
        'FWHM_WMEAN_R': (-10, 2.0),
        'FWHM_WMEAN_I': (-10, 2.0),
        'FWHM_WMEAN_Z': (-10, 2.0),
        'MAGLIM_R': (-10, 2.0),
        'MAGLIM_I': (-10, 2.0),
        'MAGLIM_Z': (-10, 2.0),
        'EBV_SFD98': (-10, 2.0),
        'unsheared/flags': (-10, 2.0),
        'unsheared/flux_r': (-20000, 2000),
        'unsheared/flux_i': (-20000, 2000),
        'unsheared/flux_z': (-20000, 2000),
        'unsheared/flux_err_r': (-10, 2.0),
        'unsheared/flux_err_i': (-10, 2.0),
        'unsheared/flux_err_z': (-10, 2.0),
        'unsheared/extended_class_sof': (-20, 2.0),
        'unsheared/flags_gold': (-10, 2.0),
        'unsheared/weight': (-10, 2.0)
    }

    dict_replace_defaults = {
        'unsheared/snr': (-7070.360705084288, -2, 2.0),
        'unsheared/T': (-9999, -2, 2.0),
        'AIRMASS_WMEAN_R': (-9999, -2, 2.0),
        'AIRMASS_WMEAN_I': (-9999, -2, 2.0),
        'AIRMASS_WMEAN_Z': (-9999, -2, 2.0),
        'FWHM_WMEAN_R': (-9999, -2, 2.0),
        'FWHM_WMEAN_I': (-9999, -2, 2.0),
        'FWHM_WMEAN_Z': (-9999, -2, 2.0),
        'MAGLIM_R': (-9999, -2, 2.0),
        'MAGLIM_I': (-9999, -2, 2.0),
        'MAGLIM_Z': (-9999, -2, 2.0)
    }



    ####################################################################################################################

    deep_field_cols = [
        # "BDF_FLUX_DERED_CALIB_U",
        # "BDF_FLUX_DERED_CALIB_G",
        "BDF_FLUX_DERED_CALIB_R",
        "BDF_FLUX_DERED_CALIB_I",
        "BDF_FLUX_DERED_CALIB_Z",
        # "BDF_FLUX_DERED_CALIB_J",
        # "BDF_FLUX_DERED_CALIB_H",
        # "BDF_FLUX_DERED_CALIB_K",
        "unsheared/flux_r",
        "unsheared/flux_i",
        "unsheared/flux_z"
    ]

    deep_field_cols_mag = [
        # "BDF_MAG_DERED_CALIB_U",
        # "BDF_MAG_DERED_CALIB_G",
        "BDF_MAG_DERED_CALIB_R",
        "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        # "BDF_MAG_DERED_CALIB_J",
        # "BDF_MAG_DERED_CALIB_H",
        # "BDF_MAG_DERED_CALIB_K",
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z"
    ]

    bdf_bins = ["R", "I", "Z"]  # "U", "G", , "J", "H", "K"
    unsheared_bins = ["r", "i", "z"]
    plot_color = False
    #
    # from astropy.table import Table
    # import h5py
    # import fitsio
    # path_deep_field1 = f"{path}/Data/y3_balrog2_v1.2_merged_select2_bstarcut_matchflag1.5asec_snr_SR_corrected_uppersizecuts.h5"
    # df_data_reg = pd.read_hdf(path_deep_field1)
    #
    infile_my = open(f"{path}/Data/balrog_cat_mcal_detect_my_df_26442133.pkl", 'rb')
    df_data_my = pd.DataFrame(pickle.load(infile_my, encoding='latin1'))
    infile_my.close()
    infile_reg = open(f"{path}/Data/y3-merged_deep_field_metacal_cuts_v1.2.pkl", 'rb')
    df_data_reg = pd.DataFrame(pickle.load(infile_reg, encoding='latin1'))
    infile_reg.close()

    # df_data_my.rename(
    #     columns={
    #         'BDF_FLUX_DERED_CALIB_KS': 'BDF_FLUX_DERED_CALIB_K',
    #         'BDF_FLUX_ERR_DERED_CALIB_KS': 'BDF_FLUX_ERR_DERED_CALIB_K'},
    #     inplace=True)
    #
    # df_data_reg['BDF_FLUX_ERR_DERED_CALIB_R'] = df_data_reg['BDF_FLUX_DERED_CALIB_R']
    # df_data_reg['BDF_FLUX_ERR_DERED_CALIB_I'] = df_data_reg['BDF_FLUX_DERED_CALIB_I']
    # df_data_reg['BDF_FLUX_ERR_DERED_CALIB_Z'] = df_data_reg['BDF_FLUX_DERED_CALIB_Z']
    df_data_reg['unsheared/flux_err_r'] = df_data_reg['unsheared/flux_r']
    df_data_reg['unsheared/flux_err_i'] = df_data_reg['unsheared/flux_i']
    df_data_reg['unsheared/flux_err_z'] = df_data_reg['unsheared/flux_z']
    # df_data_reg.rename(
    #     columns={
    #         'BDF_FLUX_DERED_CALIB_R': 'BDF_FLUX_ERR_DERED_CALIB_R',
    #         'BDF_FLUX_DERED_CALIB_R': 'BDF_FLUX_ERR_DERED_CALIB_R',
    #         'BDF_FLUX_DERED_CALIB_R': 'BDF_FLUX_ERR_DERED_CALIB_R'},
    #     inplace=True)
    #
    # # for idx, col in enumerate(deep_field_cols_mag):
    # #     df_data_my[col] = flux2mag(df_data_my[deep_field_cols[idx]])
    # #     df_data_reg[col] = flux2mag(df_data_reg[deep_field_cols[idx]])
    #
    df_data_my = calc_color(
        data_frame=df_data_my,
        mag_type=("MAG", "BDF"),
        flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
        mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
        bins=bdf_bins,
        plot_data=plot_color,
        plot_name=f"bdf_mag"
    )
    df_data_my = calc_color(
        data_frame=df_data_my,
        mag_type=("MAG", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/mag", "unsheared/mag_err"),
        bins=unsheared_bins,
        plot_data=plot_color,
        plot_name=f"unsheared/mag"
    )

    # df_data_reg = calc_color(
    #     data_frame=df_data_reg,
    #     mag_type=("MAG", "BDF"),
    #     flux_col=("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB"),
    #     mag_col=("BDF_MAG_DERED_CALIB", "BDF_MAG_ERR_DERED_CALIB"),
    #     bins=bdf_bins,
    #     plot_data=plot_color,
    #     plot_name=f"bdf_mag"
    # )
    df_data_reg = calc_color(
        data_frame=df_data_reg,
        mag_type=("MAG", "unsheared"),
        flux_col=("unsheared/flux", "unsheared/flux_err"),
        mag_col=("unsheared/mag", "unsheared/mag_err"),
        bins=unsheared_bins,
        plot_data=plot_color,
        plot_name=f"unsheared/mag"
    )

    df_data_my = df_data_my[df_data_my["detected"] == 1]
    df_data_my = unsheared_object_cuts(data_frame=df_data_my)
    df_data_my = flag_cuts(data_frame=df_data_my)
    df_data_my = unsheared_mag_cut(data_frame=df_data_my)
    df_data_my = unsheared_shear_cuts(data_frame=df_data_my)
    df_data_my = binary_cut(data_frame=df_data_my)

    # df_data_reg = df_data_reg[df_data_reg["detected"] == 1]
    # df_data_reg = unsheared_object_cuts(data_frame=df_data_reg)
    # df_data_reg = flag_cuts(data_frame=df_data_reg)
    # df_data_reg = unsheared_mag_cut(data_frame=df_data_reg)
    # df_data_reg = unsheared_shear_cuts(data_frame=df_data_reg)
    # df_data_reg = binary_cut(data_frame=df_data_reg)

    df_plot = pd.DataFrame()
    for col in deep_field_cols_mag:
        df_plot[col] = list(df_data_my[col]) + list(df_data_reg[col])

    df_plot["type"] = ["my deep field" for _ in range(len(df_data_my))] + ["regular deep field" for _ in range(len(df_data_reg))]

    for col in deep_field_cols_mag:
        mean_my_df = df_data_my[col].mean()
        mean_reg_df = df_data_reg[col].mean()
        std_my_df = df_data_my[col].std()
        std_reg_df = df_data_reg[col].std()
        print("my mean:", mean_my_df, "std:", std_my_df)
        print("reg mean:", mean_reg_df, "std:", std_reg_df)
        sns.histplot(df_plot, x=col, hue="type", stat='density', bins=100)
        plt.title(f"my mean: {mean_my_df:.4f} std: {std_my_df:.4f} reg mean: {mean_reg_df:.4f} std: {std_reg_df:.4f}")
        plt.ylim(0, .3)
        plt.show()

    exit()
    ####################################################################################################################

    create_balrog_subset(
        path_all_balrog_data=f"{path}/Data/balrog_cat_mcal_detect_df_no_cuts_26442133.pkl",  # balrog_cat_mcal_detect_df_no_cuts_26442133    balrog_cat_mcal_detect_reg_df_26442133
        path_save=f"{path}/Output",
        name_save_file="balrog_all_cuts",
        number_of_samples=no_samples,
        only_detected=True,
        apply_fill_na=None,  # "Default"
        apply_replace_defaults=None,  # "Default"
        apply_object_cut=True,
        apply_flag_cut=True,
        apply_unsheared_mag_cut=True,
        apply_unsheared_shear_cut=True,
        apply_airmass_cut=False,
        apply_binary_cut=True,
        apply_mask_cut=True,
        bdf_bins=["U", "G", "R", "I", "Z", "J", "H", "K"],
        unsheared_bins=["r", "i", "z"],
        fill_na=dict_fill_na,
        replace_defaults=dict_replace_defaults,
        protocol=2,
        plot_color=False,
        path_master=f"{path}/Data/Y3_mastercat_03_31_20.h5"
    )
