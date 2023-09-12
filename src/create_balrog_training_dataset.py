import os
import sys
import pickle
import matplotlib.pyplot as plt
from astropy.table import Table
import h5py
import fitsio
import numpy as np
import pandas as pd
import healpy as hp
# from Handler.cut_functions import *


def read_catalogs(path_metacal, path_detection, path_deep_field, path_survey, metacal_cols, detection_cols,
                  deep_field_cols, survey_cols, nside, plot_healpix=False,
                  show_plot=False, save_plot=False):
    """"""
    df_mcal = None
    df_detect = None
    df_deep_field = None
    df_survey = None

    if path_metacal is not None:
        # Read h5py file
        metacal_data = h5py.File(path_metacal, 'r')

        # Add columns to DataFrame
        df_mcal = pd.DataFrame()
        for i, col in enumerate(metacal_cols):
            if col =="unsheared/weight":
                df_mcal[col] = np.array(metacal_data['catalog/' + col])
            else:
                df_mcal[col] = np.array(metacal_data['catalog/' + col]).byteswap().newbyteorder("<")

        # Rename some columns
        df_mcal = df_mcal[metacal_cols]
        df_mcal = df_mcal.rename(columns={'unsheared/bal_id': 'bal_id'})
        df_mcal = df_mcal.rename(columns={'unsheared/coadd_object_id': 'COADD_OBJECT_ID'})

        # Verbose
        print('Length of mcal catalog: {}'.format(len(df_mcal)))
        for i, col in enumerate(df_mcal.columns):
            print(i, col)
        print(df_mcal.isnull().sum())

    if path_detection is not None:
        # Read fits file
        detection_data = Table(fitsio.read(path_detection).byteswap().newbyteorder())

        # Add columns to DataFrame
        df_detect = pd.DataFrame()
        for i, col in enumerate(detection_cols):
            df_detect[col] = detection_data[col]

        # Rename some columns
        df_detect = df_detect.rename(columns={'true_id': 'ID'})

        # Count injections
        df_detect = compute_injection_counts(df_detect)

        # Get HPIX Index
        df_detect[f"HPIX_{nside}"] = DeclRaToIndex(np.array(df_detect["true_dec"]), np.array(df_detect["true_ra"]), nside)

        # Verbose
        print('Length of detection catalog: {}'.format(len(df_detect)))
        for i, col in enumerate(df_detect.keys()):
            print(i, col)
        print(df_detect.isnull().sum())

        # Plot Data
        if plot_healpix is True:
            import healpy as hp
            arr_hpix = df_detect[f"HPIX_{nside}"].to_numpy()
            arr_flux = df_detect["ID"].to_numpy()
            npix = hp.nside2npix(nside)
            hpxmap = np.zeros(npix, dtype=np.float)
            for idx, pix in enumerate(arr_hpix):
                hpxmap[pix] = arr_flux[idx]
            hp.mollview(
                hpxmap,
                norm="hist",
                nest=True
            )
            if show_plot is True:
                plt.show()

    if path_deep_field is not None:
        # Read fits file
        deep_field_data = Table(fitsio.read(path_deep_field).byteswap().newbyteorder())

        # Add columns to DataFrame
        df_deep_field = pd.DataFrame()
        for i, col in enumerate(deep_field_cols):
            print(i, col)
            df_deep_field[col] = deep_field_data[col]

        # Verbose
        print('Length of deep field catalog: {}'.format(len(df_deep_field)))
        print(df_deep_field.isnull().sum())

    if path_survey is not None:
        df_survey = pd.DataFrame()
        for idx, file in enumerate(os.listdir(path_survey)):
            if "fits" in file:
                # Read fits file
                survey_data = Table(fitsio.read(f"{path_survey}/{file}", columns=survey_cols))

                # Add columns to DataFrame
                df_survey_tmp = pd.DataFrame()
                for i, col in enumerate(survey_cols):
                    print(i, col)
                    df_survey_tmp[col] = survey_data[col]
                if idx == 0:
                    df_survey = df_survey_tmp
                else:
                    df_survey = pd.concat([df_survey, df_survey_tmp], ignore_index=True)
                print(df_survey_tmp.shape)

        # Verbose
        print('Length of survey catalog: {}'.format(len(df_survey)))
        print(df_survey.isnull().sum())
        print(df_survey.shape)

        # Plot Data
        if plot_healpix is True:
            arr_hpix = df_survey[f"HPIX_{nside}"].to_numpy()
            arr_flux = df_survey[f"AIRMASS_WMEAN_R"].to_numpy()
            npix = hp.nside2npix(nside)
            hpxmap = np.zeros(npix, dtype=np.float)
            for idx, pix in enumerate(arr_hpix):
                hpxmap[pix] = arr_flux[idx]
            hp.mollview(
                hpxmap,
                norm="hist",
                nest=True)
            if show_plot is True:
                plt.show()
    return df_mcal, df_deep_field, df_detect, df_survey


def compute_injection_counts(det_catalog):
    # `true_id` is the DF id
    unique, ucounts = np.unique(det_catalog['ID'], return_counts=True)
    freq = pd.DataFrame()
    freq['ID'] = unique
    freq['injection_counts'] = ucounts
    return det_catalog.merge(freq, on='ID', how='left')


def merge_catalogs(metacal=None, deep_field=None, detection=None, survey=None):
    """"""
    print('Merging catalogs...')

    print('Merging metacal and detection on bal_id => mcal_detect')
    df_merged = pd.merge(detection, metacal, on='bal_id', how="left")
    print('Length of merged mcal_detect catalog: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())

    print('Merging mcal_detect and deep field on ID => mcal_detect_df')
    df_merged = pd.merge(df_merged, deep_field, on='ID', how="left")
    print('Length of merged mcal_detect_df catalog: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())

    print('Merging mcal_detect_df and survey on HPIX_4096 => mcal_detect_df_survey')
    df_merged = pd.merge(df_merged, survey, on='HPIX_4096', how="left")
    print('Length of merged mcal_detect_df_survey catalog: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())
    return df_merged


def load_healpix(path2file, hp_show=False, nest=True, partial=False, field=None):
    """
    Function to load fits datasets
    Returns:

    """
    import healpy as hp
    if field is None:
        hp_map = hp.read_map(path2file, nest=nest, partial=partial)
    else:
        hp_map = hp.read_map(path2file, nest=nest, partial=partial, field=field)
    if hp_show is True:
        hp_map_show = hp_map
        if field is not None:
            hp_map_show = hp_map[1]
        hp.mollview(
            hp_map_show,
            norm="hist",
            nest=nest
        )
        hp.graticule()
        plt.show()
    return hp_map


def match_skybrite_2_footprint(path2footprint, path2skybrite, hp_show=False,
                               nest_footprint=True, nest_skybrite=True,
                               partial_footprint=False, partial_skybrite=True, field_footprint=None,
                               field_skybrite=None):
    """
    Main function to run
    Returns:

    """
    import healpy as hp
    sky_in_footprint = hp_map_skybrite[:, hp_map_footprint != hp.UNSEEN]
    good_indices = sky_in_footprint[0, :].astype(int)
    return np.column_stack((good_indices, sky_in_footprint[1]))


def IndexToDeclRa(index, NSIDE):
    theta, phi = hp.pixelfunc.pix2ang(NSIDE, index, nest=True)
    return -np.degrees(theta - np.pi / 2.), np.degrees(np.pi * 2. - phi)


def DeclRaToIndex(decl, RA, NSIDE):
    return hp.pixelfunc.ang2pix(NSIDE, np.radians(-decl + 90.), np.radians(360. + RA), nest=True).astype(int)


def write_data_2_file(df_generated_data, save_path, number_of_sources, protocol):
    """"""
    if protocol == 2:
        with open(f"{save_path}{number_of_sources}.pkl", "wb") as f:
            pickle.dump(df_generated_data.to_dict(), f, protocol=2)
    else:
        df_generated_data.to_pickle(f"{save_path}{number_of_sources}.pkl")


def main(path_metacal, path_detection, path_deep_field, path_survey, path_save, metacal_cols, detection_cols,
         deep_field_cols, survey_cols, only_detected, nside, protocol, show_plot, save_plot, plot_healpix):
    """"""

    # Read all catalogs
    df_mcal, df_deep_field, df_detect, df_survey = read_catalogs(
        path_metacal=path_metacal,
        path_detection=path_detection,
        path_deep_field=path_deep_field,
        path_survey=path_survey,
        metacal_cols=metacal_cols,
        detection_cols=detection_cols,
        deep_field_cols=deep_field_cols,
        survey_cols=survey_cols,
        nside=nside,
        plot_healpix=plot_healpix,
        show_plot=show_plot,
        save_plot=save_plot
    )

    if df_mcal is not None and df_detect is not None and df_deep_field is not None and df_survey is not None:
        # Merge all catalogs
        df_merged = merge_catalogs(
            metacal=df_mcal,
            detection=df_detect,
            deep_field=df_deep_field,
            survey=df_survey
        )

        # Save Data to File
        write_data_2_file(
            df_generated_data=df_merged,
            save_path=path_save,
            number_of_sources=len(df_merged),
            protocol=protocol
        )


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    path_data = f"{path}/../Data"

    NSIDE = 4096

    other_metacal_cols = [
        'unsheared/coadd_object_id',
        'unsheared/ra',
        'unsheared/dec',
        'unsheared/snr',
        'unsheared/size_ratio',
        'unsheared/flags',
        'unsheared/bal_id',
        'unsheared/T',
        'unsheared/weight',
        'unsheared/extended_class_sof',
        'unsheared/flags_gold',
        'unsheared/e_1',
        'unsheared/e_2'
    ]

    main(
        path_metacal=f"{path_data}/balrog_mcal_stack-y3v02-0-riz-noNB-mcal_y3-merged_v1.2.h5",
        path_detection=f"{path_data}/balrog_detection_catalog_sof_y3-merged_v1.2.fits",
        path_deep_field=f"{path_data}/deep_field_err.pkl",
        path_survey=f"{path_data}/sct2",
        path_save=f"{path_data}/balrog_cat_mcal_detect_reg_df_",
        metacal_cols=other_metacal_cols + ['unsheared/flux_{}'.format(i) for i in 'irz'] + ['unsheared/flux_err_{}'.format(i) for i in 'irz'],
        protocol=None,
        detection_cols=[
            'bal_id',
            'true_id',
            'detected',
            'true_ra',
            'true_dec',
            'match_flag_1.5_asec',
            'flags_foreground',
            'flags_badregions',
            'flags_footprint'
        ],
        deep_field_cols=[
            "ID",
            "RA",
            "DEC",
            "BDF_T",
            "BDF_G_0",
            "BDF_G_1",
            "BDF_FLUX_DERED_CALIB_U",
            "BDF_FLUX_DERED_CALIB_G",
            "BDF_FLUX_DERED_CALIB_R",
            "BDF_FLUX_DERED_CALIB_I",
            "BDF_FLUX_DERED_CALIB_Z",
            "BDF_FLUX_DERED_CALIB_J",
            "BDF_FLUX_DERED_CALIB_H",
            "BDF_FLUX_DERED_CALIB_K",
            "BDF_FLUX_ERR_DERED_CALIB_U",
            "BDF_FLUX_ERR_DERED_CALIB_G",
            "BDF_FLUX_ERR_DERED_CALIB_R",
            "BDF_FLUX_ERR_DERED_CALIB_I",
            "BDF_FLUX_ERR_DERED_CALIB_Z",
            "BDF_FLUX_ERR_DERED_CALIB_J",
            "BDF_FLUX_ERR_DERED_CALIB_H",
            "BDF_FLUX_ERR_DERED_CALIB_K"

        ],
        survey_cols=[
            f"HPIX_{NSIDE}",
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
        ],
        only_detected=False,
        nside=NSIDE,
        show_plot=False,
        save_plot=False,
        plot_healpix=False
    )
