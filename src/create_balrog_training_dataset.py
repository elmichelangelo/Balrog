import os
import sys
import pickle
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import h5py
import fitsio
import numpy as np
import pandas as pd
import healpy as hp
from Handler import *
# from Handler.cut_functions import *


def read_catalogs(path_metacal, path_detection, path_deep_field, path_bdf_info, path_sompz_cosmos, path_survey,
                  metacal_cols, detection_cols, deep_field_cols, survey_cols, nside, lst_of_loggers, plot_healpix=False,
                  show_plot=False, save_plot=False):
    """"""
    df_mcal = None
    df_detect = None
    df_deep_field = None
    df_cosmos = None
    df_survey = None
    matchlim = 0.75

    for log in lst_of_loggers:
        log.info("Start read catalogs")

    if path_metacal is not None:
        # Read h5py file
        for log in lst_of_loggers:
            log.info(f"Read {path_metacal}")
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
        for log in lst_of_loggers:
            log.info(f"Rename Col's: unsheared/bal_id': 'bal_id and unsheared/coadd_object_id': 'COADD_OBJECT_ID")
        df_mcal = df_mcal.rename(columns={'unsheared/bal_id': 'bal_id'})
        df_mcal = df_mcal.rename(columns={'unsheared/coadd_object_id': 'COADD_OBJECT_ID'})

        # Verbose
        for log in lst_of_loggers:
            log.info('Length of mcal catalog: {}'.format(len(df_mcal)))
        print('Length of mcal catalog: {}'.format(len(df_mcal)))
        for i, col in enumerate(df_mcal.columns):
            print(i, col)
        for log in lst_of_loggers:
            log.info(df_mcal.isnull().sum())
            log.info(df_mcal.isnull().sum().sum())
        print(df_mcal.isnull().sum())
        print(df_mcal.isnull().sum().sum())

    if path_detection is not None:
        # Read fits file
        for log in lst_of_loggers:
            log.info(f"Read {path_detection}")
        detection_data = Table(fitsio.read(path_detection).byteswap().newbyteorder())

        # Add columns to DataFrame
        df_detect = pd.DataFrame()
        for i, col in enumerate(detection_cols):
            df_detect[col] = detection_data[col]

        for log in lst_of_loggers:
            log.info(f"Rename col: true_id': 'ID")
        # Rename some columns
        df_detect = df_detect.rename(columns={'true_id': 'ID'})

        # Count injections
        df_detect = compute_injection_counts(df_detect, lst_of_loggers)

        # Get HPIX Index
        for log in lst_of_loggers:
            log.info(f"Get HPIX Index")
        df_detect[f"HPIX_{nside}"] = DeclRaToIndex(np.array(df_detect["true_dec"]), np.array(df_detect["true_ra"]), nside)

        # Verbose
        print('Length of detection catalog: {}'.format(len(df_detect)))
        for log in lst_of_loggers:
            log.info('Length of detection catalog: {}'.format(len(df_detect)))
        for i, col in enumerate(df_detect.keys()):
            print(i, col)
        print(df_detect.isnull().sum())
        print(df_detect.isnull().sum().sum())
        for log in lst_of_loggers:
            log.info(df_detect.isnull().sum())
            log.info(df_detect.isnull().sum().sum())

        # Plot Data
        if plot_healpix is True:
            for log in lst_of_loggers:
                log.info(f"plot_healpix")
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
        for log in lst_of_loggers:
            log.info(f"Read {path_deep_field}")
        # Read pickle deep field file
        infile = open(path_deep_field, 'rb')
        # load pickle as pandas dataframe
        deep_field = pd.DataFrame(pickle.load(infile, encoding='latin1'))
        # close file
        infile.close()

        # Read fits bdf info file
        for log in lst_of_loggers:
            log.info(f"Read fits bdf info file")
        bdf_info = Table(fitsio.read(path_bdf_info).byteswap().newbyteorder()).to_pandas()

        for log in lst_of_loggers:
            log.info(f"merge deep field and bdf")
        df_deep_field = bdf_info.merge(deep_field, left_on='ID', right_on='ID', how='inner')

        # Code from Justins script sompz_getready_data_0.py ############################################################
        df_deep_field.rename(columns={
            'BDF_T_y': 'BDF_T',
            'BDF_T_ERR_y': 'BDF_T_ERR',
            'BDF_G_0_y': 'BDF_G_0',
            'BDF_G_1_y': 'BDF_G_1',
        }, inplace=True)

        del df_deep_field['TILENAME']  #h5 doesn't like the TILENAME columns, so removing now
        del df_deep_field['BDF_T_x']
        del df_deep_field['BDF_T_ERR_x']
        del df_deep_field['BDF_G_0_x']
        del df_deep_field['BDF_G_1_x']

        for log in lst_of_loggers:
            log.info(f"Copy COSMOS Data")
        df_cosmos = df_deep_field[df_deep_field['FIELD'] == 'COSMOS'].copy()

        # READ IN REDSHIFT CATALOGUE
        for log in lst_of_loggers:
            log.info(f"Read {path_sompz_cosmos}")
        df_cosmos_z = pd.read_hdf(path_sompz_cosmos)

        print(len(df_deep_field))
        print(len(df_cosmos), 'DES DEEP')
        print(len(df_cosmos_z), 'Laigle')

        for log in lst_of_loggers:
            log.info(f"Len deep field {len(df_deep_field)}")
            log.info(f"len cosmos DES {len(df_cosmos)}")
            log.info(f"len Laigle {len(df_cosmos_z)}")

        for log in lst_of_loggers:
            log.info(f"Calc SkyCoord")
        c = SkyCoord(ra=df_cosmos_z['ALPHA_J2000'].values * u.degree, dec=df_cosmos_z['DELTA_J2000'].values * u.degree)
        catalog = SkyCoord(ra=df_cosmos['RA'].values * u.degree, dec=df_cosmos['DEC'].values * u.degree)

        for log in lst_of_loggers:
            log.info(f"SkyCoord: {c}, {catalog}")
            log.info(f"match_to_catalog_sky")

        idx, d2d, d3d = catalog.match_to_catalog_sky(c)

        for log in lst_of_loggers:
            log.info(f"match_to_catalog_sky: idx: {idx}, d2d:{d2d}, d3d:{d3d}")
        is_match = d2d < matchlim * u.arcsec
        for log in lst_of_loggers:
            log.info(f"is match: {is_match}, d2d={d2d} < matchlim={matchlim} * u.arcsec")

        zpdfcols = ["Z{:.2f}".format(s).replace(".", "_") for s in np.arange(0, 6.01, 0.01)]

        print('add Z info')
        for log in lst_of_loggers:
            log.info(f"add Z info")
        df_cosmos['Z'] = -1
        df_cosmos.loc[is_match, 'Z'] = df_cosmos_z.iloc[idx[is_match], df_cosmos_z.columns.get_loc('PHOTOZ')].values

        print('add pz info')
        for log in lst_of_loggers:
            log.info(f"add pz info")
        df_cosmos[zpdfcols] = pd.DataFrame(
            -1 * np.ones((len(df_cosmos),len(zpdfcols))), columns=zpdfcols, index=df_cosmos.index)

        zpdfcols_indices = [df_cosmos_z.columns.get_loc(_) for _ in zpdfcols]

        df_cosmos.loc[is_match, zpdfcols] = df_cosmos_z.iloc[idx[is_match], zpdfcols_indices].values

        print('add Laigle ID info')
        for log in lst_of_loggers:
            log.info(f"add Laigle ID info")
        df_cosmos.loc[is_match, 'LAIGLE_ID'] = df_cosmos_z.iloc[idx[is_match], df_cosmos_z.columns.get_loc('ID')].values
        ids, counts = np.unique(df_cosmos.loc[is_match, 'LAIGLE_ID'], return_counts=True)
        print('n duplicated Laigle', len(counts[counts > 1]))
        print("all cosmos deep: ", len(df_cosmos['BDF_MAG_DERED_CALIB_R']))
        print("matched cosmos deep: ", len(df_cosmos['BDF_MAG_DERED_CALIB_R'].loc[is_match]))
        print("unmatched cosmos deep: ", len(df_cosmos['BDF_MAG_DERED_CALIB_R'][df_cosmos['Z'] == -1]))
        print('We found       {:,}/{:,} of the COSMOS deep catalog in the Laigle cat'.format(
            np.sum(df_cosmos['Z'] != -1),len(df_cosmos)))
        print('We didn\'t find {:,}/{:,} \"\"\" '.format(np.sum(df_cosmos['Z'] == -1), len(df_cosmos)))
        for log in lst_of_loggers:
            log.info('n duplicated Laigle', len(counts[counts > 1]))
            log.info("all cosmos deep: ", len(df_cosmos['BDF_MAG_DERED_CALIB_R']))
            log.info("matched cosmos deep: ", len(df_cosmos['BDF_MAG_DERED_CALIB_R'].loc[is_match]))
            log.info("unmatched cosmos deep: ", len(df_cosmos['BDF_MAG_DERED_CALIB_R'][df_cosmos['Z'] == -1]))
            log.info('We found       {:,}/{:,} of the COSMOS deep catalog in the Laigle cat'.format(
                np.sum(df_cosmos['Z'] != -1),len(df_cosmos)))
            log.info('We didn\'t find {:,}/{:,} \"\"\" '.format(np.sum(df_cosmos['Z'] == -1), len(df_cosmos)))
        df_cosmos = df_cosmos[df_cosmos['Z'] != -1].copy()
        df_cosmos = df_cosmos[["ID", "Z", "FIELD"]]

        ################################################################################################################

        # Verbose
        print('Length of deep field catalog: {}'.format(len(df_deep_field)))
        print(df_deep_field.isnull().sum())
        print(df_deep_field.isnull().sum().sum())

        print('Length of cosmos catalog: {}'.format(len(df_cosmos)))
        print(df_cosmos.isnull().sum())
        print(df_cosmos.isnull().sum().sum())
        for log in lst_of_loggers:
            log.info('Length of deep field catalog: {}'.format(len(df_deep_field)))
            log.info(df_deep_field.isnull().sum())
            log.info(df_deep_field.isnull().sum().sum())
            log.info('Length of cosmos catalog: {}'.format(len(df_cosmos)))
            log.info(df_cosmos.isnull().sum())
            log.info(df_cosmos.isnull().sum().sum())

    if path_survey is not None:
        df_survey = pd.DataFrame()
        for idx, file in enumerate(os.listdir(path_survey)):
            if "fits" in file:
                # Read fits file
                for log in lst_of_loggers:
                    log.info(f"Read {path_survey}/{file}")
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
                for log in lst_of_loggers:
                    log.info(df_survey_tmp.shape)

        # Verbose
        print('Length of survey catalog: {}'.format(len(df_survey)))
        print(df_survey.isnull().sum())
        print(df_survey.isnull().sum().sum())
        print(df_survey.shape)
        for log in lst_of_loggers:
            log.info('Length of survey catalog: {}'.format(len(df_survey)))
            log.info(df_survey.isnull().sum())
            log.info(df_survey.isnull().sum().sum())
            log.info(df_survey.shape)

        # Plot Data
        if plot_healpix is True:
            for log in lst_of_loggers:
                log.info(f"Plot Healpix")
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
    return df_mcal, df_deep_field, df_cosmos, df_detect, df_survey


def compute_injection_counts(det_catalog, lst_of_loggers):
    # `true_id` is the DF id
    unique, ucounts = np.unique(det_catalog['ID'], return_counts=True)
    freq = pd.DataFrame()
    freq['ID'] = unique
    freq['injection_counts'] = ucounts
    for log in lst_of_loggers:
        log.info(f"Compute injection counts: {unique}, {ucounts}")
    return det_catalog.merge(freq, on='ID', how='left')


def merge_catalogs(lst_of_loggers, metacal=None, deep_field=None, detection=None, survey=None):
    """"""
    print('Merging catalogs...')
    for log in lst_of_loggers:
        log.info(f"Merging catalogs...")

    print('Merging metacal and detection on bal_id => mcal_detect')
    for log in lst_of_loggers:
        log.info(f"Merging metacal and detection on bal_id => mcal_detect")
    df_merged = pd.merge(detection, metacal, on='bal_id', how="left")
    print('Length of merged mcal_detect catalog: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())
    print(df_merged.isnull().sum().sum())
    for log in lst_of_loggers:
        log.info('Length of merged mcal_detect catalog: {}'.format(len(df_merged)))
        log.info(df_merged.isnull().sum())
        log.info(df_merged.isnull().sum().sum())

    print('Merging mcal_detect and deep field on ID => mcal_detect_df')
    for log in lst_of_loggers:
        log.info('Merging mcal_detect and deep field on ID => mcal_detect_df')
    df_merged = pd.merge(df_merged, deep_field, on='ID', how="left")
    print('Length of merged mcal_detect_df catalog: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())
    print(df_merged.isnull().sum().sum())
    for log in lst_of_loggers:
        log.info('Length of merged mcal_detect_df catalog: {}'.format(len(df_merged)))
        log.info(df_merged.isnull().sum())
        log.info(df_merged.isnull().sum().sum())

    print('Merging mcal_detect_df and survey on HPIX_4096 => mcal_detect_df_survey')
    for log in lst_of_loggers:
        log.info('Merging mcal_detect_df and survey on HPIX_4096 => mcal_detect_df_survey')
    df_merged = pd.merge(df_merged, survey, on='HPIX_4096', how="left")

    print('Length of merged mcal_detect_df_survey catalog with AIRMASS NANs: {}'.format(len(df_merged)))
    for log in lst_of_loggers:
        log.info('Length of merged mcal_detect_df_survey catalog with AIRMASS NANs: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())
    print(df_merged.isnull().sum().sum())
    for log in lst_of_loggers:
        log.info('Length of merged mcal_detect_df_survey catalog: {}'.format(len(df_merged)))
        log.info(df_merged.isnull().sum())
        log.info(df_merged.isnull().sum().sum())

    df_merged = df_merged[~df_merged['AIRMASS_WMEAN_R'].isnull()]
    print('Length of merged mcal_detect_df_survey catalog without AIRMASS NANs: {}'.format(len(df_merged)))
    for log in lst_of_loggers:
        log.info('Length of merged mcal_detect_df_survey catalog without AIRMASS NANs: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())
    print(df_merged.isnull().sum().sum())
    for log in lst_of_loggers:
        log.info('Length of merged mcal_detect_df_survey catalog: {}'.format(len(df_merged)))
        log.info(df_merged.isnull().sum())
        log.info(df_merged.isnull().sum().sum())
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


def write_data_2_file(df_generated_data, save_path, save_name, protocol, lst_of_loggers):
    """"""
    for log in lst_of_loggers:
        log.info(f"Save data...")
    if protocol == 2:
        for log in lst_of_loggers:
            log.info(f"Use protocol 2")
            log.info(f"Save as {save_path}{save_name}")
        with open(f"{save_path}{save_name}", "wb") as f:
            pickle.dump(df_generated_data.to_dict(), f, protocol=2)
    else:
        for log in lst_of_loggers:
            log.info(f"Use protocol 5")
            log.info(f"Save as {save_path}{save_name}")
        df_generated_data.to_pickle(f"{save_path}{save_name}")


def main(path_metacal, path_detection, path_deep_field, path_bdf_info, path_sompz_cosmos, path_survey, path_save,
         path_log, metacal_cols, detection_cols, deep_field_cols, survey_cols, only_detected, nside, protocol,
         show_plot, save_plot, plot_healpix):
    """"""

    # Initialize the logger
    start_window_logger = LoggerHandler(
        logger_dict= {"logger_name": "start window",
                      "info_logger": INFO_LOGGER,
                      "error_logger": ERROR_LOGGER,
                      "debug_logger": DEBUG_LOGGER,
                      "stream_logger": STREAM_LOGGER,
                      "stream_logging_level": LOGGING_LEVEL},
        log_folder_path=path_log
    )

    # Get the list of loggers
    lst_of_loggers = start_window_logger.lst_of_loggers
    # Write status to logger
    for log in lst_of_loggers:
        log.info("Start create balrog training dataset")

    # Read all catalogs
    df_mcal, df_deep_field, df_cosmos, df_detect, df_survey = read_catalogs(
        path_metacal=path_metacal,
        path_detection=path_detection,
        path_deep_field=path_deep_field,
        path_bdf_info=path_bdf_info,
        path_sompz_cosmos=path_sompz_cosmos,
        path_survey=path_survey,
        metacal_cols=metacal_cols,
        detection_cols=detection_cols,
        deep_field_cols=deep_field_cols,
        survey_cols=survey_cols,
        nside=nside,
        plot_healpix=plot_healpix,
        lst_of_loggers=lst_of_loggers,
        show_plot=show_plot,
        save_plot=save_plot
    )

    if df_mcal is not None and df_detect is not None and df_deep_field is not None and df_survey is not None:
        # Merge all catalogs
        df_merged = merge_catalogs(
            metacal=df_mcal,
            detection=df_detect,
            deep_field=df_deep_field,
            survey=df_survey,
            lst_of_loggers=lst_of_loggers,
        )

        # Save Data to File
        write_data_2_file(
            df_generated_data=df_merged,
            save_path=path_save,
            save_name=f"balrog_training_no_cuts_{len(df_merged)}.pkl",
            protocol=protocol,
            lst_of_loggers=lst_of_loggers
        )

        write_data_2_file(
            df_generated_data=df_cosmos,
            save_path=path_save,
            save_name=f"deep_cosmos_{len(df_cosmos)}.pkl",
            protocol=protocol,
            lst_of_loggers=lst_of_loggers
        )


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    path_data = "/Users/P.Gebhardt/Development/PhD/data"
    path_output = "/Users/P.Gebhardt/Development/PhD/output/Balrog"

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
        # path_deep_field=f"{path_data}/deep_ugriz.mof02_sn.jhk.ff04_c.jhk.ff02_052020_realerrors_May20calib.pkl",
        path_deep_field=f"{path_data}/deep_field_flux_mag_T_G_ID_tilename.pkl",
        path_bdf_info=f"{path_data}/DF_BDFT_BDFG.fits",
        path_sompz_cosmos=f"{path_data}/sompz_cosmos.h5",
        path_survey=f"{path_data}/sct2",
        path_save=f"{path_output}/Catalogs/",
        path_log=f"{path_output}/Logs/",
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
            "BDF_FLUX_DERED_CALIB_KS",
            "BDF_FLUX_ERR_DERED_CALIB_U",
            "BDF_FLUX_ERR_DERED_CALIB_G",
            "BDF_FLUX_ERR_DERED_CALIB_R",
            "BDF_FLUX_ERR_DERED_CALIB_I",
            "BDF_FLUX_ERR_DERED_CALIB_Z",
            "BDF_FLUX_ERR_DERED_CALIB_J",
            "BDF_FLUX_ERR_DERED_CALIB_H",
            "BDF_FLUX_ERR_DERED_CALIB_KS"

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
