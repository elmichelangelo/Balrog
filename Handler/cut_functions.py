from Handler.helper_functions import *
import pandas as pd
import numpy as np


def unsheared_object_cuts(data_frame, prob=False):
    """"""
    print("Apply unsheared object cuts")
    cuts = (data_frame["unsheared/extended_class_sof"] >= 0) & (data_frame["unsheared/flags_gold"] < 2)
    if prob:
        data_frame.loc[~cuts, 'is_in'] = 0
    else:
        data_frame = data_frame[cuts]
    print('Length of catalog after applying unsheared object cuts: {}'.format(len(data_frame)))
    return data_frame

def unsheared_object_cuts_gandalf(data_frame):
    """"""
    print("Apply unsheared object cuts")
    cuts = (data_frame["unsheared/extended_class_sof"] >= 2) & (data_frame["unsheared/flags_gold"] < 2)
    data_frame = data_frame[cuts]
    print('Length of catalog after applying unsheared object cuts: {}'.format(len(data_frame)))
    return data_frame


def unsheared_object_cuts_gatti(data_frame):
    """"""
    print("Apply unsheared object cuts")
    cuts = (data_frame["unsheared/flags_gold"] < 8)
    data_frame = data_frame[cuts]
    print('Length of catalog after applying unsheared object cuts: {}'.format(len(data_frame)))
    return data_frame


def flag_cuts(data_frame, prob=False):
    """"""
    print("Apply flag cuts")
    cuts = (data_frame["match_flag_1.5_asec"] < 2) & \
           (data_frame["flags_foreground"] == 0) & \
           (data_frame["flags_badregions"] < 2) & \
           (data_frame["flags_footprint"] == 1)
    if prob:
        data_frame.loc[~cuts, 'is_in'] = 0
    else:
        data_frame = data_frame[cuts]
    print('Length of catalog after applying flag cuts: {}'.format(len(data_frame)))
    return data_frame


def flag_cuts_gandalf(data_frame):
    """"""
    print("Apply flag cuts")
    cuts = (data_frame["match_flag_1.5_asec"] < 2) & \
           (data_frame["flags_foreground"] == 0) & \
           (data_frame["flags_badregions"] < 2) & \
           (data_frame["flags_footprint"] == 1)

    data_frame = data_frame[cuts]
    print('Length of catalog after applying flag cuts: {}'.format(len(data_frame)))
    return data_frame


def flag_cuts_gatti(data_frame):
    """"""
    print("Apply flag cuts")
    cuts = (data_frame["flags_foreground"] == 0) & \
           (data_frame["flags_badregions"] < 2)
    data_frame = data_frame[cuts]
    print('Length of catalog after applying flag cuts: {}'.format(len(data_frame)))
    return data_frame


def unsheared_mag_cut(data_frame):
    """"""
    print("Apply unsheared mag cuts")
    cuts = (
            (18 < data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_i"] < 23.5) &
            (15 < data_frame["unsheared/mag_r"]) &
            (data_frame["unsheared/mag_r"] < 26) &
            (15 < data_frame["unsheared/mag_z"]) &
            (data_frame["unsheared/mag_z"] < 26) &
            (-1.5 < data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"] < 4) &
            (-4 < data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"] < 1.5)
    )
    data_frame = data_frame[cuts]
    print('Length of catalog after applying unsheared mag cuts: {}'.format(len(data_frame)))
    return data_frame


def unsheared_mag_cut_gandalf(data_frame):
    """"""
    print("Apply unsheared mag cuts")
    cuts = (
            (16 < data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_i"] < 30) &
            (13 < data_frame["unsheared/mag_r"]) &
            (data_frame["unsheared/mag_r"] < 32) &
            (13 < data_frame["unsheared/mag_z"]) &
            (data_frame["unsheared/mag_z"] < 32) &
            (-3.5 < data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"] < 6) &
            (-3.5 < data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"] < 6)
    )
    data_frame = data_frame[cuts]
    print('Length of catalog after applying unsheared mag cuts: {}'.format(len(data_frame)))
    return data_frame


def unsheared_mag_cut_gatti(data_frame):
    """"""
    print("Apply unsheared mag cuts")
    cuts = (
            (18 < data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_i"] < 23.5) &
            (15 < data_frame["unsheared/mag_r"]) &
            (data_frame["unsheared/mag_r"] < 26) &
            (15 < data_frame["unsheared/mag_z"]) &
            (data_frame["unsheared/mag_z"] < 26) &
            (-1.5 < data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"] < 4) &
            (-1.5 < data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"] < 4)
    )
    data_frame = data_frame[cuts]
    print('Length of catalog after applying unsheared mag cuts: {}'.format(len(data_frame)))
    return data_frame


def unsheared_shear_cuts(data_frame, prob=False):
    """"""
    print("Apply unsheared shear cuts")
    initial_cuts = (
            (10 < data_frame["unsheared/snr"]) &
            (data_frame["unsheared/snr"] < 1000) &
            (0.5 < data_frame["unsheared/size_ratio"]) &
            (data_frame["unsheared/T"] < 10)
    )

    additional_condition = ~((2 < data_frame["unsheared/T"]) & (data_frame["unsheared/snr"] < 30))

    combined_cuts = initial_cuts & additional_condition

    if prob:
        data_frame.loc[~combined_cuts, 'is_in'] = 0
    else:
        data_frame = data_frame[combined_cuts]
    print('Length of catalog after applying unsheared shear cuts: {}'.format(len(data_frame)))
    return data_frame

def unsheared_shear_cuts_gandalf(data_frame):
    """"""
    print("Apply unsheared shear cuts")
    initial_cuts = (
            (1 < data_frame["unsheared/snr"]) &
            (data_frame["unsheared/snr"] < 2000) &
            (0.1 < data_frame["unsheared/size_ratio"]) &
            (data_frame["unsheared/T"] < 20)
    )

    additional_condition = ~((2 < data_frame["unsheared/T"]) & (data_frame["unsheared/snr"] < 30))

    combined_cuts = initial_cuts & additional_condition

    data_frame = data_frame[combined_cuts]
    print('Length of catalog after applying unsheared shear cuts: {}'.format(len(data_frame)))
    return data_frame


def unsheared_shear_cuts_gatti(data_frame):
    """"""
    print("Apply unsheared shear cuts")
    initial_cuts = (
            (10 < data_frame["unsheared/snr"]) &
            (data_frame["unsheared/snr"] < 1000) &
            (0.5 < data_frame["unsheared/size_ratio"]) &
            (data_frame["unsheared/T"] < 10)
    )

    additional_condition = ~((2 < data_frame["unsheared/T"]) & (data_frame["unsheared/snr"] < 30))

    combined_cuts = initial_cuts & additional_condition

    data_frame = data_frame[combined_cuts]
    print('Length of catalog after applying unsheared shear cuts: {}'.format(len(data_frame)))
    return data_frame


def mask_cut(data_frame, master, prob=False):
    """"""
    import healpy as hp
    import h5py
    print("define mask")
    f = h5py.File(master)
    # theta = (np.pi / 180.) * (90. - data_frame['unsheared/dec'].to_numpy())
    # phi = (np.pi / 180.) * data_frame['unsheared/ra'].to_numpy()
    # gpix = hp.ang2pix(16384, theta, phi, nest=True)
    # mask_cut = np.in1d(gpix // (hp.nside2npix(16384) // hp.nside2npix(4096)), f['index/mask/hpix'][:],
    #                    assume_unique=False)
    hpix = data_frame['HPIX_4096'].to_numpy()  # Angenommen, 'HPIX_4096' ist der Spaltenname für Ihre HPIX 4096 Daten
    mask_cut = np.in1d(hpix, f['index/mask/hpix'][:], assume_unique=False)
    if prob:
        data_frame.loc[~mask_cut, 'is_in'] = 0
    else:
        data_frame = data_frame[mask_cut]
    npass = np.sum(mask_cut)
    print('pass: ', npass)
    print('fail: ', len(mask_cut) - npass)
    return data_frame


def binary_cut(data_frame, prob=False):
    """"""
    highe_cut = np.greater(np.sqrt(np.power(data_frame['unsheared/e_1'], 2.) + np.power(data_frame['unsheared/e_2'], 2)), 0.8)
    c = 22.5
    m = 3.5
    magT_cut = np.log10(data_frame['unsheared/T']) < (c - flux2mag(data_frame['unsheared/flux_r'])) / m
    binaries = highe_cut * magT_cut

    print("perform binaries cut")
    if prob:
        data_frame.loc[binaries, 'is_in'] = 0
    else:
        data_frame = data_frame[~binaries]

    print('len w/ binaries', len(data_frame))
    return data_frame

def binary_cut_gandalf(data_frame):
    """"""
    highe_cut = np.greater(np.sqrt(np.power(data_frame['unsheared/e_1'], 2.) + np.power(data_frame['unsheared/e_2'], 2)), 0.8)
    c = 22.5
    m = 2.5
    magT_cut = np.log10(data_frame['unsheared/T']) < (c - data_frame['unsheared/mag_r']) / m
    binaries = highe_cut * magT_cut

    print("perform binaries cut")
    data_frame = data_frame[~binaries]

    print('len w/o binaries', len(data_frame))
    return data_frame

def binary_cut_gatti(data_frame):
    """"""
    highe_cut = np.greater(np.sqrt(np.power(data_frame['unsheared/e_1'], 2.) + np.power(data_frame['unsheared/e_2'], 2)), 0.8)
    c = 22.5
    m = 2.5
    magT_cut = np.log10(data_frame['unsheared/T']) < (c - flux2mag(data_frame['unsheared/flux_r'])) / m
    binaries = highe_cut * magT_cut

    print("perform binaries cut")
    data_frame = data_frame[~binaries]

    print('len w/ binaries', len(data_frame))
    return data_frame

def mask_cut_healpy(data_frame, master):
    """"""
    import healpy as hp
    import h5py
    print("define mask")
    dec = data_frame['unsheared/dec'].to_numpy()
    invalid_dec_mask = ~np.isfinite(dec) | (dec < -90) | (dec > 90)
    if invalid_dec_mask.any():
        print("Invalid dec values detected:")
        print(dec[invalid_dec_mask])

    f = h5py.File(master)
    theta = (np.pi / 180.) * (90. - data_frame['unsheared/dec'].to_numpy())
    phi = (np.pi / 180.) * data_frame['unsheared/ra'].to_numpy()
    gpix = hp.ang2pix(16384, theta, phi, nest=True)
    mask_cut = np.in1d(gpix // (hp.nside2npix(16384) // hp.nside2npix(4096)), f['index/mask/hpix'][:],
                       assume_unique=False)
    data_frame = data_frame[mask_cut]
    npass = np.sum(mask_cut)
    print('pass: ', npass)
    print('fail: ', len(mask_cut) - npass)
    return data_frame


def mask_cut_healpy_gandalf(data_frame, master):
    """"""
    import healpy as hp
    import h5py
    print("define mask")
    dec = data_frame['unsheared/dec'].to_numpy()
    invalid_dec_mask = ~np.isfinite(dec) | (dec < -90) | (dec > 90)
    if invalid_dec_mask.any():
        print("Invalid dec values detected:")
        print(dec[invalid_dec_mask])

    f = h5py.File(master)
    theta = (np.pi / 180.) * (90. - data_frame['unsheared/dec'].to_numpy())
    phi = (np.pi / 180.) * data_frame['unsheared/ra'].to_numpy()
    gpix = hp.ang2pix(16384, theta, phi, nest=True)
    mask_cut = np.in1d(gpix // (hp.nside2npix(16384) // hp.nside2npix(4096)), f['index/mask/hpix'][:],
                       assume_unique=False)
    data_frame = data_frame[mask_cut]
    npass = np.sum(mask_cut)
    print('pass: ', npass)
    print('fail: ', len(mask_cut) - npass)
    return data_frame


def mask_cut_healpy_gatti(data_frame, master):
    """"""
    import healpy as hp
    import h5py
    print("define mask")
    dec = data_frame['unsheared/dec'].to_numpy()
    invalid_dec_mask = ~np.isfinite(dec) | (dec < -90) | (dec > 90)
    if invalid_dec_mask.any():
        print("Invalid dec values detected:")
        print(dec[invalid_dec_mask])

    f = h5py.File(master)
    theta = (np.pi / 180.) * (90. - data_frame['unsheared/dec'].to_numpy())
    phi = (np.pi / 180.) * data_frame['unsheared/ra'].to_numpy()
    gpix = hp.ang2pix(16384, theta, phi, nest=True)
    mask_cut = np.in1d(gpix // (hp.nside2npix(16384) // hp.nside2npix(4096)), f['index/mask/hpix'][:],
                       assume_unique=False)
    data_frame = data_frame[mask_cut]
    npass = np.sum(mask_cut)
    print('pass: ', npass)
    print('fail: ', len(mask_cut) - npass)
    return data_frame


def apply_cuts(cfg, data_frame, flag_cat="", merge_with_flag=False):
    """"""
    if merge_with_flag is True:
        df_merged = pd.merge(data_frame, flag_cat, on='bal_id', how='left')
    else:
        df_merged = data_frame
    df_merged = unsheared_object_cuts(data_frame=df_merged)
    df_merged = flag_cuts(data_frame=df_merged)
    df_merged = unsheared_shear_cuts(data_frame=df_merged)
    df_merged = binary_cut(data_frame=df_merged)
    df_merged = mask_cut_healpy(
        data_frame=df_merged,
        master=f"{cfg['PATH_DATA']}/{cfg['FILENAME_MASTER_CAT']}"
    )
    df_merged = unsheared_mag_cut(data_frame=df_merged)
    return df_merged


def apply_galaxy_cuts(data_frame, flag_cat="", merge_with_flag=False):
    """"""
    if merge_with_flag is True:
        df_merged = pd.merge(data_frame, flag_cat, on='bal_id', how='left')
    else:
        df_merged = data_frame
    df_merged = flag_cuts_gandalf(data_frame=df_merged)
    return df_merged


def apply_photometric_cuts(cfg, data_frame, flag_cat="", merge_with_flag=False):
    """"""
    if merge_with_flag is True:
        df_merged = pd.merge(data_frame, flag_cat, on='bal_id', how='left')
    else:
        df_merged = data_frame
    df_merged = unsheared_object_cuts_gandalf(data_frame=df_merged)
    df_merged = unsheared_shear_cuts_gandalf(data_frame=df_merged)
    df_merged = binary_cut_gandalf(data_frame=df_merged)
    df_merged = mask_cut_healpy_gandalf(
        data_frame=df_merged,
        master=f"{cfg['PATH_DATA']}/{cfg['FILENAME_MASTER_CAT']}"
    )
    df_merged = unsheared_mag_cut_gandalf(data_frame=df_merged)
    return df_merged


def apply_gatti_cuts(cfg, data_frame, flag_cat="", merge_with_flag=False):
    """"""
    if merge_with_flag is True:
        df_merged = pd.merge(data_frame, flag_cat, on='bal_id', how='left')
    else:
        df_merged = data_frame
    df_merged = unsheared_object_cuts_gatti(data_frame=df_merged)
    df_merged = flag_cuts_gatti(data_frame=df_merged)
    df_merged = unsheared_shear_cuts_gatti(data_frame=df_merged)
    df_merged = binary_cut_gatti(data_frame=df_merged)
    df_merged = mask_cut_healpy_gatti(
        data_frame=df_merged,
        master=f"{cfg['PATH_DATA']}/{cfg['FILENAME_MASTER_CAT']}"
    )
    df_merged = unsheared_mag_cut_gatti(data_frame=df_merged)
    return df_merged
