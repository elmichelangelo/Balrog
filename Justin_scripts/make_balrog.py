import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
import fitsio
import healpy as hp
from argparse import ArgumentParser
from astropy.table import Table, vstack

parser = ArgumentParser()

parser.add_argument(
    '--mcal_file',
    type=str,
    default='../Data/balrog_mcal_stack-y3v02-0-riz-noNB-mcal_y3-merged_v1.2.h5',
    help='Filename of Balrog mcal catalog to use for sompz run'
    )
parser.add_argument(
    '--detection_file',
    type=str,
    default='../Data/balrog_detection_catalog_sof_y3-merged_v1.0.fits',
    help='Filename of Balrog detection catalog to use for sompz run'
    )
parser.add_argument(
    '--outfile',
    type=str,
    default='y3_balrog2_v1.2_merged_select2_bstarcut_matchflag1.5asec_snr_SR_corrected_uppersizecuts.h5',
    help='Filename of output file'
    )
parser.add_argument(
    '--outdir',
    type=str,
    default='../Output/',
    help='Directory location for output file'
    )
parser.add_argument(
    '--extra_mcal',
    type=str,
    default=None,
    # '/project/projectdirs/des/severett/Balrog/run2a/stacked_catalogs/1.4/mcal/balrog_mcal_stack-y3v02-0-riz-noNB-mcal_run2a_v1.4.h5',
    help='Pass a supplemental Balrog mcal catalog to stack with the main one' +
          '(e.g. run2 + run2a)'
    )
parser.add_argument(
    '--extra_detection',
    type=str,
    default=None,
    #'/project/projectdirs/des/severett/Balrog/run2a/stacked_catalogs/1.4/sof/balrog_detection_catalog_sof_run2a_v1.4.fits',
    help='Pass a supplemental Balrog detection catalog to stack with the main one' +
          '(e.g. run2 + run2a)'
    )
parser.add_argument(
    '--version',
    type=float,
    default=1.4,
    help='Version of Balrog catalog'
    )
parser.add_argument(
    '--mastercat',
    type=str,
    default='../Data/Y3_mastercat_03_31_20.h5',   # Y3_mastercat_03_16_20_highsnr.h5
    help='mastercat to load')
parser.add_argument(
    '--skip_plots',
    action='store_true',
    default=False,
    help='Set to skip making plots, only build catalog')
parser.add_argument(
    '--vb',
    action='store_true',
    default=True,
    help='Set to print out more information'
    )



plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.figsize'] = (16.,8.)
dpi = 150

plt.rcParams.update({
'lines.linewidth':1.0,
'lines.linestyle':'-',
'lines.color':'black',
'font.family':'serif',
'font.weight':'bold', #normal
'font.size':16.0, #10.0
'text.color':'black',
'text.usetex':False,
'axes.edgecolor':'black',
'axes.linewidth':1.0,
'axes.grid':False,
'axes.titlesize':'x-large',
'axes.labelsize':'x-large',
'axes.labelweight':'bold', #normal
'axes.labelcolor':'black',
'axes.formatter.limits':[-4,4],
'xtick.major.size':7,
'xtick.minor.size':4,
'xtick.major.pad':8,
'xtick.minor.pad':8,
'xtick.labelsize':'x-large',
'xtick.minor.width':1.0,
'xtick.major.width':1.0,
'ytick.major.size':7,
'ytick.minor.size':4,
'ytick.major.pad':8,
'ytick.minor.pad':8,
'ytick.labelsize':'x-large',
'ytick.minor.width':1.0,
'ytick.major.width':1.0,
'legend.numpoints':1,
'legend.fontsize':'x-large',
'legend.shadow':False,
'legend.frameon':False})

def flux2mag(flux, zero_pt=30):
    return zero_pt - 2.5 * np.log10(flux)

def compute_injection_counts(det_catalog):
    '''
    Expects det_catalog to be a pandas DF
    '''
    # `true_id` is the DF id
    unique, ucounts = np.unique(det_catalog['true_id'], return_counts=True)
    
    freq = pd.DataFrame()
    freq['true_id'] = unique
    freq['injection_counts'] = ucounts
    
    return det_catalog.merge(freq, on='true_id', how='left')

def main():
    args = parser.parse_args()
    datafile = args.mcal_file
    detectionfile = args.detection_file
    outfile = args.outfile
    outdir = args.outdir
    extra_mcal_file = args.extra_mcal
    extra_detection_file = args.extra_detection
    version = args.version
    mastercat = args.mastercat
    skip_plots = args.skip_plots
    vb = args.vb

    # If you are passing a supplemental catalog, need to pass both!
    if extra_mcal_file is not None:
        assert extra_detection_file is not None
    if extra_detection_file is not None:
        assert extra_mcal_file is not None

    if vb is True:
        print('begin make_balrog.py')

    # Match flag colnames depend on version
    # (There are 7 different match_flags after v1.4,
    #  but we use 1.5" for cosmology)
    if version < 1.4:
        match_flag_colname = 'match_flag'
    else:
        match_flag_colname = 'match_flag_1.5_asec'

    detection_cols = ['bal_id',
                      'true_id',
                      'flags_footprint',
                      'flags_foreground',
                      'flags_badregions',
                      'meas_FLAGS_GOLD_SOF_ONLY',
                      match_flag_colname]

    data = h5py.File(datafile, 'r')
    detection = Table(fitsio.read(detectionfile, columns=detection_cols).byteswap().newbyteorder())
    
    if vb is True:
        print('Detection cols loaded')

    # If you want to use a supplemental catalog, add it here
    if (extra_mcal_file is not None) and (extra_detection_file is not None):
        extra_data = h5py.File(extra_mcal_file, 'r')
        extra_detection = Table(fitsio.read(extra_detection_file, columns=detection_cols))

        # Combine with main catalogs...
        detection = vstack([detection, extra_detection]).to_pandas()

        if vb is True:
            print('Extra detection cols loaded')
        # data = ... #NOTE: We would do something like the above, but this is
        # quite difficult with h5 files and we don't want to actually produce
        # another file. Will instead stack the data frames below

    fluxes = ['sheared_{}/flux_{}'.format(i,j) for i in ['1m','1p','2m','2p'] for j in 'irz']
    fluxes = fluxes  + ['unsheared/flux_{}'.format(i) for i in 'irz']
    fluxerrs = [_.replace('flux','flux_err') for _ in fluxes]
    other_metacal_cols = ['unsheared/ra',
                          'unsheared/dec',
                          'unsheared/coadd_object_id',
                          'unsheared/snr',
                          'unsheared/size_ratio',
                          'unsheared/flags',
                          'unsheared/bal_id',
                          'unsheared/T',
                          'unsheared/T_err',
                          'unsheared/e_1',
                          'unsheared/e_2',
                          'unsheared/R11',
                          'unsheared/R22',
                          'unsheared/weight']

    df = pd.DataFrame()

    for i, col in enumerate(fluxes + fluxerrs + other_metacal_cols):
        if vb is True:
            print(i, col)
        df[col] = np.array(data['catalog/' + col]).byteswap().newbyteorder()
    df = df.rename(columns={'unsheared/bal_id' : 'bal_id'} )

    # See NOTE above; much easier to combine mcal h5 data at the data frame level
    if extra_mcal_file is not None:
        extra_df = pd.DataFrame()
        for i, col in enumerate(fluxes + fluxerrs + other_metacal_cols):
            if vb is True:
                print('Extra mcal cat: {}'.format((i, col)))
            extra_df[col] = np.array(extra_data['catalog/' + col]).byteswap().newbyteorder()
        extra_df = extra_df.rename(columns={'unsheared/bal_id' : 'bal_id'} )

        # combine with main catalog
        if vb is True:
            print('Combining with main catalog...')
        df = pd.concat([df, extra_df])

    if vb is True:
        print('Length of mcal catalog: {}'.format(len(df)))

    df_detect = pd.DataFrame()
    for i, col in enumerate(detection_cols):
        if vb is True:
            print(i, col)
        df_detect[col] = detection[col]
    if vb is True:
        print('Length of detection catalog: {}'.format(len(df_detect)))
        
    # Keep track of how many times a given DF object was injected (regardless of detection)
    if vb is True:
        print('Computing injection counts..')
    df_detect = compute_injection_counts(df_detect)

    if vb is True:
        print('Merging catalogs...')
    df_merged = pd.merge(df, df_detect, on='bal_id', how='inner')
    
    if vb is True:
        print('Length of merged catalog: {}'.format(len(df_merged)))

    if skip_plots is False:
        fig, ax = plt.subplots()
        plt.hist(flux2mag(df_merged['unsheared/flux_i']),
                 range=(5,40),bins=50,histtype='step',
                 label='balrog_metacal imag %s' %len(df_merged['unsheared/flux_i']))
        plt.xlabel('wide mcal mag')
        plt.yscale('log')
        plt.legend()
        plt.savefig(outdir + 'balrog_hist_before_wl.png', dpi=100)

    # TODO
    # if vb is True:
    #     print('masking')
    # f = h5py.File(mastercat)
    # theta = (np.pi / 180.) * (90. - df_merged['unsheared/dec'])
    # phi = (np.pi / 180.) * df_merged['unsheared/ra']
    # gpix = hp.ang2pix(16384, theta, phi, nest=True)
    # mask_cut = np.in1d(gpix // (hp.nside2npix(16384) // hp.nside2npix(4096)),
    #                    f['index/mask/hpix'][:],
    #                    assume_unique=False)
    # npass = np.sum(mask_cut)

    if vb is True:
        # print('pass: ', npass)
        # print('fail: ', len(mask_cut) - npass)
        print('Starting shape cuts')

    unsheared_snr_cut = (df_merged['unsheared/snr'] > 10) & (df_merged['unsheared/snr'] < 1000)
    unsheared_size_ratio_cut = df_merged['unsheared/size_ratio'] > 0.5
    unsheared_flags_cut = df_merged['unsheared/flags'] == 0
    #unsheared_new_cut = (df_merged['unsheared/snr'] > 30) & (df_merged['unsheared/T'] < 2)
    unsheared_size_cut = (df_merged['unsheared/T'] < 10) 
    unsheared_shape_cuts = unsheared_snr_cut & unsheared_size_ratio_cut & unsheared_flags_cut  & unsheared_size_cut #& unsheared_new_cut
    

    if vb is True:
        l = len(unsheared_shape_cuts[unsheared_shape_cuts == True])
        print('{} objects pass shape cuts'.format(l))
        print('Starting flags cuts')
    flags_foreground_cut = df_merged['flags_foreground'] == 0
    flags_badregions_cut = df_merged['flags_badregions'] < 2
    flags_gold_cut = df_merged['meas_FLAGS_GOLD_SOF_ONLY'] < 2
    flags_footprint_cut = df_merged['flags_footprint'] == 1
    gold_flags_cut = flags_foreground_cut & flags_badregions_cut & flags_gold_cut & flags_footprint_cut
    if vb is True:
        l = len(gold_flags_cut[gold_flags_cut == True])
        print('{} objects pass gold cuts'.format(l))
        print('len w/ flags and shape cuts but not mask cut',
              len(df_merged[gold_flags_cut & unsheared_shape_cuts]))
    # TODO
    df_merged = df_merged[gold_flags_cut & unsheared_shape_cuts ]  # & mask_cut

    if vb is True:
        print('len w/ flags and shape cuts and     mask cut', len(df_merged))
        print('binary star cut')

    highe_cut = np.greater(np.sqrt(np.power(df_merged['unsheared/e_1'],2.) +
                                   np.power(df_merged['unsheared/e_2'],2)), 0.8)
    c = 22.5
    m = 3.5
    magT_cut = np.log10(df_merged['unsheared/T']) < (c - flux2mag(df_merged['unsheared/flux_r']))/m
    binaries = highe_cut*magT_cut
    df_merged = df_merged[~binaries]

    if vb is True:
        print('len w/ flags and shape cuts and mask cut and binaries', len(df_merged))
        print('additional cuts')

    unsheared_imag_cut = (flux2mag(df_merged['unsheared/flux_i'])>18) & \
                         (flux2mag(df_merged['unsheared/flux_i'])<23.5)
    unsheared_rmag_cut = (flux2mag(df_merged['unsheared/flux_r'])>15) & \
                         (flux2mag(df_merged['unsheared/flux_r'])<26)
    unsheared_zmag_cut = (flux2mag(df_merged['unsheared/flux_z'])>15) & \
                         (flux2mag(df_merged['unsheared/flux_z'])<26)
    unsheared_zi_cut = ((flux2mag(df_merged['unsheared/flux_z'])-flux2mag(df_merged['unsheared/flux_i'])) <1.5) & \
                       ((flux2mag(df_merged['unsheared/flux_z'])-flux2mag(df_merged['unsheared/flux_i'])) >-4)
    unsheared_ri_cut = ((flux2mag(df_merged['unsheared/flux_r'])-flux2mag(df_merged['unsheared/flux_i'])) <4) & \
                       ((flux2mag(df_merged['unsheared/flux_r'])-flux2mag(df_merged['unsheared/flux_i'])) >-1.5)
    df_merged = df_merged[unsheared_imag_cut &
                          unsheared_rmag_cut &
                          unsheared_zmag_cut &
                          unsheared_zi_cut &
                          unsheared_ri_cut]
    
    unsheared_new_cut = (df_merged['unsheared/snr'] < 30) & (df_merged['unsheared/T'] > 2)
    df_merged = df_merged[~unsheared_new_cut]

    if vb is True:
        print('len w/ flags and shape cuts and mask cut and binaries and wl selection 2',
              len(df_merged))

    df_merged = df_merged[df_merged[match_flag_colname] < 2]
    if vb is True:
        print('len w/ flags and shape cuts and mask cut and binaries and wl selection 2 and {}'.format(
            match_flag_colname), len(df_merged))
        print('len w/ selection 2 cut', len(df_merged))

    of = os.path.join(outdir, outfile)
    if vb is True:
        print('write ' + of)
    df_merged.to_hdf(of, key='df', mode='w')
    os.system('chmod a+r {}'.format(of))

    if skip_plots is False:
        fig, ax = plt.subplots()
        plt.hist(flux2mag(df_merged['unsheared/flux_i']),
                 range=(5,40),bins=50,
                 histtype='step',
                 label='balrog_metacal imag %s' %len(df_merged['unsheared/flux_i']))
        plt.xlabel('wide mcal mag')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(outdir, 'balrog_hist_after_wl.png'), dpi=100)
        if vb is True:
            print('save', os.path.join(outdir, 'balrog_hist_after_wl.png'))

    return


if __name__ == '__main__':
    main()
