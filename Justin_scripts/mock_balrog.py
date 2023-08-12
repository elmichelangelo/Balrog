import pdb
import os
import sys
import numpy as np
#import pickle as p
import cPickle as p
import pandas as pd
import h5py
import yaml
import fitsio
import time
import datetime
from astropy.coordinates import SkyCoord
import astropy.units as u

import matplotlib as mpl
mpl.use('pdf')
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
plt.style.use('/global/homes/j/jmyles/.matplotlib/stylelib/jmyles.mplstyle')

sys.path.insert(0, '/global/homes/j/jmyles/repositories/jmutil/')
from jmutils import save_figure, upload_to_dropbox

sys.path.insert(0, '/global/homes/j/jmyles/repositories/sompz/sompz/')
from utils import mag2flux, flux2mag, fluxerr2magerr #, magerr2fluxerr
from plots import plot_true_vs_wide, plot_true_vs_true_wide_diff

sys.path.insert(0, '/global/homes/j/jmyles/repositories/sompz/test/full_run_on_data')
from mock_error import setup_balrog_error_model, balrog_error_apply

datadir = '/global/project/projectdirs/des/jmyles/sompz_data/v0.50/'
inject_all_deep = True
run_name = 'v0.50.{}'.format(int(inject_all_deep))

today = str(datetime.datetime.now())[:10]
outdir = '/global/cscratch1/sd/jmyles/pitpz/{}/{}/'.format(run_name, today)

if not os.path.exists(outdir):
    print('mkdir -p {}'.format(outdir))
    os.system('mkdir -p {}'.format(outdir))

# Set parameters of run
cfgfile = '/global/homes/j/jmyles/repositories/pitpz/cfg/pitpz.cfg'
with open(cfgfile, 'r') as fp:
    cfg = yaml.load(fp)

bands = 'IGRZUJHK'
control = False
deep_columns = ['BDF_MAG_DERED_CALIB_' + band for band in bands]
deep_flux_columns = [colname.replace('MAG','FLUX') for colname in deep_columns]
zp_unc = np.array([0.0039, 0.0039, 0.0039, 0.0039, 0.0548, 0.0054, 0.0054, 0.0054]) # IGRZUJHK
nreal = 100 # number of realizations to make (e.g. 300)
int_cols = ['ID',
            'FLAGS',
            'MASK_FLAGS',
            'FLAGS_jhk',
            'bal_id',
            'true_id',
            'ninject',
            'ndetect',
            'nselect']
N_BALROG_DATA_DATA = 2417437
N_SPEC_DATA_DATA = 451776

if len(sys.argv) == 4:
    nstart, nend = int(sys.argv[1]),int(sys.argv[2])
    realizations = range(nstart, nend+1)
    control = sys.argv[3] == 'control'
elif len(sys.argv) == 3:
    nstart, nend = int(sys.argv[1]),int(sys.argv[2])
    realizations = range(nstart, nend+1)
else:
    realizations = range(nreal)
t0 = time.time()
print('{:.1f}: Doing realizations {} through {} inclusive. Control: {}'.format(time.time()-t0,
                                                                               realizations[0],
                                                                               realizations[-1],
                                                                               control))

spec_data_one_row_per_deep_cache_file = os.path.join(outdir,'spec_data_one_row_per_deep_cached.pkl')
balrog_data_one_row_per_deep_cache_file = os.path.join(outdir,'balrog_data_one_row_per_deep_cached.pkl')
make_tables = not (os.path.exists(balrog_data_one_row_per_deep_cache_file) and os.path.exists(spec_data_one_row_per_deep_cache_file))

### Load up balrog_data from data ###
if not make_tables:
    print('Loading cached file {}'.format(balrog_data_one_row_per_deep_cache_file))
    balrog_data_one_row_per_deep = pd.read_pickle(balrog_data_one_row_per_deep_cache_file)
else:

    print('{:.1f}: Making cache file {}'.format(time.time()-t0, balrog_data_one_row_per_deep_cache_file))
    detection_cat = fitsio.read(cfg['DetectionFile'])
    matched_cat = h5py.File(cfg['MatchedCatFile'])

    df_detection_cat = pd.DataFrame({'bal_id' : detection_cat['bal_id'].byteswap().newbyteorder(),
                                     'true_id' :  detection_cat['true_id'].byteswap().newbyteorder(),
                                     'detected' :  detection_cat['detected'].byteswap().newbyteorder(),
                                     'match_flag_1.5_asec' : detection_cat['match_flag_1.5_asec'].byteswap().newbyteorder()})
    del detection_cat, matched_cat

    balrog_data_file = os.path.join(datadir, 'deep_balrog.pkl')
    balrog_data = pd.read_pickle(balrog_data_file)
    balrog_data["FIELD"] = balrog_data["FIELD"].astype("category")
    nselect = balrog_data.groupby('true_id')['true_id'].count()
    print('Calculating injection and detection counts')

    if inject_all_deep:
        print('inject_all_deep True')
        ### BEGIN OPTION A BLOCK: Inject all deep
        infile_deep = '/global/cscratch1/sd/aamon/deep_ugriz.mof02_sn.jhk.ff04_c.jhk.ff02_052020_realerrors_May20calib.pkl'
        deep_data = p.load(open(infile_deep,'rb'))
        injected_at_least_once = np.isin(deep_data['ID'], df_detection_cat[df_detection_cat['match_flag_1.5_asec'] < 2]['true_id'])
        deep_data_injected = deep_data[injected_at_least_once]
        deep_data_injected = deep_data_injected.drop(['RA_jhk','DEC_jhk','MASK_FLAGS_jhk',
                                                      'TILENAME','TILENAME_jhk'],axis=1)

        for col in ['bal_id',
                    'unsheared/flux_i','unsheared/flux_r','unsheared/flux_z',
                    'unsheared/mag_i' ,'unsheared/mag_r' ,'unsheared/mag_z',
                    'unsheared/R11'   ,'unsheared/R22'   ,
                    'unsheared/weight','injection_counts',]:
            deep_data_injected.loc[:,col] = np.full_like(deep_data_injected['ID'].to_numpy(), np.nan)
        deep_data_injected.loc[:,'true_id'] = deep_data_injected['ID'].to_numpy()

        balrog_data_one_row_per_deep = deep_data_injected.copy()
        ### END OPTION A BLOCK
    else:
        ### BEGIN OPTION B BLOCK: Inject only deep galaxies with at least one selected realization in data
        balrog_data_one_row_per_deep = balrog_data.drop_duplicates('true_id')
        for fluxtype in ['1m','1p','2m','2p']:
            for band in 'irz':
                balrog_data_one_row_per_deep = balrog_data_one_row_per_deep.drop('sheared_{}/flux_{}'.format(fluxtype, band), axis=1)
                balrog_data_one_row_per_deep = balrog_data_one_row_per_deep.drop('sheared_{}/flux_ivar_{}'.format(fluxtype, band), axis=1)
        ### END OPTION B BLOCK
    
    groupby = df_detection_cat.groupby('true_id')
    ndetect = groupby['detected'].sum()
    ninject = groupby['detected'].count()

    balrog_data_one_row_per_deep['ninject'] = balrog_data_one_row_per_deep['true_id'].map(ninject)
    balrog_data_one_row_per_deep['ninject_cumsum_prior'] = np.concatenate([[0],np.cumsum(balrog_data_one_row_per_deep['ninject'].to_numpy()[:-1])])
    balrog_data_one_row_per_deep['ndetect'] = balrog_data_one_row_per_deep['true_id'].map(ndetect)
    balrog_data_one_row_per_deep['nselect'] = balrog_data_one_row_per_deep['true_id'].map(nselect)
    balrog_data_one_row_per_deep['nselect'] = balrog_data_one_row_per_deep['nselect'].fillna(0)
    
    ### Correct dtype of pandas.DataFrame ###
    print("{:.1f} Correct dtype of pandas.DataFrame".format(time.time()-t0))
    dtype_balrog_data = {col : np.float32 for col in balrog_data_one_row_per_deep.columns}
    for col in int_cols:
        dtype_balrog_data[col] = np.int64
    dtype_balrog_data['FIELD'] = "category"
    balrog_data_one_row_per_deep = balrog_data_one_row_per_deep.astype(dtype_balrog_data)
    balrog_data_one_row_per_deep.to_pickle(balrog_data_one_row_per_deep_cache_file)
print(len(balrog_data_one_row_per_deep), 'unique metacal-detected-selected deep galaxies')

### Load of spec_data from data ###
if not make_tables:
    print('Loading cached file {}'.format(spec_data_one_row_per_deep_cache_file))
    spec_data_one_row_per_deep = pd.read_pickle(spec_data_one_row_per_deep_cache_file)
else:
    print('{:.1f}: Making cache file {}'.format(time.time()-t0, spec_data_one_row_per_deep_cache_file))
    spec_data_file1 = os.path.join(datadir, 'redshift_deep_balrog.pkl')
    spec_data_file2 = os.path.join(datadir, 'redshift_deep_balrog2.pkl')
    spec_data1 = p.load(open(spec_data_file1, 'rb'))
    spec_data2 = p.load(open(spec_data_file2, 'rb'))
    spec_data = pd.concat([spec_data1,spec_data2], ignore_index=True)
    nselect = spec_data.groupby('true_id')['true_id'].count()

    if inject_all_deep:
        print('inject_all_deep True')
        ### BEGIN OPTION A BLOCK: Inject all deep
        matchlim= 0.75 # cfg['matchlim']
        z_file = '/global/project/projectdirs/des/jmyles/sompz_cosmos.h5'
        cosmos=deep_data_injected[deep_data_injected['FIELD'] == 'COSMOS']
        cosmos_z = pd.read_hdf(z_file)
        c = SkyCoord(ra=cosmos_z['ALPHA_J2000'].to_numpy()*u.degree, dec=cosmos_z['DELTA_J2000'].to_numpy()*u.degree)
        catalog = SkyCoord(ra=cosmos['RA'].to_numpy()*u.degree, dec=cosmos['DEC'].to_numpy()*u.degree)
        idx, d2d, d3d = catalog.match_to_catalog_sky(c)
        is_match = d2d < matchlim * u.arcsec
        zpdfcols = ["Z{:.2f}".format(s).replace(".","_") for s in np.arange(0,6.01,0.01)]
        print('add Z info')
        cosmos['Z'] = -1
        cosmos.loc[is_match, 'Z'] = cosmos_z.iloc[idx[is_match], cosmos_z.columns.get_loc('PHOTOZ')].to_numpy()
        print('add pz info')
        cosmos[zpdfcols] = pd.DataFrame(-1 * np.ones((len(cosmos), len(zpdfcols))), columns=zpdfcols, index=cosmos.index)
        zpdfcols_indices = [cosmos_z.columns.get_loc(_) for _ in zpdfcols]
        cosmos.loc[is_match, zpdfcols] = cosmos_z.iloc[idx[is_match], zpdfcols_indices].to_numpy()
        print('add Laigle ID info')
        cosmos.loc[:,'LAIGLE_ID'] = -1
        cosmos.loc[is_match, 'LAIGLE_ID'] = cosmos_z.iloc[idx[is_match], cosmos_z.columns.get_loc('ID')].to_numpy()
        ids, counts = np.unique(cosmos.loc[is_match, 'LAIGLE_ID'], return_counts=True)
        print('n duplicated Laigle', len(counts[counts > 1]))

        for col in ['unsheared/flux_i','unsheared/flux_r','unsheared/flux_z',
                    'unsheared/mag_i', 'unsheared/mag_r', 'unsheared/mag_z',
                    'unsheared/R11'   ,'unsheared/R22'   ,
                    'unsheared/weight','injection_counts',]:
            cosmos.loc[:,col] = np.full_like(cosmos['ID'].to_numpy(), np.nan)
        cosmos.loc[:,'true_id'] = cosmos['ID'].to_numpy()
        cosmos.loc[:,'bal_id'] =np.full_like(cosmos['ID'].to_numpy(), -1)
        spec_data_one_row_per_deep = cosmos[cosmos['Z'] != -1].copy()
        ### END OPTION A BLOCK
    else:
        ### BEGIN OPTION B BLOCK: Inject only deep galaxies with at least one selected realization in data
        spec_data_one_row_per_deep = spec_data.drop_duplicates('ID')
        for fluxtype in ['1m','1p','2m','2p']:
            for band in 'irz':
                spec_data_one_row_per_deep = spec_data_one_row_per_deep.drop('sheared_{}/flux_{}'.format(fluxtype, band), axis=1)
                spec_data_one_row_per_deep = spec_data_one_row_per_deep.drop('sheared_{}/flux_ivar_{}'.format(fluxtype, band), axis=1)
        ### END OPTION B BLOCK
    
    ndetect = groupby['detected'].sum()
    ninject = groupby['detected'].count()

    spec_data_one_row_per_deep.loc[:,'ninject'] = spec_data_one_row_per_deep['true_id'].map(ninject)
    spec_data_one_row_per_deep.loc[:,'ndetect'] = spec_data_one_row_per_deep['true_id'].map(ndetect)
    spec_data_one_row_per_deep.loc[:,'nselect'] = spec_data_one_row_per_deep['true_id'].map(nselect)
    spec_data_one_row_per_deep.loc[:,'nselect'] = spec_data_one_row_per_deep['nselect'].fillna(0)

    dtype_spec_data = {col : np.float32 for col in spec_data_one_row_per_deep.columns}
    for col in int_cols + ['LAIGLE_ID']:
        dtype_spec_data[col] = np.int64
    dtype_spec_data['FIELD'] = "category"

    spec_data_one_row_per_deep = spec_data_one_row_per_deep.astype(dtype_spec_data)
    spec_data_one_row_per_deep.to_pickle(spec_data_one_row_per_deep_cache_file)
print(len(spec_data_one_row_per_deep), 'unique metacal-detected-selected deep galaxies with redshifts')

### Build cache storing ninjects for each galaxy ###
result_ninjects_cache_file = os.path.join(outdir, 'result_ninjects_cache_file.p')
if os.path.exists(result_ninjects_cache_file):
    print('Loading cached file {}'.format(result_ninjects_cache_file))
    result_ninjects = p.load(open(result_ninjects_cache_file, 'rb'))
else:
    # Construct array with index in x of every element in y in x
    print('{:.1f}: Begin write {}'.format(time.time()-t0, result_ninjects_cache_file))
    #assert np.sum(balrog_data_one_row_per_deep['ninject'] == 0) == 0
    true_id_to_ninject_cumsum = balrog_data_one_row_per_deep.set_index('true_id')['ninject_cumsum_prior'].to_dict()
    spec_data_one_row_per_deep['ninject_cumsum_prior'] = spec_data_one_row_per_deep['true_id'].map(true_id_to_ninject_cumsum).astype(np.int64)
    result_ninjects_summand_1 = np.repeat(spec_data_one_row_per_deep['ninject_cumsum_prior'].to_numpy(),
                                          spec_data_one_row_per_deep['ninject'].to_numpy(), axis=0)
    add_these_values_because_balrog_data_gets_repeated_too = map(lambda x: np.arange(0,x),
                                                                 spec_data_one_row_per_deep['ninject'].to_numpy())
    result_ninjects_summand_2 = np.concatenate(add_these_values_because_balrog_data_gets_repeated_too).ravel()

    # for each row in balrog_data_ninject_rows_per_deep, the row number in spec_data_ninject_rows_per_deep
    # i.e. index for each row in balrog_data_ninject_rows_per_deep to the corresponding row of spec_data_ninject_rows_per_deep
    result_ninjects = result_ninjects_summand_1 + result_ninjects_summand_2

    p.dump(result_ninjects, open(result_ninjects_cache_file, 'wb'))
    
### Generate (or load from disk) zp offsets ###
np.random.seed(11235)
outfile_offsets = os.path.join(outdir, 'zp_offsets_all.fits')
if os.path.exists(outfile_offsets):
    print('Loading cached file {}'.format(outfile_offsets))
    offsets_all = fitsio.read(outfile_offsets)
else:
    nfields = len(balrog_data_one_row_per_deep['FIELD'].unique())    
    offsets_all = np.random.normal(loc=0, scale=zp_unc, size=(nreal, nfields, len(deep_columns)))
    print('{:.1f}: Begin write {}'.format(time.time()-t0, outfile_offsets))
    fitsio.write(outfile_offsets, offsets_all)

### Build mock Balrog error model (KDTree) ###
print('{:.1f}: Building Balrog error model to use'.format(time.time()-t0))
balrog_bands = cfg['BalrogBands']
usebalmags = cfg['UseBalMags'] # [2,3,4] 
detection_file = cfg['DetectionFile']
matched_cat_file = cfg['MatchedCatFile']

detection_catalog, true_deep_cat, matched_catalog, matched_cat_sorter = setup_balrog_error_model(detection_file, matched_cat_file)

### Begin loop to make mock Balrog realizations ###
for i in realizations:
    t0 = time.time()
    print('{:.1f}: Begin realization {:03d} at {}'.format(time.time()-t0, i, datetime.datetime.now()))
    # Draw zp offset from zp offset uncertainty distribution
    if control:
        offsets = np.zeros_like(offsets_all[i]) # this variable should never actually get used
        control_suffix = '_control'
    else:
        offsets = offsets_all[i]
        control_suffix = ''

    outdir_realization = os.path.join(outdir, '{:03d}_of_{:03d}{}'.format(i, nreal, control_suffix))
    os.system('mkdir -p {}'.format(outdir_realization))

    made_spec_data = os.path.exists(os.path.join(outdir_realization, 'redshift_deep_balrog.pkl'))
    made_balrog_data = os.path.exists(os.path.join(outdir_realization, 'deep_balrog.pkl'))
    if made_spec_data and made_balrog_data:
        print('{:.1f}: Already done realization {:03d} of {:03d}. Skipping.'.format(time.time()-t0,i,nreal))
        continue

    # apply zp offset to BDF_MAG_DERED_CALIB_ and recompute BDF_FLUX_DERED_CALIB_
    offset_balrog_data_one_row_per_deep = balrog_data_one_row_per_deep.copy()
    offset_spec_data_one_row_per_deep = spec_data_one_row_per_deep.copy()
    if not control:
        print('{:.1f}: Offseting deep field mags and fluxes'.format(time.time()-t0))
        for j, field in enumerate(balrog_data_one_row_per_deep['FIELD'].unique()):
            select_field = balrog_data_one_row_per_deep['FIELD'] == field
            offset_balrog_data_one_row_per_deep.loc[select_field, deep_columns] = balrog_data_one_row_per_deep.loc[select_field, deep_columns] + offsets[j] # copy
            fluxes = mag2flux(offset_balrog_data_one_row_per_deep.loc[select_field, deep_columns])
            offset_balrog_data_one_row_per_deep.loc[select_field, deep_flux_columns] = np.copy(fluxes)

            select_field = spec_data_one_row_per_deep['FIELD'] == field
            offset_spec_data_one_row_per_deep.loc[select_field, deep_columns] = spec_data_one_row_per_deep.loc[select_field, deep_columns] + offsets[j] # copy
            fluxes = mag2flux(offset_spec_data_one_row_per_deep.loc[select_field, deep_columns])
            offset_spec_data_one_row_per_deep.loc[select_field, deep_flux_columns] = np.copy(fluxes)

    outfile_offset_balrog_data_one_row_per_deep = os.path.join(outdir_realization, 'offset_balrog_data_one_row_per_deep.pkl')
    offset_balrog_data_one_row_per_deep.to_pickle(outfile_offset_balrog_data_one_row_per_deep)
    outfile_offset_spec_data_one_row_per_deep = os.path.join(outdir_realization, 'offset_spec_data_one_row_per_deep.pkl')
    offset_spec_data_one_row_per_deep.to_pickle(outfile_offset_spec_data_one_row_per_deep)
    
    # now make wide field flux using error model as implemented in mock_error.py
    print('{:.1f}: Make omag'.format(time.time()-t0))
    omag = np.repeat(offset_balrog_data_one_row_per_deep[['BDF_MAG_DERED_CALIB_' + band for band in 'RIZ']].to_numpy(),
                     offset_balrog_data_one_row_per_deep['ninject'].to_numpy(),
                     axis=0).astype(np.float32)
    true_ids = np.repeat(offset_balrog_data_one_row_per_deep['true_id'].to_numpy(),
                         offset_balrog_data_one_row_per_deep['ninject'].to_numpy(),
                         axis=0).astype(np.int)
    np.save(os.path.join(outdir_realization,'true_ids.npy'), true_ids)
    print('len omag: {}'.format(len(omag)))

    print('{:.1f}: Call balrog_error_apply'.format(time.time()-t0))
    (flux_bal, fluxerr_bal, dist,
     _, _, _, _,
     _, _, R11, R22,
     injection_counts, weight, bal_id) = balrog_error_apply(detection_catalog,
                                                            true_deep_cat,
                                                            matched_catalog,
                                                            omag,
                                                            matched_cat_sorter=matched_cat_sorter,
                                                            zp = 30,
                                                            zp_data = 30,
                                                            true_cat_mag_cols=usebalmags,
                                                            return_all_metacal_quantities=True)
    np.save(os.path.join(outdir_realization, 'neighbor_dist.npy'), dist)
    del omag
    """
    select_r = np.isfinite(flux_bal[:,0])
    select_i = np.isfinite(flux_bal[:,1])
    select_z = np.isfinite(flux_bal[:,2])
    select = select_r & select_i & select_z
    """
    print('{:.1f}: Make Dataframe'.format(time.time()-t0))
    balrog_data_realization = offset_balrog_data_one_row_per_deep.iloc[np.repeat(np.arange(len(offset_balrog_data_one_row_per_deep)),
                                                                                 offset_balrog_data_one_row_per_deep['ninject'].to_numpy())].copy()
    spec_data_realization = offset_spec_data_one_row_per_deep.iloc[np.repeat(np.arange(len(offset_spec_data_one_row_per_deep)),
                                                                                 offset_spec_data_one_row_per_deep['ninject'].to_numpy())].copy()

    mag_bal = flux2mag(flux_bal, zero_pt=30)
    magerr_bal = fluxerr2magerr(flux_bal, fluxerr_bal)

    for i, band in enumerate('riz'):
        balrog_data_realization['unsheared/flux_'+band] = flux_bal[:,i].astype(np.float32).copy() #.loc[:, on LHS
        balrog_data_realization['unsheared/flux_ivar_'+band] = fluxerr_bal[:,i].astype(np.float32).copy()
        balrog_data_realization['unsheared/mag_'+band] = mag_bal[:,i].astype(np.float32).copy()
        balrog_data_realization['unsheared/mag_err_'+band] = magerr_bal[:,i].astype(np.float32).copy()

        spec_data_realization['unsheared/flux_'+band] = flux_bal[result_ninjects,i].astype(np.float32).copy()
        spec_data_realization['unsheared/flux_ivar_'+band] = fluxerr_bal[result_ninjects,i].astype(np.float32).copy()
        spec_data_realization['unsheared/mag_'+band] = mag_bal[result_ninjects,i].astype(np.float32).copy()
        spec_data_realization['unsheared/mag_err_'+band] = magerr_bal[result_ninjects,i].astype(np.float32).copy()
    del flux_bal, fluxerr_bal, mag_bal, magerr_bal
    balrog_data_realization['bal_id'] = bal_id[:].astype(np.int64).copy()
    balrog_data_realization['unsheared/R11'] = R11[:].astype(np.float32).copy()
    balrog_data_realization['unsheared/R22'] = R22[:].astype(np.float32).copy()
    balrog_data_realization['injection_counts'] = injection_counts[:].astype(np.float32).copy()
    balrog_data_realization['unsheared/weight'] = weight[:].astype(np.float32).copy()

    spec_data_realization['bal_id'] = bal_id[result_ninjects].astype(np.int64).copy()
    spec_data_realization['unsheared/R11'] = R11[result_ninjects].astype(np.float32).copy()
    spec_data_realization['unsheared/R22'] = R22[result_ninjects].astype(np.float32).copy()
    spec_data_realization['injection_counts'] = injection_counts[result_ninjects].astype(np.float32).copy()
    spec_data_realization['unsheared/weight'] = weight[result_ninjects].astype(np.float32).copy()

    """
    print('{:.1f}: begin save many rows fits'.format(time.time()-t0))
    outfile_balrog_data_realization = os.path.join(outdir_realization, 'offset_balrog_data_realization.fits')
    fitsio.write(outfile_balrog_data_realization, balrog_data_realization.to_records(), clobber=True)
    print('{:.1f}: finished balrog, start spec'.format(time.time()-t0))
    outfile_spec_data_realization = os.path.join(outdir_realization, 'offset_spec_data_realization.fits')
    fitsio.write(outfile_spec_data_realization, spec_data_realization.drop(labels=["Z{:.2f}".format(s).replace(".","_") for s in np.arange(0,6.01,0.01)],
                                                                                         axis=1).to_records(), clobber=True)
    """
    ### Apply selection functions to balrog_data and spec_data
    print('{:.1f}: Apply selection'.format(time.time()-t0))
    select_balrog_i = np.isfinite(balrog_data_realization['unsheared/flux_i'])
    select_balrog_r = np.isfinite(balrog_data_realization['unsheared/flux_r'])
    select_balrog_z = np.isfinite(balrog_data_realization['unsheared/flux_z'])
    select_balrog = select_balrog_i & select_balrog_r & select_balrog_z
    np.save(os.path.join(outdir_realization, 'select_balrog.npy'), select_balrog)
    select_spec_i = np.isfinite(spec_data_realization['unsheared/flux_i'])
    select_spec_r = np.isfinite(spec_data_realization['unsheared/flux_r'])
    select_spec_z = np.isfinite(spec_data_realization['unsheared/flux_z'])
    select_spec = select_spec_i & select_spec_r & select_spec_z
    np.save(os.path.join(outdir_realization, 'select_spec.npy'), select_spec)
    
    balrog_data_realization_select = balrog_data_realization[select_balrog]#.copy()
    spec_data_realization_select = spec_data_realization[select_spec]#.copy()

    balrog_data_realization_boot = balrog_data_realization_select.sample(n=N_BALROG_DATA_DATA,replace=True)
    spec_data_realization_boot = spec_data_realization_select.sample(n=N_SPEC_DATA_DATA,replace=True)

    print(len(balrog_data_realization), len(balrog_data_realization_select), len(balrog_data_realization_boot))
    print(len(spec_data_realization), len(spec_data_realization_select), len(spec_data_realization_boot))
    
    ### Write balrog_data_realization and spec_data_realization to disk
    for (balrog_data_filename,
         spec_data_filename,
         balrog_data_obj,
         spec_data_obj) in zip(['deep_balrog_boot.pkl'         ,'deep_balrog.pkl', 'deep_balrog_offset_one_row_per_deep.pkl'],
                               ['redshift_deep_balrog_boot.pkl','redshift_deep_balrog.pkl', 'redshift_deep_balrog_offset_one_row_per_deep.pkl'], 
                               [balrog_data_realization_boot   , balrog_data_realization_select, offset_balrog_data_one_row_per_deep],
                               [  spec_data_realization_boot   , spec_data_realization_select, offset_spec_data_one_row_per_deep]):
        balrog_data_realization_outfile = os.path.join(outdir_realization, balrog_data_filename) #'deep_balrog.pkl')
        print('{:.1f}: Begin write {}'.format(time.time() -t0, balrog_data_realization_outfile))
        try:
            balrog_data_obj.to_pickle(balrog_data_realization_outfile) #balrog_data_realization
        except Exception as e:
            print('to_pickle command FAILED. Error: {}.\ntrying fitsio instead'.format(e))
            fitsio.write(balrog_data_realization_outfile.replace('pkl','fits'),
                         balrog_data_obj.to_records(), clobber=True) #balrog_data_realization

        spec_data_realization_outfile = os.path.join(outdir_realization, spec_data_filename) # 'redshift_deep_balrog_no_boot.pkl'
        print('{:.1f}: Begin write {}'.format(time.time() - t0, spec_data_realization_outfile))
        try:
            #lenint=int(len(spec_data_obj)/2)
            #spec_data1 = spec_data_obj.iloc[:lenint]
            #spec_data2 = spec_data_obj.iloc[lenint:]
            #spec_data_pkl_file = spec_data_realization_outfile
            #spec_data_pkl_file2 = spec_data_realization_outfile.replace('.pkl','2.pkl')

            #spec_data1.to_pickle(spec_data_pkl_file)
            #spec_data2.to_pickle(spec_data_pkl_file2)
            spec_data_obj.to_pickle(spec_data_realization_outfile) #spec_data_realization
        except Exception as e:
            print('to_pickle command FAILED. Error: {}.\ntrying fitsio instead'.format(e))
            fitsio.write(spec_data_realization_outfile.replace('pkl','fits'),
                         spec_data_obj.to_records(), clobber=True) #spec_data_realization
    
    ### Checkplots
    """
    omag_for_plot = balrog_data_realization[['BDF_MAG_DERED_CALIB_' + band for band in 'RIZ']].to_numpy()
    mag_bal_for_plot = balrog_data_realization[['unsheared/mag_' + band for band in 'riz']].to_numpy()
    del balrog_data_realization
    plot_true_vs_wide(omag_for_plot, mag_bal_for_plot, outdir_realization, suffix='_balrog_{:03d}'.format(i))
    plot_true_vs_true_wide_diff(omag_for_plot, mag_bal_for_plot, outdir_realization, suffix='_balrog_{:03d}'.format(i))
    
    omag_for_plot = spec_data_realization[['BDF_MAG_DERED_CALIB_' + band for band in 'RIZ']].to_numpy()
    mag_bal_for_plot = spec_data_realization[['unsheared/mag_' + band for band in 'riz']].to_numpy()
    del spec_data_realization
    plot_true_vs_wide(omag_for_plot, mag_bal_for_plot, outdir_realization, suffix='_spec_{:03d}'.format(i))
    plot_true_vs_true_wide_diff(omag_for_plot, mag_bal_for_plot, outdir_realization, suffix='_spec_{:03d}'.format(i))
    """
