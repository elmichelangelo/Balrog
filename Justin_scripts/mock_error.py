from glob import glob
import pdb
from mpi4py import MPI
from rot_mock_tools import rot_mock_file
from sklearn.neighbors import KDTree
import h5py
import numpy.lib.recfunctions as rf
import numpy as np
import healpy as hp
import h5py as h5
import healpix_util as hu
import fitsio
import pickle
import yaml
import sys
import os
import time
import pandas as pd
sys.path.insert(0, '/global/homes/j/jmyles/repositories/sompz/sompz/')
from utils import get_balrog_selection_mask

models = {
    'DR8':
        {'maglims':[20.425,21.749,21.239,20.769,19.344],
        'exptimes' : [21.00,159.00,126.00,99.00,15.00],
        'lnscat' : [0.284,0.241,0.229,0.251,0.264]
         },

    'STRIPE82':
        {
        'maglims' : [22.070,23.432,23.095,22.649,21.160],
        'exptimes' : [99.00,1172.00,1028.00,665.00,138.00],
        'lnscat' : [0.280,0.229,0.202,0.204,0.246]
        },

    'CFHTLS':
        {
        'maglims' : [24.298,24.667,24.010,23.702,22.568],
        'exptimes' : [2866.00,7003.00,4108.00,3777.00,885.00],
        'lnscat' : [0.259,0.244,0.282,0.258,0.273]
        },
    'DEEP2':
        {
        'maglims' : [24.730,24.623,24.092],
        'exptimes' : [7677.00,8979.00,4402.00],
        'lnscat' : [0.300,0.293,0.300]
        },
    'FLAMEX':
        {
        'maglims' : [21.234,20.929],
        'exptimes' : [259.00,135.00],
        'lnscat' : [0.300,0.289]
        },
    'IRAC':
        {
        'maglims' : [19.352,18.574],
        'exptimes' : [8.54,3.46],
        'lnscat' : [0.214,0.283]
        },
    'NDWFS':
        {
        'maglims' : [25.142,23.761,23.650],
        'exptimes' : [6391.00,1985.00,1617.00],
        'lnscat' : [0.140,0.294,0.272]
        },

    'RCS':
        {
        'maglims' : [23.939,23.826,23.067,21.889],
        'exptimes' : [2850.00,2568.00,1277.00,431.00],
        'lnscat' : [0.164,0.222,0.250,0.271]
        },
    'VHS':
        {
        'maglims' : [20.141,19.732,19.478],
        'exptimes' : [36.00,31.00,23.00],
        'lnscat' : [0.097,0.059,0.069]
        },
    'VIKING':
        {
        'maglims' : [21.643,20.915,20.768,20.243,20.227],
        'exptimes' : [622.00,246.00,383.00,238.00,213.00],
        'lnscat' : [0.034,0.048,0.052,0.040,0.066]
        },
    'DC6B':
        {
        'maglims' : [24.486,23.473,22.761,22.402],
        'exptimes' : [2379.00,1169.00,806.00,639.00],
        'lnscat' : [0.300,0.300,0.300,0.300]
        },

    'DES':
        {
        'maglims' : [24.956,24.453,23.751,23.249,21.459],
        'exptimes' : [14467.00,12471.00,6296.00,5362.00,728.00],
        'lnscat' : [0.2,0.2,0.2,0.2,0.2]
        },

    'BCS_LO':
        {
        'maglims' : [23.082,22.618,22.500,21.065],
        'exptimes' : [809.00,844.00,641.00,108.00],
        'lnscat' : [0.277,0.284,0.284,0.300]
        },

    'BCS':
        {
        'maglims' : [23.360,23.117,22.539,21.335],
        'exptimes' : [838.00,1252.00,772.00,98.00],
        'lnscat' : [0.276,0.272,0.278,0.279]
        },

    'DES_SV':
    {
	'maglims' : [23.621,23.232,23.008,22.374,20.663],
	'exptimes' : [4389.00,1329.00,1405.00,517.00,460.00],
	'lnscat' : [0.276,0.257,0.247,0.241,0.300]
        },

    'DES_SV_OPTIMISTIC':
        {
        'maglims' : [23.621+0.5,23.232+0.5,23.008,22.374,20.663],
        'exptimes' : [4389.00,1329.00,1405.00,517.00,460.00],
        'lnscat' : [0.276,0.257,0.247,0.241,0.300]
        },
    'WISE':
        {
        'maglims' : [19.352,18.574],
        'exptimes' : [8.54,3.46],
        'lnscat' : [0.214,0.283]
        },

    'DECALS':
        {
       'maglims' : [23.3,23.3,22.2,20.6,19.9],
       'exptimes' : [1000,3000,2000,1500,1500],
       'lnscat' : [0.2,0.2,0.2,0.2,0.2]
       },

}

def calc_nonuniform_errors(exptimes,limmags,mag_in,nonoise=False,zp=22.5,
                            nsig=10.0,fluxmode=False,lnscat=None,b=None,
                            inlup=False,detonly=False):

    f1lim = 10**((limmags-zp)/(-2.5))
    fsky1 = ((f1lim**2)*exptimes)/(nsig**2) - f1lim
    fsky1[fsky1<0.001] = 0.001

    if inlup:
        bnmgy = b*1e9
        tflux = exptimes*2.0*bnmgy*np.sinh(-np.log(b)-0.4*np.log(10.0)*mag_in)
    else:
        tflux = exptimes*10**((mag_in - zp)/(-2.5))

    noise = np.sqrt(fsky1*exptimes + tflux)

    if lnscat is not None:
        noise = np.exp(np.log(noise) + lnscat*np.random.randn(len(mag_in)))

    if nonoise:
        flux = tflux
    else:
        flux = tflux + noise*np.random.randn(len(mag_in))

    #convert to nanomaggies
        flux = flux/exptimes
        noise = noise/exptimes

        flux  = flux * 10 ** ((zp - 22.5)/-2.5)
        noise = noise * 10 ** ((zp - 22.5)/-2.5)

    if fluxmode:
        mag = flux
        mag_err = noise
    else:
        if b is not None:
            bnmgy = b*1e9
            flux_new = flux
            noise_new = noise
            mag = 2.5*np.log10(1.0/b) - asinh2(0.5*flux_new/(bnmgy))/(0.4*np.log(10.0))

            mag_err = 2.5*noise_new/(2.*bnmgy*np.log(10.0)*np.sqrt(1.0+(0.5*flux_new/(bnmgy))**2.))

        else:
            mag = 22.5-2.5*np.log10(flux)
            mag_err = (2.5/np.log(10.))*(noise/flux)

            #temporarily changing to cut to 10-sigma detections in i,z
            bad = np.where((np.isfinite(mag)==False))
            nbad = len(bad)

            if detonly:
                mag[bad]=99.0
                mag_err[bad]=99.0

    return mag, mag_err


def calc_uniform_errors(model, tmag, maglims, exptimes, lnscat, zp=22.5):

    nmag=len(maglims)
    ngal=len(tmag)

    tmag = tmag.reshape(len(tmag),nmag)

    #calculate fsky1 -- sky in 1 second
    flux1_lim = 10**((maglims-zp)/(-2.5))
    flux1_lim[flux1_lim < 120/exptimes] = 120/exptimes[flux1_lim < 120/exptimes]
    fsky1 = (flux1_lim**2*exptimes)/100. - flux1_lim

    oflux=np.zeros((ngal, nmag))
    ofluxerr=np.zeros((ngal, nmag))
    omag=np.zeros((ngal, nmag))
    omagerr=np.zeros((ngal, nmag))
    offset = 0.0

    for i in range(nmag):
        tflux = exptimes[i] * 10**((tmag[:,i]-offset-zp)/(-2.5))
        noise = np.exp(np.log(np.sqrt(fsky1[i]*exptimes[i] + tflux))
                    + lnscat[i]*np.random.randn(ngal))

        flux = tflux + noise*np.random.randn(ngal)

        oflux[:,i] = flux / exptimes[i]
        ofluxerr[:,i] = noise/exptimes[i]

        oflux[:,i]    *= 10 ** ((zp - 22.5) / -2.5)
        ofluxerr[:,i] *= 10 ** ((zp - 22.5) / -2.5)

        omag[:,i] = 22.5-2.5*np.log10(oflux[:,i])
        omagerr[:,i] = (2.5/np.log(10.))*(ofluxerr[:,i]/oflux[:,i])

        bad,=np.where(~np.isfinite(omag[:,i]))
        nbad = len(bad)
        if (nbad > 0) :
            omag[bad,i] = 99.0
            omagerr[bad,i] = 99.0


    return omag, omagerr, oflux, ofluxerr

def make_output_structure(ngals, dbase_style=False, bands=None, nbands=None,
                          all_obs_fields=True, blind_obs=False,
                          balrog_bands=None):

    if all_obs_fields & dbase_style:
        if bands is None:
            raise(ValueError("Need names of bands in order to use database formatting!"))

        fields = [('ID', np.int), ('RA', np.float), ('DEC', np.float),
                  ('EPSILON1', np.float), ('EPSILON2', np.float),
                  ('SIZE', np.float), ('PHOTOZ_GAUSSIAN', np.float)]

        for b in bands:
            fields.append(('MAG_{0}'.format(b.upper()), np.float))
            fields.append(('MAGERR_{0}'.format(b.upper()), np.float))
            fields.append(('FLUX_{0}'.format(b.upper()), np.float))
            fields.append(('IVAR_{0}'.format(b.upper()), np.float))

        if balrog_bands is not None:
            for b in balrog_bands:
                fields.append(('MCAL/MAG_{0}'.format(b.upper()), np.float))
                fields.append(('MCAL_MAGERR_{0}'.format(b.upper()), np.float))
                fields.append(('METACAL/flux_{0}'.format(b), np.float))
                fields.append(('METACAL/flux_ivar_{0}'.format(b), np.float))

    if all_obs_fields & (not dbase_style):

        fields = [('ID', np.int), ('RA', np.float), ('DEC', np.float),
                  ('EPSILON1', np.float), ('EPSILON2', np.float),
                  ('SIZE', np.float), ('PHOTOZ_GAUSSIAN', np.float),
                  ('MAG', (np.float, nbands)), ('FLUX', (np.float, nbands)),
                  ('MAGERR', (np.float, nbands)), ('IVAR', (np.float, nbands))]

    if (not all_obs_fields) & dbase_style:
        fields = [('ID', np.int)]
        for b in bands:
            fields.append(('MAG_{0}'.format(b.upper()), np.float))
            fields.append(('MAGERR_{0}'.format(b.upper()), np.float))
            fields.append(('FLUX_{0}'.format(b.upper()), np.float))
            fields.append(('IVAR_{0}'.format(b.upper()), np.float))

    if (not all_obs_fields) & (not dbase_style):
        fields = [('ID', np.int), ('MAG', (np.float, nbands)),
                  ('FLUX', (np.float, nbands)), ('MAGERR', (np.float, nbands)),
                  ('IVAR', (np.float, nbands))]

    if blind_obs:
        fields.extend([('M200', np.float), ('Z', np.float),
                       ('CENTRAL', np.int), ('HALOID', np.int64),
                       ('R200', np.float), ('Z_COS', np.float)])

    odtype = np.dtype(fields)

    out = np.zeros(ngals, dtype=odtype)

    return out


def setup_deep_bal_cats(detection_catalog):

    # only keep things with good matches
    match_idx = detection_catalog['match_flag_1.5_asec'] < 2
    detection_catalog = detection_catalog[match_idx]

    # get unique deep field galaxies
    _, uidx = np.unique(detection_catalog['true_id'], return_index=True)
    true_deep_cat = detection_catalog[uidx]

    # rename ids so that they are contiguous
    sidx = true_deep_cat['true_id'].argsort()
    true_deep_cat = true_deep_cat[sidx]
    old_id = np.copy(true_deep_cat['true_id'])
    true_deep_cat['true_id'] = np.arange(len(true_deep_cat))
    map_dict = dict(zip(old_id, true_deep_cat['true_id']))
    detection_catalog['true_id'] = np.array(
        [map_dict[detection_catalog['true_id'][i]] for i in range(len(detection_catalog['true_id']))])

    # sort detection catalog by true_id
    deep_sidx = detection_catalog['true_id'].argsort()
    detection_catalog = detection_catalog[deep_sidx]

    return detection_catalog, true_deep_cat


def generate_bal_id(detection_catalog, true_deep_cat, sim_mag_true,
                    outdir_realization = '/global/cscratch1/sd/jmyles/pitpz/v0.50.1/2021-09-20/000_of_100/'):
    # count number of injections for each true_id in detection_catalog
    # (equivalently, each true_id in true_deep_cat)
    n_injections, _ = np.histogram(detection_catalog['true_id'],
                                   np.arange(len(true_deep_cat) + 1))
    # make cumulative sum of number of injections
    cum_injections = np.cumsum(n_injections)
    # build K-dimensional tree of true deep BDF mags
    deep_tree = KDTree(true_deep_cat['true_bdf_mag_deredden'][:, 1:])
    # for each galaxy in sim_mag_true, find its nearest neighbor in the K-dim tree
    # _ stores the distance to the nearest neighbor
    # deep_idx stores the index (i.e. an index of true_deep_cat)
    dist, deep_idx = deep_tree.query(sim_mag_true)

    # random number drawn from U[0,1] for each galaxy in sim_mag_true (i.e. each mock injection truth)
    rand = np.random.uniform(size=len(sim_mag_true))
    # for each galaxy in sim_mag_true
    # use idx (deep_idx) of nearest neighbor in the KDTree of true deep mags
    # to get the 1) cumulative injections for that true deep gal and
    # 2) the number of injections for that true deep gal
    # multiply 2 by the random number and take the floor.
    # the result is a number between zero and the number of injections
    # subtract that off the total number of cumulative injections for that true deep gal
    # to yield an id that has the property of being
    # between the index associated with the first and the last injection for that true deep gal
    bal_id = cum_injections[deep_idx].flatten() - cum_injections[0] #+ np.floor(rand * n_injections[deep_idx].flatten())
    bal_id = bal_id.astype(np.int)

    if outdir_realization is not None:
        np.save(os.path.join(outdir_realization,'n_injections.npy'), n_injections)
        np.save(os.path.join(outdir_realization,'deep_idx.npy'), deep_idx)
        np.save(os.path.join(outdir_realization,'cum_injections.npy'), cum_injections)
        np.save(os.path.join(outdir_realization,'rand.npy'), rand)
        np.save(os.path.join(outdir_realization,'bal_cat_idx.npy'), bal_id)
        np.save(os.path.join(outdir_realization,'bal_id.npy'), detection_catalog['bal_id'][bal_id])
    
    # return actual balrog catalog bal_id, as well as our index into the detection_catalog to yield that bal_id
    return detection_catalog['bal_id'][bal_id], bal_id, dist


def balrog_error_apply(detection_catalog, # Balrog detection catalog (includes all injections)
                       true_deep_cat, # true deep field catalog injected by Balrog
                       matched_balrog_cat, # matched Balrog metacal catalog
                       mag_in, # simulated or zero-point-offset or otherwise mock deep/true fluxes
                       matched_cat_sorter=None, # 
                       zp=22.5, zp_data=30.,
                       matched_cat_flux_cols=['flux_r', 'flux_i', 'flux_z'],
                       matched_cat_flux_err_cols=[
                           'flux_err_r', 'flux_err_i', 'flux_err_z'],
                       true_cat_mag_cols=[1, 2, 3],
                       return_all_metacal_quantities=False):
    print('Running Balrog error model code')
    # JTM 2021-09-17 add comment:
    # maintaining original naming of variables but FYI:
    # flux_err is at various points of the code a flux, magnitude
    # flux_err is never an _error_
    # flux_err_report is an error, sometimes in flux, sometimes in mag
    """ 
    # original code from JdR
    flux_out = np.zeros_like(mag_in)
    flux_err = np.zeros_like(mag_in)
    flux_err_report = np.zeros_like(mag_in)
    """
    flux_out = np.full_like(mag_in, np.nan)
    flux_err = np.full_like(mag_in, np.nan)
    flux_err_report = np.full_like(mag_in, np.nan)

    # JTM 2021-09-17 add:
    # save quantities other than just flux
    snr_out = np.full_like(mag_in[:,0], np.nan)
    size_ratio_out = np.full_like(mag_in[:,0], np.nan)
    flags_out = np.full_like(mag_in[:,0], np.nan)
    T_out = np.full_like(mag_in[:,0], np.nan)
    e_1_out = np.full_like(mag_in[:,0], np.nan)
    e_2_out = np.full_like(mag_in[:,0], np.nan)

    R11_out = np.full_like(mag_in[:,0], np.nan)
    R22_out = np.full_like(mag_in[:,0], np.nan)
    injection_counts_out = np.full_like(mag_in[:,0], np.nan)
    weight_out = np.full_like(mag_in[:,0], np.nan)

    # get balrog injection ids for all simulated galaxies
    # JTM 2021-08-12 add comment:
    # for each zp offset deep gal in mag_in,
    #     identify the bal_id of the real Balrog injection with the most similar deep mag-color.
    #     Store that in bal_id.
    #     Store the index into detection_catalog to get that bal_id in bal_cat_idx
    bal_id, bal_cat_idx, dist = generate_bal_id(detection_catalog,
                                                true_deep_cat,
                                                mag_in)

    # determine which are detected
    detected = detection_catalog['detected'][bal_cat_idx].astype(np.bool)
    injection_counts_out = detection_catalog['injection_counts'][bal_cat_idx]
    
    # JTM 2021-09-17 add comment:
    # find matches in matched cat to get wide field measured fluxes
    # matched_idx stores indices of elements in bal_id[detected] such that
    # if these indices were inserted into matched_balrog_cat['catalog/unsheared/bal_id'],
    # then order in the latter would be maintained
    matched_idx = matched_balrog_cat['catalog/unsheared/bal_id'][:].searchsorted(bal_id[detected],
                                                                                 sorter=matched_cat_sorter)
    # JM add 2021-08-12
    snr_out[detected] = matched_balrog_cat['catalog/unsheared/snr'][:][matched_cat_sorter][matched_idx]
    size_ratio_out[detected] = matched_balrog_cat['catalog/unsheared/size_ratio'][:][matched_cat_sorter][matched_idx]
    flags_out[detected] = matched_balrog_cat['catalog/unsheared/flags'][:][matched_cat_sorter][matched_idx]
    T_out[detected] = matched_balrog_cat['catalog/unsheared/T'][:][matched_cat_sorter][matched_idx]
    e_1_out[detected] = matched_balrog_cat['catalog/unsheared/e_1'][:][matched_cat_sorter][matched_idx]
    e_2_out[detected] = matched_balrog_cat['catalog/unsheared/e_2'][:][matched_cat_sorter][matched_idx]

    R11_out[detected] = matched_balrog_cat['catalog/unsheared/R11'][:][matched_cat_sorter][matched_idx]
    R22_out[detected] = matched_balrog_cat['catalog/unsheared/R22'][:][matched_cat_sorter][matched_idx]
    weight_out[detected] = matched_balrog_cat['catalog/unsheared/weight'][:][matched_cat_sorter][matched_idx]
    
    # calculate error
    if matched_cat_sorter is not None:
        for i in range(len(matched_cat_flux_cols)):
            flux_err[detected, i] = matched_balrog_cat['catalog/unsheared/{}'.format(
                matched_cat_flux_cols[i])][:][matched_cat_sorter][matched_idx]
            flux_err_report[detected, i] = matched_balrog_cat['catalog/unsheared/{}'.format(
                matched_cat_flux_err_cols[i])][:][matched_cat_sorter][matched_idx]

    else:
        for i in range(len(matched_cat_flux_cols)):
            flux_err[detected, i] = matched_balrog_cat['catalog/unsheared/{}'.format(
                matched_cat_flux_cols[i])][:][matched_idx]
            flux_err_report[detected, i] = matched_balrog_cat['catalog/unsheared/{}'.format(
                matched_cat_flux_err_cols[i])][:][matched_idx]
    # JTM 2021-08-23 comment out code that handles different mock zero point than data
    """
    for i in range(len(matched_cat_flux_cols)):
        # turn the flux stored in flux_err, which is the metacal flux of a balrog detection,  into a mag
        flux_err[detected, i] = zp_data - 2.5 * np.log10(flux_err[detected, i])
        # subtract off injection mag to yield difference between balrog output and injection mag
        flux_err[detected, i] -= detection_catalog['true_bdf_mag_deredden'][bal_cat_idx[detected.astype(
            np.bool)], true_cat_mag_cols[i]-1]
        # add difference between balrog output and injection mag to mag_in
        flux_out[detected, :] = mag_in[detected, :] + flux_err[detected, :]

        flux_out[~detected, i] = np.nan
        flux_err_report[~detected, i] = np.nan

    flux_out = 10**((flux_out - zp) / -2.5) # the flux_out on the RHS is a magnitude
    flux_err_report *= 10 ** ((zp_data - zp) / -2.5)
    """
    # JTM add 2021-08-23 for ZP offset PIT use
    # for which zero point is consistently 30
    flux_out = flux_err     

    # JTM add 2021-08-??
    # Build dataframe with detection catalog detected galaxies to determine which were selected
    df = pd.DataFrame({'flags_foreground': detection_catalog['flags_foreground'][bal_cat_idx[detected]].byteswap().newbyteorder(),
                       'flags_badregions' : detection_catalog['flags_badregions'][bal_cat_idx[detected]].byteswap().newbyteorder(),
                       'meas_FLAGS_GOLD_SOF_ONLY' : detection_catalog['meas_FLAGS_GOLD_SOF_ONLY'][bal_cat_idx[detected]].byteswap().newbyteorder(),
                       'flags_footprint' : detection_catalog['flags_footprint'][bal_cat_idx[detected]].byteswap().newbyteorder(),
                       'unsheared/snr' : snr_out[detected],
                       'unsheared/size_ratio' : size_ratio_out[detected],
                       'unsheared/flags' : flags_out[detected],
                       'unsheared/T' : T_out[detected],
                       'unsheared/e_1' : e_1_out[detected],
                       'unsheared/e_2' : e_2_out[detected],
                       'unsheared/flux_i' : flux_out[detected,1],
                       'unsheared/flux_r' : flux_out[detected,0],
                       'unsheared/flux_z' : flux_out[detected,2]})
    selected_len_detected = get_balrog_selection_mask(df)
    selected_len_all = np.zeros(len(mag_in))
    selected_len_all[detected] = np.copy(selected_len_detected)
    detected_but_not_selected = detected & ~selected_len_all.astype(np.bool)
    print('len(detection_catalog): {}'.format(len(detection_catalog)))
    print('len(mag_in)           : {}'.format(len(mag_in)))
    print('N detections          : {} = {} = {}'.format(np.sum(detected), len(df), len(selected_len_detected)))
    print('N selections          : {}'.format(np.sum(selected_len_detected)))
    print('N detected but not sel: {}'.format(np.sum(detected_but_not_selected)))
    for i in range(len(matched_cat_flux_cols)):
        flux_out[detected_but_not_selected,i] = np.nan
        flux_err_report[detected_but_not_selected,i] = np.nan

    if return_all_metacal_quantities:
        return (flux_out, flux_err_report, dist,
                snr_out, size_ratio_out, flags_out, T_out,
                e_1_out, e_2_out, R11_out, R22_out,
                injection_counts_out, weight_out, bal_id)
    return flux_out, flux_err_report, dist

def apply_nonuniform_errormodel(g, obase, odir, d, dhdr,
                                survey, magfile=None, usemags=None,
                                nest=False, bands=None, balrog_bands=None,
                                usebalmags=None, all_obs_fields=True,
                                dbase_style=True, use_lmag=True,
                                sigpz=0.03, blind_obs=False, filter_obs=True,
                                refbands=None, zp=22.5,
                                detection_catalog=None,
                                true_deep_cat=None,
                                matched_catalog=None,
                                matched_cat_sorter=None):

    if magfile is not None:
        mags = fitsio.read(magfile)
        if use_lmag:
            if ('LMAG' in mags.dtype.names) and (mags['LMAG']!=0).any():
                imtag = 'LMAG'
                omag = mags['LMAG']
            else:
                raise KeyError
        else:
            try:
                imtag = 'TMAG'
                omag = mags['TMAG']
            except:
                imtag = 'OMAG'
                omag = mags['OMAG']
    else:
        if use_lmag:
            if ('LMAG' in g.dtype.names) and (g['LMAG']!=0).any():
                imtag = 'LMAG'
                omag = g['LMAG']
            else:
                raise ValueError
        else:
            try:
                imtag = 'TMAG'
                omag = g['TMAG']
            except:
                imtag = 'OMAG'
                omag = g['OMAG']

    if balrog_bands is not None:
        apply_balrog_errors = True

    if dbase_style:
        mnames  = ['MAG_{0}'.format(b.upper()) for b in bands]
        menames = ['MAGERR_{0}'.format(b.upper()) for b in bands]
        fnames  = ['FLUX_{0}'.format(b.upper()) for b in bands]
        fenames = ['IVAR_{0}'.format(b.upper()) for b in bands]

        if apply_balrog_errors:
            '''
            bfnames = ['MCAL_FLUX_{0}'.format(b.upper()) for b in balrog_bands]
            bfenames = ['MCAL_IVAR_{0}'.format(b.upper()) for b in balrog_bands]
            '''
            bfnames = ['METACAL/flux_{0}'.format(b) for b in balrog_bands]
            bfenames = ['METACAL/flux_ivar_{0}'.format(b) for b in balrog_bands]


        if filter_obs & (refbands is not None):
            refnames = ['MAG_{}'.format(b.upper()) for b in refbands]
        elif filter_obs:
            refnames = mnames
    else:
        if filter_obs & (refbands is not None):
            refnames = refbands
        elif filter_obs:
            refnames = list(range(len(usemags)))


    fs = fname.split('.')
    #oname = "{0}/{1}_obs.{2}.fits".format(odir,obase,fs[-2])

    #get mags to use
    if usemags is None:
        nmag = omag.shape[1]
        usemags = list(range(nmag))
    else:
        nmag = len(usemags)

    #make output structure
    obs = make_output_structure(len(g), dbase_style=dbase_style, bands=bands,
                                nbands=len(usemags),
                                all_obs_fields=all_obs_fields,
                                blind_obs=blind_obs,
                                balrog_bands=balrog_bands)


    if ("Y1" in survey) | ("Y3" in survey) | (survey=="DES") | (survey=="SVA") | (survey=='Y3'):
        mindec = -90.
        maxdec = 90
        minra = 0.0
        maxra = 360.

    elif survey=="DR8":
        mindec = -20
        maxdec = 90
        minra = 0.0
        maxra = 360.

    maxtheta=(90.0-mindec)*np.pi/180.
    mintheta=(90.0-maxdec)*np.pi/180.
    minphi=minra*np.pi/180.
    maxphi=maxra*np.pi/180.

    #keep pixels in footprint
    #theta, phi = hp.pix2ang(dhdr['NSIDE'],d['HPIX'])
    #infp = np.where(((mintheta < theta) & (theta < maxtheta)) & ((minphi < phi) & (phi < maxphi)))
    #d = d[infp]

    #match galaxies to correct pixels of depthmap

    theta = (90-g['DEC'])*np.pi/180.
    phi   = (g['RA']*np.pi/180.)

    pix   = hp.ang2pix(dhdr['NSIDE'],theta, phi, nest=nest)

    guse = np.in1d(pix, d['HPIX'])
    guse, = np.where(guse==True)

    if not any(guse):
        print("No galaxies in this pixel are in the footprint")
        return

    pixind = d['HPIX'].searchsorted(pix[guse],side='right')
    pixind -= 1

    oidx = np.zeros(len(omag), dtype=bool)
    oidx[guse] = True

    if apply_balrog_errors:
        idx = np.zeros_like(omag, dtype=np.bool)
        for i in range(omag.shape[1]):
            if i not in usebalmags: continue
            idx[guse, i] = True

        flux_bal, fluxerr_bal = balrog_error_apply(detection_catalog,
                                                   true_deep_cat,
                                                   matched_catalog,
                                                   omag[idx].reshape(-1,len(usebalmags)),
                                                   matched_cat_sorter=matched_cat_sorter,
                                                   zp_data=zp,
                                                   true_cat_mag_cols=usebalmags)

    bal_idx = dict(zip(usebalmags, np.arange(len(usebalmags))))

    for ind,i in enumerate(usemags):
        # flux & fluxerr,assuming zp = 22.5, of each of injected galaxies at their new positions.
        flux, fluxerr = calc_nonuniform_errors(d['EXPTIMES'][pixind,ind],
                                               d['LIMMAGS'][pixind,ind],
                                               omag[guse,i], fluxmode=True,
                                               zp=zp)

        if not dbase_style:

            obs['OMAG'][:,ind] = 99
            obs['OMAGERR'][:,ind] = 99

            obs['FLUX'][guse,ind] = flux # zero point is 22.5 -- possibly misinterpreted later in pipeline
            obs['IVAR'][guse,ind] = 1/fluxerr**2
            obs['OMAG'][guse,ind] = 22.5 - 2.5*np.log10(flux)
            obs['OMAGERR'][guse,ind] = 1.086*fluxerr/flux

            bad = (flux<=0)

            obs['OMAG'][guse[bad],ind] = 99.0
            obs['OMAGERR'][guse[bad],ind] = 99.0

            r = np.random.rand(len(pixind))
            assert len(np.unique(d['FRACGOOD'])) <= 2 # may introduce OMAG/ flux discrepancy (OMAG 99 but flux is valid) if 0 < fracgood < 1

            if len(d['FRACGOOD'].shape)>1:
                bad = r>d['FRACGOOD'][pixind,ind]
            else:
                bad = r>d['FRACGOOD'][pixind]

            if len(bad)>0:
                obs['OMAG'][guse[bad],ind] = 99.0
                obs['OMAGERR'][guse[bad],ind] = 99.0

            if filter_obs and (ind in refnames):
                oidx &= obs['OMAG'][:,ind] < d['LIMMAGS'][pixind,ind]

        else:
            obs[mnames[ind]]  = 99.0
            obs[menames[ind]] = 99.0

            obs[fnames[ind]][guse]  = flux
            obs[fenames[ind]][guse] = 1/fluxerr**2
            obs[mnames[ind]][guse]  = 22.5 - 2.5*np.log10(flux)
            obs[menames[ind]][guse] = 1.086*fluxerr/flux

            bad = (flux<=0)

            obs[mnames[ind]][guse[bad]] = 99.0
            obs[menames[ind]][guse[bad]] = 99.0

            #Set fluxes, magnitudes of non detections to zero, 99
            ntobs = ~np.isfinite(flux)
            obs[fnames[ind]][guse[ntobs]]  = 0.0
            obs[fenames[ind]][guse[ntobs]] = 0.0
            obs[mnames[ind]][guse[ntobs]] = 99.0
            obs[mnames[ind]][guse[ntobs]] = 99.0

            if apply_balrog_errors:
                if i in usebalmags:
                    obs[bfnames[bal_idx[i]]][guse] = flux_bal[:, bal_idx[i]]
                    obs[bfenames[bal_idx[i]]][guse] = 1 / fluxerr_bal[:, bal_idx[i]]**2
                    bad = (flux_bal[:, bal_idx[i]] <= 0)
                    obs[bfnames[bal_idx[i]]][guse[bad]] = 0.0
                    obs[bfenames[bal_idx[i]]][guse[bad]] = 0.0
                    '''
                    obs[bfnames[bal_idx[ind]]][guse] = flux_bal[:, bal_idx[ind]]
                    obs[bfenames[bal_idx[ind]]][guse] = 1 / fluxerr_bal[:, bal_idx[ind]]**2
                    bad = (flux_bal[:, bal_idx[ind]] <= 0)
                    obs[bfnames[bal_idx[ind]]][guse[bad]] = 0.0
                    obs[bfenames[bal_idx[ind]]][guse[bad]] = 0.0
                    '''

            r = np.random.rand(len(pixind))

            if len(d['FRACGOOD'].shape)>1:
                bad = r>d['FRACGOOD'][pixind,ind]
            else:
                bad = r>d['FRACGOOD'][pixind]
            if any(bad):
                obs[mnames[ind]][guse[bad]]  = 99.0
                obs[menames[ind]][guse[bad]] = 99.0

            if (filter_obs) and (mnames[ind] in refnames):
                oidx[guse] &= obs[mnames[ind]][guse] < d['LIMMAGS'][pixind,ind]


    obs['RA']              = g['RA']
    obs['DEC']             = g['DEC']
    obs['ID']              = g['ID']
    #obs['EPSILON1']        = g['EPSILON'][:,0]
    #obs['EPSILON2']        = g['EPSILON'][:,1]
    #obs['SIZE']            = g['SIZE']
    obs['PHOTOZ_GAUSSIAN'] = g['Z'] + sigpz * (1 + g['Z']) * (np.random.randn(len(g)))

    if blind_obs:
        obs['M200']    = g['M200']
        obs['CENTRAL'] = g['CENTRAL']
        obs['Z']       = g['Z']

    fitsio.write(oname, obs, clobber=True)
    print('File saved at :', oname)

    if filter_obs:
        soname = oname.split('.')
        soname[-3] += '_rmp'
        roname = '.'.join(soname)
        fitsio.write(roname, obs[oidx], clobber=True)

    return oidx


def apply_uniform_errormodel(g, oname, survey, magfile=None, usemags=None,
                              bands=None, all_obs_fields=True,
                              dbase_style=True, use_lmag=True,
                              sigpz=0.03, blind_obs=False, filter_obs=True,
                              refbands=None, zp=22.5):

    if magfile is not None:
        mags = fitsio.read(magfile)
        if use_lmag:
            if ('LMAG' in mags.dtype.names) and (mags['LMAG']!=0).any():
                imtag = 'LMAG'
                omag = mags['LMAG']
            else:
                raise KeyError
        else:
            try:
                imtag = 'TMAG'
                omag = mags['TMAG']
            except:
                imtag = 'OMAG'
                omag = mags['OMAG']
    else:
        if use_lmag:
            if ('LMAG' in g.dtype.names) and (g['LMAG']!=0).any():
                imtag = 'LMAG'
                omag = g['LMAG']
            else:
                raise ValueError
        else:
            try:
                imtag = 'TMAG'
                omag = g['TMAG']
            except:
                imtag = 'OMAG'
                omag = g['OMAG']

    if dbase_style:
        mnames  = ['MAG_{0}'.format(b.upper()) for b in bands]
        menames = ['MAGERR_{0}'.format(b.upper()) for b in bands]
        fnames  = ['FLUX_{0}'.format(b.upper()) for b in bands]
        fenames = ['IVAR_{0}'.format(b.upper()) for b in bands]

        if filter_obs & (refbands is not None):
            refnames = ['MAG_{}'.format(b.upper()) for b in refbands]
        elif filter_obs:
            refnames = mnames
    else:
        if filter_obs & (refbands is not None):
            refnames = refbands
        elif filter_obs:
            refnames = list(range(len(usemags)))

    fs = fname.split('.')
    oname = "{0}/{1}_obs.{2}.fits".format(odir,obase,fs[-2])

    #get mags to use
    if usemags is None:
        nmag = omag.shape[1]
        usemags = list(range(nmag))
    else:
        nmag = len(usemags)

    #make output structure
    obs = make_output_structure(len(g), dbase_style=dbase_style, bands=bands,
                                nbands=len(usemags),
                                all_obs_fields=all_obs_fields,
                                blind_obs=blind_obs)

    if ("Y1" in survey) | (survey=="DES") | (survey=="SVA"):
        mindec = -90.
        maxdec = 90
        minra = 0.0
        maxra = 360.

    elif survey=="DR8":
        mindec = -20
        maxdec = 90
        minra = 0.0
        maxra = 360.

    maxtheta=(90.0-mindec)*np.pi/180.
    mintheta=(90.0-maxdec)*np.pi/180.
    minphi=minra*np.pi/180.
    maxphi=maxra*np.pi/180.

    maglims = np.array(models[model]['maglims'])
    exptimes = np.array(models[model]['exptimes'])
    lnscat = np.array(models[model]['lnscat'])

    oidx = np.ones(len(omag), dtype=bool)

    for ind,i in enumerate(usemags):

        _, _, flux, fluxerr = calc_uniform_errors(model, omag[:,i],
                                                  np.array([maglims[ind]]),
                                                  np.array([exptimes[ind]]),
                                                  np.array([lnscat[ind]]),
                                                  zp=zp)

        flux    = flux.reshape(len(flux))
        fluxerr = fluxerr.reshape(len(fluxerr))

        if not dbase_style:

            obs['FLUX'][:,ind] = flux
            obs['IVAR'][:,ind] = 1/fluxerr**2
            obs['OMAG'][:,ind] = 22.5 - 2.5*np.log10(flux)
            obs['OMAGERR'][:,ind] = 1.086*fluxerr/flux

            bad = (flux<=0)

            obs['OMAG'][bad,ind] = 99.0
            obs['OMAGERR'][bad,ind] = 99.0

            if filter_obs and (ind in refnames):
                oidx &= obs['OMAG'][:,ind] < maglims[ind]

        else:
            obs[mnames[ind]]  = 99.0
            obs[menames[ind]] = 99.0

            obs[fnames[ind]]  = flux
            obs[fenames[ind]] = 1/fluxerr**2
            obs[mnames[ind]]  = 22.5 - 2.5*np.log10(flux)
            obs[menames[ind]] = 1.086*fluxerr/flux

            bad = (flux<=0)

            obs[mnames[ind]][bad] = 99.0
            obs[menames[ind]][bad] = 99.0

            if (filter_obs) and (mnames[ind] in refnames):
                print('filtering {}'.format(mnames[ind]))
                oidx &= obs[mnames[ind]] < maglims[ind]
            else:
                print('mnames[ind]: {}'.format(mnames[ind]))

    print('filter_obs: {}'.format(filter_obs))
    print('refnames: {}'.format(refnames))
    print('maglims: {}'.format(maglims))
    print('oidx.any(): {}'.format(oidx.any()))

    obs['RA']              = g['RA']
    obs['DEC']             = g['DEC']
    obs['ID']           = g['ID']
    obs['EPSILON1']        = g['EPSILON'][:,0]
    obs['EPSILON2']        = g['EPSILON'][:,1]
    obs['SIZE']            = g['SIZE']
    obs['PHOTOZ_GAUSSIAN'] = g['Z'] + sigpz * (1 + g['Z']) * (np.random.randn(len(g)))

    if blind_obs:
        obs['M200']    = g['M200']
        obs['CENTRAL'] = g['CENTRAL']
        obs['Z']       = g['Z']

    fitsio.write(oname, obs, clobber=True)

    if filter_obs:
        soname = oname.split('.')
        soname[-3] += '_rmp'
        roname = '.'.join(soname)
        fitsio.write(roname, obs[oidx], clobber=True)


def setup_balrog_error_model(detection_file, matched_cat_file):

    detection_catalog = fitsio.read(detection_file,
                                    columns=['match_flag_1.5_asec',
                                             'true_id',
                                             'true_bdf_mag_deredden',
                                             'bal_id',
                                             'detected',
                                             'flags_footprint',
                                             'flags_foreground',
                                             'flags_badregions',
                                             'meas_FLAGS_GOLD_SOF_ONLY',
                                             'injection_counts'])

    detection_catalog, true_deep_cat = setup_deep_bal_cats(detection_catalog)
    matched_catalog = h5.File(matched_cat_file, 'r')

    bal_id_sidx = matched_catalog['catalog/unsheared/bal_id'][:].argsort()

    return detection_catalog, true_deep_cat, matched_catalog, bal_id_sidx


if __name__ == "__main__":

    t0 = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cfgfile = sys.argv[1]
    with open(cfgfile, 'r') as fp:
        cfg = yaml.load(fp)

    #####################################################################################################################
    job_id = sys.argv[2]
    realisations = int(sys.argv[3])
    print(time.time()-t0, 'Number of realisation of the deep fields : {0}'.format(realisations))
    #####################################################################################################################

    model = cfg['Model']
    obase = cfg['DeepFieldOutputBase']
    odir  = cfg['data_dir'] #+ '/{}/'.format(cfg['run_name'])
    GalBaseName  = cfg['GalBaseName']
    MagBaseName  = cfg['MagBaseName']
    gpath = '{0}/{1}.{2}.fits'.format(odir, GalBaseName, job_id)
    mpath = '{0}/{1}.{2}.fits'.format(odir, MagBaseName, job_id)

    #fnames = np.array(glob(gpath))
    fnames = [gpath]*realisations
    mnames = [mpath]*realisations
    print(time.time()-t0, 'List of files', fnames)
    print(time.time()-t0, 'List of files', mnames)

    if 'DepthFile' in list(cfg.keys()):
        dfile = cfg['DepthFile']
        uniform = False
        if 'Nest' in list(cfg.keys()):
            nest = bool(cfg['Nest'])
        else:
            nest = False
        print(time.time()-t0, 'Start opening depth map')
        d,dhdr = fitsio.read(dfile, header=True)
        print(time.time()-t0, 'Depth map opened')
        pidx = d['HPIX'].argsort()
        d = d[pidx]
    else:
        uniform = True

    if 'UseMags' in list(cfg.keys()):
        usemags = cfg['UseMags']
    else:
        usemags = None

    if ('DataBaseStyle' in list(cfg.keys())) & (cfg['DataBaseStyle']==True):
        if ('Bands' in list(cfg.keys())):
            dbstyle = True
            bands   = cfg['Bands']
        else:
            raise KeyError
    else:
        dbstyle = False

    if ('AllObsFields' in list(cfg.keys())):
        all_obs_fields = bool(cfg['AllObsFields'])
    else:
        all_obs_fields = True

    if ('BlindObs' in list(cfg.keys())):
        blind_obs = bool(cfg['BlindObs'])
    else:
        blind_obs = True

    if ('UseLMAG' in list(cfg.keys())):
        use_lmag = bool(cfg['UseLMAG'])
    else:
        use_lmag = False

    if ('FilterObs' in list(cfg.keys())):
        filter_obs = bool(cfg['FilterObs'])
    else:
        filter_obs = True

    if ('RefBands' in list(cfg.keys())):
        refbands = cfg['RefBands']
    else:
        refbands = None

    zp = cfg.pop('zp', 22.5)
    print('zp: {}'.format(zp))

    truth_only = cfg.pop('TruthOnly',False)

    if rank==0:
        try:
            os.makedirs(odir)
        except Exception as e:
            pass

    if ('RotOutDir' in list(cfg.keys())):
        if ('MatPath' in list(cfg.keys())):
            rodir = cfg['RotOutDir']
            robase = cfg['RotBase']
            rpath = cfg['MatPath']
            with open(rpath, 'r') as fp:
                rot    = pickle.load(fp)
            try:
                os.makedirs(rodir)
            except Exception as e:
                pass
        else:
            raise KeyError

    else:
        rodir = None
        rpath = None
        rot   = None
        robase= None

    if 'BalrogBands' in cfg.keys():
        balrog_bands = cfg['BalrogBands']
        usebalmags = cfg['UseBalMags']
        detection_file = cfg['DetectionFile']
        matched_cat_file = cfg['MatchedCatFile']

    else:
        balrog_bands = None
        usebalmags = None
        detection_file = None
        matched_cat_file = None

    print("Rank {0} assigned {1} files".format(rank, len(fnames[rank::size])))

    if balrog_bands is not None:
        detection_catalog, true_deep_cat, \
        matched_catalog, matched_cat_sorter = \
        setup_balrog_error_model(detection_file, matched_cat_file)
    else:
        detection_catalog = None
        true_deep_cat = None
        matched_catalog = None
        matched_cat_sorter = None

    for real, fname, mname in zip(np.arange(realisations)[rank::size], fnames[rank::size],mnames[rank::size]):
        print(time.time() - t0, '******Computing error on realisation {0}******'.format(real))
        if rodir is not None:
            p = fname.split('.')[-2]
            nfname = "{0}/{1}.{2}.fits".format(rodir,robase,p)
            g = rot_mock_file(fname,rot,nfname,
                    footprint=d,nside=dhdr['NSIDE'],nest=nest)

            #if returns none, no galaxies in footprint
            if g is None: continue
        else:
            g = fitsio.read(fname)

        #####################################################################################################################
        #find where the galaxies can be pasted
        #assign each galaxy a new injection position
        #hpix = hu.HealPix('ring', dhdr['NSIDE'])
        density = np.zeros(hp.nside2npix(dhdr['NSIDE']))
        density[d['HPIX']] = 1
        dmap = hu.DensityMap('ring', density)
        new_ra, new_dec = dmap.genrand(len(g))
        g['RA'] = new_ra
        g['DEC'] = new_dec
        print(time.time() - t0, 'new radec')
        ########################################################################################################################
        fs = fname.split('.')
        fp = fs[-2]

        ########################################################################################################################
        oname = "{0}/{1}_{3}.{4}.fits".format(odir, obase, model, fp, real)
        print(time.time() - t0, 'File will be saved at {0}'.format(oname))
        ########################################################################################################################
        if truth_only: continue

        if uniform:
            apply_uniform_errormodel(g, oname, model, magfile=mname,
                                     usemags=usemags,
                                     bands=bands,
                                     all_obs_fields=all_obs_fields,
                                     dbase_style=dbstyle,
                                     use_lmag=use_lmag,
                                     blind_obs=blind_obs,
                                     filter_obs=filter_obs,
                                     refbands=refbands,
                                     zp=zp)

        else:
            oidx = apply_nonuniform_errormodel(g, obase, odir, d, dhdr,
                                               model, magfile=mname,
                                               usemags=usemags,
                                               usebalmags=usebalmags,
                                               balrog_bands=balrog_bands,
                                               nest=nest, bands=bands,
                                               all_obs_fields=all_obs_fields,
                                               dbase_style=dbstyle,
                                               use_lmag=use_lmag,
                                               blind_obs=blind_obs,
                                               filter_obs=filter_obs,
                                               refbands=refbands,
                                               zp=zp,
                                               detection_catalog=detection_catalog,
                                               true_deep_cat=true_deep_cat,
                                               matched_catalog=matched_catalog,
                                               matched_cat_sorter=matched_cat_sorter)

        print(time.time() - t0, 'Error model computed on realisation {0}'.format(real))

    if rank==0:
        print("*******Rotation and error model complete!*******")
