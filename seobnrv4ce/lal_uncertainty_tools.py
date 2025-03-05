
# Tools in this module require Roberto's modified lalsuite that allows
# for free choice of calibration parameters for SEOBNRv4.
#
# The purpose of these tools is to generate a database of waveforms
# with calibration parameters sampled from the posteriors used in
# Boh\'e et al (2017).
#

import h5py
import numpy as np
import pandas as pd

# Must use Roberto's modified lalsuite that allows for free choice of
# calibration posteriors.
import lal
import lalsimulation as LS

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy import stats
from scipy.interpolate import interp1d, Rbf
from numpy.polynomial.chebyshev import Chebyshev
import scipy

calibration_file = "../MCMC_Aug25.h5"
data_dir = r"../data/"

#
# Calibration parameter sampling tools
#

# Parameters for manual pruning of posteriors
cut_params = {'SXS_BBH_0180': [(0, .8, 0), (1, 3, 0)],
              'SXS_BBH_0004': [(1, 0, 0), (2, 50, 1)],
              'SXS_BBH_0005': [(0, .56, 0), (2, 0, 1)],
              'SXS_BBH_0007': [],
              'SXS_BBH_0013': [(0, .5, 0)],
              'SXS_BBH_0016': [(0, 2, 0)],
              'SXS_BBH_0019': [(0, 2, 0)],
              'SXS_BBH_0025': [(0, .6, 0)],
              'SXS_BBH_0030': [(1, 7.5, 0)],
              'SXS_BBH_0036': [(1, 0, 0)],
              'SXS_BBH_0045': [(1, 0, 0)],
              'SXS_BBH_0046': [(0, 2, 0)],
              'SXS_BBH_0047': [(0, 2, 0)],
              'SXS_BBH_0056': [(0, 1.7, 0)],
              'SXS_BBH_0060': [(1, 0, 0)],
              'SXS_BBH_0061': [(1, 60, 0)],
              'SXS_BBH_0063': [(0, 2, 0)],
              'SXS_BBH_0064': [(1, 80, 0)],
              'SXS_BBH_0065': [(0, 1, 0)],
              'SXS_BBH_0148': [(2, 50, 1)],
              'SXS_BBH_0149': [(1, -50, 0), (2, 0, 1), (3, 12 ,0)],
              'SXS_BBH_0150': [(0, .55, 0), (2, -50, 1), (3, 18, 0)],
              'SXS_BBH_0151': [(0, 2, 0)],
              'SXS_BBH_0152': [(0, .45, 0), (1, 80, 0)],
              'SXS_BBH_0153': [(0, .2, 0)],
              'SXS_BBH_0154': [(0, 2, 0)],
              'SXS_BBH_0155': [(0, .3, 0), (1, 0, 0)],
              'SXS_BBH_0156': [(0, .6, 0)],
              'SXS_BBH_0157': [(0, .2, 0), (1, 0, 0), (2, 25, 1)],
              'SXS_BBH_0158': [(0, .2, 0), (1, 0, 0), (2, 25, 1)],
              'SXS_BBH_0159': [(0, .75, 0)],
              'SXS_BBH_0160': [(0, .2, 0), (1, 0, 0)],
              'SXS_BBH_0166': [(0, 2, 0)],
              'SXS_BBH_0167': [(0, 2, 0)],
              'SXS_BBH_0169': [(1, 6, 0)],
              'SXS_BBH_0170': [(0, .6, 0)],
              'SXS_BBH_0172': [(0, .2, 0), (1, 0, 0), (2, 20, 1)],
              'SXS_BBH_0174': [(0, .7, 0)],
              'PRIVATE_BBH_0059': [(0, 2, 0)],
              'PRIVATE_BBH_0103': [(1, 6, 0)],
              'PRIVATE_BBH_0104': [(1, 7, 0)],
              'PRIVATE_BBH_0105': [(1, 7, 0)],
              'PRIVATE_BBH_0115': [(1, 7, 0)],
              'PRIVATE_BBH_0117': [(1, 8, 0)],
              'PRIVATE_BBH_0050': [(0, .7, 0)],
              'PRIVATE_BBH_0118': [(0, 1.8, 0)],
              'PRIVATE_BBH_0057': [(0, 2, 0)],
              'PRIVATE_BBH_0058': [(0, 2, 0)],
              'PRIVATE_BBH_0119': [(0, 2, 0)],
              'PRIVATE_BBH_0066': [(1, 0, 0), (3, 3, 0)],
              'PRIVATE_BBH_0067': [(0, .56, 0)],
              'PRIVATE_BBH_0068': [(2, 0, 1)],
              'PRIVATE_BBH_0069': [(0, 2, 0)],
              'PRIVATE_BBH_0070': [(0, 2, 0)],
              'PRIVATE_BBH_0075': [(1, 0, 0)],
              'PRIVATE_BBH_0060': [(1, -50, 1), (3, 12, 0)],
              'PRIVATE_BBH_0076': [(1, -50, 0), (3, 4, 0)],
              'PRIVATE_BBH_0077': [(2, 0, 1)],
              'PRIVATE_BBH_0078': [(0, .56, 0), (3, 12, 0)],
              'PRIVATE_BBH_0051': [(0, .5, 0)],
              'PRIVATE_BBH_0052': [(0, 2, 0)],
              'PRIVATE_BBH_0079': [(2, 0, 1)],
              'PRIVATE_BBH_0080': [(1, 0, 0), (2, 0, 1)],
              'PRIVATE_BBH_0121': [(1, 3, 0)],
              'PRIVATE_BBH_0081': [(1, -50, 0), (2, 0, 1)],
              'PRIVATE_BBH_0053': [(2, 100, 0)],
              'PRIVATE_BBH_0054': [(0, 2, 0)],
              'PRIVATE_BBH_0029': [(0, .5, 0)],
              'PRIVATE_BBH_0021': [(0, .7, 0)],
              'PRIVATE_BBH_0022': [(0, .7, 0)],
              'PRIVATE_BBH_0023': [(0, .7, 0)],
              'PRIVATE_BBH_0123': [(0, .7, 0)],
              'PRIVATE_BBH_0024': [(1, 50, 0)],
              'PRIVATE_BBH_0055': [(0, .75, 0)],
              'PRIVATE_BBH_0030': [(0, .75, 0)],
              'PRIVATE_BBH_0056': [(0, .7, 0), (1, 0, 0)],
              'PRIVATE_BBH_0031': [(0, .3, 0), (1, 0, 0)],
              'PRIVATE_BBH_0061': [(0, 2, 0)],
              'PRIVATE_BBH_0062': [(1, -50, 1)],
              'PRIVATE_BBH_0033': [(2, 0, 1)],
              'PRIVATE_BBH_0032': [(0, 2, 0)],
              'PRIVATE_BBH_0065': [(0, .6, 0), (1, 0, 0)],
              'PRIVATE_BBH_0034': [(0, .6, 0)],
              'PRIVATE_BBH_0035': [(2, 0, 1)],
              'PRIVATE_BBH_0071': [(1, -100, 0), (2, 0, 1), (3, 3, 0)],
              'PRIVATE_BBH_0036': [(0, .3, 0)],
              'PRIVATE_BBH_0072': [(0, .6, 0)],
              'PRIVATE_BBH_0073': [(0, 2, 0)],
              'PRIVATE_BBH_0074': [(0, .4, 0)],
              'PRIVATE_BBH_0082': [(0, .4, 0)],
              'PRIVATE_BBH_0083': [(1, -50, 1), (2, -80, 1)],
              'PRIVATE_BBH_0037': [(0, .5, 0)],
              'PRIVATE_BBH_0038': [(0, .5, 0)],
              'PRIVATE_BBH_0084': [(0, 2, 0)],
              'PRIVATE_BBH_0025': [(1, 40, 0)],
              'PRIVATE_BBH_0039': [(0, 2, 0)],
              'PRIVATE_BBH_0040': [(0, 2, 0)],
              'PRIVATE_BBH_0085': [(1, 50, 0)],
              'PRIVATE_BBH_0041': [(0, .56, 0)],
              'PRIVATE_BBH_0086': [(0, .55, 0)],
              'PRIVATE_BBH_0087': [(0, .55, 0)],
              'PRIVATE_BBH_0124': [(0, 2, 0)],
              'PRIVATE_BBH_0092': [(0, 2, 0)],
              'PRIVATE_BBH_0093': [(0, 2, 0)],
              'PRIVATE_BBH_0042': [(2, 35, 0)],
              'PRIVATE_BBH_0088': [(0, 2, 0)],
              'PRIVATE_BBH_0089': [(0, .6, 1), (1, -80, 0), (2, 60, 1), (3, 8, 0)],
              'PRIVATE_BBH_0090': [(0, 2, 0)],
              'PRIVATE_BBH_0043': [(0, 2, 0)],
              'PRIVATE_BBH_0094': [(0, 2, 0)],
              'PRIVATE_BBH_0096': [(0, 2, 0)],
              'PRIVATE_BBH_0044': [(0, .3, 1), (1, -70, 0), (2, 50, 1), (3, 15, 0)],
              'PRIVATE_BBH_0097': [(0, 2, 0)],
              'PRIVATE_BBH_0047': [(0, 2, 0)],
              'PRIVATE_BBH_0098': [(0, 2, 0)],
              'PRIVATE_BBH_0099': [(2, 40, 1)],
              'PRIVATE_BBH_0101': [(0, 2, 0)],
              'PRIVATE_BBH_0091': [(0, 2, 0)],
              'PRIVATE_BBH_0045': [(0, 2, 0)],
              'PRIVATE_BBH_0046': [(0, 2, 0)],
              'PRIVATE_BBH_0102': [(0, 2, 0)],
              'SXS_BBH_0205': [(1, 0, 0), (3, 15, 0)],
              'SXS_BBH_0203': [(0, 2, 0)],
              'SXS_BBH_0206': [(1, -50, 0), (3, 15, 0)],
              'SXS_BBH_0204': [(0, 2, 0)],
              'SXS_BBH_0207': [(1, 0, 0), (3, 18, 0)],
              'SXS_BBH_0202': [(0, 1.2, 0)],
              'IAN_BBH_0001': [(1, -20, 0), (2, 0, 1)],
              'IAN_BBH_0002': [(2, 0, 1)],
              'IAN_BBH_0003': [(1, 80, 0)],
              'IAN_BBH_0004': [(0, .6, 0)],
              'IAN_BBH_0006': [(0, .6, 0)],
              'IAN_BBH_0007': [(0, .6, 0)],
              'IAN_BBH_0008': [(0, .5, 0)],
              'IAN_BBH_0010': [(0, .5, 0)],
              'IAN_BBH_0011': [(0, .3, 0)],
              'SXS_BBH_0177': [(0, .18, 0), (1, 0, 0), (2, 20, 1)],
              'SXS_BBH_0178': [(0, .15, 0), (1, 0, 0), (2, 20, 1)],
              'SXS_BBH_0306': [(0, .5, 0)],
              'EAS_BBH_0006': [(0, 2, 0)],
              'EAS_BBH_0008': [(0, 2, 0)],
              'SXS_BBH_0000': [(0, 2, 0)],
              'EAS_BBH_0009': [(0, .3, 0), (1, 0, 0), (2, 0, 1)],
              'PRIVATE_BBH_0064': [(0, .3, 0)]}

def cut_extra(dat=None, param_idx=0, cut_line=1., mode=0):
    """ mode = 0 keeps the left part, = 1 keeps right part
        - For nonspinning: param_idx = 0 for K, = 1 for dNQC
        - For spinning: param_idx = 0,1,2,3 for K,dSO,dSS,dNQC
    """
    if mode == 0:
        return np.array([x for x in dat if x[param_idx] < cut_line])
    else:
        return np.array([x for x in dat if x[param_idx] > cut_line])

def cut_full(cfg, h5file=calibration_file, verbose=False):
    f = h5py.File(h5file, 'r')
    g = f[cfg]
    dat = g['chain']
    if verbose:
        print((' *** Original length:  %d' % len(dat)))
    # Normal cut
    dat = dat[int(len(dat)/2):]  # burnin
    dat = np.array([list(x) for x in dat])
    dat = np.array([x for x in dat if x[-4] >= .99 and np.fabs(x[-1]) <= 5])
    #labels = ['K', 'dSO', 'dSS', 'dNQC']
    #if g.attrs['mode'] == b'nonspinning':
    #    dat = dat[:, [0,3]]
    #    labels = ['K', 'dNQC']
    if verbose:
        print((' *** After normal cut: %d' % len(dat)))
    # Extra cut
    if cfg in list(cut_params.keys()):
        for (param_idx, cut_line, cut_mode) in cut_params[cfg]:
            dat = cut_extra(dat, param_idx, cut_line, cut_mode)
    if verbose:
        print((' *** After extra cut:  %d' % len(dat)))
    f.close()
    return dat

def cut_extra_df(df, param_idx=0, cut_line=1., mode=0):
    """ mode = 0 keeps the left part, = 1 keeps right part
        - For nonspinning: param_idx = 0 for K, = 1 for dNQC
        - For spinning: param_idx = 0,1,2,3 for K,dSO,dSS,dNQC

        Variant that uses a dataframe instead of an array.
    """
    Q = df.columns[param_idx]
    if mode == 0:
        return df[df[Q] < cut_line]
    else:
        return df[df[Q] > cut_line]

def full_cut_df(grp, cfg):
    """Function that applies the cuts as in `cut_full()`,
    but here the h5py object is passed in by the user and
    the function returns a DataFrame instead of an array.
    """
    df = pd.DataFrame(grp['chain'][:])

    # Normal cut
    df = df.iloc[int(len(df)/2):]
    df = df[(df['match_min'] >= 0.99) & (np.abs(df['dT']) <= 5.0)]

    # Extra cut
    if cfg in list(cut_params.keys()):
        for (param_idx, cut_line, cut_mode) in cut_params[cfg]:
            df = cut_extra_df(df, param_idx, cut_line, cut_mode)
    return df

def GetIntrinsicParameters(h5file=calibration_file, cfg='SXS_BBH_0205'):
    fp = h5py.File(h5file, "r")
    q = fp[cfg].attrs['q']
    chi1 = fp[cfg].attrs['chi1']
    chi2 = fp[cfg].attrs['chi2']
    fp.close()

    return q, chi1, chi2

def GetCalibrationRandomChoicesFullCut(h5file=calibration_file,
                                       cfg='SXS_BBH_0205', nsamples=1):
    data_cut = cut_full(cfg=cfg, h5file=h5file)
    samples = []
    for n in range(nsamples):
        i = np.random.randint(0, len(data_cut))
        sample = data_cut[i].tolist()
        samples.append((sample[0], sample[1], sample[2], sample[3], sample[9]))
    return samples

#
# Waveform generation tools
#

def SEOBNRv4FD(Mtot, q, chi1, chi2, CalPars=None,
               deltaF=0.1, f_min=20.0, f_max=2048.0,
               r=100e6*lal.PC_SI, inc=0.0, phiC=0.0):
    """Generate SEOBNRv4 FD waveform with arbitrary calibration
    parameters.

    """

    m1SI = Mtot*q/(1.0+q) * lal.MSUN_SI
    m2SI = Mtot/(1.0+q) * lal.MSUN_SI

    if CalPars:
        assert len(CalPars) == 4
        KappaCal, dSOCal, dSSCal, DT22Cal = CalPars
        LALParams = lal.CreateDict()
        LS.SimInspiralWaveformParamsInsertKappaCal(LALParams, KappaCal)
        LS.SimInspiralWaveformParamsInsertdSOCal(LALParams, dSOCal)
        LS.SimInspiralWaveformParamsInsertdSSCal(LALParams, dSSCal)
        LS.SimInspiralWaveformParamsInsertDT22Cal(LALParams, DT22Cal)
    else:
        LALParams = None

    f_ref = f_min
    longAscNodes, eccentricity, meanPerAno = 0.0, 0.0, 0.0
    Hp, Hc = LS.SimInspiralFD(m1SI, m2SI,
                     0, 0, chi1, 0, 0, chi2,
                     r, inc, phiC,
                     longAscNodes, eccentricity, meanPerAno,
                     deltaF, f_min, f_max, f_ref,
                     LALParams, LS.SEOBNRv4) # LS.SEOBNRv4_ROM

    f = np.arange(Hp.data.length)*Hp.deltaF
    H = Hp.data.data + 1j*Hc.data.data
    amp = np.abs(H)
    phi = np.unwrap(np.angle(H))
    idx = np.nonzero(amp)
    ampI = spline(f[idx], amp[idx])
    phiI = spline(f[idx], phi[idx])

    return f, ampI, phiI, H

def rescale_Mtot(Mtot_new, Mtot_old, f_old, ampI_old, phiI_old, H_old,
                 deltaF_new=0.1, f_min_new=20.0, f_max_new=2048.0):
    """Given a frequency domain waveform, rescale frequency to give a new
    waveform at a different total mass.

    """

    idx = np.nonzero(H_old)[0]

    f_min_min = f_old[idx[0]]*Mtot_old/Mtot_new
    f_max_max = f_old[-1]*Mtot_old/Mtot_new

    if f_min_min > f_min_new:
        print("Error: f_min_new is too low.")
        return
    elif f_max_max < f_max_new:
        print("Error: f_max_new is too high.")
        return
    else:
        f_new = np.arange(int(round(f_max_new/deltaF_new))+1)*deltaF_new
        H_new = ampI_old(f_new * Mtot_new / Mtot_old) \
                * np.exp(1j * phiI_old(f_new * Mtot_new / Mtot_old))
        H_new[f_new < f_min_new] = 0.0

        return f_new, H_new

def match_improved(h1, h2, psdfun, f, f_min=20.0, zpf=5):
    """
    Compute the match between FD waveforms h1, h2

    :param h1, h2: data from frequency series [which start at f=0Hz]
    :param psdfun: power spectral density as a function of frequency in Hz
    :param zpf:    zero-padding factor
    """
    assert(len(h1) == len(h2))
    n = len(h1)
    psd_ratio = psdfun(100) / np.array(list(map(psdfun, f)))
    psd_ratio[0] = psd_ratio[1] # get rid of psdfun(0) = nan

    h1_z = np.copy(h1)
    h2_z = np.copy(h2)

    # Zero the waveforms for frequencies < f_min
    h1_z[f < f_min] = 0.0
    h2_z[f < f_min] = 0.0

    h1abs = np.abs(h1_z)
    h2abs = np.abs(h2_z)
    norm1 = np.dot(h1abs, h1abs*psd_ratio)
    norm2 = np.dot(h2abs, h2abs*psd_ratio)
    integrand = h1_z * h2_z.conj() * psd_ratio # different name!

    # New array length should be a power of 2, approximately (2*zpf +
    # 1) times the old length New length = 2^a
    a = int(round( np.log2(len(integrand)) + np.log2(2*zpf + 1) ))
    pad_total = int(2.**a - len(integrand))
    pad_left = int(pad_total/2)
    pad_right = pad_total - pad_left

    integrand_zp = np.concatenate([np.zeros(pad_left), integrand,
                                   np.zeros(pad_right)])
    csnr = np.asarray(np.fft.fft(integrand_zp)) # complex snr

    return np.max(np.abs(csnr)) / np.sqrt(norm1*norm2) # This is
                                                       # really the
                                                       # "faithfulness"

#
#  Build the model of uncertainty
#

def create_samples_db(Mtot, ndraws=50, f_min=20.0, f_max=2048.0,
                      deltaF=0.01, f_pts=1000):
    """Sample many waveforms for each point in physical parameter space,
    and save them in a .hdf5 file for later use.

    This saves the amplitude and phase sampled at f_pts frequencies
    between f_min and f_max.

    """

    fn = r'samples_Mtot-{0}_fmin-{1}_ndraws-{2}.hdf5'.format(Mtot,
                                                             f_min, ndraws)
    fn = data_dir + fn
    fs = h5py.File(fn, 'w')

    fs.attrs['Mtot'] = Mtot
    fs.attrs['f_min'] = f_min
    fs.attrs['f_max'] = f_max
    fs.attrs['deltaF'] = deltaF

    f_grid = np.linspace(f_min, f_max, f_pts)

    for n, cfg in enumerate(cfgs):
        cfg_grp = fs.create_group(cfg)

        q, chi1, chi2 = GetIntrinsicParameters(h5file=cal_file, cfg=cfg)
        cfg_grp.attrs['q'] = q
        cfg_grp.attrs['chi1'] = chi1
        cfg_grp.attrs['chi2'] = chi2

        sample_pars = \
            GetCalibrationRandomChoicesFullCut(h5file=calibration_file,
                                               cfg=cfg, nsamples=ndraws)
        match_mins = [x[-1] for x in sample_pars]
        cal_draws = [x[0:4] for x in sample_pars]

        cfg_grp.create_dataset("cal_pars", data=np.array(cal_draws),
                               compression="gzip", compression_opts=9)

        # Generate SEOBNRv4 waveform
        f_eob, ampI_eob, phiI_eob, H_eob \
            = SEOBNRv4FD(Mtot, q, chi1, chi2, CalPars=None, deltaF=deltaF,
                         f_min=f_min, f_max=f_max)

        cfg_grp.create_dataset('f_grid', data=f_grid, compression="gzip",
                               compression_opts=9)
        cfg_grp.create_dataset('amp_EOB', data=ampI_eob(f_grid),
                               compression="gzip", compression_opts=9)
        cfg_grp.create_dataset('phi_EOB', data=phiI_eob(f_grid),
                               compression="gzip", compression_opts=9)

        # Generate drawn waveforms
        unfaithfulness_array = []
        amp_array = []
        phi_array = []
        for i, CalParsRnd in enumerate(cal_draws):
            print(("Generating waveform " + str(i+1) + " of "
                   + str(ndraws) + ' for cfg ' + str(n+1) + ' of '
                   + str(len(cfgs)) + "   \r",))
            f_draw, ampI_draw, phiI_draw, H_draw \
                = SEOBNRv4FD(Mtot, q, chi1, chi2, CalPars=CalParsRnd,
                             deltaF=deltaF, f_min=f_min, f_max=f_max)
            unfaithfulness = \
                1. - match_improved(H_eob, H_draw,
                                    LS.SimNoisePSDaLIGOZeroDetHighPower,
                                    f_draw, f_min=f_min, zpf=5)
            amp_array.append(ampI_draw(f_grid))
            phi_array.append(phiI_draw(f_grid))
            unfaithfulness_array.append(unfaithfulness)

        cfg_grp.create_dataset('unfaithfulness_vs_eob',
                               data=np.array(unfaithfulness_array),
                               compression="gzip", compression_opts=9)
        cfg_grp.create_dataset('amp', data=np.array(amp_array),
                               compression="gzip", compression_opts=9)
        cfg_grp.create_dataset('phi', data=np.array(phi_array),
                               compression="gzip", compression_opts=9)

    fs.close()

    return
