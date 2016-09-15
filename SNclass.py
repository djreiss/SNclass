import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import gzip

import george
from george import kernels
from scipy.optimize import minimize

filters = [b'g', b'r', b'i', b'z']  # , b'Y']

__all__ = ("SNClass")

def SNPhotCC_Parser(filename):
    '''
    Reads and returns supernovae data and metadata.
    * filename is a string containing the path to the supernovae light curve data
    * survey is a string containing the survey name
    * snid is an integer containing the supernova ID
    * ra is a float containing the RA of the supernova
    * dec is a float containing the Dec of the supernova
    * mwebv is a float describing the dust extinction
    * hostid is an integer containing the host galaxy ID
    * hostz is an array of floats containing the photometric redshift of the galaxy and the error on the measurement
    * spec is an array of floats containing the redshift
    * sim_type is a string containing the supernova type
    * sim_z is a float containing the redshift of the supernova
    * obs is a pandas dataframe containing [observation time, filter, fluxes, fluxe errors]
    '''

    survey = snid = ra = decl = mwebv = hostid = hostz = spec = sim_type = sim_z = None
    with gzip.open(filename, 'rb') as f:
        for lineno,line in enumerate(f):
            s = line.split(':')
            if len(s) <= 0:
                continue
            if s[0] == 'SURVEY':
                survey = s[1].strip()
            elif s[0] == 'SNID':
                snid = int(s[1].strip())
            elif s[0] == 'SNTYPE':
                sn_type = int(s[1].strip())
            elif s[0] == 'RA':
                ra = float(s[1].split('deg')[0].strip())
            elif s[0] == 'DECL':
                decl = float(s[1].split('deg')[0].strip())
            elif s[0] == 'MWEBV':
                mwebv = float(s[1].split('MW')[0].strip())
            elif s[0] == 'HOST_GALAXY_GALID':
                hostid = int(s[1].strip())
            elif s[0] == 'HOST_GALAXY_PHOTO-Z':
                hostz = float(s[1].split('+-')[0].strip()), float(s[1].split('+-')[1].strip())
            elif s[0] == 'REDSHIFT_SPEC':
                spec = float(s[1].split('+-')[0].strip()), float(s[1].split('+-')[1].strip())
            elif s[0] == 'SIM_COMMENT':
                sim_type = s[1].split('SN Type =')[1].split(',')[0].strip()
            elif s[0] == 'SIM_REDSHIFT':
                sim_z = float(s[1])
            elif s[0] == 'VARLIST':
                break
    obs = pd.read_table(gzip.GzipFile(filename), sep='\s+', header=0, skiprows=lineno, skipfooter=1,
                        comment='DETECTION', engine='python',
                        usecols=[1,2,4,5,6,7,8,9])
    metadata = {'survey': survey, 'snid': snid, 'sn_type': sn_type, 'sim_type': sim_type,
                'sim_z': sim_z, 'ra': ra, 'decl': decl,
                'mwebv': mwebv, 'hostid': hostid, 'hostz': hostz, 'spec': spec, 'filename': filename}
    return obs, metadata


def SNPhot_fitter_filt(obs, filt=b'r', verbose=False, kernelMultiplier=5.):
    fname = None
    if isinstance(obs, str):  # assume it's a filename
        fname = obs
        obs, metadata = SNPhotCC_Parser(obs)
        metadata['filename'] = fname

    df = obs[obs.FLT == filt]
    #print fname, filt, df.shape
    x = df.MJD.values
    y = df.FLUXCAL.values
    dy = df.FLUXCALERR.values

    kernel = kernelMultiplier * np.var(y) * kernels.ExpSquaredKernel(100.)
    gp = george.GP(kernel)
    gp.compute(x, np.abs(dy))

    if verbose:
        print("Initial ln-likelihood: {0:.2f}".format(gp.lnlikelihood(y)))

    def neg_ln_like(p):
        gp.set_vector(p)
        return -gp.lnlikelihood(y)

    def grad_neg_ln_like(p):
        gp.set_vector(p)
        return -gp.grad_lnlikelihood(y)

    result = minimize(neg_ln_like, gp.get_vector(), jac=grad_neg_ln_like)
    if verbose:
        print(result)

    gp.set_vector(result.x)
    if verbose:
        print("\nFinal ln-likelihood: {0:.2f}".format(gp.lnlikelihood(y)))
    return gp


def SNPhot_fitter(obs, verbose=False):
    metadata = fname = None
    if isinstance(obs, str):  # assume it's a filename
        fname = obs
        obs, metadata = SNPhotCC_Parser(obs)
        metadata['filename'] = fname

    gps = {}
    for filt in filters:
        try:
            gps[filt] = SNPhot_fitter_filt(obs, filt, verbose=verbose)
        except:
            try:
                print 'ERROR(1): %s %s -- trying again' % (fname, filt)
                gps[filt] = SNPhot_fitter_filt(obs, filt, verbose=verbose, kernelMultiplier=10.)
            except:
                try:
                    print 'ERROR(2): %s %s -- trying again' % (fname, filt)
                    gps[filt] = SNPhot_fitter_filt(obs, filt, verbose=verbose, kernelMultiplier=25.)
                except:
                    print 'ERROR FATAL: %s %s' % (fname, filt)
    return obs, metadata, gps


def SNPhot_plotter_filt(obs, gp, filt=b'r'):
    df = obs[obs.FLT == filt]
    x = df.MJD.values
    y = df.FLUXCAL.values
    dy = df.FLUXCALERR.values
    colors = {b'g':'g', b'r':'r', b'i':'c', b'z':'m'}

    if gp is not None:
        x_pred = np.linspace(x.min()-100., x.max()+100., 1000)
        pred, pred_var = gp.predict(y, x_pred, return_var=True)
        #pred, pred_cov = gp.predict(y, x_pred, return_cov=True)
        plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var), color=colors[filt], alpha=0.2)
        plt.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)
        #plt.ylim((pred-np.sqrt(pred_var)*1.1).min(), (pred+np.sqrt(pred_var)*1.1).max())
    #else:
    plt.ylim((y-dy*1.1).min(), (y+dy*1.1).max())
    plt.errorbar(x, y, yerr=dy, fmt=".k", capsize=0)
    plt.xlim(x.min()-30, x.max()+30)
    plt.title(filt)
    return plt

def SNPhot_plotter(obs, gps):
    if gps is None:
        gps = dict(zip(filters, [None, None, None, None]))
    for i, filt in enumerate(filters):
        plt.subplot(2, 2, i+1)
        gp = None
        if filt in gps:
            gp = gps[filt]
        SNPhot_plotter_filt(obs, gp, filt)


class SNclass(object):
    def __init__(self, filename):
        self.obs, self.metadata, self.gps = SNPhot_fitter(filename)

    def fit(self, verbose=False):
        self.gps = SNPhot_fitter(self.obs, verbose=verbose)

    def plot(self):
        SNPhot_plotter(self.obs, self.gps)
