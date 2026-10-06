import numpy  as np
from astropy.io import fits

def load_spec(slit_num_str, spec_scale=1E-17,
              binfac=5): # bin=1 == 0.61A/bin
    slit_num = int(slit_num_str)
    spec1dfolder   = 'D:/_RschArchives/RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles/'
    spec1dfilename = spec1dfolder + f'spec1d.829.{slit_num:03d}.{slit_num+100305:06d}.fits'
    spectrum     = read_spec1d(spec1dfilename, binfac)
    spectrum[1:] = spectrum[1:] * spec_scale
    spectrum     = spectrum.T

    return spectrum

def read_spec1d(spec1dfilepath, binfactor):
    hdulist = fits.open(spec1dfilepath)
    hdr01data = hdulist[1].data
    w = hdr01data["LAMBDA"][0]
    f = hdr01data["FLUX"][0]
    e = hdr01data["IVAR"][0]**(-1/2)
    arr_obs_wave = binning_1d(w, binfactor)
    arr_obs_flux = binning_1d(f, binfactor)
    arr_obs_erro = binning_1d(e, binfactor)
    arr = np.concatenate(([arr_obs_wave],[arr_obs_flux],[arr_obs_erro]), axis=0)
    return arr

def binning_1d(arr, binfac):
    """ Bins up two or three column spectral data by a specified factor. """
    binfac = int(binfac)
    nbins  = len(arr) // binfac
    binned = np.zeros((nbins, 1))
    for i in range(len(binned)):
        spec_slice = arr[i*binfac:(i+1)*binfac]
        binned[i] = np.mean(spec_slice)
    return binned[:,0]

