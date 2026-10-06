import numpy as np
from   astropy.io  import fits


def read_spec2d(spec2dfilepath):
    hdul = fits.open(spec2dfilepath)
    
    fdata = hdul[1].data['FLUX']
    wdata = hdul[1].data['LAMBDA']
    
    farr = fdata[0,:,:]
    warr = wdata[0,:,:]
    
    farr = np.nan_to_num(farr, nan=-200) # replace nan with negatives
    warr = np.nan_to_num(warr, nan=0)
    
    return {'flux': farr, 'wave': warr}


def readinfodat(infdatfilepath):
    infile_dat = open(infdatfilepath)
    dat_dict   = {}
    for sen in infile_dat:
        if sen[:2] != 'ID':
            dat_dict[sen.split('       ')[0]] = float(sen.split('       ')[1][:-1])
    return dat_dict


def stack_spec2d(arr2d01a, arr2d01b, arr2d01c, 
                 ivar01a=None, ivar01b=None, ivar01c=None):
    # stacked spec2d row size
    max_rows = np.max([arr2d01a.shape[0], arr2d01b.shape[0], arr2d01c.shape[0]])
    
    # fill extra pixels with -1000 (or 0 or NaN)
    def _pad_array(arr, max_rows):
        pad_width = ((0, max_rows - arr.shape[0]), (0, 0))
        return np.pad(arr, pad_width, mode='constant', constant_values=-1000)
    
    # padding
    arr2d01a_padded = _pad_array(arr2d01a, max_rows)
    arr2d01b_padded = _pad_array(arr2d01b, max_rows)
    arr2d01c_padded = _pad_array(arr2d01c, max_rows)
    if ivar01a is not None:
        ivar01a_padded = _pad_array(ivar01a, max_rows)
        ivar01b_padded = _pad_array(ivar01b, max_rows)
        ivar01c_padded = _pad_array(ivar01c, max_rows)
    
    # stacking
    if ivar01a is not None:
        stacked_spectrum = (arr2d01a_padded * ivar01a_padded + 
                            arr2d01b_padded * ivar01b_padded + 
                            arr2d01c_padded * ivar01c_padded) / \
                           (ivar01a_padded + ivar01b_padded + ivar01c_padded)
    else:
        stacked_spectrum = (arr2d01a_padded + arr2d01b_padded + arr2d01c_padded)/3
    
    return stacked_spectrum


def cutoutspec(spec2d, wave, width=30):
    lamb = spec2d['wave']
    mask = (lamb >= wave-width/2) & (lamb <= wave+width/2)
    
    spec_out = {}
    for key, arr in spec2d.items():
        spec_out[key] = arr[:, mask[0, :]]
    
    return spec_out


