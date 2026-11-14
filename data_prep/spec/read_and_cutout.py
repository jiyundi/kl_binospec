import numpy as np
from pathlib import Path
from   astropy.io  import fits


def read_spec2d(spec2dfilepath, with_variance=False):
    with fits.open(spec2dfilepath) as hdul:
        farr = hdul[1].data['FLUX'][0, :, :].copy()
        warr = hdul[1].data['LAMBDA'][0, :, :].copy()
    
    farr = np.nan_to_num(farr, nan=-200) # replace nan with negatives
    warr = np.nan_to_num(warr, nan=0)
    
    result = {'flux': farr, 'wave': warr}
    if with_variance:
        path = Path(spec2dfilepath)
        target = int(path.stem.split('.')[-1])
        folder = path.parent.parent
        with fits.open(folder / 'obj_counts_slits_lin.fits') as hdul:
            matches = [h for h in hdul[1:] if h.header.get('SLITTARG') == target]
            if len(matches) != 1:
                raise ValueError(f'Expected one counts extension for target {target}')
            counts = matches[0]
            # The split export clips flux to [-200, 10000], including NaN sentinels.
            reference = np.nan_to_num(np.clip(counts.data, -200, 10000), nan=-200)
            if reference.shape != farr.shape or not np.array_equal(reference, farr):
                raise ValueError(f'Split/counts pixel mismatch: {path}')
            keys = ('SLITTARG', 'SIDE', 'SLITID', 'CRPIX1', 'CRVAL1',
                    'CDELT1', 'CTYPE1', 'CUNIT1', 'CRPIX2', 'CRVAL2', 'CDELT2')
            with fits.open(folder / 'obj_counts_err_slits_lin.fits') as errors:
                err = errors[counts.name]
                if (err.data.shape != farr.shape or
                        any(err.header.get(k) != counts.header.get(k) for k in keys)):
                    raise ValueError(f'Counts/error coordinate mismatch: {path}')
                sigma = np.asarray(err.data, dtype=float)
                # Pipeline's sigma=1 floor is not a reliable noise measurement.
                result['var_total'] = np.where(
                    np.isfinite(sigma) & (sigma > 1), sigma**2, np.nan)
    return result


def stack_mean_variance(*variances):
    """Variance of independent spectra averaged by stack_spec2d (equal weights).

    Rows absent from any input remain invalid rather than treating padding as
    another measured exposure. Flux stacking and saved crop hashes are unchanged.
    """
    if not variances or any(v.ndim != 2 for v in variances):
        raise ValueError('Expected 2D input variances')
    if len({v.shape[1] for v in variances}) != 1:
        raise ValueError('Variance wavelength dimensions differ')
    max_rows = max(v.shape[0] for v in variances)
    padded = [np.pad(np.asarray(v, dtype=float),
                     ((0, max_rows-v.shape[0]), (0, 0)), constant_values=np.nan)
              for v in variances]
    return np.sum(padded, axis=0) / len(variances)**2


def readinfodat(infdatfilepath):
    dat_dict   = {}
    with open(infdatfilepath) as infile_dat:
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


