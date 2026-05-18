import numpy as np


def stack_spec2d(arr2d01a, arr2d01b, arr2d01c, 
                 ivar01a=None, ivar01b=None, ivar01c=None):
    # stacked spec2d row size
    max_rows = max(arr2d01a.shape[0], arr2d01b.shape[0], arr2d01c.shape[0])
    
    # fill extra pixels with -1000 (or 0 or NaN)
    def pad_array(arr, max_rows):
        pad_width = ((0, max_rows - arr.shape[0]), (0, 0))
        return np.pad(arr, pad_width, mode='constant', constant_values=-1000)
    
    # padding
    arr2d01a_padded = pad_array(arr2d01a, max_rows)
    arr2d01b_padded = pad_array(arr2d01b, max_rows)
    arr2d01c_padded = pad_array(arr2d01c, max_rows)
    if ivar01a is not None:
        ivar01a_padded = pad_array(ivar01a, max_rows)
        ivar01b_padded = pad_array(ivar01b, max_rows)
        ivar01c_padded = pad_array(ivar01c, max_rows)
    
    # stacking
    if ivar01a is not None:
        stacked_spectrum = (arr2d01a_padded * ivar01a_padded + 
                            arr2d01b_padded * ivar01b_padded + 
                            arr2d01c_padded * ivar01c_padded) / \
                           (ivar01a_padded + ivar01b_padded + ivar01c_padded)
    else:
        stacked_spectrum = (arr2d01a_padded + arr2d01b_padded + arr2d01c_padded)/3
    
    return stacked_spectrum


def cutoffspec(spec2d_A, spec2d_B, spec2d_C, wave, width=30):
    wave_C = spec2d_C['wave']
    mask   = (wave_C >= wave-width/2) & (wave_C <= wave+width/2)
    for spec2d_dict in [spec2d_A, spec2d_B, spec2d_C]:
        for key, arr in spec2d_dict.items():
            spec2d_dict[key] = arr[:, mask[0, :]]
    return spec2d_A, spec2d_B, spec2d_C, mask



