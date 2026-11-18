import numpy as np


def from_total_variance(data, mask, total_variance):
    """
    Keep the legacy data + var contract while using pipeline total variance.
    """
    total = np.asarray(total_variance, dtype=float)
    if data.shape != total.shape or mask.shape != data.shape:
        raise ValueError('Data, mask and total variance must have identical shapes')
    
    valid = np.isfinite(data) & np.isfinite(total) & (total > 0)

    variance = np.zeros(data.shape, dtype=float)
    variance[valid] = total[valid] # - data[valid]

    return variance, mask & valid


def build_var(data, mask, option=2): 
    """ This generates a uniform Gaussian noise level for all pixels. """
    
    # OPTION 1: 75th percentile filter
    if option == 1:
        data_masked = data[mask]
        noise_high  = np.percentile(data_masked, 75)
        noise_level = np.std(data[
            (np.where(mask, data, np.inf) < noise_high) & mask
            ])
        
        back_noise  = np.ones(data.shape) * noise_level
        spec_var    = back_noise**2
    
    # OPTION 2: avoid central emission line
    elif option == 2:
        left, right = data.shape[1]//4, data.shape[1] - data.shape[1]//4
        up,   down  = data.shape[0]//4, data.shape[0] - data.shape[0]//4
        
        mask_var = np.ones(data.shape, dtype=bool)
        mask_var[up:down, left:right] = False
        
        if np.sum(mask_var & mask) <= 4:
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 'SNR & sky mask skipped all pixels. Gauss background noise may be not robust.')
            gaussian_background_noise = np.nanstd(data[mask_var       ])
        else:
            gaussian_background_noise = np.nanstd(data[mask_var & mask])
        
        spec_var = np.ones(data.shape) * gaussian_background_noise**2
    
    # Rescaled var may be: var + data <= 0. Mask out these px
    mask_add   = ((data + spec_var) > 0)
    mask_added = mask & mask_add
    
    return spec_var, mask_added


def rescale_var(spec_data, spec_var_raw, spec_mask=None, option=2):
    if spec_mask is None:
        spec_mask = np.ones(spec_data.shape, dtype=bool)

    # OPTION 2: avoid central flux
    if option == 2:
        mask_var    = np.ones(spec_data.shape, dtype=bool)
        left, right = spec_data.shape[1]//4, spec_data.shape[1] - spec_data.shape[1]//4
        up,   down  = spec_data.shape[0]//4, spec_data.shape[0] - spec_data.shape[0]//4
        mask_var[up:down, left:right] = False

        raw_var_amp = np.nanmax(spec_var_raw[mask_var & spec_mask])
        gaussian_background_noise = np.nanstd(spec_data[mask_var & spec_mask])

        rescale_fac = 1/raw_var_amp * gaussian_background_noise**2
        assert rescale_fac > 0, f'Rescale factor is negative: {rescale_fac}'
        spec_var = spec_var_raw * rescale_fac
        print(f'Variance of spec has been rescaled by {rescale_fac:.2e}')
    
    return spec_var