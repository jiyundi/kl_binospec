import numpy as np


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
        
        # mask_sides  = np.ones(mask.shape, dtype=bool)
        # mask_sides[:,mask.shape[1]//4:-mask.shape[1]//4] = False
        # mask_bkgrnd = mask_sides & mask
        
        # # at least one entry are True
        # if mask_bkgrnd.any() == True: 
        #     data_masked = data[mask_bkgrnd]
        # # all entries are False
        # elif mask_bkgrnd.any() == False: 
        #     data_masked = data[mask]
        
        # noise_level = np.std(data_masked)
        # back_noise  = np.ones(data.shape) * noise_level
        # spec_var    = back_noise**2
    
    # Rescaled var may be: var + data <= 0. Mask out these px
    mask_add   = ((data + spec_var) > 0)
    mask_added = mask & mask_add
    
    return spec_var, mask_added