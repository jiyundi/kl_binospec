import numpy as np


def rescale_var(image_data, image_var_raw, image_mask=None, option=2):
    if image_mask is None:
        image_mask = np.ones(image_data.shape, dtype=bool)

    # OPTION 1: 75th percentile filter
    if option == 1:
        bkg_var_high = np.percentile(image_data[   image_mask], 75)
        raw_var_high = np.percentile(image_var_raw[image_mask], 75)
        bkg_pixels   = (np.where(image_mask, image_data, np.inf) < bkg_var_high) & image_mask

        raw_var_amp = raw_var_high
        noise_level = np.nanstd(image_data[bkg_pixels])

        image_var = image_var_raw / raw_var_amp * (noise_level)**2
        print(f'Variance of image has been rescaled by {1/raw_var_amp * noise_level**2:.2e}')

    # OPTION 2: avoid central flux
    if option == 2:
        mask_var    = np.ones(image_data.shape, dtype=bool)
        left, right = image_data.shape[1]//4, image_data.shape[1] - image_data.shape[1]//4
        up,   down  = image_data.shape[0]//4, image_data.shape[0] - image_data.shape[0]//4
        mask_var[up:down, left:right] = False

        raw_var_amp = np.nanmax(image_var_raw[mask_var & image_mask])
        gaussian_background_noise = np.nanstd(image_data[mask_var & image_mask])
        
        image_var = image_var_raw / raw_var_amp * gaussian_background_noise**2
        print(f'Variance of image has been rescaled by {1/raw_var_amp * gaussian_background_noise**2:.2e}')
    
    return image_var