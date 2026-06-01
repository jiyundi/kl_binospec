import numpy as np


def rescale_var(image_data, image_var_raw, image_mask=None):
    if image_mask is None:
        image_mask = np.ones(image_data.shape, dtype=bool)
    
    noise_high  = np.percentile(image_data[image_mask], 75)
    noise_level = np.std(image_data[
        (np.where(image_mask, image_data, np.inf) < noise_high) & image_mask
        ])
    
    var_high    = np.percentile(image_var_raw, 75)
    image_var   = image_var_raw / var_high * (noise_level)**2
    
    return image_var