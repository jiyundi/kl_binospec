import numpy as np

from core.find_bkg import bkg_estimate


def snr_estimate(
    image, 
    bkg_is_subtracted=True, mock_bkg=None, restore_CCD_flux=True, 
    data_is_noisy=None, noise_var=0, 
    var_given=None
):
    if bkg_is_subtracted == False:
        data = image
        var_poisson = image
            
        if data_is_noisy == False: # common for simulated/modelled images
            var = var_poisson

        else: # data_is_noisy
            if noise_var == 0:
                raise TypeError(
                    f'Noise variance is required since you specified data is noisy.'
                    )

            if noise_var == 'extract': # common for real data, not knowing it
                noise_var = bkg_estimate(image)[2]**2

            var = var_poisson + noise_var

    else: # bkg_is_subtracted, common for Binospec data
        if mock_bkg is None:
            background_level = bkg_estimate(image)[0]
        else:
            background_level = mock_bkg
        
        if restore_CCD_flux:
            data = image - background_level
        else: 
            data = image

        var_poisson = image - background_level # must be restored the original CCD counts
            
        if data_is_noisy == False: # simulated/modelled images
            var = var_poisson

        else: # data noisy, common for Binospec data
            if var_given is not None: # common for Binospec data
                var = var_given

            else:
                if noise_var == 0:
                    raise TypeError(
                        f'Noise variance is required since you specified data is noisy.'
                        )

                if noise_var == 'extract': # common for real data, not knowing it
                    noise_var = bkg_estimate(image)[2]**2

                var = var_poisson + noise_var

    if np.nansum(var) > 0:
        return np.nansum(data) / np.sqrt(np.nansum(var))

    else:
        raise ValueError("S/N is not valid due to negative denominator.")