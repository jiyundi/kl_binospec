import numpy as np
from   scipy.ndimage import median_filter


# Mask out sky lines' pixels
sky_wav_min  = [8378,   8785, 8428, 8411, 8427, 8835,   8463, 8999, 8825, 8834]
sky_wav_max  = [8382.5, 8789, 8432, 8416, 8432, 8837.5, 8467, 9003, 8828, 8838]

sky_wav_min += [8834, 8824, 7339, 8884, 9304, 9311, 9321, 8833, 8765, 8757]
sky_wav_max += [8838, 8829, 7342, 8889, 9308, 9315, 9326, 8838, 8770, 8763]

sky_wav_min += [7327, 7314, 8883, 8917, 8941, 8865, 8341, 7716, 8955, 7850]
sky_wav_max += [7330, 7318, 8888, 8921, 8944, 8870, 8345, 7722, 8959, 7854]

sky_wav_min += [8984, 7910, 7919, 7791, 7819, 7273, 7748, 7758, 7990, 8060]
sky_wav_max += [8989, 7916, 7923, 7796, 7823, 7277, 7752, 7761, 7994, 8063]

sky_wav_min += [8296, 7805, 8023, 8277, 8900, 7366, 7887, 8350, 8308, 8847]
sky_wav_max += [8300, 7810, 8027, 8281, 8904, 7370, 7891, 8353, 8311, 8851]

sky_wav_min += [9035, 8309, 6862, 7977, 7243, 7709, 8396, 7436, 8492, 7838]
sky_wav_max += [9039, 8312, 6865, 7980, 7246, 7714, 8399, 7439, 8494, 7842]

sky_wav_min += [8662, 8128]
sky_wav_max += [8666, 8133]


def mask_out_sky_lines(cutA, mask_before, 
                       out_mask=[sky_wav_min, sky_wav_max]):
    """
    Apply a rectangular mask bounded by lower and upper wavelength values
    
    Parameters
    ----------
    cutA : dict
        contains keys of 'flux', 'wave', 'mask'
    out_mask : list 
        list of 2. Size: (2, ) or (2, N), N >= 2. 
        Position 1: lower limits of the mask let. 
        Position 2: upper limits of the mask let. 
        Usage: for two masks of 7000--7002 and 9000--9002:
            mask_out_pixels(cutA.copy(), 
                            out_mask=[[7000, 9000], 
                                      [7002, 9002]])
    """
    mask = np.ones(mask_before.shape, dtype=bool)
    
    for i in range(len(out_mask[0])):
        sky_min, sky_max = out_mask[0][i], out_mask[1][i]
        wave = cutA['wave'][0]
        in_sky = (wave >= sky_min) & (wave <= sky_max)
        mask   = np.where(in_sky, False, mask)
    
    # Pixels that are not in sky masked regions
    mask_after = mask & mask_before
    
    return mask_after


def bad_pixel_mask(image, window_size=5, k1=6):
    # calculate median and MAD (median absolute deviation)
    med = median_filter(image, size=window_size, mode='reflect')
    abs_dev = np.abs(image - med)
    mad = median_filter(abs_dev, size=window_size, mode='reflect')
    sigma_local = 1.4826 * mad  # --> effective sigma
    
    # isolated bad pixel is detected: 
    # higher than single-point value and neighbor's value
    overexposed = ((image > med + k1 * sigma_local) | 
                   (image < med - k1 * sigma_local))
    
    return overexposed # mask whose True == bad pixels


if __name__ == '__main__':
    import joblib
    pkl_folder     = '../../scripts/binospec_pkl/'
    pkl_folder_new = '../../scripts/binospec_pkl_modified/'
    
    for slit_num in range(1, 143):
        try:
            with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as fi:
                data_info = joblib.load(fi)
                
        except FileNotFoundError:
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Slit {slit_num:3d} skipped because no PKL found.')
            continue
        
        for spec_data in data_info['spec']:
            spec_data['mask'] = mask_out_sky_lines(
                {'wave': spec_data['meta']['lambda_grid'].value}, 
                spec_data['mask'], 
                out_mask=[sky_wav_min, sky_wav_max]
                )
        
        with open(f'{pkl_folder_new}pkl/slit_{slit_num:03d}.pkl', "wb") as fo:
            joblib.dump(data_info, fo)
        
        import os
        os.chdir('../')
        
        # Plot
        from data_structure import RealDataInfo, ImageData, SpecData
        from plot import make_exam_plots
        specs_data_info_list = []
        for spec_data in data_info['spec']:
            specs_data_info_list.append(
                SpecData(
                    data = spec_data['data'],
                    var  = spec_data['var'],
                    cont = spec_data['cont'],
                    mask = spec_data['mask'],
                    meta = spec_data['meta']
                    )
                )
        image_data_info = ImageData(data = data_info['image']['data'], 
                                    mask = data_info['image']['mask'], 
                                    var  = data_info['image']['var'], 
                                    var_raw = data_info['image']['var_raw'], 
                                    meta = data_info['image']['meta'])
        data_info = RealDataInfo(spec   = specs_data_info_list,
                                 image  = image_data_info,
                                 galaxy = data_info['galaxy'])
        make_exam_plots(data_info, f'{slit_num:03d}', pkl_folder=pkl_folder_new[3:])
        
        os.chdir('./spec/')
        
        print( "\033[32m" + 'INFO   :' + "\033[0m " + 
              f'Slit {slit_num:3d} Done.')