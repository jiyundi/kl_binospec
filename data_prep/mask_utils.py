import numpy as np
from   scipy.ndimage import median_filter


def mask_out_pixels(cutA, out_mask=None, 
                    idx=None, spec_width=20, spec_height=20, limit=90):
    # Regular mode: rectangular mask
    if idx is None: 
        if out_mask is None:
            print('Provide a mask (e.g. outmask=[7000, 7002]) given a range to mask out.')
            return cutA
        
        if isinstance(out_mask, list)==False:
            print('Provided mask unreadable. Error.')
            return cutA
        else:
            mask_before = cutA['mask']
            mask = np.ones(mask_before.shape, dtype=bool)
            
            # Basic mask for -500 - -3000 flux
            mask = np.where(cutA['flux']>-500, mask, False)
            
            for i in range(len(out_mask[0])):
                sky_min, sky_max = out_mask[0][i], out_mask[1][i]
                wave = cutA['wave'][0]
                in_sky = (wave >= sky_min) & (wave <= sky_max)
                mask   = np.where(in_sky, False, mask)
                # for key, spec in cutA.items():
                    # if   key == 'flux':
                    #     flux_clean = np.where(mask, spec, 0)
                    #     cutA[key]  = flux_clean
                    # elif key == 'ivar':
                    #     ivar_clean = np.where(mask, spec, 0.001)
                    #     cutA[key]  = ivar_clean
            mask_after   = mask & mask_before
            cutA['mask'] = mask_after
            return cutA
    
    # Manual mode: low-filter
    # elif not isinstance(limit, bool):
    #     # 用户输入中心坐标与矩形大小
    #     x0, y0 = idx[0], idx[1] # 中心像素

    #     # 计算矩形边界 + 创建矩形掩膜
    #     x1 = int(x0 - spec_width/2) - 2
    #     x2 = int(x0 + spec_width/2) + 2
    #     y1 = int(y0 - spec_height/2) - 2
    #     y2 = int(y0 + spec_height/2) + 2
        
    #     cut_ = cutA.copy()
    #     flux = np.where(cutA['flux']>0, cutA['flux'], 0)
    #     ivar = cutA['ivar']
        
    #     mask = np.zeros_like(flux, dtype=bool)
    #     mask[y1:y2, x1:x2] = True

    #     # 定义阈值（limit=90, 例如全图亮度第90百分位）
    #     threshold = np.percentile(flux, limit)
    #     final_mask = ~mask & (flux > threshold)

    #     # 对矩形外像素进行低通处理（归零超亮像素）
    #     filtered = flux.copy()
    #     filtered[final_mask] = 0
    #     ivar[final_mask] = 0.001
        
    #     cut_['flux'] = filtered
    #     cut_['ivar'] = ivar
    #     return cut_
    
    else: # if skip low-filter
        return cutA


# def mask_min_max_flux(flux, ivar=None, flux_min=0, flux_min_to=None, mask=None):
#     if flux_min_to is None:
#         flux_min_to = np.nan
    
#     # A rough filter by given min & max flux
#     if mask is None:
#         mask = (flux > flux_min)
#     else:
#         assert (mask.shape == flux.shape)
    
#     # Add mask to remove bad pixels
#     _, bad_px_mask = remove_bad_pixels(flux)
#     mask = mask & (~bad_px_mask)
    
#     # Overwrite masked flux
#     masked_flux = np.where(mask, flux, flux_min_to)
    
#     # Overwrite masked ivar
#     if ivar is None:
#         return masked_flux, None
#     if ivar is not None:
#         masked_ivar = np.where(mask, ivar, 0.001)
#         return masked_flux, masked_ivar


def remove_bad_pixels(image, window_size=5, k1=5.5):
    # 计算局部中位数与MAD (median absolute deviation)
    med = median_filter(image, size=window_size, mode='reflect')
    abs_dev = np.abs(image - med)
    mad = median_filter(abs_dev, size=window_size, mode='reflect')
    sigma_local = 1.4826 * mad  # 转换为等效sigma
    
    # 检测孤立坏像素（比单点值高，比邻域高）mask
    high = image > med + k1 * sigma_local
    
    return np.where(high, 0, image), high 


