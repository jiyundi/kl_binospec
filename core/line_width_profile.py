import numpy as np
from   scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt


def _gaussian(xx, 
              mean1, std, amp):
    return amp * np.exp(-0.5 * ((xx - mean1) / std) ** 2)


def _gaussian_nonzero(xx, 
                      mean1, std, amp, y0):
    return amp * np.exp(-0.5 * ((xx - mean1) / std) ** 2) + y0


def _double_gaussian(xx, 
                     mean1, std1, amp1, dmean, std2, amp2):
    mean2 = mean1 + dmean
    yy1 = amp1 * np.exp(-0.5 * ((xx - mean1) / std1) ** 2)
    yy2 = amp2 * np.exp(-0.5 * ((xx - mean2) / std2) ** 2)
    return yy1 + yy2


def _double_gaussian_nonzero(xx, 
                             mean1, std1, amp1, dmean, std2, amp2, y0):
    mean2 = mean1 + dmean
    yy1 = amp1 * np.exp(-0.5 * ((xx - mean1) / std1) ** 2)
    yy2 = amp2 * np.exp(-0.5 * ((xx - mean2) / std2) ** 2)
    return yy1 + yy2 + y0


def spike_outlier(arr, window=5, threshold=2.5):
    arr = np.array(arr, dtype=float)
    arr_cleaned = arr.copy()
    n = len(arr)
    is_outlier = np.zeros(n, dtype=bool)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbors = np.concatenate([arr[lo:i], arr[i+1:hi]])

        if len(neighbors) < 2:
            continue

        median = np.median(neighbors)
        mad = np.median(np.abs(neighbors - np.median(neighbors)))
        
        if mad == 0:
            continue
        
        score = abs(arr[i] - median) / mad
        is_outlier[i] = score > threshold
        if is_outlier[i]: 
            arr_cleaned[i] = np.mean(neighbors)

    return is_outlier, arr_cleaned


def find_line_sigma(arr, line, fit_mode, lambda_scale=0.24, verbose=False):
    ny,  nx  = arr.shape
    noise    = np.std(arr[(arr < 3*np.nanstd(arr)) & (arr > -1*np.nanstd(arr))])
    
    fit_func = _double_gaussian if line == "O2" else _gaussian
    
    if fit_mode == 'y0=0':
        x0_sigma_amp_1 = np.zeros((1+ny, 3)).astype(str)
        x0_sigma_amp_2 = np.zeros((1+ny, 3)).astype(str)
        x0_sigma_amp_1[0] = ['mean1', 'std1', 'amp1']
        x0_sigma_amp_2[0] = ['mean2', 'std2', 'amp2']
        
        # params:  mean1,   std1,    amp1
        bound1 = ((    4,      1,   noise), 
                  ( nx-4,   nx/2,  np.inf) )
        p0_1   = [  nx/2,  nx/10, 1+noise]
        
        # params:  mean1,   std1,    amp1,   dmean,   std2,    amp2
        bound2 = ((    4,      1,   noise,       4,      1,   noise), 
                  ( nx-4,   nx/2,  np.inf,      10,   nx/2,  np.inf) )
        p0_2   = [  nx/3,  nx/10, 1+noise,       5,  nx/10, 1+noise]
    
        bounds = bound2 if line == "O2" else bound1
        p0     = p0_2   if line == "O2" else p0_1
    
    else:
        x0_sigma_amp_1 = np.zeros((1+ny, 4)).astype(str)
        x0_sigma_amp_2 = np.zeros((1+ny, 4)).astype(str)
        x0_sigma_amp_1[0] = ['mean1', 'std1', 'amp1', 'shared_bkg']
        x0_sigma_amp_2[0] = ['mean2', 'std2', 'amp2', 'shared_bkg']
        
        # params:  mean1,   std1,    amp1,      y0,
        bound1 = ((    4,      1,   noise,  -noise), 
                  ( nx-4,   nx/2,  np.inf,   noise) )
        p0_1   = [  nx/2,   nx/8, 1+noise,       0]
        
        # params:  mean1,   std1,    amp1,   dmean,   std2,    amp2,      y0
        bound2 = ((    4,      1,   noise,       4,      1,   noise,  -noise), 
                  ( nx-4,   nx/2,  np.inf,      10,   nx/2,  np.inf,   noise) )
        p0_2   = [  nx/3,   nx/8, 1+noise,       5,   nx/8, 1+noise,       0]
        
        fit_func = _double_gaussian_nonzero if line == "O2" else _gaussian_nonzero
        bounds = bound2 if line == "O2" else bound1
        p0     = p0_2   if line == "O2" else p0_1
    
    for y in range(ny):
        if verbose:
            print(f'Fitting row #{y}...')
            
        nan_mask = np.isnan(arr[y])
        xx = np.arange(nx)[~nan_mask]
        yy =        arr[y][~nan_mask]
        # sigma = np.ones_like(yy) * np.exp((y-ny/2)**2/(2*ny)) # row uncertainty
        
        # Panelty for outliers: greater uncertainty, less weights
        panelty_mask = (yy > 3 * np.std(yy)) | (yy < -3 * np.std(yy))
        panelty_idx  = [i for i, val in enumerate(panelty_mask) if val]
        xx    = np.delete(xx,    panelty_idx)
        yy    = np.delete(yy,    panelty_idx)
        # sigma = np.delete(sigma, panelty_idx)
        
        if verbose:
            if len(panelty_idx) != 0:
                print(f'[INFO] Outliers of #{y}: x = {panelty_idx}')
        
        try:
            print(f'bounds: {bounds}')
            print(f'p0: {p0}')
            popt, _ = curve_fit(
                fit_func, xx, yy, 
                bounds=bounds, p0=p0, maxfev=10000
                )
        
        except RuntimeError:
            print(f'[INFO]: Row #{y}: curve_fit failed. Try using least_squares...')
            
            def _residuals(p, x, y):
                return fit_func(x, *p) - y
            
            result = least_squares(
                _residuals, p0, args=(xx, yy),
                bounds=(bounds[0], bounds[1]),
                max_nfev=10000,
                loss='soft_l1', 
                f_scale=1.0
            )
        
            popt = result.x
        
        if verbose:
            fig, ax = plt.subplots(figsize=(4,2))
            ax.scatter(xx, yy)
            ax.scatter(xx, fit_func(xx, *popt))
            plt.show()
            plt.close()
        
        if fit_mode == 'y0=0':
            if line == "O2":
                mean1, std1, amp1, dmean, std2, amp2 = popt
                mean2 = mean1 + dmean
                x0_sigma_amp_1[y+1] = mean1, std1 * lambda_scale, amp1
                x0_sigma_amp_2[y+1] = mean2, std2 * lambda_scale, amp2
            else:
                mean1, std1, amp1 = popt
                x0_sigma_amp_1[y+1] = mean1, std1 * lambda_scale, amp1
                
        else:
            if line == "O2":
                mean1, std1, amp1, dmean, std2, amp2, y0 = popt
                mean2 = mean1 + dmean
                x0_sigma_amp_1[y+1] = mean1, std1 * lambda_scale, amp1, y0
                x0_sigma_amp_2[y+1] = mean2, std2 * lambda_scale, amp2, y0
            else:
                mean1, std1, amp1, y0 = popt
                x0_sigma_amp_1[y+1] = mean1, std1 * lambda_scale, amp1, y0
    
    # Rewrite outliers of sigma & amp with smoothed values
    _, sigmas1 = spike_outlier(x0_sigma_amp_1[1:, 1], 
                               window=10, threshold=5)
    _, amps1   = spike_outlier(x0_sigma_amp_1[1:, 2], 
                               window=10, threshold=5)
    x0_sigma_amp_1[1:, 1] = sigmas1
    x0_sigma_amp_1[1:, 2] = amps1
    
    if line == "O2":
        _, sigmas2 = spike_outlier(x0_sigma_amp_2[1:, 1], 
                                   window=10, threshold=5)
        _, amps2   = spike_outlier(x0_sigma_amp_2[1:, 2], 
                                   window=10, threshold=5)
        x0_sigma_amp_2[1:, 1] = sigmas2 
        x0_sigma_amp_2[1:, 2] = amps2
    
    print('[INFO]: Line width fitting finished.')
    
    return x0_sigma_amp_1, x0_sigma_amp_2

