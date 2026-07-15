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


def find_line_sigma(arr, line, lambda_scale=0.61, verbose=False):
    ny, nx = arr.shape
    
    # fit_func = _gaussian_nonzero
    x0_sigma_amp_1 = np.zeros((1+ny, 4)).astype(str)
    x0_sigma_amp_2 = np.zeros((1+ny, 4)).astype(str)
    x0_sigma_amp_1[0] = ['mean1', 'std1', 'amp1', 'shared_bkg']
    x0_sigma_amp_2[0] = ['mean2', 'std2', 'amp2', 'shared_bkg']

    # 3 rows together
    for y in range(ny):
        if y==0:
            data = np.concatenate(([arr[0]],  arr[0:2, :]), axis=0)
        elif y==ny-1:
            data = np.concatenate((arr[-2:, :], [arr[-1]]), axis=0)
        else:
            data = arr[y-1:y+2]

        xx_all = []
        yy_all = []
        row_id = []
        for i in range(3):
            nan_mask = np.isnan(data[i])

            xx = np.arange(nx)[~nan_mask]
            yy = data[i][~nan_mask]

            xx_all.extend(xx)
            yy_all.extend(yy)
            row_id.extend(np.ones_like(xx)*i)

        xx_all = np.array(xx_all)
        yy_all = np.array(yy_all)
        row_id = np.array(row_id)
        
        yy_some = yy_all[(yy_all < 3*np.std(yy_all)) & (yy_all > -1*np.std(yy_all))]
        # All pixels are noisy
        if len(yy_some) == 0:
            noise = np.std(yy_all) + 1
        else:
            noise = np.std(yy_some) + 1

        # -------- single-line model --------
        if not (line == 'O2'):
            def model(x, mean, std, amp1, amp2, amp3, y01, y02, y03):
                amps = np.array([amp1, amp2, amp3])
                y0s  = np.array([y01, y02, y03])
                irow = row_id.astype(int)
                return amps[irow] * np.exp(-(x-mean)**2/(2*std**2)) + y0s[irow]
    
            # params:  mean1, std1,    amp1,    amp1,    amp1,      y0,      y0,      y0
            bound1 = ((    2,    1,       0,       0,       0,  -noise,  -noise,  -noise), 
                      ( nx-2,    5,  np.inf,  np.inf,  np.inf,   noise,   noise,   noise) )
            
            # p0_1:  [  nx/2,    3,  amp1_1,   amp1_2, amp1_3,       0,       0,       0]
            def estimate_p0(xx_all, yy_all, row_id):
                mean_list, amp_list = [], []
                for r in np.unique(row_id):
                    mask = row_id == r
                    x_r, y_r = xx_all[mask], yy_all[mask]
                    peak_idx = np.argmax(y_r)
                    amp_list.append(y_r[peak_idx] if y_r[peak_idx]>=1e-2 else 1e-2)
                    mean_list.append(x_r[peak_idx])
                mean0 = np.mean(mean_list)   # shared mean
                if (mean0 < bound1[0][0]) or (mean0 > bound1[1][0]):
                    mean0 = (bound1[0][0] + bound1[1][0]) / 2
                
                return [mean0, 3.0] + amp_list + [1e-2, 1e-2, 1e-2]
            
            p0_1 = estimate_p0(xx_all, yy_all, row_id)
            for p0, p_min, p_max in zip(p0_1, bound1[0], bound1[1]):
                assert (p0 >= p_min) and (p0 <= p_max), \
                    f'Found p0 = {p0} outside of bounds ({p_min}, {p_max}). \np0_1 = {p0_1}. \nBounds = {bound1}.'
            #     print(f'Found p0 = {p0} in bounds ({p_min}, {p_max}). \np0_1 = {p0_1}. \nBounds = {bound1}.')
            # print('------------------------------------------------')
            try:
                popt,_ = curve_fit(model, xx_all, yy_all, 
                                   p0=p0_1, bounds=bound1, maxfev=10000)
            except RuntimeError:
                print(f'Row {y-1}-{y+1} failed. Skippped.')
                continue
    
            x0_sigma_amp_1[y+1] = popt[0], popt[1] * lambda_scale, popt[3], popt[6]
            x0_sigma_amp_2[y+1] = 0, 0, 0, 0
        
        else:
            # -------- double-line model --------
            def model_double(x, mean1, std1,  amp1,  amp2,  amp3,
                                dmean, std2, damp1, damp2, damp3,
                                y01, y02, y03):
                amps1  = np.array([ amp1,  amp2,  amp3])
                amps2  = np.array([damp1, damp2, damp3])
                y0s    = np.array([y01,   y02,   y03  ])
                irow   = row_id.astype(int)
                mean2  = mean1 + dmean
                g1 = amps1[irow] * np.exp(-(x - mean1)**2 / (2*std1**2))
                g2 = amps2[irow] * np.exp(-(x - mean2)**2 / (2*std2**2))
                return g1 + g2 + y0s[irow]

            # p: mean1, std1,   amp1,   amp2,   amp3, 
            #    dmean, std2,  damp1,  damp2,  damp3,     y01,     y02,     y03
            bound2 = (
                (    2,    1,      0,      0,      0,   
                     4,    1,      0,      0,      0,  -noise,  -noise,  -noise),
                ( nx-2,    5, np.inf, np.inf, np.inf, 
                    12,    5, np.inf, np.inf, np.inf,   noise,   noise,   noise)
            )

            def estimate_p0_double(xx_all, yy_all, row_id):
                amp_list = []
                mean1_list = []
                dmean_list = []
                for r in np.unique(row_id):
                    mask = row_id == r
                    x_r, y_r = xx_all[mask], yy_all[mask]
                    
                    # highest peak
                    peak1_idx = np.argmax(y_r)
                    peak1_x = x_r[peak1_idx]
                    
                    # Exclude ±6 pixels near highest peak, find 2nd heightest
                    y_r_masked = y_r.copy()
                    suppress = np.abs(x_r - peak1_x) < 6
                    y_r_masked[suppress] = 0
                    peak2_idx = np.argmax(y_r_masked)
                    peak2_x = x_r[peak2_idx]
                    
                    # mean1 = left peak's
                    left_x = min(peak1_x, peak2_x)
                    diff_x = np.abs(peak1_x - peak2_x)
                    amp_list.append(y_r[peak1_idx] if y_r[peak1_idx]>=1e-2 else 1e-2)
                    mean1_list.append(left_x)
                    dmean_list.append(diff_x)
                
                mean0 = np.mean(mean1_list)
                dmean = np.mean(dmean_list)
                if (mean0 < bound2[0][0]) or (mean0 > bound2[1][0]):
                    mean0 = bound2[0][0]
                if (dmean < bound2[0][5]) or (dmean > bound2[1][5]):
                    dmean = (bound2[1][5] - bound2[0][5]) / 2
                return [mean0, 3.0] + amp_list + [dmean, 3.0] + amp_list + [1e-2, 1e-2, 1e-2]

            p0_2 = estimate_p0_double(xx_all, yy_all, row_id)
            for p0, p_min, p_max in zip(p0_2, bound2[0], bound2[1]):
                assert (p0 >= p_min) and (p0 <= p_max), \
                    f'Found p0 = {p0} outside of bounds ({p_min}, {p_max}). \np0_2 = {p0_2}. \nBounds = {bound2}'
            try:
                popt, _ = curve_fit(model_double, xx_all, yy_all,
                                    p0=p0_2, bounds=bound2, maxfev=10000)
            except RuntimeError:
                print(f'Row {y-1}-{y+1} failed. Skipped.')
                continue

            mean1, std1 = popt[0], popt[1]
            dmean, std2  = popt[5], popt[6]
            amp1_row1,  amp2_row1  = popt[3],  popt[8] # pick middle (row=1)
            y0_row1                = popt[11]
            x0_sigma_amp_1[y+1] = mean1,         std1 * lambda_scale, amp1_row1, y0_row1
            x0_sigma_amp_2[y+1] = mean1 + dmean, std2 * lambda_scale, amp2_row1, y0_row1

    
    # Remove x0 completely hidden by masked region (x0-std -- x0+std)
    nan_mask = np.isnan(arr)
    for x0_sigma_amp in [x0_sigma_amp_1, x0_sigma_amp_2]:
        for y in range(1, len(x0_sigma_amp)):
            x0  = int(np.round(x0_sigma_amp[y, 0].astype(float)))
            std = int(         x0_sigma_amp[y, 1].astype(float) )
            accept = False
            for x in range(x0-std, x0+std+1):
                try:
                    accept_this_col = (~np.all(nan_mask[:, x]))
                except IndexError: 
                    accept_this_col = False
                accept |= accept_this_col
                
            # Overwrite amp = 0
            if not accept:
                x0_sigma_amp[y, 2] = 0 # as a reference -- directly modify the original
    
    return x0_sigma_amp_1, x0_sigma_amp_2
















def find_line_sigma__(arr, line, fit_mode='y0!=0', 
                    lambda_scale=0.61, verbose=False):
    ny,  nx  = arr.shape
    noise    = np.std(arr[(arr < 3*np.nanstd(arr)) & (arr > -1*np.nanstd(arr))])
    
    fit_func = _double_gaussian if line == "O2" else _gaussian
    
    if fit_mode == 'y0!=0':
        x0_sigma_amp_1 = np.zeros((1+ny, 4)).astype(str)
        x0_sigma_amp_2 = np.zeros((1+ny, 4)).astype(str)
        x0_sigma_amp_1[0] = ['mean1', 'std1', 'amp1', 'shared_bkg']
        x0_sigma_amp_2[0] = ['mean2', 'std2', 'amp2', 'shared_bkg']
        
        # params:  mean1,   std1,    amp1,      y0,
        bound1 = ((    2,      1,   noise,  -noise), 
                  ( nx-2,      6, 9*noise,   noise) )
        p0_1   = [  nx/2,      3, 1+noise,       0]
        
        # params:  mean1,   std1,    amp1,   dmean,   std2,    amp2,      y0
        bound2 = ((    2,      1,   noise,       4,      1,   noise,  -noise), 
                  ( nx-2,      6, 9*noise,      10,      6, 9*noise,   noise) )
        p0_2   = [  nx/3,      3, 1+noise,       5,      3, 1+noise,       0]
        
        fit_func = _double_gaussian_nonzero if line == "O2" else _gaussian_nonzero
        bounds = bound2 if line == "O2" else bound1
        p0     = p0_2   if line == "O2" else p0_1
    
    else:
        x0_sigma_amp_1 = np.zeros((1+ny, 3)).astype(str)
        x0_sigma_amp_2 = np.zeros((1+ny, 3)).astype(str)
        x0_sigma_amp_1[0] = ['mean1', 'std1', 'amp1']
        x0_sigma_amp_2[0] = ['mean2', 'std2', 'amp2']
        
        # params:  mean1,   std1,    amp1
        bound1 = ((    2,      1,   noise), 
                  ( nx-2,      6, 9*noise) )
        p0_1   = [  nx/2,      3, 1+noise]
        
        # params:  mean1,   std1,    amp1,   dmean,   std2,    amp2
        bound2 = ((    2,      1,   noise,       4,      1,   noise), 
                  ( nx-2,      6, 9*noise,      10,      6, 9*noise) )
        p0_2   = [  nx/3,      3, 1+noise,       5,      3, 1+noise]
    
        bounds = bound2 if line == "O2" else bound1
        p0     = p0_2   if line == "O2" else p0_1
    
    # === Before the for y in range(ny) loop, scan one line === 
    # ===           to determine the initial x0.            ===
    # Estimate the position of the emission line 
    # using the average value in the column direction
    col_sum = np.nansum(arr, axis=0)
    x0_init = np.argmax(col_sum) # rough-estimate of center
    
    if line == "O2":
        p0[0] = x0_init * 0.6 # mean1 rough-estimate
        p0[3] = 5             # dmean 保持
    else:
        p0[0] = x0_init       # mean1 rough-estimate
    
    # Warm start
    popt_prev = None
    
    for y in range(ny):
        if verbose:
            print(f'Fitting row #{y}...')
            
        # NaN check: If all values near the expected position of 
        # the emission line are NaN, skip the fitting for this line.
        expected_x0 = int(round(popt_prev[0])) if popt_prev is not None \
                      else int(round(x0_init))
        check_cols  = np.arange(max(0, expected_x0 - 2),
                                min(nx, expected_x0 + 3))
        if np.all(np.isnan(arr[y][check_cols])):
            print(f'[SKIP] Row #{y}: x0 region (x≈{expected_x0}) is all NaN, '
                  f'inheriting prev popt')
            if popt_prev is not None:
                if fit_mode == 'y0=0':
                    if line == "O2":
                        m1, s1, a1, dm, s2, a2 = popt_prev
                        m2 = m1 + dm
                        x0_sigma_amp_1[y+1] = m1, s1 * lambda_scale, a1
                        x0_sigma_amp_2[y+1] = m2, s2 * lambda_scale, a2
                    else:
                        m1, s1, a1 = popt_prev
                        x0_sigma_amp_1[y+1] = m1, s1 * lambda_scale, a1
                else:
                    if line == "O2":
                        m1, s1, a1, dm, s2, a2, y0 = popt_prev
                        m2 = m1 + dm
                        x0_sigma_amp_1[y+1] = m1, s1 * lambda_scale, a1, y0
                        x0_sigma_amp_2[y+1] = m2, s2 * lambda_scale, a2, y0
                    else:
                        m1, s1, a1, y0 = popt_prev
                        x0_sigma_amp_1[y+1] = m1, s1 * lambda_scale, a1, y0
            continue  # 跳过本行拟合，popt_prev 保持不变

        # Normal fitting process
        nan_mask = np.isnan(arr[y])
        xx = np.arange(nx)[~nan_mask]
        yy_=        arr[y][~nan_mask]
        
        # Panelty for outliers: greater uncertainty, less weights
        panelty_mask = (yy_> 3 * np.std(yy_)) | (yy_< -3 * np.std(yy_))
        panelty_idx  = [i for i, val in enumerate(panelty_mask) if val]
        xx    = np.delete(xx,    panelty_idx)
        yy    = np.delete(yy_,   panelty_idx)
        
        assert len(yy)!=0, \
            f'ValueError: Please check Row {y} and raw yy = \n{yy_}'
        
        if verbose and len(panelty_idx) != 0:
            print(f'[INFO] Outliers of #{y}: x = {panelty_idx}')
        
        # warm start: Use the result of the previous line as the initial guess
        p0_use = popt_prev.tolist() if popt_prev is not None else p0
        
        try:
            popt, _ = curve_fit(
                fit_func, xx, yy, 
                bounds=bounds, p0=p0_use, maxfev=10000
                )
        
        except RuntimeError:
            print(f'[INFO] Row #{y}: curve_fit failed. Try using least_squares...')
            
            def _residuals(p, x, y):
                return fit_func(x, *p) - y
            
            result = least_squares(
                _residuals, p0_use, args=(xx, yy),
                bounds=(bounds[0], bounds[1]),
                max_nfev=10000,
                loss='soft_l1', 
                f_scale=1.0
            )
            popt = result.x
            
        # Rationality check: x0 should not deviate too far from the previous row.
        if popt_prev is not None:
            x0_cur  = popt[0]
            x0_last = popt_prev[0]
            if abs(x0_cur - x0_last) > 5: # The threshold can be adjusted, for example, from 3 to 5 pixels.
                print(f'[WARN] Row #{y}: x0 jumped {x0_cur:.1f} -> {x0_last:.1f}, using prev popt as fallback')
                popt = popt_prev.copy()   # fallback to previous row
        
        if verbose:
            fig, ax = plt.subplots(figsize=(4,2))
            ax.scatter(xx, yy)
            ax.scatter(xx, fit_func(xx, *popt), label='fitted')
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
        
        popt_prev = popt.copy() # update: warm start
        
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
    
    print('[INFO] Line width fitting finished.')
    
    return x0_sigma_amp_1, x0_sigma_amp_2

