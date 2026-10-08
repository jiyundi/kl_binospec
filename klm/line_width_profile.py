import numpy as np
from   scipy.optimize import curve_fit

def find_line_sigma(arr, line, lambda_scale, verbose=False):
    """
    Please note lambda_scale is for converting pixels --> Angstrom.
    If you want mean and std in pixels, set lambda_scale = 1.
    """
    ny, nx = arr.shape
    
    x0_sigma_amp_1 = np.zeros((1+ny, 5)).astype(str)
    x0_sigma_amp_2 = np.zeros((1+ny, 5)).astype(str)
    x0_sigma_amp_1[0] = ['mean1', 'std1', 'amp1', 'shared_bkg', 'reliability']
    x0_sigma_amp_2[0] = ['mean2', 'std2', 'amp2', 'shared_bkg', 'reliability']

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

            try:
                popt,_ = curve_fit(model, xx_all, yy_all, 
                                   p0=p0_1, bounds=bound1, maxfev=10000)
            except RuntimeError:
                print(f'Row {y-1}-{y+1} failed. Skippped.')
                continue

            # Check if this row's fitted mean is close to edges
            if (popt[0] < bound1[0][0] + 1) or (popt[0] > bound1[1][0] - 1):
                reliability = False
            else:
                reliability = True

            # Note: mean1 in pixels, std1 in Angstrom
            x0_sigma_amp_1[y+1] = popt[0], popt[1] * lambda_scale, popt[3], popt[6], reliability
            x0_sigma_amp_2[y+1] = 0, 0, 0, 0, 0
        
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

            # Check if this row's fitted mean is close to edges
            if (popt[0] < bound2[0][0] + 1) or (popt[0] + popt[5] > bound2[1][0] - 1):
                reliability = False
            else:
                reliability = True
            
            # Note: mean1 in pixels, std1 in Angstrom
            mean1, std1 = popt[0], popt[1]
            dmean, std2  = popt[5], popt[6]
            amp1_row1,  amp2_row1  = popt[3],  popt[8] # pick middle (row=1)
            y0_row1                = popt[11]
            x0_sigma_amp_1[y+1] = mean1,         std1 * lambda_scale, amp1_row1, y0_row1, reliability
            x0_sigma_amp_2[y+1] = mean1 + dmean, std2 * lambda_scale, amp2_row1, y0_row1, reliability

    
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
    
    return x0_sigma_amp_1, x0_sigma_amp_2 # std in Angstrom


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


def exam_line_profile(LPFs, spec_data, spec_mask, 
                      line_species, lambda_scale, patch_size=3, filename='test_LPF.jpg'):
    from scipy.ndimage import median_filter
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

    arr = np.where(spec_mask, spec_data, np.nan)
    smoothed = median_filter(arr, size=patch_size)
    
    x0_sigma_amp_1, x0_sigma_amp_2 = LPFs
    
    # sigma Angstrom --> pixels for line restoration only
    xsa1_in_pixels = x0_sigma_amp_1.copy() # COPY: OTHERWISE YOU WILL 
    xsa2_in_pixels = x0_sigma_amp_2.copy() #       DESTROY x0_sigma_amp_1s
    xsa1_in_pixels[1:, 1] = xsa1_in_pixels[1:, 1].astype(float) / lambda_scale
    xsa2_in_pixels[1:, 1] = xsa2_in_pixels[1:, 1].astype(float) / lambda_scale
    
    ny, nx = arr.shape
    lw_restored = np.tile([np.arange(nx)], (1, ny)).reshape(ny, nx)
    
    fit_func = _double_gaussian_nonzero if line_species == "O2" else _gaussian_nonzero
    
    for y in range(ny):
        params = xsa1_in_pixels[y+1][:-1].astype(float)
        if line_species == "O2":
            params = np.append(params[:-1],  # keep one shared_bkg
                               xsa2_in_pixels[y+1]).astype(float)
            idx_mean1 = np.where(xsa1_in_pixels[0] == 'mean1')[0][0]
            idx_mean2 = np.where(xsa2_in_pixels[0] == 'mean2')[0][0] + xsa1_in_pixels.shape[1]-1
            params[idx_mean2] -= params[idx_mean1] # mean2 -> dmean
        lw_restored[y] = fit_func(lw_restored[y], *params)
    
    # ---------------------------------------------------
    # Plotting
    # ---------------------------------------------------
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,6))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    ax[0,2].remove()
    
    # Note: spec_data[0, 0] was defined at spec's lower-left corner
    im0 = ax[0,0].imshow(
        np.where(spec_mask, spec_data, np.nan), 
        aspect='auto', cmap='viridis', 
        origin='lower'
        )
    
    vmin = np.nanmin(smoothed)
    vmax = np.nanmax(smoothed)
    im1 = ax[1,0].imshow(
        np.where(spec_mask, smoothed, np.nan), 
        aspect='auto', cmap='viridis', 
        origin='lower'
        )
    im2 = ax[1,1].imshow(
        lw_restored, 
        aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, 
        origin='lower'
        )
    im3 = ax[1,2].imshow(
        np.where(spec_mask, smoothed - lw_restored, np.nan), 
        aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax, 
        origin='lower'
        )
    plt.colorbar(im0, ax=ax[0,0])
    plt.colorbar(im1, ax=ax[1,0])
    plt.colorbar(im2, ax=ax[1,1])
    plt.colorbar(im3, ax=ax[1,2])
    
    x0_sigma_amp_1 = x0_sigma_amp_1[1:].copy()
    x0_sigma_amp_2 = x0_sigma_amp_2[1:].copy()
    
    # sigma Angstrom --> pixels
    x0_sigma_amp_1[1:, 1] = x0_sigma_amp_1[1:, 1].astype(float) / lambda_scale
    x0_sigma_amp_2[1:, 1] = x0_sigma_amp_2[1:, 1].astype(float) / lambda_scale
    
    max_alpha_1 = np.max(x0_sigma_amp_1[:, 2].astype(float))
    max_alpha_2 = np.max(x0_sigma_amp_2[:, 2].astype(float))
    max_alpha   = np.max([max_alpha_1, max_alpha_2])
    alphas_1 = x0_sigma_amp_1[:, 2].astype(float) / max_alpha
    if x0_sigma_amp_2[0,2] != 0:
        alphas_2 = x0_sigma_amp_2[:, 2].astype(float) / max_alpha
    
    for y in range(len(x0_sigma_amp_1)):
        ax[0,1].errorbar(
            x0_sigma_amp_1[y, 0].astype(float), 
            y, 
            xerr=x0_sigma_amp_1[y, 1].astype(float), fmt='o', capsize=5, 
            label='Line 1', color='blue', alpha=alphas_1[y]
            )
        
        if x0_sigma_amp_2[0,1] != 0:
            ax[0,1].errorbar(
                x0_sigma_amp_2[y, 0].astype(float), 
                y, 
                xerr=x0_sigma_amp_2[y, 1].astype(float), fmt='x', capsize=5, 
                label='Line 2', color='blue', alpha=alphas_2[y]
                )
    
    add_colorbar_by_alpha(plt, fig, ax[0,1], 'blue', label='amp', 
                          bar_low=0, bar_high=max_alpha)
    
    # ax[0,1].invert_yaxis()
    ax[0,1].set_xlim(0, arr.shape[1])
    ax[0,1].set_title('line width fit')
    
    ax[0,0].set_title(f'Data')
    ax[1,0].set_title(f'median filtered ({patch_size}x{patch_size})')
    ax[1,1].set_title('line width restored')
    ax[1,2].set_title('Filtered - restored')

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    return


def add_colorbar_by_alpha(plt, fig, ax, color, label='label', bar_low=0, bar_high=1):
    # 创建一个只改变 alpha 的自定义 cmap，转换成 RGB，然后定义渐变
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    cdict = {
        'red':   [(0.0, rgb[0], rgb[0]), (1.0, rgb[0], rgb[0])],
        'green': [(0.0, rgb[1], rgb[1]), (1.0, rgb[1], rgb[1])],
        'blue':  [(0.0, rgb[2], rgb[2]), (1.0, rgb[2], rgb[2])],
        'alpha': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] # 这里控制透明度渐变
    }
    alpha_cmap = mcolors.LinearSegmentedColormap('AlphaMap', cdict)

    # 添加色条，创建归一化映射
    norm = mcolors.Normalize(vmin=bar_low, vmax=bar_high)
    
    # 创建 ScalarMappable 并传给 colorbar
    sm = plt.cm.ScalarMappable(cmap=alpha_cmap, norm=norm)
    sm.set_array([]) # 必须设置一个空数组

    # 在指定的子图 ax 旁边添加色条
    cbar = fig.colorbar(sm, ax=ax, label=label)
    
    return cbar
