import joblib
import numpy as np
from   scipy.optimize import curve_fit, least_squares
from   scipy.ndimage  import median_filter
import matplotlib.pyplot as plt

from core.post_fitting import deduplicate_ordered, complete_fit_params
from core.make_config_dic import make_config_dic

def _gaussian(xx, 
              mean1, std, amp):
    return amp * np.exp(-0.5 * ((xx - mean1) / std) ** 2) # + y0


def _double_gaussian(xx, 
                     mean1, std1, amp1, dmean, std2, amp2):
    mean2 = mean1 + dmean
    yy1 = amp1 * np.exp(-0.5 * ((xx - mean1) / std1) ** 2)
    yy2 = amp2 * np.exp(-0.5 * ((xx - mean2) / std2) ** 2)
    return yy1 + yy2 # + y0


# def make_config_dic(linespecies, fitting_params, fid_params, 
#                     log10_Mstar=9.30, log10_Mstar_err=0.05, 
#                     use_line_profile=None):
#     config_dic = {
#         'galaxy_params': {
#             'obs_type':     'slit', 
#             'line_species':    linespecies, 
#             'log10_Mstar':     log10_Mstar, 
#             'log10_Mstar_err': log10_Mstar_err, 
#             'line_profile_path': use_line_profile,
#             }, 
#         'likelihood': {
#             'fit_image':  True, 
#             'fit_spec':   True, 
#             'set_non_analytic_prior': None,
#             'fid_params': fid_params
#             }, 
#         'TFprior': {
#             'use_TFprior': True, 
#             'log10_vTF':   None,
#             'sigmaTF':     None, 
#             'a':            None, 
#             'b':            None, 
#             'sigmaTF_intr': None, 
#             'relation':     None
#             }, 
#         'params': fitting_params,
#         'truevalues': None
#         }
#     return config_dic


def another_load_mock(pkl_folder='mock/', slit_num=95):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
        data_info = joblib.load(f)
    
    import galsim
    ap_wcs  = data_info['image']['par_meta']['ap_wcs']
    data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    
    return data_info


# Save mocks
def _pop_and_save(pkl_folder, slit_num, datainfo):
    slit_name = f'{slit_num:03d}'
    
    # Unfortunately, my galsim.wcs objects cannot be packed in PKL files.
    # To pack galsim.wcs, DELETE it before packing in PKL.
    # To read, always regenerate by using ap_wcs. (by JD)
    datainfo['image']['par_meta'].pop('wcs') # delete it!
    
    with open(f'{pkl_folder}pkl/slit_{slit_name}.pkl', "wb") as f:
        joblib.dump(datainfo, f)   
        
    return


def add_colorbar_by_alpha(ax, color, label='label', bar_low=0, bar_high=1):
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


def make_spec_from_best_fit(inference, best_fit_dict, fitting_par, 
                            i, set_num):
    # Resolve unique ordered line list from best_fit_dict keys
    lines = deduplicate_ordered(
        k.split('_')[0]
        for k in best_fit_dict
        if k.split('_')[0] != 'shared'
    )
    line = lines[i // 3]

    # Normalise fitting_par to flat form if a nested dict was passed in
    if 'shared_params' in fitting_par:
        fitting_par = complete_fit_params(
            fitting_par,
            inference.config.galaxy_params.line_species,
        )
    
    inference.spec_model[i]._init_observable(
        data_info['galaxy'],
        data_info['spec'][i]['par_meta'],
    )
    
    # Collapse per-set intensity keys (I01_specN → I01) for this row
    best_one_level = {
        **best_fit_dict['shared_params'],
        **best_fit_dict[f'{line}_params'],
    }
    for k in ('I01', 'I02'):
        best_one_level[k] = best_one_level.get(f'{k}_spec{set_num}')
    
    spec_fit = inference.spec_model[i].get_observable(best_one_level)
    
    return spec_fit


def spike_outlier(arr, window=5, threshold=2.5):
    """适合检测局部尖峰/突刺"""
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
        mad = np.median(np.abs(neighbors - np.median(neighbors)))  # MAD更鲁棒
        
        if mad == 0:
            continue
        
        score = abs(arr[i] - median) / mad
        is_outlier[i] = score > threshold
        if is_outlier[i]: 
            arr_cleaned[i] = np.mean(neighbors)

    return is_outlier, arr_cleaned


def find_line_sigma(arr, line, verbose=False):
    ny,  nx  = arr.shape
    x0_sigma_amp_1 = np.zeros((ny, 3))
    x0_sigma_amp_2 = np.zeros((ny, 3))
    
    fit_func = _double_gaussian if line == "O2" else _gaussian
    bound1 = ((    1,      0,      0), 
              (   nx, np.inf, np.inf) )
    p0_1   = [  nx/2,  nx/10,     10]
    # params:  mean1,   std1,   amp1,  dmean,   std2,   amp2
    bound2 = ((    1,      0,      0,      4,      0,      0), 
              (   nx, np.inf, np.inf,   nx/2, np.inf, np.inf) )
    p0_2   = [  nx/4,  nx/10,     10,   nx/4,  nx/10,     10]
    bounds = bound2 if line == "O2" else bound1
    p0     = p0_2   if line == "O2" else p0_1
    
    for y in range(ny):
        if verbose:
            print(f'Fitting row #{y}...')
            
        nan_mask = np.isnan(arr[y])
        xx = np.arange(nx)[~nan_mask]
        yy =        arr[y][~nan_mask]
        sigma = np.ones_like(yy) * np.exp((y-ny/2)**2/(2*ny))
        
        # Panelty for outliers: greater error, less weights
        panelty_mask = (np.abs(yy) > 3 * np.std(yy))
        panelty_idx  = [i for i, val in enumerate(panelty_mask) if val]
        xx    = np.delete(xx,    panelty_idx)
        yy    = np.delete(yy,    panelty_idx)
        sigma = np.delete(sigma, panelty_idx)
        
        if verbose:
            if len(panelty_idx) != 0:
                print(f'[INFO] Outliers of #{y}: x = {panelty_idx}')
        
        try:
            popt, _ = curve_fit(
                fit_func, xx, yy, 
                bounds=bounds, p0=p0, maxfev=10000
                )
        
        except RuntimeError:
            print(f'[INFO] Row #{y}: curve_fit failed. Try using least_squares...')
            
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
        
        if line == "O2":
            mean1, std1, amp1, dmean, std2, amp2 = popt
            mean2 = mean1 + dmean
            x0_sigma_amp_1[y] = mean1, std1, amp1
            x0_sigma_amp_2[y] = mean2, std2, amp2
        else:
            mean1, std1, amp1 = popt
            x0_sigma_amp_1[y] = mean1, std1, amp1
    
    # Rewrite outliers of sigma & amp with smoothed values
    _, sigmas1 = spike_outlier(x0_sigma_amp_1[:, 1], 
                               window=10, threshold=5)
    _, amps1   = spike_outlier(x0_sigma_amp_1[:, 2], 
                               window=10, threshold=5)
    x0_sigma_amp_1[:, 1] = sigmas1
    x0_sigma_amp_1[:, 2] = amps1
    
    if line == "O2":
        _, sigmas2 = spike_outlier(x0_sigma_amp_2[:, 1], 
                                   window=10, threshold=5)
        _, amps2   = spike_outlier(x0_sigma_amp_2[:, 2], 
                                   window=10, threshold=5)
        x0_sigma_amp_2[:, 1] = sigmas2
        x0_sigma_amp_2[:, 2] = amps2
    
    print('[INFO]: Line width fitting finished.')
    
    return x0_sigma_amp_1, x0_sigma_amp_2




if __name__ == '__main__':
    redo = True
    pkl_folder='../scripts_beta/binospec_pkl/'
    for slit_num in [95]: # , 97
        # Load
        data_info  = another_load_mock(pkl_folder, 
                                       slit_num=slit_num)
        
        smootheds, lw_restoreds, x0_sigma_amp_1s, x0_sigma_amp_2s = [], [], [], []
        for spec_idx in range(len(data_info['spec'])):
        
            # ABOVE: Before line profile
            # =================================================================
            # BELOW: Extract line profile
        
            spec_data  = data_info['spec'][spec_idx]['data']
            spec_mask  = data_info['spec'][spec_idx]['mask']
            
            line_species  = data_info['spec'][spec_idx]['par_meta']['line_species']
            line_sig_amps = data_info['spec'][spec_idx]['par_meta']['line_sig_amps']
            
            arr = np.where(spec_mask, spec_data, np.nan)
            
            patch_size = 3
            smoothed = median_filter(arr, size=patch_size)
            smootheds.append(smoothed)
            
            if redo:
                x0_sigma_amp_1, x0_sigma_amp_2 = find_line_sigma(
                    smoothed, line_species
                    )
                # data_info['spec'][spec_idx]['par_meta']['line_sig_amps'] = np.array([x0_sigma_amp_1, x0_sigma_amp_2])
                x0_sigma_amp_1s.append(x0_sigma_amp_1)
                x0_sigma_amp_2s.append(x0_sigma_amp_2)
                
            else:
                x0_sigma_amp_1, x0_sigma_amp_2 = data_info['spec'][spec_idx]['par_meta']['line_sig_amps']
            
            ny, nx = arr.shape
            lw_restored = np.tile([np.arange(nx)], (1, ny)).reshape(ny, nx)
            fit_func = _double_gaussian if line_species == "O2" else _gaussian
            for y in range(ny):
                params = x0_sigma_amp_1[y]
                if line_species == "O2":
                    params = np.append(params, x0_sigma_amp_2[y])
                    params[3] -= params[0] # mean2 -> dmean
                lw_restored[y] = fit_func(lw_restored[y], *params)
                
            lw_restoreds.append(lw_restored)
            
        del x0_sigma_amp_1, x0_sigma_amp_2
        
        # _pop_and_save(pkl_folder, slit_num, data_info)
        data_info = another_load_mock(pkl_folder, slit_num)
        
        # ABOVE: Extract line profile
        # =====================================================================
        # BELOW: Update line profile in inference class

        import yaml
        run_dir_new = f'../scripts_beta/Slit_{slit_num:03d}_original/'
        date_of_run_new = './'
        fiduci_yaml =  "../config/binospec_fid_params.yaml"
        fittin_yaml =  "../config/binospec_fitting_params.yaml"
        
        with open(fiduci_yaml, "r", encoding="utf-8") as file1:
            fid_params     = yaml.safe_load(file1)
        with open(fittin_yaml, "r", encoding="utf-8") as file2:
            fitting_params = yaml.safe_load(file2)
        
        linespecies = []
        for spec in data_info['spec']:
            linespecies.append(spec['par_meta']['line_species'])
        
        config_dic = make_config_dic(
            linespecies, fitting_params, fid_params, 
            log10_Mstar=data_info['galaxy']['log10_Mstar'], 
            log10_Mstar_err=data_info['galaxy']['log10_Mstar_err'],
            use_line_profile = 'meta', #   'extracted'
            )
        
        from klm.nautilus_sampler import NautilusSampler
        nautilus_sampler = NautilusSampler(data_info, config_dic)
        for par, prior in nautilus_sampler.config.params.prior.items(): print(par, prior)
        
        from core.fitting_result_utils import load_best_fit_json
        json_filename = run_dir_new + date_of_run_new + 'best_fit.json'
        _, best_fit_params, fitting_par = load_best_fit_json(
            nautilus_sampler, fitting_params, json_filename
            )
        
        # best_fit_params['O2_params']['I01_spec1'] /= 2
        # best_fit_params['O2_params']['I02_spec1'] /= 2
        # best_fit_params['O2_params']['I01_spec2'] /= 2
        # best_fit_params['O2_params']['I02_spec2'] /= 2
        # best_fit_params['O2_params']['I01_spec3'] /= 2
        # best_fit_params['O2_params']['I02_spec3'] /= 2
        
        # config_dic['galaxy_params']['line_profile_path'] = 'extracted'
        nautilus_sampler_new = NautilusSampler(data_info, config_dic)
        spec_fits_new = []
        for spec_idx in range(len(data_info['spec'])):
            spec_fit_new = make_spec_from_best_fit(
                nautilus_sampler_new, best_fit_params, fitting_params, 
                spec_idx, set_num=spec_idx+1
                )
            spec_fits_new.append(spec_fit_new)
            
            
            
            
            
            
        for spec_idx in range(len(data_info['spec'])):
            fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,8))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            ax[0,2].remove()
            
            spec_data  = data_info['spec'][spec_idx]['data']
            spec_mask  = data_info['spec'][spec_idx]['mask']
            
            im0 = ax[0,0].imshow(
                np.where(spec_mask, spec_data, np.nan), 
                aspect='auto', cmap='viridis')
            
            vmin = np.nanmin(smootheds[spec_idx])
            vmax = np.nanmax(smootheds[spec_idx])
            im1 = ax[1,0].imshow(
                np.where(spec_mask, smootheds[spec_idx], np.nan), 
                aspect='auto', cmap='viridis')
            im2 = ax[1,1].imshow(
                lw_restoreds[spec_idx], 
                aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            im3 = ax[1,2].imshow(
                np.where(spec_mask, smootheds[spec_idx] - lw_restoreds[spec_idx], np.nan), 
                aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
            # im5 = ax[1,1].imshow(np.where(spec_mask, spec_fits[spec_idx], np.nan), 
            #                      aspect='auto', cmap='viridis')
            # im6 = ax[1,2].imshow(np.where(spec_mask, spec_data - spec_fits[spec_idx], np.nan), 
            #                      aspect='auto', cmap='viridis')
            im4 = ax[2,0].imshow(np.where(spec_mask, spec_fits_new[spec_idx], np.nan), 
                                 aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            im5 = ax[2,1].imshow(np.where(spec_mask, spec_data - spec_fits_new[spec_idx], np.nan), 
                                 aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
            im6 = ax[2,2].imshow(np.where(spec_mask, smootheds[spec_idx] - spec_fits_new[spec_idx], np.nan), 
                                 aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
            plt.colorbar(im0, ax=ax[0,0])
            plt.colorbar(im1, ax=ax[1,0])
            plt.colorbar(im2, ax=ax[1,1])
            plt.colorbar(im3, ax=ax[1,2])
            plt.colorbar(im4, ax=ax[2,0])
            plt.colorbar(im5, ax=ax[2,1])
            plt.colorbar(im6, ax=ax[2,2])
            # plt.colorbar(im8, ax=ax[2,1])
            # plt.colorbar(im9, ax=ax[2,2])
            
            # x0_sigma_amp_1, x0_sigma_amp_2 = data_info['spec'][spec_idx]['par_meta']['line_sig_amps']
            x0_sigma_amp_1 = x0_sigma_amp_1s[spec_idx]
            x0_sigma_amp_2 = x0_sigma_amp_2s[spec_idx]
            max_alpha_1 = np.max(x0_sigma_amp_1[:, 2])
            max_alpha_2 = np.max(x0_sigma_amp_2[:, 2])
            max_alpha   = np.max([max_alpha_1, max_alpha_2])
            alphas_1 = x0_sigma_amp_1[:, 2] / max_alpha
            if x0_sigma_amp_2[0,1] != 0:
                alphas_2 = x0_sigma_amp_2[:, 2] / max_alpha
            
            for y in range(len(x0_sigma_amp_1)):
                ax[0,1].errorbar(
                    x0_sigma_amp_1[y, 0], 
                    y, 
                    xerr=x0_sigma_amp_1[y, 1], fmt='o', capsize=5, 
                    label='Line 1', color='blue', alpha=alphas_1[y]
                    )
                
                if x0_sigma_amp_2[0,1] != 0:
                    ax[0,1].errorbar(
                        x0_sigma_amp_2[y, 0], 
                        y, 
                        xerr=x0_sigma_amp_2[y, 1], fmt='x', capsize=5, 
                        label='Line 2', color='blue', alpha=alphas_2[y]
                        )
            
            add_colorbar_by_alpha(ax[0,1], 'blue', label='amp', 
                                  bar_low=0, bar_high=max_alpha)
            ax[0,1].invert_yaxis()
            ax[0,1].set_xlim(0, arr.shape[1])
            ax[0,1].set_title('line width fit')
            
            ax[0,0].set_title(f'#{slit_num} spec {spec_idx+1}: Data')
            ax[1,0].set_title(f'median filtered ({patch_size}x{patch_size})')
            ax[1,1].set_title('line width restored')
            ax[1,2].set_title('Filtered - restored')
            if redo:
                ax[2,0].set_title('Model by using new line profile')
            else:
                ax[2,0].set_title('Model by using current line profile')
            ax[2,1].set_title('Data - Model')
            ax[2,2].set_title('Filtered - Model')
            # ax[3,1].set_title('Best-fit model by line width profile\n(each x0 not fixed)')
            # ax[3,2].set_title('Residual by line width profile\n(each x0 not fixed)')
            plt.savefig(f'{slit_num}_{spec_idx+1}_line_profile_redo{redo}.jpg', dpi=100, bbox_inches='tight')

        print('Done.')

