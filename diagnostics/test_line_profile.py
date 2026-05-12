import joblib
import numpy as np
from   scipy.ndimage  import median_filter
import matplotlib.pyplot as plt

from core.post_fitting import deduplicate_ordered, complete_fit_params
from core.make_config_dic import make_config_dic
from core.line_width_profile import find_line_sigma, _gaussian, _double_gaussian


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





if __name__ == '__main__':
    redo = False #  True 
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
        run_dir_new = f'../scripts_beta/Slit_{slit_num:03d}/'
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
            use_line_profile = 'raw', #  'redo' 
            )
        
        from klm.nautilus_sampler import NautilusSampler
        from core.fitting_result_utils import load_best_fit_json
        
        # nautilus_sampler = NautilusSampler(data_info, config_dic)
        # for par, prior in nautilus_sampler.config.params.prior.items(): print(par, prior)
        nautilus_sampler_new = NautilusSampler(data_info, config_dic)
        
        json_filename = run_dir_new + date_of_run_new + 'best_fit.json'
        _, best_fit_params, fitting_par = load_best_fit_json(
            nautilus_sampler_new, fitting_params, json_filename
            )
        
        # config_dic['galaxy_params']['line_profile_path'] = 'extracted'
        spec_fits_new = []
        for spec_idx in range(len(data_info['spec'])):
            spec_fit_new = make_spec_from_best_fit(
                nautilus_sampler_new, best_fit_params, fitting_params, 
                spec_idx, set_num=spec_idx+1
                )
            spec_fits_new.append(spec_fit_new)
            
            
            
            
            
            
        for spec_idx in range(len(data_info['spec'])):
            fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10,12))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            ax[0,2].remove()
            ax[2,0].remove()
            ax[3,0].remove()
            
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
            im5 = ax[2,1].imshow(np.where(spec_mask, spec_fits_new[spec_idx], np.nan), 
                                 aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            im6 = ax[2,2].imshow(np.where(spec_mask, spec_data - spec_fits_new[spec_idx], np.nan), 
                                 aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
            im8 = ax[3,1].imshow(np.where(spec_mask, median_filter(spec_fits_new[spec_idx], size=patch_size), np.nan), 
                                 aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            im9 = ax[3,2].imshow(np.where(spec_mask, smootheds[spec_idx] - median_filter(spec_fits_new[spec_idx], size=patch_size), np.nan), 
                                 aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
            plt.colorbar(im0, ax=ax[0,0])
            plt.colorbar(im1, ax=ax[1,0])
            plt.colorbar(im2, ax=ax[1,1])
            plt.colorbar(im3, ax=ax[1,2])
            plt.colorbar(im5, ax=ax[2,1])
            plt.colorbar(im6, ax=ax[2,2])
            plt.colorbar(im8, ax=ax[3,1])
            plt.colorbar(im9, ax=ax[3,2])
            
            if redo:
                x0_sigma_amp_1 = x0_sigma_amp_1s[spec_idx]
                x0_sigma_amp_2 = x0_sigma_amp_2s[spec_idx]
            else:
                x0_sigma_amp_1, x0_sigma_amp_2 = data_info['spec'][spec_idx]['par_meta']['line_sig_amps']
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
                ax[2,1].set_title('Model by using new line profile')
            else:
                ax[2,1].set_title('Model by using current line profile')
            ax[2,2].set_title('Data - Model')
            ax[3,1].set_title(f'median filtered model ({patch_size}x{patch_size})')
            ax[3,2].set_title('Filtered data - Filtered model')
            plt.savefig(f'{slit_num}_{spec_idx+1}_line_profile_redo{redo}.jpg', dpi=100, bbox_inches='tight')

        print('Done.')

