import json
# import h5py
import yaml
import numpy as np
from klm.parameters import Parameters
from klm.safe_plot import setup; setup() # must before plt
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
plt.style.use('default')
# plt.rcParams.update({
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
#     "font.serif": "Helvetica",
# })
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
# plt.rcParams['mathtext.rm'] = 'Helvetica' # 'cursive' not found

# Don't mess up with "complete_flattened_fit_params()" (👇)
def complete_fit_params(fitting_params, line_species, 
                        need_sorted_flattened=True):
    lines        = list(dict.fromkeys(line_species))
    doublet_lines  = ['O2']
    doub_pars      = ['v_0_2','dx_vel_2','dy_vel_2','I02','f2_1','f2_2']
    doub_pars_twin = ['v_0',  'dx_vel',  'dy_vel',  'I01','f1_1','f1_2']
    
    fitting_par = {}
    for line in lines: fitting_par[f'{line}_params'] = {}
    
    for key, dic in fitting_params.items():
        if key == 'shared_params':
            fitting_par[key] = dic
        
        elif key == 'line_params':
            for pname_w_spec, prior_dic in dic.items():
                for line in lines:
                    line_p = f'{pname_w_spec}'
                    fitting_par[f'{line}_params'][line_p] = prior_dic
                    
                    if line in doublet_lines:
                        if 'spec' in pname_w_spec.split('_')[1]: # I01_spec1
                            pname_wo_spec = pname_w_spec.split('_')[0]
                            i_q    = doub_pars_twin.index(pname_wo_spec)
                            q2name = doub_pars[i_q]
                            line_q = f'{q2name}_{pname_w_spec.split("_")[1]}'
                            fitting_par[f'{line}_params'][line_q] = prior_dic
                            
                        else: # v_0
                            i_q    = doub_pars_twin.index(pname_w_spec)
                            q2name = doub_pars[i_q]
                            line_q = f'{q2name}'
                            fitting_par[f'{line}_params'][line_q] = prior_dic
    
    if need_sorted_flattened:
        # --- flatten ---
        flattened_shared_keys = []
        for k1, subdict in fitting_par.items():
            if k1 == 'shared_params':
                for k2 in subdict.keys():
                    flattened_shared_keys.append(f"{k1}-{k2}")
        
        # --- flatten ---
        flattened_line_keys = []
        for k1, subdict in fitting_par.items():
            if k1 in [f'{line}_params' for line in lines]:
                for k2 in subdict.keys():
                    flattened_line_keys.append((f"{k1}-{k2}", k2))
        
        # 按二级 key 的首次出现顺序排序
        seen_order = []
        for _, k2 in flattened_line_keys:
            if k2 not in seen_order:
                seen_order.append(k2)
        line_order = [f"{line}_params" for line in lines]
        flattened = sorted(
            flattened_line_keys,
            key=lambda x: (
                line_order.index(x[0].split('-')[0]),  # 先按 line 排
                seen_order.index(x[1])                # 再按参数顺序排
            )
        )
        
        # 对O2等双线的第二线参数进行插队
        flattened_linekeys = []
        skip_next = set()
        
        for i, key_par in enumerate(flattened):
            key = key_par[0]
            if key in skip_next:
                continue
        
            flattened_linekeys.append(key)
        
            # 处理 O2 的双线参数
            if key.startswith("O2_params-"):
                base = key.split("-", 1)[1]
        
                # 特殊规则：v_0 → v_0_2
                if base == "v_0" and "O2_params-v_0_2" in flattened:
                    flattened_linekeys.append("O2_params-v_0_2")
                    skip_next.add("O2_params-v_0_2")
        
                # 规则：I01_specN → I02_specN
                if base.startswith("I01_spec"):
                    i02 = key.replace("I01_", "I02_")
                    if i02 in flattened:
                        flattened_linekeys.append(i02)
                        skip_next.add(i02)
        
        # Append shared_params. Assign back latex and prior values
        flattened_keys = flattened_shared_keys + flattened_linekeys
        sorted_par = {}
        for key in flattened_keys: 
            sorted_par[key] = fitting_par[key.split('-')[0]][key.split('-')[1]]
            
        fitting_par = sorted_par
    
    return fitting_par


# Don't mess up with "complete_fit_params()" (ABOVE👆)
def complete_flattened_fit_params(fitting_params_flat, line_species):
    lines          = list(dict.fromkeys(line_species))
    doublet_lines  = ['O2']
    doub_pars      = ['v_0_2','dx_vel_2','dy_vel_2','I02','f2_1','f2_2']
    doub_pars_twin = ['v_0',  'dx_vel',  'dy_vel',  'I01','f1_1','f1_2']
    fitting_par    = {}
    pars_this_line = {}
    
    for key, subdic in fitting_params_flat.items():
        if key.split('-')[0] == 'shared_params':
            fitting_par[key] = subdic
        
        elif key.split('-')[0] == 'line_params':
            pname_w_spec = key.split('-')[1] # v_0, I01_spec1, ...
            pars_this_line[pname_w_spec] = {}
            for line in lines:
                line_p = f'{line}_params-{pname_w_spec}'
                pars_this_line[pname_w_spec][line] = {'standard_key': line_p,
                                                      'latex_prior':  subdic }
                
                if line in doublet_lines:
                    if 'spec' in pname_w_spec.split('_')[1]: # I01_spec1
                        pname  = pname_w_spec.split('_')[0]
                        q2name = doub_pars[doub_pars_twin.index(pname)]
                        q2name_w_spec = f'{q2name}_{pname_w_spec.split("_")[1]}'
                        line_q        = f'{line}_params-{q2name_w_spec}'
                        latex2 = subdic['latex_name'][:-1] + "\,_{(2)}$"
                        
                        if pars_this_line.get(q2name, None) is None:
                            pars_this_line[q2name_w_spec] = {}
                        
                        pars_this_line[q2name_w_spec][line] = {
                            'standard_key': line_q,
                            'latex_prior':  {'latex_name': latex2,
                                             'prior':      subdic['prior']}
                            }
                        
                    else: # v_0
                        pname  = pname_w_spec
                        q2name = doub_pars[doub_pars_twin.index(pname)]
                        line_q = f'{line}_params-{q2name}'
                        latex2 = subdic['latex_name'][:-1] + "\,_{(2)}$"
                        
                        if pars_this_line.get(q2name, None) is None:
                            pars_this_line[q2name] = {}
                        
                        pars_this_line[q2name][line] = {
                            'standard_key': line_q,
                            'latex_prior':  {'latex_name': latex2,
                                             'prior':      subdic['prior']}
                            }
                        
        elif key.split('-')[0].split('_')[0] in lines:
            pname_w_spec = key.split('-')[1] # v_0, I01_spec1, ...
            line = key.split('-')[0].split('_')[0]
            
            line_p = f'{line}_params-{pname_w_spec}'
            fitting_par[line_p] = subdic
            
            # 重要：凡是涉及到字典修改/赋值操作，
            # 即使不作为函数输出，也必须先deep copy！
            # 因为你使用了
            # for key, emptydic in fitting_params_flat.items()
            #     emptydic['latex_name'] = latex_n
            import copy
            subdic_ = copy.deepcopy(subdic)
            
            if line in doublet_lines:
                if 'spec' in pname_w_spec.split('_')[1]: # I01_spec1
                    pname_wo_spec = pname_w_spec.split('_')[0]
                    i_q    = doub_pars_twin.index(pname_wo_spec)
                    q2name = doub_pars[i_q]
                    line_q = f'{line}_params-{q2name}_{pname_w_spec.split("_")[1]}'
                    latex_name = subdic['latex_name']
                    latex_n    = latex_name[:-1] + "\,_{(2)}$"
                    subdic_['latex_name'] = latex_n
                    fitting_par[line_q]  = subdic_
                    
                else: # v_0
                    i_q    = doub_pars_twin.index(pname_w_spec)
                    q2name = doub_pars[i_q]
                    line_q = f'{line}_params-{q2name}'
                    latex_name = subdic['latex_name']
                    latex_n    = latex_name[:-1] + "\,_{(2)}$"
                    subdic_['latex_name'] = latex_n
                    fitting_par[line_q]  = subdic_
    
    # Since real line fitting params are sorted by line > spec_i...
    # re-organize and sort "cleandic", by line in lines
    if len(pars_this_line) != 0:
        for line in lines:
            for par, this_par_lines_with_dic in pars_this_line.items():
                try:
                    standard_key = this_par_lines_with_dic[line]['standard_key']
                    latex_prior  = this_par_lines_with_dic[line]['latex_prior' ]
                
                # In secondary dict for doublet lines, it's OK to see 
                # one line's par being not in doublet line's secondary dict
                except KeyError:
                    pass
                
                fitting_par[standard_key] = latex_prior
                
    return fitting_par


def analyze_percentile(samples, config_filename, 
                       line_species=['O2','O2','O2','Hg','Hg','Hg']):
    # load fitting config
    with open(config_filename, "r", encoding="utf-8") as file1:
        config = yaml.safe_load(file1)
    
    config_par = complete_fit_params(config, line_species)
    
    config_param_names = [key for key, _ in config_par.items()]
    
    dic_percent = {}
    nparams = len(samples[0])
    arr = np.zeros((nparams, 3))
    
    for j in range(nparams):
        samp_points = samples[:,j]
        x123 = np.percentile(samp_points, [16, 50, 84])
        err_lo, mean, err_hi = x123[0]-x123[1], x123[1], x123[2]-x123[1]
        arr[j] = np.around([mean, err_lo, err_hi], decimals=4)
        
        try:
            dic_percent[config_param_names[j]] = {'mean':   arr[j,0], 
                                                  'err_lo': arr[j,1], 
                                                  'err_hi': arr[j,2]}
        except IndexError:
            # print(f'nparams = {nparams}, \n'+f'j = {j}, \n'+
            #       f'len(config_param_names) = {len(config_param_names)}\n'+
            #       f'config_param_names = {config_param_names}')
            dic_percent = 'ERROR'
            break

    return dic_percent


def load_best_fit_json(inference, fitting_params, filename, 
                       config_to_check=None):
    # Load JSON for estimates dict
    with open(filename, 'r', encoding='utf-8') as f:
        estimates  = json.load(f)['maximum_likelihood']['point']
    # elif filename[-5:] == '.hdf5':
    #     # Nautilus HDF5
    #     with h5py.File(filename, "r") as f:
    #         g = f["sampler"]

    #         pts  = np.vstack([g[k][:] for k in g.keys() 
    #                           if k.startswith("points_") and k != "points_t"])
    #         logl = np.concatenate([g[k][:] for k in g.keys() 
    #                                if k.startswith("log_l_") and k != "log_l_t"])

    #     estimates = pts[np.argmax(logl)]
    
    if isinstance(estimates, dict): 
        estimates = list(estimates.values())
    
    # Check if you are using correct config YAML file
    if config_to_check is not None:
        config_params = {}
        config_params = Parameters._flatten(config_to_check['params'], level=1)
        n_param_estimates = len(estimates)
        n_param_fitting   = len(config_params)
        if n_param_estimates != n_param_fitting:
            raise Exception(f'Your JSON file ({filename}) \n'+ 
                  '           does not match with the fitting parameters used in the run \n'+
                  '           by your config. Here they have different # of parameters.')
    
    # Complete flattened fit_params
    fitting_params_flat = Parameters._flatten(fitting_params, level=1)
    fitting_par = complete_flattened_fit_params(
        fitting_params_flat, 
        line_species=inference.config.galaxy_params.line_species
        )
    best_fit_dict  = inference.params.gen_param_dict(fitting_par.keys(), 
                                                     estimates)
    
    return estimates, best_fit_dict, fitting_par


def ax_compass(ax, x0, y0, dx, dy, color='black'):
    # 北：向上（y 减小）
    ax.arrow(x0, y0, 0, dy,
             head_width=0.02, head_length=0.02,
             fc=color, ec=color, linewidth=1.5, 
             transform=ax.transAxes)
    ax.text(x0 + 0.02, y0 + dy, 'N',
            color=color, ha='left', va='center', fontsize=12, 
            transform=ax.transAxes)
    
    # 东：一般地，向左（x 减小）
    ax.arrow(x0, y0, -dx, 0,
             head_width=0.02, head_length=0.02,
             fc=color, ec=color, linewidth=1.5, 
             transform=ax.transAxes)
    ax.text(x0 - dx, y0 + 0.02, 'E',
            color=color, ha='center', va='bottom', fontsize=12, 
            transform=ax.transAxes)
    return


def plot_obs_fit_res(data_info, 
                     inference, best_fit_dict, fitting_par, 
                     run, save_path=None, other_path_filename=None):
    line_list ={'O2':  r"[O II] $\lambda\lambda$3726,3729",
                'Ha':  r"H$\alpha$",
                'Hb':  r"H$\beta$", 
                'Hg':  r"H$\gamma$", 
                'O3a': r"[O III] $\lambda$4959", 
                'O3b': r"[O III] $\lambda$5007", 
                'N2a': r"[N II] $\lambda$6549", 
                'N2b': r"[N II] $\lambda$6583",
                }
    nspec = len(data_info['spec'])
    lines = []
    for k in best_fit_dict.keys(): 
        if k.split('_')[0] != 'shared':
            lines.append(k.split('_')[0])
    
    """
    # Note the inference-returned array was based on WCS's 
    # increasing RA/DEC direction. 
    data[ΔDEC = 0, ΔRA = 0] = South-West corner!!!
    
    Note: image_obs & image_fit used real sky projection = WCS.
    E.g. A 2D WCS array = 
                (5', 10') --------------- (5', 0)
                   |   NE               NW   |
                   |                         |
                   |   SE               SW   |
                (0 , 10') --------------- (0 , 0) = ΔDEC, ΔRA
    
    To compare with model fit, MUST CONVERT image_obs to ΔRA. 
    
    Step 1. FLIP horizontal axis: (IMPORTANT)
                (0, 10') ---------------- (5', 10')
                   |   NW               NE   |
                   |                         |
                   |   SW               SE   |
    ΔDEC, ΔRA = (0 , 0) ----------------- (0 , 10')
                                           
    Step 2. MUST MANDIDATE origin='lower' IN ax.imshow().
        This is an astro common sense in matplotlib. 
        
    In summary:
        image_obs: left = East (ΔRA > 0), right = West (ΔRA < 0).
        Need to flip to left = (ΔRA < 0), right = (ΔRA > 0):
    """
    fig   = plt.figure(figsize=(16, 4*(1+nspec)))  # (length, height)
    plt.subplots_adjust(hspace=0.2, wspace=0.2) # h=height
    gs    = fig.add_gridspec(nrows=1+nspec, ncols=3, 
                             height_ratios=[1]*(1+nspec), 
                             width_ratios =[1,1,1])
    ax_img_obs = fig.add_subplot(gs[0, 0])
    ax_img_fit = fig.add_subplot(gs[0, 1])
    ax_img_res = fig.add_subplot(gs[0, 2])
    
    n_par = 0
    for key in fitting_par.keys():
        if len(key.split('-')) == 1: # Have subkeys (g1, g2, v0...)
            for k in fitting_par[key].keys():
                n_par += 1
        else:
            n_par = len(fitting_par)
    
    image_obs = np.flip(inference.data_image, axis=1)
    image_msk = np.flip(inference.mask_image, axis=1)
    image_var = np.flip(inference.var_image,  axis=1)
    image_fit = np.flip(inference.image_model.get_image(best_fit_dict['shared_params']), 
                        axis=1)
    image_chi2 = inference.calc_image_loglike(best_fit_dict)
    
    noise = np.nanstd(image_obs[image_msk])
    ny_img, nx_img = image_obs.shape
    imgDOF    = ny_img * nx_img - n_par
    pix_scale = data_info['image']['par_meta']['pixScale']
    im1 = ax_img_obs.imshow(np.where(image_msk, image_obs, np.nan), 
                            vmin=0, vmax=0 + 5*noise, 
                            extent = [
                                -nx_img*pix_scale/2,  nx_img*pix_scale/2, 
                                 # nx_img*pix_scale/2, -nx_img*pix_scale/2, 
                                -ny_img*pix_scale/2, ny_img*pix_scale/2],
                            cmap='cividis', origin='lower', aspect='equal')
    im2 = ax_img_fit.imshow(image_fit, 
                            vmin=0, vmax=0 + 5*noise, 
                            extent = [
                                -nx_img*pix_scale/2,  nx_img*pix_scale/2, 
                                 # nx_img*pix_scale/2, -nx_img*pix_scale/2, 
                                -ny_img*pix_scale/2, ny_img*pix_scale/2],
                            cmap='cividis', origin='lower', aspect='equal')
    im3 = ax_img_res.imshow(np.where(image_msk, image_obs - image_fit, np.nan), 
                            vmin=-5*noise, vmax=5*noise, 
                            extent = [
                                # -nx_img*pix_scale/2,  nx_img*pix_scale/2, 
                                 nx_img*pix_scale/2, -nx_img*pix_scale/2, 
                                -ny_img*pix_scale/2, ny_img*pix_scale/2],
                            cmap='coolwarm', origin='lower', aspect='equal')
    fig.colorbar(im1, ax=ax_img_obs)
    fig.colorbar(im2, ax=ax_img_fit)
    fig.colorbar(im3, ax=ax_img_res)
    
    # Imaging Chi2
    strk_txt2 = ax_img_res.text(1, 1, 
                                r"$\chi^2=$"+f"{image_chi2:.0f}", 
                                c='yellow', fontsize=30, weight='bold', 
                                ha='right', va='top', transform=ax_img_res.transAxes)
    strk_txt2.set_path_effects([path_effects.Stroke(linewidth=5, 
                                                    foreground='black'),
                                path_effects.Normal()]) # stroked-text
    ax_img_res.text(0.98, 0.8, 
                    f"DOF = {imgDOF}", 
                    fontsize=18, color='black', ha='right', va='top', 
                    transform=ax_img_res.transAxes)
    
    # Flatten fitting_par if first key is either shared_params or line_params
    if 'shared_params' in fitting_par.keys():
        fitting_par = complete_fit_params(
            fitting_par, inference.config.galaxy_params.line_species)
    
    # Annotate imaging best fit parameters
    aximagtxts = ''
    for level1key, subdict in fitting_par.items():
        if level1key.split('-')[0] == 'shared_params':
            # No lower level dict
            # if len(level1key.split('-')) == 0: 
                par_name   = level1key.split('-')[1]
                latex_name = subdict['latex_name']
                value      = best_fit_dict['shared_params'][par_name]
                aximagtxts += (latex_name + ' = ' + '{:.2g}'.format(value) + '\n')
    ax_img_fit.text(1, 1, aximagtxts, fontsize=12, color='white', ha='right', va='top', 
                    transform=ax_img_fit.transAxes)
    
    image_SNR = np.sum(image_obs[image_msk]) / np.sum(image_var)**0.5
    ax_img_obs.text(0.98, 0.97, 
                    f'N_RA = {nx_img} px'+'\n'+
                    f'N_DEC = {ny_img} px'+'\n'+
                    f'SNR = {image_SNR:.0f}', 
                    fontsize=12, color='white', ha='right', va='top', 
                    transform=ax_img_obs.transAxes,
                    bbox=dict(facecolor='black', alpha=0.75))
    
    ax_compass(ax_img_obs, x0=0.05, y0=0.05, dx=-0.12, dy=0.12, color='white')
    ax_compass(ax_img_fit, x0=0.05, y0=0.05, dx=-0.12, dy=0.12, color='white')
    ax_compass(ax_img_res, x0=0.05, y0=0.05, dx=-0.12, dy=0.12, color='black')
    
    ax_img_obs.set_ylabel(r'${\bf Imaging}$'+'\n'+r'$\Delta$ DEC (arcsec)', 
                          fontsize=18)
    ax_img_obs.set_xlabel(r'$\Delta$ RA (arcsec)', )
    ax_img_fit.set_xlabel(r'$\Delta$ RA (arcsec)', )
    ax_img_res.set_xlabel(r'$\Delta$ RA (arcsec)', )
    ax_img_obs.set_title('Observation', fontsize=18)
    ax_img_fit.set_title('Best fit model', fontsize=18)
    ax_img_res.set_title('Residual (= obs - model)', fontsize=18)
    ax_img_obs.grid(linestyle=':', color='white', alpha=0.5)
    ax_img_fit.grid(linestyle=':', color='white', alpha=0.5)
    ax_img_res.grid(linestyle=':', color='white', alpha=0.5)
    
    # ================ Spectrum sets ==================
    set_names = ['A', 'C', 'B']
    for i in range(nspec):
        inference.spec_model[i]._init_observable(data_info['galaxy'], 
                                                 data_info['spec'][i]['par_meta'])
        
        # Update by set. Assign I01 from I01_spec1/2/3 now
        line    = lines[i//3]
        set_num = i % 3 + 1
        best_only_one_level = {**best_fit_dict[ 'shared_params'],
                               **best_fit_dict[f'{line}_params'] }
        for k in ['I02', 'I01']: #, 'dx_vel', 'dy_vel', 'bkg_level']:
            best_only_one_level[k] = None
            if k+f'_spec{set_num}' in best_only_one_level.keys():
                best_only_one_level[k] = best_only_one_level[k+f'_spec{set_num}']
        
        spec0_obs = inference.data_spec[i]
        spec0_msk = inference.mask_spec[i]
        svar0_obs = inference.var_spec[i]
        scon0_obs = inference.cont_model[i]
        spec0_fit = inference.spec_model[i].get_observable(best_only_one_level)
        spec0_chi2 = inference._loglike_one_slit(data_spec=spec0_obs, 
                                                 mask_spec=spec0_msk,
                                                 var_spec=svar0_obs, 
                                                 cont_spec=scon0_obs,
                                                 model_spec=spec0_fit)
        
        ny_spec, nx_spec = spec0_obs.shape
        specDOF = ny_spec * nx_spec - len(fitting_par)
        extent = [data_info['spec'][i]['par_meta']['lambda_grid'][0][ 0].value, 
                  data_info['spec'][i]['par_meta']['lambda_grid'][0][-1].value, 
                  inference.spec_model[i].slit_x[0], 
                  inference.spec_model[i].slit_x[-1]]
    
        ax_spe_obs = fig.add_subplot(gs[i+1, 0])
        ax_spe_fit = fig.add_subplot(gs[i+1, 1])
        ax_spe_res = fig.add_subplot(gs[i+1, 2])
        
        spec0_obs_to_show = np.where(spec0_msk, spec0_obs, np.nan)
        spec0_res_to_show = np.where(spec0_msk, spec0_obs - spec0_fit, np.nan)
        noise = np.nanstd(spec0_obs_to_show)
        im1 = ax_spe_obs.imshow(spec0_obs_to_show, 
                                vmin=0, vmax=0 + 5*noise, 
                                # origin='lower', 
                                extent=extent, 
                                cmap='viridis', aspect='auto')
        im2 = ax_spe_fit.imshow(spec0_fit, 
                                vmin=0, vmax=0 + 5*noise, 
                                # origin='lower', 
                                extent=extent, 
                                cmap='viridis', aspect='auto')
        im3 = ax_spe_res.imshow(spec0_res_to_show, 
                                vmin=-5*noise, vmax=5*noise, 
                                # origin='lower', 
                                extent=extent, 
                                cmap='coolwarm', aspect='auto')
        fig.colorbar(im1, ax=ax_spe_obs)
        fig.colorbar(im2, ax=ax_spe_fit)
        fig.colorbar(im3, ax=ax_spe_res)
        
        # Spec Chi2
        strk_txt1 = ax_spe_res.text(1, 1, 
                                    r"$\chi^2=$"+f"{spec0_chi2:.0f}", 
                                    c='yellow', fontsize=30, weight='bold', 
                                    ha='right', va='top', 
                                    transform=ax_spe_res.transAxes)
        strk_txt1.set_path_effects([path_effects.Stroke(linewidth=5, 
                                                        foreground='black'), 
                                    path_effects.Normal()]) # stroked-text
        ax_spe_res.text(0.98, 0.82, 
                        r"$\chi^2$"+f"/DOF = {spec0_chi2/specDOF:.1f}", 
                        fontsize=18, color='black', ha='right', va='top', 
                        transform=ax_spe_res.transAxes)
        
        
        # Annotate spectrum best fit parameters
        axspectxts = ''
        for level1key, subdict in fitting_par.items():
            if level1key.split('-')[0] == f'{line}_params':
                par_name   = level1key.split('-')[1]
                latex_name = subdict['latex_name']
                value      = best_fit_dict[f'{line}_params'][par_name]
                axspectxts += (latex_name+' = '+'{:.1f}'.format(value)+'\n')
        
        # Choose params for the current set
        axspectxts_ = ''
        for s in axspectxts.split('\n'): 
            if (f'Spec {i%3+1}' in s) or ('Spec' not in s): 
                if len(s) != 0:
                    axspectxts_ += (s+'\n')
        ax_spe_fit.text(1, 1, axspectxts_, fontsize=12, color='white', 
                        ha='right', va='top', 
                        transform=ax_spe_fit.transAxes)
    
        spec_SNR = np.sum(spec0_obs[spec0_msk]) / (np.sum(spec0_obs[spec0_msk]) + np.sum(svar0_obs[spec0_msk]))**0.5
        ax_spe_obs.text(0.98, 0.97, 
                        'N_'+r'$\lambda$'+f' = {nx_spec} px'+'\n'+
                        'N_slit'+f' = {ny_spec} px'+'\n'+
                        f'SNR = {spec_SNR:.0f}', 
                        fontsize=12, color='white', ha='right', va='top', 
                        transform=ax_spe_obs.transAxes,
                        bbox=dict(facecolor='black', alpha=0.75))
        ax_spe_obs.text(0.02, 0.02, 
                        line_list[line], 
                        fontsize=18, color='white', ha='left', va='bottom', 
                        transform=ax_spe_obs.transAxes,
                        bbox=dict(facecolor='black', alpha=0.75))
        
        ax_spe_obs.xaxis.get_major_formatter().set_useOffset(False)
        ax_spe_fit.xaxis.get_major_formatter().set_useOffset(False)
        ax_spe_res.xaxis.get_major_formatter().set_useOffset(False)
        ax_spe_obs.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
        ax_spe_fit.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
        ax_spe_res.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
        ax_spe_obs.set_ylabel(r'${\bf Spec}$'+f' set {set_names[set_num-1]}'+
                              '\n'+'Slit Position (arcsec)', fontsize=18)
        ax_spe_obs.grid(linestyle=':', color='white', alpha=0.5)
        ax_spe_fit.grid(linestyle=':', color='white', alpha=0.5)
        ax_spe_res.grid(linestyle=':', color='white', alpha=0.5)
    
    # Bottom plot
    ax_spe_obs.set_xlabel(r'Wavelength ($\AA$)')
    ax_spe_fit.set_xlabel(r'Wavelength ($\AA$)')
    ax_spe_res.set_xlabel(r'Wavelength ($\AA$)')
    
    if other_path_filename is None:
        fig_path = f'{save_path}/best_fit_spec.png'
    else: 
        fig_path = other_path_filename
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    return
    
