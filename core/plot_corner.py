from pathlib import Path
import yaml
import numpy as np
import astropy.units as u

import getdist.plots

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']  = 100
plt.rcParams['savefig.dpi'] = 100


def get_max_num_subdir(root_path):
    p = Path(root_path)
    
    # 1. 匹配所有二级文件夹（即 */*）
    # 2. 确保它是目录 (is_dir)
    # 3. 确保文件夹名是纯数字 (name.isdigit)
    valid_subdirs = []
    
    for d in p.glob("*"):
        if not d.is_dir():
            continue
        
        last_part = d.name.split('_')[-1]
        
        if last_part.isdigit():   # 只保留能转成整数的
            valid_subdirs.append((d, int(last_part)))
    
    if not valid_subdirs:
        return None
    
    # 取数字最大的
    max_subdir = max(valid_subdirs, key=lambda x: x[1])[0]
    
    return max_subdir.name


def diagnose_posterior(samples, weights):
    w = weights / np.sum(weights)

    # Effective Sample Size
    ESS = 1.0 / np.sum(w**2)
    
    if ESS > 1:
        # 协方差矩阵
        cov = np.cov(samples.T, aweights=w)
    
        # 特征值
        eigvals = np.linalg.eigvalsh(cov)
        
    else:
        eigvals = None

    return ESS, eigvals


def read_post(full_path, params_NOT_to_plot=None):
    try:
        runsample = np.loadtxt(full_path, dtype=str, skiprows=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"{full_path} not found. \n"+
                                "Perhaps fitting was not completed for this slit?")
        
    # Choose your param NOT to plot
    if params_NOT_to_plot is not None:
        for Not_to in params_NOT_to_plot:
           runsample = np.delete(
               runsample, np.where(runsample[0] == Not_to), axis=1)
            
    par_names = runsample[0,  2:]
    weights   = runsample[1:, 0 ].astype(float)
    loglikes  = runsample[1:, 1 ].astype(float)
    samples   = runsample[1:, 2:].astype(float)
    mask      = np.ones(weights.shape, dtype=bool)
    
    too_few = False
    ESS, eigvals = diagnose_posterior(samples, weights)
    if (ESS < len(par_names)**2) or (eigvals is None):
        too_few = True
        print("\033[43m" + '[WARNING]' + "\033[0m " + 
              f"Only  {int(ESS):6d} effective points. Skipped for plotting {full_path}.")
    else:
        print(f"[INFO   ] ESS = {int(ESS):6d} effective points.")
    
    return runsample, par_names, weights, loglikes, samples, mask, too_few


def complete_post_and_par(tbd_samples, tbd_names, good_names):
    good_names  = ['weight', 'logl'] + list(good_names)
    tbd_names   = ['weight', 'logl'] + list(tbd_names)
    tbd_samples = tbd_samples
    out = np.zeros((tbd_samples.shape[0], len(good_names))).astype(str)
    name_to_idx = {n:i for i,n in enumerate(tbd_names)}
    for j, name in enumerate(good_names):
        name = str(name)
        if name in name_to_idx:
            out[:, j] = tbd_samples[:, name_to_idx[name]]
        else: # Found unmatched columns
            if \
                name == 'shared_params-g1':
                    out[0,  j] = name
                    out[1:, j] = 0 + np.random.normal(scale=0.001, 
                                                  size=(len(out)-1))
            elif \
                name == 'shared_params-g2':
                    out[0,  j] = name
                    out[1:, j] = 0 + np.random.normal(scale=0.001, 
                                                  size=(len(out)-1))
            else:
                raise IndexError('Something wrong here...')
    return out


def plot_corner(full_run_dirs, labels, 
                params_to_plot=None, 
                params_NOT_to_plot=None,
                read_latex_from=None,
                corner_name='corner_compare.png', 
                contour_color='dimgray', 
                contour_alpha={'alpha': 0.75},
                true_values=None,
                test=False,):
    no_plot1, no_plot2 = True, True
    # Step 1. Read posteriors (weighted)
    if len(full_run_dirs) == 1:
        full_run_path1 = full_run_dirs[0]
        label1         = labels[0]
        
        full_run_path2 = None
        label2         = None
        
        run_samples1, par_names1, weights1, \
        loglikes1, samples1, mask1, no_plot1 = read_post(
            full_run_path1, 
            params_NOT_to_plot,
            # percentile=percentile,
            # equal_weights=change_to_equal_weights_in_case, 
            # force_to_use_weights=force_to_use_weights
            )
        
        par_names1 = run_samples1[0,  2:]
        samples1   = run_samples1[1:, 2:].astype(float)

    else:
        full_run_path1 = full_run_dirs[0]
        full_run_path2 = full_run_dirs[1]
        label1         = labels[0]
        label2         = labels[1]
        
        run_samples1, par_names1, weights1, \
        loglikes1, samples1, mask1, no_plot1 = read_post(
            full_run_path1, 
            params_NOT_to_plot,
            # percentile=percentile,
            # equal_weights=change_to_equal_weights_in_case, 
            # force_to_use_weights=force_to_use_weights
            )
        
        run_samples2, par_names2, weights2, \
        loglikes2, samples2, mask2, no_plot2 = read_post(
            full_run_path2, 
            params_NOT_to_plot,
            # percentile=percentile,
            # equal_weights=change_to_equal_weights_in_case, 
            # force_to_use_weights=force_to_use_weights
            )
    
        # Step 2.1 Complete param of posteriors (Optional)
        run_samples2 = complete_post_and_par(
            tbd_samples = run_samples2, 
            tbd_names   = par_names2, 
            good_names  = par_names1,
            )
        par_names1 = run_samples1[0,  2:]
        par_names2 = run_samples2[0,  2:]
        samples1   = run_samples1[1:, 2:].astype(float)
        samples2   = run_samples2[1:, 2:].astype(float)
    
    # Test mode
    if test:
        print("\033[42m" + 'INFO:' + "\033[0m " + 
              'Testing done. OK ✅\n')
        return
    
    # Step 3. Choose your param to plot
    if params_to_plot is not None:
        par_names_idx1 = [
            list(par_names1).index(params_to_plot[i]) 
            for i in range(len(params_to_plot))
            ]
        par_names1 = par_names1[par_names_idx1].tolist()
        samples1   = samples1[:,par_names_idx1]
        
        if len(full_run_dirs) > 1: # not only one post
            par_names_idx2 = [
                list(par_names2).index(params_to_plot[i]) 
                for i in range(len(params_to_plot))
                ]
            par_names2 = par_names2[par_names_idx2]
            samples2   = samples2[:,par_names_idx2]
            
    else:
        params_to_plot = par_names1.tolist()
    
    # Step 4. Read emission lines and fitting param to get latex names
    lines1 = []
    for par in par_names1.tolist():
        lv1_key, lv2_key = par.split('-')
        if lv1_key != 'shared_params':
            lines1.append(lv1_key.split('_')[0])

    if read_latex_from is not None:
        fit_par1 = read_latex_from
    else:
        with open('../config/binospec_fitting_params.yaml', 
                "r", encoding="utf-8") as yamlfile:
            fit_par1 = yaml.safe_load(yamlfile)
    
    latex_names, latex_to_parname = [], []
    for lv1key, subdict in fit_par1.items():
        if lv1key == 'shared_params':
            for lv2key, par_dict in subdict.items():
                if f'shared_params-{lv2key}' in params_to_plot:
                    latex_names.append(
                        par_dict['latex_name'][1:-1]
                        )
                    latex_to_parname.append(f'shared_params-{lv2key}')
                elif params_NOT_to_plot is not None:
                    if f'shared_params-{lv2key}' in params_NOT_to_plot:
                        latex_names.pop()
                        latex_to_parname.pop()
                else:
                    raise ValueError('Something wrong here...')

        else: # line_params
            for line in list(dict.fromkeys(lines1)):
                for lv2key, par_dict in subdict.items():
                    
                    config_par_to_find = f'{lv1key}-{lv2key}'
                    if config_par_to_find in params_to_plot:
                        latex_names.append(
                            '\mathrm{'+f'{line}'+': }'+par_dict['latex_name'][1:-1]
                            )
                        latex_to_parname.append(f'{lv1key}-{lv2key}')
                        if line == 'O2':
                            if any(a in lv2key for a in ('bkg_level', 'sersic_spec')):
                                # Do nothing, because we want bkg is shared between O2 and O2b
                                # It is supposed to be one parameter across two lines
                                pass
                                
                            # Duplicate params since the line is a doublet
                            else:
                                latex_names.append(
                                '\mathrm{'+f'{line}b'+': }'+par_dict['latex_name'][1:-1]
                                ) # Note: this is "b" here
                            
                                # Match latex names to config names, add "2" into parameter names, indicate that this is the second line of O2 doublet
                                if lv2key == 'v_0':
                                    latex_to_parname.append(f'{line}_params-v_0_2')
                                else:
                                    if lv2key.split("_")[0][:2] == 'I0':
                                        latex_to_parname.append(f'{line}_params-I02_{lv2key.split("_")[1]}')
                                    elif lv2key.split("_")[0][:2] == 'f1':
                                        latex_to_parname.append(f'{line}_params-f2_{lv2key.split("_")[1]}_{lv2key.split("_")[2]}')
                                    elif lv2key.split('_')[0][:2] == 'dx':
                                        latex_to_parname.append(f'{line}_params-dx_{lv2key.split("_")[1]}_2_{lv2key.split("_")[2]}')
                                    else:
                                        raise ValueError(f'Something wrong here: lv2key is {lv2key}. Cannot match I0, f1, dx, or similar.')
                    
                    elif lv1key.split('_')[0] == 'line':
                        config_par_to_find = f'{line}_params-{lv2key}'
                        if config_par_to_find in params_to_plot:
                            latex_names.append(
                                '\mathrm{'+f'{line}'+': }'+par_dict['latex_name'][1:-1]
                                )
                            latex_to_parname.append(f'{line}_params-{lv2key}')
                            if line == 'O2':
                                if any(a in lv2key for a in ('bkg_level', 'sersic_spec')):
                                    # Do nothing, because we want bkg is shared between O2 and O2b
                                    # It is supposed to be one parameter across two lines
                                    pass

                                # Duplicate params since the line is a doublet
                                else:
                                    latex_names.append(
                                        '\mathrm{'+f'{line}b'+': }'+par_dict['latex_name'][1:-1]
                                        ) # Note: this is "b" here

                                    # Match latex names to config names, add "2" into parameter names, indicate that this is the second line of O2 doublet
                                    if lv2key == 'v_0':
                                        latex_to_parname.append(f'{line}_params-v_0_2')
                                    else:
                                        if lv2key.split("_")[0][:2] == 'I0':
                                            latex_to_parname.append(f'{line}_params-I02_{lv2key.split("_")[1]}')
                                        elif lv2key.split("_")[0][:2] == 'f1':
                                            latex_to_parname.append(f'{line}_params-f2_{lv2key.split("_")[1]}_{lv2key.split("_")[2]}')
                                        elif lv2key.split('_')[0][:2] == 'dx':
                                            latex_to_parname.append(f'{line}_params-dx_{lv2key.split("_")[1]}_2_{lv2key.split("_")[2]}')
                                        else:
                                            raise ValueError(f'Something wrong here: lv2key is {lv2key}. Cannot match I0, f1, dx, or similar.')

                    elif params_NOT_to_plot is not None:
                        if f'{lv1key}-{lv2key}' not in params_NOT_to_plot:
                            latex_names.pop()
                            latex_to_parname.pop()
                            if line == 'O2':
                                latex_names.pop()
                                latex_to_parname.pop()

                    else:
                        raise ValueError('Something wrong here...')
    latex_names = np.array(latex_names)
    assert len(latex_names) == len(latex_to_parname)
    
    # Re-order
    if params_to_plot is not None:
        latex_names_idx1 = [
            list(latex_to_parname).index(params_to_plot[i]) 
            for i in range(len(params_to_plot))
            ]
        latex_names = latex_names[latex_names_idx1]
    
    # (Optional 1) Special limits of g1/g2 priors
    # g1_idx_in_sample1 = list(par_names1).index('shared_params-g1')
    # g2_idx_in_sample1 = list(par_names1).index('shared_params-g2')
    # good_idx_sample1 = [bool(
    #     (samples1[i, g1_idx_in_sample1] > -0.2) and 
    #     (samples1[i, g1_idx_in_sample1] <  0.2) and 
    #     (samples1[i, g2_idx_in_sample1] > -0.2) and 
    #     (samples1[i, g2_idx_in_sample1] <  0.2)
    #     for i in range(len(samples1))
    #     )]
    # mask1 &= good_idx_sample1
    # if len(full_run_dirs) > 1: # not only one post
    #     g1_idx_in_sample2 = list(par_names2).index('shared_params-g1')
    #     g2_idx_in_sample2 = list(par_names2).index('shared_params-g2')
    #     good_idx_sample2 = [bool(
    #         (samples2[i, g1_idx_in_sample2] > -0.2) and 
    #         (samples2[i, g1_idx_in_sample2] <  0.2) and 
    #         (samples2[i, g2_idx_in_sample2] > -0.2) and 
    #         (samples2[i, g2_idx_in_sample2] <  0.2))
    #         for i in range(len(samples2)
    #         )]
    #     mask2 &= good_idx_sample2
    
    # Step 5. Pack in getdist/MCSamples
    mc1 = getdist.MCSamples(
        names    = par_names1,
        weights  = weights1[ mask1],
        loglikes = loglikes1[mask1],
        samples  = samples1[ mask1],
        labels   = latex_names, 
        )
    if len(full_run_dirs) > 1: # not only one post
        mc2 = getdist.MCSamples(
            names    = par_names2,
            weights  = weights2[ mask2],
            loglikes = loglikes2[mask2],
            samples  = samples2[ mask2],
            labels   = latex_names, 
            )
    
    # Step 6. Plot settings
    getdist_plotter = getdist.plots.get_subplot_plotter(subplot_size = 1.6)
    getdist_plotter.settings.legend_fontsize = 20
    # getdist_plotter.settings.progress = True
    
    if isinstance(contour_color, str):
        contour_colors = [contour_color]
    else:
        contour_colors = contour_color
        
    if isinstance(contour_alpha, dict):
        contour_alphas = [contour_alpha]
    else:
        contour_alphas = contour_alpha
    
    # contours (1 & 2) will be OK
    if no_plot1 is not True: # 1.1. Plot 1st post
        if no_plot2 is True: # 1.2. DO NOT plot 2nd post

            # For mock analysis ONLY:
            if true_values is not None and len(true_values) > 0:
                assert len(true_values) == len(latex_names), \
                    f"Length of true_values must match par_names1: \n{par_names1}"
                true_dict = dict(zip(par_names1, true_values))
            else:
                true_dict = None
            
            if contour_color is None: 
                contour_color = 'green'
                
            getdist_plotter.triangle_plot(mc1, 
                filled        = True, 
                legend_labels = [label1],
                contour_colors= contour_colors,
                contour_args  = contour_alphas,
                title_limit   = 1, # 1σ
                title_fmt     = '.2f', 
                smooth1d = 0, # bypass KDE smoother
                smooth2d = 0, # bypass KDE smoother
                markers=true_dict,
                marker_args={'color': 'red', 'ls': '--', 'lw': 1.2},
                )
            
            post_corner_hist(getdist_plotter, latex_names, no_plot1, no_plot2, 
                             par_names1=par_names1, par_names2=None, 
                             mc1=mc1, mc2=None)
            plt.savefig(corner_name, bbox_inches='tight')
            plt.close()
            return run_samples1, None
        
        else: # 2.2. Plot 1st & 2nd post
            # For mock analysis ONLY:
            if true_values is not None and len(true_values) > 0:
                assert len(true_values) == len(latex_names), \
                    f"Length of true_values must match par_names1: \n{par_names1}"
                true_dict = dict(zip(par_names1, true_values))
            else:
                true_dict = None
            
            getdist_plotter.triangle_plot([mc1, mc2], 
                filled        = True, 
                legend_labels = ['A: '+label1, 'B: '+label2],
                contour_colors= contour_colors,
                contour_args  = contour_alphas,
                title_limit   = 1, # 1σ
                title_fmt     = '.2f', 
                smooth1d = 0, # bypass KDE smoother
                smooth2d = 0, # bypass KDE smoother
                markers  = true_dict,
                marker_args={'color': 'red', 'ls': '--', 'lw': 1.2},
                )
            post_corner_hist(getdist_plotter, latex_names, no_plot1, no_plot2, 
                             par_names1=par_names1, par_names2=par_names2, 
                             mc1=mc1, mc2=mc2)
            plt.savefig(corner_name, bbox_inches='tight')
            plt.close()
            return run_samples1, run_samples2
    
    # contours will fail
    else: # 3.1. DO NOT plot 1st post1
        if no_plot2 is not True: # 3.2. But plot 2nd post
            getdist_plotter.triangle_plot(mc2, 
                filled        = True, 
                legend_labels = ['B: '+label2+'\n< A: No posterior >'],
                contour_colors= contour_colors,
                contour_args  = contour_alphas,
                title_limit   = 1, # 1σ
                title_fmt     = '.2f', 
                smooth1d = 0, # bypass KDE smoother
                smooth2d = 0, # bypass KDE smoother
                markers  = true_dict,
                marker_args = {'color': 'red', 'ls': '--', 'lw': 1.2},
                )
            post_corner_hist(getdist_plotter, latex_names, no_plot1, no_plot2, 
                             par_names1=None, par_names2=par_names2, 
                             mc1=None, mc2=mc2)
            plt.savefig(corner_name, bbox_inches='tight')
            plt.close()
            return None, run_samples2
        
        else: # 4.2 DO NOT plot 1st + 2nd post
            print("\033[43m" + 'WARNING:' + "\033[0m " + 
                  'No plot generated due to either/both posterior(s) failed.')
            return None, None


def post_corner_hist(getdist_plotter, latex_names, no_plot1, no_plot2, 
                     par_names1=None, par_names2=None, mc1=None, mc2=None):
    # Step 7. More 1D histogram settings
    # Plot 1 enabled + Plot 2 disabled
    if (no_plot1 is not True) & (no_plot2 is True):
        for i, p in enumerate(par_names1):
            ax = getdist_plotter.subplots[i, i]
        
            m1, s1 = mc1.mean(p), mc1.std(p)
            latex_name = latex_names[i]
        
            ax.set_title(
                rf'${latex_name}$' '\n'
                f'{m1:.2f} ' r'$\pm$' f' {s1:.2f}',
                fontsize=12
            )
            ax.tick_params(labelsize=8, top=True, labeltop=True)
            ax.grid(linestyle=':', axis='x', alpha=1)
    
    # Plot 1 disabled + Plot 2 enabled
    elif (no_plot1 is True) & (no_plot2 is not True):
        for i, p in enumerate(par_names2):
            ax = getdist_plotter.subplots[i, i]
        
            m2, s2 = mc2.mean(p), mc2.std(p)
            latex_name = latex_names[i]
        
            ax.set_title(
                rf'${latex_name}$' '\n'
                f'{m2:.2f} ' r'$\pm$' f' {s2:.2f}',
                fontsize=12
            )
            ax.tick_params(labelsize=8, top=True, labeltop=True)
            ax.grid(linestyle=':', axis='x', alpha=1)
    
    # Both plots are enabled
    else:
        for i, p in enumerate(par_names2):
            ax = getdist_plotter.subplots[i, i]
        
            m1, s1 = mc1.mean(p), mc1.std(p)
            m2, s2 = mc2.mean(p), mc2.std(p)
            latex_name = latex_names[i]
        
            ax.set_title(
                rf'${latex_name}$' '\n'
                f'A: {m1:.2f} ' r'$\pm$' f' {s1:.2f}' '\n'
                f'B: {m2:.2f} ' r'$\pm$' f' {s2:.2f}',
                fontsize=12
            )
            ax.tick_params(labelsize=8, top=True, labeltop=True)
            ax.grid(linestyle=':', axis='x', alpha=1)
        
    # Step 8. More 2D contours settings
    par_names1 = par_names1 if par_names2 is None else par_names2
    for i in range(len(par_names1)):
        for j in range(i):
            ax = getdist_plotter.subplots[i, j]
            ax.tick_params(labelsize=12)
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
            ax.grid(linestyle=':', alpha=1)
    return


if __name__ == '__main__':
    import joblib
    test = False #   True
    
    print('Corner plotting is started...')

    # runs_folder_new = '/xdisk/timeifler/jiyundi/kl_mmt/scripts_Pranjal/runs_RC+img/' # os.getcwd()
    import os
    runs_folder_new = os.getcwd() + '/'
    runs_folder_old = runs_folder_new + '../runs_spec+img_given_LPF/'

    label_new = r'$\chi^2_{\rm image} + \chi^2_{\rm RC}$ (no LPF given)'
    label_old = r'$\chi^2_{\rm image} + \chi^2_{\rm spec}$ (given LPF)'
    
    color_new, alpha_new = 'orange', 1.0
    color_old, alpha_old = 'green', 1.0

    # mock_folder = '/xdisk/timeifler/jiyundi/kl_mmt/scripts_Pranjal/008b_vary_thetaint_slitLPA_major/'
    mock_folder = runs_folder_new + '../008b_vary_thetaint_slitLPA_major/'
    
    # fit_config_yaml = "/xdisk/timeifler/jiyundi/kl_mmt/scripts_Pranjal/binospec_fitting_params_Pranjal.yaml"
    fit_config_yaml = runs_folder_new + "../binospec_fitting_params_Pranjal.yaml"
    with open(fit_config_yaml, "r", encoding="utf-8") as file2:
        fitting_params  = yaml.safe_load(file2)

    for slit_num in np.arange(1, 11): 

        new_slit_folder  = f'{runs_folder_new}Slit_{slit_num:03d}/'
        old_slit_folder  = f'{runs_folder_old}Slit_{slit_num:03d}/'
        new_posttxt_path = f'{runs_folder_new}Slit_{slit_num:03d}/post.txt'
        old_posttxt_path = f'{runs_folder_old}Slit_{slit_num:03d}/post.txt'
        
        print('\n============================================================')
        if Path(new_posttxt_path).exists() is False:
            print(f'posterior for {slit_num} does not exist: {new_posttxt_path}')
            continue
        print(f'Plotting for Slit {slit_num}...')

        # Mock params
        special_idxs = ['0.0', '0.3', '0.7', '1.0', '1.4', '1.6', '1.7', '2.1', '2.4', '2.8', '3.1']
        with open(f'{mock_folder}/slit_002_{special_idxs[slit_num]}.pkl', "rb") as f:
            data_info = joblib.load(f)
        
        # Calibrate v_0 due to different definitions of Hb
        c_kms = 299792.458
        Hb_wave_Pranjal = 486.1333 * u.nm
        Hb_wave_JD = 4862.683 * u.Angstrom
        wave_shift = Hb_wave_Pranjal - Hb_wave_JD
        v0_shift   = c_kms * wave_shift / Hb_wave_JD
        v0_Pranjal = data_info['fid_params']['Hb_params']['v_0']
        data_info['fid_params']['Hb_params']['v_0'] = v0_Pranjal + v0_shift.decompose() # convert units

        true_values = [
            data_info['fid_params']['shared_params']['gamma_t'],   # g_t
            data_info['fid_params']['shared_params']['theta_int'], # theta_int
            data_info['fid_params']['shared_params']['vcirc'],     # vcirc
            data_info['fid_params']['shared_params']['cosi'],      # cosi
            data_info['fid_params']['shared_params']['r_hl_disk'], # r_hl_disk
            data_info['fid_params']['shared_params']['flux'],      # flux
            data_info['fid_params']['shared_params']['dx_disk'],   # dx_disk
            data_info['fid_params']['shared_params']['dy_disk'],   # dy_disk
            data_info['fid_params']['shared_params']['vscale'],    # vscale
            data_info['fid_params']['Hb_params']['v_0'], # v0
            data_info['fid_params']['Hb_params']['I01'], # I0
            data_info['fid_params']['Hb_params']['bkg_level'], # bkg_level
            data_info['fid_params']['Hb_params']['dx_vel'], # dx_vel
        ]
        
        # If found an old post
        if (old_posttxt_path is not None): 
            if Path(old_posttxt_path).exists():
                plot_corner(
                    [new_posttxt_path, old_posttxt_path],  
                    [f'#{slit_num} {label_new}', f'#{slit_num} {label_old}'],  
                    contour_color = [color_new, color_old],
                    contour_alpha = [{'alpha': alpha_new}, {'alpha': alpha_old}],
                    read_latex_from=fitting_params,
                    true_values=true_values,
                    corner_name=f'{new_slit_folder}corner_compare.png',
                    test=test,
                    )
            else:
                print(f'Old posterior for {slit_num} does not exist: {old_posttxt_path}')
        
        # In case there is no a good posterior for old
        else:
            plot_corner(
                [new_posttxt_path],  
                [f'#{slit_num}'],  
                read_latex_from=fitting_params,
                true_values=true_values,
                corner_name=f'{new_slit_folder}corner_plot.png',
                test=test,
                )


