import yaml
import joblib
import getdist.plots
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']  = 300
plt.rcParams['savefig.dpi'] = 300
# plt.rcParams.update({
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
#     "font.serif": "Helvetica",
# })
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['font.sans-serif'] = ['Helvetica']
# plt.rcParams['mathtext.it'] = 'Helvetica:italic'
# plt.rcParams['mathtext.bf'] = 'Helvetica:bold'

def read_post(full_path, params_NOT_to_plot=None, percentile=0, 
              equal_weights=False, force_to_use_weights=False):
    
    runsample = np.loadtxt(full_path, dtype=str, skiprows=0)
    
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
    
    if not force_to_use_weights:
        mask  = weights > np.percentile(weights, percentile)
        weights_masked = weights[mask]
        weight_high    = np.percentile(weights_masked, 90)
    
        try:
            assert weight_high >= 1/len(samples),  \
                   'No enough well-weighted samples. Consider not to include weights.'
        
        except AssertionError:
            if equal_weights:
                weights[ mask] = 1/len(weights_masked)
                weights[~mask] = 0
                print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                      f'{full_path}: No enough well-weighted samples. Changed to equal weights.\n')
            
            else:
                raise AssertionError(
                    'No enough well-weighted samples. Consider not to include weights.'
                    )
    
    return runsample, par_names, weights, loglikes, samples, mask

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
        # Found unmatched columns
        else:
            # Recover first-row header
            out[0,  j] = name
            
            # Assign special name by desired value
            if \
                name == 'shared_params-g1':
                out[1:, j] = 0 + np.random.normal(scale=0.001, 
                                                  size=(len(out)-1))
            elif \
                name == 'shared_params-g2':
                out[1:, j] = 0 + np.random.normal(scale=0.001, 
                                                  size=(len(out)-1))
            
            elif \
                name == 'shared_params-dx_bulge':
                out[1:, j] = 0 + np.random.normal(scale=0.001, 
                                                  size=(len(out)-1))
            
            elif \
                name == 'shared_params-dy_bulge':
                out[1:, j] = 0 + np.random.normal(scale=0.001, 
                                                  size=(len(out)-1))
            
            # Assign other param name by a reasonable artificial value
            elif \
                name == 'shared_params-vcirc':
                out[1:, j] = 200 * np.random.lognormal(mean=0.0, 
                                                       sigma=1.0, 
                                                       size=(len(out)-1))
            elif \
                name == 'shared_params-vscale' or \
                name == 'shared_params-r_hl_disk' or \
                name == 'shared_params-cosi':
                out[1:, j] = 0.5 * np.random.lognormal(mean=0.0, 
                                                       sigma=1.0, 
                                                       size=(len(out)-1))
            else:
                out[1:, j] = np.random.normal(loc=0, 
                                              scale=0.1, 
                                              size=(len(out)-1))
    return out

def plot_corner(full_run_dirs, labels, 
                params_to_plot=None, 
                params_NOT_to_plot=None,
                percentile=95, 
                change_to_equal_weights_in_case=False, 
                force_to_use_weights=False,
                corner_name='corner_compare.png'):
    # Step 1. Read posteriors (weighted)
    if isinstance(full_run_dirs, str):
        full_run_path1 = full_run_dirs
        label1         = labels
        
        full_run_path2 = None
        label2         = None
        
        run_samples1, par_names1, weights1, \
        loglikes1, samples1, mask1 = read_post(
            full_run_path1, 
            params_NOT_to_plot,
            percentile=percentile,
            equal_weights=change_to_equal_weights_in_case, 
            force_to_use_weights=force_to_use_weights)
        
        par_names1 = run_samples1[0,  2:]
        samples1   = run_samples1[1:, 2:].astype(float)

    else:
        full_run_path1 = full_run_dirs[0]
        full_run_path2 = full_run_dirs[1]
        label1         = labels[0]
        label2         = labels[1]
        
        run_samples1, par_names1, weights1, \
        loglikes1, samples1, mask1 = read_post(
            full_run_path1, 
            params_NOT_to_plot,
            percentile=percentile,
            equal_weights=change_to_equal_weights_in_case, 
            force_to_use_weights=force_to_use_weights)
        
        run_samples2, par_names2, weights2, \
        loglikes2, samples2, mask2 = read_post(
            full_run_path2, 
            params_NOT_to_plot,
            percentile=percentile,
            equal_weights=change_to_equal_weights_in_case, 
            force_to_use_weights=force_to_use_weights)
    
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
                
    # Step 3. Choose your param to plot
    if params_to_plot is not None:
        par_names_idx1 = [
            list(par_names1).index(params_to_plot[i]) 
            for i in range(len(params_to_plot))
            ]
        if not isinstance(full_run_dirs, str): # not only one post
            par_names_idx2 = [
                list(par_names2).index(params_to_plot[i]) 
                for i in range(len(params_to_plot))
                ]
    else:
        par_names_idx1 = list(np.arange(len(par_names1)))
        if not isinstance(full_run_dirs, str): # not only one post
            par_names_idx2 = list(np.arange(len(par_names2)))
    
    # Step 4. Read emission lines and fitting param to get latex names
    lines1 = []
    for par in par_names1:
        lv1_key, lv2_key = par.split('-')
        if lv1_key != 'shared_params':
            lines1.append(lv1_key.split('_')[0])
    
    with open('./config/binospec_fitting_params.yaml', 
              "r", encoding="utf-8") as yamlfile:
        fit_par1 = yaml.safe_load(yamlfile)
    
    latex_names = []
    for lv1key, subdict in fit_par1.items():
        if lv1key == 'shared_params':
            for lv2key, par_dict in subdict.items():
                if f'shared_params-{lv2key}' not in params_NOT_to_plot:
                    latex_names.append(
                        par_dict['latex_name'][1:-1]
                        )
        else:
            for line in list(dict.fromkeys(lines1)):
                for lv2key, par_dict in subdict.items():
                    if f'{lv1key}-{lv2key}' not in params_NOT_to_plot:
                        latex_names.append(
                            '\mathrm{'+f'{line}'+': }'+par_dict['latex_name'][1:-1]
                            )
                        if line == 'O2':
                            latex_names.append(
                                '\mathrm{'+f'{line}b'+': }'+par_dict['latex_name'][1:-1]
                                )
    latex_names = np.array(latex_names)
    
    # Step 5. Pack in getdist/MCSamples
    mc1 = getdist.MCSamples(
        names    = par_names1[par_names_idx1],
        weights  = weights1[ mask1],
        loglikes = loglikes1[mask1],
        samples  = samples1[ mask1][:, par_names_idx1],
        labels   = latex_names[par_names_idx1], 
        )
    if not isinstance(full_run_dirs, str): # not only one post
        mc2 = getdist.MCSamples(
            names    = par_names2[par_names_idx2],
            weights  = weights2[ mask2],
            loglikes = loglikes2[mask2],
            samples  = samples2[ mask2][:, par_names_idx2],
            labels   = latex_names[par_names_idx2], 
            )
    
    # Step 6. Plot settings
    getdist_plotter = getdist.plots.get_subplot_plotter(subplot_size = 1.6)
    getdist_plotter.settings.legend_fontsize = 30
    getdist_plotter.settings.progress = True
    if isinstance(full_run_dirs, str):
        getdist_plotter.triangle_plot(mc1, 
            filled        = True, 
            legend_labels = 'A: '+label1,
            contour_args  = {'alpha': 0.5},
            title_limit   = 1, # 1σ
            title_fmt     = '.2f', 
            smooth1d = 0, # bypass KDE smoother
            smooth2d = 0, # bypass KDE smoother
            )
    else:
        getdist_plotter.triangle_plot([mc1, mc2], 
            filled        = True, 
            legend_labels = ['A: '+label1, 'B: '+label2],
            contour_colors= ['green',   'deepskyblue'],
            contour_args  = {'alpha': 0.5},
            title_limit   = 1, # 1σ
            title_fmt     = '.2f', 
            smooth1d = 0, # bypass KDE smoother
            smooth2d = 0, # bypass KDE smoother
            )
    
    # Step 7. More 1D histogram settings
    if isinstance(full_run_dirs, str):
        for i, p in enumerate(par_names1[par_names_idx1]):
            ax = getdist_plotter.subplots[i, i]
        
            m1, s1 = mc1.mean(p), mc1.std(p)
            latex_name = latex_names[par_names_idx1][i]
        
            ax.set_title(
                rf'${latex_name}$' '\n'
                f'{m1:.2f} ' r'$\pm$' f' {s1:.2f}',
                fontsize=12
            )
            ax.tick_params(labelsize=8, top=True, labeltop=True)
            ax.grid(linestyle=':', axis='x', alpha=1)
    else:
        for i, p in enumerate(par_names2[par_names_idx2]):
            ax = getdist_plotter.subplots[i, i]
        
            m1, s1 = mc1.mean(p), mc1.std(p)
            m2, s2 = mc2.mean(p), mc2.std(p)
            latex_name = latex_names[par_names_idx2][i]
        
            ax.set_title(
                rf'${latex_name}$' '\n'
                f'A: {m1:.2f} ' r'$\pm$' f' {s1:.2f}' '\n'
                f'B: {m2:.2f} ' r'$\pm$' f' {s2:.2f}',
                fontsize=12
            )
            ax.tick_params(labelsize=8, top=True, labeltop=True)
            ax.grid(linestyle=':', axis='x', alpha=1)
        
    # Step 8. More 2D contours settings
    for i in range(len(par_names1[par_names_idx1])):
        for j in range(i):
            ax = getdist_plotter.subplots[i, j]
            ax.tick_params(labelsize=12)
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
            ax.grid(linestyle=':', alpha=1)
    
    # (Optional) Step 9. Mark true values for mock
    for i in range(len(par_names1[par_names_idx1])):
        for j in range(i):
            ax = getdist_plotter.subplots[i, j]
            # truth lines
            ax.axvline(true_to_plot[par_names1[par_names_idx1][j]], 
                       color='black', linestyle=':')
            ax.axhline(true_to_plot[par_names1[par_names_idx1][i]], 
                       color='black', linestyle=':')
    for k in range(len(par_names1[par_names_idx1])):
        ax = getdist_plotter.subplots[k,k]
        ax.vlines(true_to_plot[par_names1[par_names_idx1][k]], color='black', linestyle='--', 
                  ymin=0, ymax=1, transform=ax.get_xaxis_transform())
    
    plt.savefig(corner_name, bbox_inches='tight')
    
    if isinstance(full_run_dirs, str):
        return run_samples1
    else:
        return run_samples1, run_samples2

if __name__ == '__main__':
    # params_to_plot = [
    #     'shared_params-g1',
    #     'shared_params-g2',
    #     'shared_params-theta_int',
    #     'shared_params-vcirc',
    #     'shared_params-vscale',
    #     'shared_params-cosi',
    #     'shared_params-r_hl_disk',
    #     'shared_params-dx_disk',
    #     'shared_params-dy_disk',
    #     'Hb_params-v_0',
    #     'Hb_params-I01_spec1', 
    #     'Hb_params-I01_spec2',
    #     'Hb_params-I01_spec3', 
    #     'O3b_params-v_0', 
    #     'O3b_params-I01_spec1',
    #     'O3b_params-I01_spec2', 
    #     'O3b_params-I01_spec3'
    # ]
    
    params_NOT_to_plot = [
        # 'shared_params-vscale',
        'shared_params-flux',
        'shared_params-dx_disk',
        'shared_params-dy_disk',
        'shared_params-dx_bulge',
        'shared_params-dy_bulge',
        "O2_params-v_0",
        "O2_params-v_0_2",
        "O2_params-I01_spec1",
        "O2_params-I02_spec1",
        "O2_params-I01_spec2",
        "O2_params-I02_spec2",
        "O2_params-I01_spec3",
        "O2_params-I02_spec3",
        "Hb_params-v_0",
        "Hb_params-I01_spec1",
        "Hb_params-I01_spec2",
        "Hb_params-I01_spec3",
        
        'shared_params-r_hl_disk', 
        'shared_params-r_hl_bulge', 
        'shared_params-flux_bulge',
        'shared_params-beta', 
        'shared_params-image_snr', 
        'shared_params-spec_snr', 
        'line_params-dx_vel', 
        'line_params-dy_vel', 
        'line_params-dx_vel_2', 
        'line_params-dy_vel_2', 
        'line_params-bkg_level', 
        'line_params-v_0', 
        'line_params-v_0_2'
        ]
    
    slit_num = 3
    for case_num in [1,2,3,4,5,
                     11,12,13,14,15,
                     21,22,23,24,25,
                     31,32,33,34,35,
                     41,42,43,44,45,
                     51,52,53,54,55]: # 5,23,55,72,95,
        full_run_dir_1 = f'../../../../../RSCH3/kl_github/mock/slit_003_runs/Slit_{slit_num:03d}_{case_num:03d}/post.txt'
        full_run_dir_2 = f'Slit_{slit_num:03d}_runs_img/run0.01/run1/chains/weighted_post.txt'
        
        # (Optional) See Step 9.
        with open(f'binospec_multi_mock/pkl/mock_{slit_num:03d}_{case_num:03d}.pkl', "rb") as f:
            mock_params = joblib.load(f)['par_fit']
        true_to_plot = mock_params
        for key in params_NOT_to_plot:
            if key.split('-')[0] == 'shared_params' \
                or key.split('-')[0] == 'line_params':
                true_to_plot.pop(key)
            else:
                try:
                    if '_spec' in key.split('-')[1]:
                        true_to_plot.pop('line_params-'+key.split('-')[1].split('_')[0])
                except KeyError:
                    pass
        
        # try:
        plot_corner(full_run_dir_1, 
                    'Nautilus (new)',
                    # params_to_plot = params_to_plot,
                    params_NOT_to_plot = params_NOT_to_plot,
                    percentile=0, 
                    change_to_equal_weights_in_case=True, 
                    force_to_use_weights=True,
                    corner_name=f'corner_{slit_num:03d}_{case_num:03d}.jpg'
                    )
        # except:
        #     pass




