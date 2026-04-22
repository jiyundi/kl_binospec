from pathlib import Path
import yaml
import numpy as np

import getdist.plots

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']  = 300
plt.rcParams['savefig.dpi'] = 300


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


def read_post(full_path, params_NOT_to_plot=None, percentile=0, 
              equal_weights=False, force_to_use_weights=False, 
              post_is_bad=True):
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
                percentile=95, 
                change_to_equal_weights_in_case=False,
                force_to_use_weights=False,
                corner_name='corner_compare.png', test=False, 
                nautilus_color=None):
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
        loglikes1, samples1, mask1, no_plot1 = read_post(
            full_run_path1, 
            params_NOT_to_plot,
            percentile=percentile,
            equal_weights=change_to_equal_weights_in_case, 
            force_to_use_weights=force_to_use_weights)
        
        run_samples2, par_names2, weights2, \
        loglikes2, samples2, mask2, no_plot2 = read_post(
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
                else:
                    if params_NOT_to_plot is not None:
                        if f'shared_params-{lv2key}' in params_NOT_to_plot:
                            latex_names.pop()
                            latex_to_parname.pop()
        else:
            for line in list(dict.fromkeys(lines1)):
                for lv2key, par_dict in subdict.items():
                    
                    config_par_to_find = f'{lv1key}-{lv2key}'
                    if config_par_to_find in params_to_plot:
                        latex_names.append(
                            '\mathrm{'+f'{line}'+': }'+par_dict['latex_name'][1:-1]
                            )
                        latex_to_parname.append(f'{lv1key}-{lv2key}')
                        if line == 'O2':
                            latex_names.append(
                                '\mathrm{'+f'{line}b'+': }'+par_dict['latex_name'][1:-1]
                                ) # Note: this is "b" here
                            if lv2key == 'v_0':
                                latex_to_parname.append(f'{line}_params-v_0_2')
                            else:
                                latex_to_parname.append(f'{line}_params-I02_{lv2key.split("_")[1]}')
                
                    elif lv1key.split('_')[0] == 'line':
                        config_par_to_find = f'{line}_params-{lv2key}'
                        if config_par_to_find in params_to_plot:
                            latex_names.append(
                                '\mathrm{'+f'{line}'+': }'+par_dict['latex_name'][1:-1]
                                )
                            latex_to_parname.append(f'{line}_params-{lv2key}')
                            if line == 'O2':
                                latex_names.append(
                                    '\mathrm{'+f'{line}b'+': }'+par_dict['latex_name'][1:-1]
                                    ) # Note: this is "b" here
                                if lv2key == 'v_0':
                                    latex_to_parname.append(f'{line}_params-v_0_2')
                                else:
                                    latex_to_parname.append(f'{line}_params-I02_{lv2key.split("_")[1]}')
                    
                    else:
                        if params_NOT_to_plot is not None:
                            if f'{lv1key}-{lv2key}' not in params_NOT_to_plot:
                                latex_names.pop()
                                latex_to_parname.pop()
                                if line == 'O2':
                                    latex_names.pop()
                                    latex_to_parname.pop()
    latex_names = np.array(latex_names)
    
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
    
    # contours (1 & 2) will be OK
    if no_plot1 is not True: # 1.1. Plot 1st post
        if no_plot2 is True: # 1.2. DO NOT plot 2nd post
            
            # Scripts removed one post
            if nautilus_color is None: 
                getdist_plotter.triangle_plot(mc1, 
                    filled        = True, 
                    legend_labels = ['A: '+label1+'\n< B: No posterior >'],
                    contour_colors= ['green'],
                    contour_args  = {'alpha': 0.5},
                    title_limit   = 1, # 1σ
                    title_fmt     = '.2f', 
                    smooth1d = 0, # bypass KDE smoother
                    smooth2d = 0, # bypass KDE smoother
                    )
            
            # User mandates post to be Nautilus
            else: 
                getdist_plotter.triangle_plot(mc1, 
                    filled        = True, 
                    legend_labels = [label1],
                    contour_colors= [nautilus_color],
                    contour_args  = {'alpha': 0.75},
                    title_limit   = 1, # 1σ
                    title_fmt     = '.2f', 
                    smooth1d = 0, # bypass KDE smoother
                    smooth2d = 0, # bypass KDE smoother
                    )
            
            post_corner_hist(getdist_plotter, latex_names, no_plot1, no_plot2, 
                             par_names1=par_names1, par_names2=None, 
                             mc1=mc1, mc2=None)
            plt.savefig(corner_name, bbox_inches='tight')
            return run_samples1, None
        
        else: # 2.2. Plot 1st & 2nd post
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
            post_corner_hist(getdist_plotter, latex_names, no_plot1, no_plot2, 
                             par_names1=par_names1, par_names2=par_names2, 
                             mc1=mc1, mc2=mc2)
            plt.savefig(corner_name, bbox_inches='tight')
            return run_samples1, run_samples2
    
    # contours will fail
    else: # 3.1. DO NOT plot 1st post1
        if no_plot2 is not True: # 3.2. But plot 2nd post
            getdist_plotter.triangle_plot(mc2, 
                filled        = True, 
                legend_labels = ['B: '+label2+'\n< A: No posterior >'],
                contour_colors= ['deepskyblue'],
                contour_args  = {'alpha': 0.5},
                title_limit   = 1, # 1σ
                title_fmt     = '.2f', 
                smooth1d = 0, # bypass KDE smoother
                smooth2d = 0, # bypass KDE smoother
                )
            post_corner_hist(getdist_plotter, latex_names, no_plot1, no_plot2, 
                             par_names1=None, par_names2=par_names2, 
                             mc1=None, mc2=mc2)
            plt.savefig(corner_name, bbox_inches='tight')
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
    params_to_plot = [
        'shared_params-g1',
        'shared_params-g2',
        'shared_params-cosi',
        'shared_params-theta_int',
        'shared_params-vcirc',
        'shared_params-r_hl_disk',
        'shared_params-vscale',
    ]
    
    # params_NOT_to_plot = [
    #     'shared_params-vscale',
    #     'shared_params-dx_bulge',
    #     'shared_params-dy_bulge',
    #     "O3b_params-v_0",
    #     "O3b_params-I01_spec1",
    #     "O3b_params-I01_spec2",
    #     "O3b_params-I01_spec3",
    #     "Hb_params-v_0",
    #     "Hb_params-I01_spec1",
    #     "Hb_params-I01_spec2",
    #     "Hb_params-I01_spec3",
    # ]
    
    test = False #   True
    slit_nums = [113] #  np.arange(116, 142)# 118,32,100,113, 18,64,65,67,70,95,97,102,108,128]# np.arange(1, 143)
    
    # slits_not_done  = [29, 57, 59, 112, 122]#, 48, 112, 116, 122]
    # slits_no_formal_nautilus = [48, 116]
    # slits_not_to_plot     = slits_not_done + slits_no_formal_nautilus
    # slits_not_to_plot_idx = [list(slit_nums).index(s) 
    #                          for s in slits_not_to_plot if s in list(slit_nums)]
    # slit_nums = np.delete(slit_nums, slits_not_to_plot_idx)
    
    for slit_num in slit_nums: 
        run_dir_old, date_of_run_old, post_path_old = None, None, None
        base_dir = '../../../RSCH3/kl_github/'
        # run_dir_old = f'{base_dir}runs_nautilus/Slit_{slit_num:03d}/'
        run_dir_new = f'{base_dir}runs_nautilus/Slit_{slit_num:03d}/'
        run_dir_new = f'{base_dir}the_converted/Slit_{slit_num:03d}/'
        # date_of_run_old = 'runs_20260328/'
        date_of_run_new = 'runs_20260421/'
        # post_path_old = f'{run_dir_old}{date_of_run_old}post.txt'
        post_path_new = f'{run_dir_new}{date_of_run_new}post.txt'
        corner_png_dir = run_dir_new + date_of_run_new
                
        print('\n============================================================')
        if Path(run_dir_new).exists() is False:
            print(f'Slit {slit_num} does not exist: {run_dir_new}')
            continue
        print(f'Plotting for Slit {slit_num}...')
        
        # If found an old post
        if (post_path_old is not None): 
            if Path(post_path_old).exists():
                plot_corner([post_path_old, post_path_new],  
                            [f'#{slit_num} Nautilus', f'#{slit_num} Nautilus \n         (g1, g2 fixed to 0)'],  
                            params_to_plot = params_to_plot,
                            # params_NOT_to_plot = params_NOT_to_plot,
                            percentile=0, 
                            change_to_equal_weights_in_case=True,
                            corner_name=f'{corner_png_dir}corner_compare.png',
                            test=test,
                            )
        
        # In case there is no a good posterior for old
        else:
            plot_corner([post_path_new],  
                        [f'#{slit_num} (2026-04-21)'],  
                        nautilus_color='dimgray', # deepskyblue
                        # params_to_plot = params_to_plot,
                        percentile=0, 
                        change_to_equal_weights_in_case=True,
                        corner_name=f'{corner_png_dir}corner_all.png',
                        test=test,
                        )


