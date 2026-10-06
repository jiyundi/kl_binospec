import os
os.chdir('../')

import json
import yaml
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.fitting_result_utils import complete_fit_params
from core.plot_corner  import read_post

from summary.survey_bad_g1g2 import check_g1g2_post

from klm.safe_plot import setup; setup() # must before plt
plt.style.use('default')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})



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
        err_lo, median, err_hi = x123[0]-x123[1], x123[1], x123[2]-x123[1]
        
        if ('-g1' in config_param_names[j]) or \
            ('-g2' in config_param_names[j]):
                
            is_g_bad = check_g1g2_post(
                samp_points, pmin=-0.15, pmax=0.15
                )
            
            if is_g_bad:
                # flip signs of low/high errors on purpose
                err_lo *= -1
                err_hi *= -1
                print(f'Bad shear detected: {config_param_names[j]}')
        
        arr[j] = np.around([median, err_lo, err_hi], decimals=4)
        
        try:
            dic_percent[config_param_names[j]] = {'median': arr[j,0], 
                                                  'err_lo': arr[j,1], 
                                                  'err_hi': arr[j,2]}
        except IndexError:
            dic_percent = 'ERROR'
            break

    return dic_percent




















if __name__ == '__main__':
    base_dir        = './scripts_Pranjal/'
    post_dir        = base_dir
    run_dir_suffix  = '_flipped'
    real_pkl_folder = './scripts_Pranjal/'
    pkl_suffix      = '_flipped'
    
    # If any, specify slits that were not finished
    slits_not_finished = [2]
    
    # Add some columns
    g_cat = np.array([
        ['slit_ID', 'slit_name', 'slit_RA', 'slit_DEC', 'log10_Mstar', 'log10_Mstar_err'],
        [        1,    'b_007' ,   np.nan ,    np.nan ,       np.nan ,           np.nan ],
        [        2,    'b_008' ,   np.nan ,    np.nan ,       np.nan ,           np.nan ],
        [        3,    'c_007' ,   np.nan ,    np.nan ,       np.nan ,           np.nan ], 
        ])
    
    for slit_num in [1, 2, 3]:
        
        try:
            with open(f'{real_pkl_folder}pkl/slit_{slit_num:03d}{pkl_suffix}.pkl', "rb") as f:
                data_info = joblib.load(f)
        except FileNotFoundError: 
            print( "\033[43m" + 'WARNING: ' + "\033[0m " + 
                  f'Slit {slit_num} skipped because no pkl found.\n')
            continue
        
        # Match RA/DEC
        g_cat[slit_num, 2:4] = data_info['galaxy']['RA'], data_info['galaxy']['Dec']
        
        # Check if already solved M_stellar
        log10_Mstar     = data_info['galaxy']['log10_Mstar']
        log10_Mstar_err = data_info['galaxy']['log10_Mstar_err']
        g_cat[slit_num, 4:6] = log10_Mstar, log10_Mstar_err
        
        if slit_num in slits_not_finished:
            print( "\033[43m" + 'WARNING: ' + "\033[0m " + 
                  f'Slit {slit_num} skipped since you specified the fitting is not finished.\n')
            continue
        
        full_run_dir_1 = f'{post_dir}Slit_{slit_num:03d}{run_dir_suffix}/'
        if Path(full_run_dir_1).exists() is False:
            print( "\033[43m" + 'WARNING: ' + "\033[0m " + 
                  f'Slit {slit_num} does not exist.\n')
            continue
        
        post_path1 = f'{full_run_dir_1}post.txt'
        best_path1 = f'{full_run_dir_1}best_fit.json'
        
        run_samples1, par_names1, weights1, \
        loglikes1, samples1, mask1, no_plot1 = read_post(
            post_path1, percentile=95)
        samples1 = run_samples1[1:, 2:].astype(float)
        
        # Check bad g1/g2, pass posterior txt --> dict
        alllinespecies = []
        for spec in data_info['spec']:
            alllinespecies.append(spec['meta']['line_species'])
        percentile = analyze_percentile(samples1, 
                                        "config/binospec_fitting_params_Pranjal.yaml", 
                                        alllinespecies)
        if not isinstance(percentile, dict): 
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Slit {slit_num} hint: your run does not match your current pkl or config yaml. Skipped.\n')
            continue
        
        # Add best params into percentile array
        with open(best_path1, "r") as f:
            best_par = json.load(f)['maximum_likelihood']['point']
        with open(best_path1, "r") as f:
            peak_par = json.load(f)['posterior_mode'    ]['point']
        for key, vals in percentile.items():
            percentile[key]['best'] = best_par[key]
            percentile[key]['peak'] = peak_par[key]
        
        # Add more columns to contain posteriors
        n_rows, n_cols = g_cat.shape
        for key in list(percentile.keys()):
            # We only do once in the slit for loop!
            if n_cols <= 6: 
                if key.split('-')[0] == 'shared_params':
                    par = key.split('-')[1]
                    g_cat = np.append(
                        g_cat, 
                        np.array([
                            [f'{par}'       ]+[np.nan]*(n_rows-1), # col 19
                            [f'{par}_peak'  ]+[np.nan]*(n_rows-1), # col 20
                            [f'{par}_median']+[np.nan]*(n_rows-1), # col 21
                            [f'{par}_err-'  ]+[np.nan]*(n_rows-1), # col 22
                            [f'{par}_err+'  ]+[np.nan]*(n_rows-1), # col 23
                            ]).T, 
                        axis=1)
            
            # Assign values from posterior
            par = key.split('-')[1]
            for j in range(len(g_cat[1])):
                head_key = str(g_cat[0, j])
                if par == head_key:
                    g_cat[slit_num, j:j+5] = list([
                        percentile[key]['best'], 
                        percentile[key]['peak'], 
                        percentile[key]['median'], # or mean
                        percentile[key]['err_lo'], 
                        percentile[key]['err_hi'], 
                        ])

        print("\033[42m" + 'INFO:    ' + "\033[0m " + 
              f'Slit {slit_num} recorded. 👍\n')
    
    
    # g1/g2 --> gt/gx
    A2261_ctr_RA  = 260.612917 # ( (17)+(22)/60+(26.986)/3600 ) * 15 
    A2261_ctr_DEC =  32.133889 # ( (32)+( 7)/60+(57.89 )/3600 )
    RAs, DECs = g_cat[1:, 2].astype(float), g_cat[1:, 3].astype(float)
    
    treat_best_fit_as_gs = False
    treat_mode_as_gs     = False
    treat_median_as_gs   = True
    if treat_best_fit_as_gs:
        g1s, g2s = g_cat[1:, 6].astype(float), g_cat[1:, 11].astype(float)
    elif treat_mode_as_gs:
        g1s, g2s = g_cat[1:, 7].astype(float), g_cat[1:, 12].astype(float)
    elif treat_median_as_gs:
        g1s, g2s = g_cat[1:, 8].astype(float), g_cat[1:, 13].astype(float)
    else:
        raise IndexError('What do you want?')
    
    # according to 68% CI around median if assuming Gaussian post
    g1_errs_nega, g1_errs_posi = g_cat[1:,  9].astype(float), g_cat[1:, 10].astype(float)
    g2_errs_nega, g2_errs_posi = g_cat[1:, 14].astype(float), g_cat[1:, 15].astype(float)
    g1_errs = np.nanmean([-g1_errs_nega, g1_errs_posi], axis=0) # average
    g2_errs = np.nanmean([-g2_errs_nega, g2_errs_posi], axis=0) # average
    assert np.all([(g1_err > 0 or np.isnan(g1_err)) for g1_err in g1_errs]) 
    assert np.all([(g2_err > 0 or np.isnan(g2_err)) for g2_err in g2_errs])
    
    # Polar coordinates
    Rs = 60 * np.sqrt((RAs - A2261_ctr_RA)**2 + 
                      (DECs - A2261_ctr_DEC)**2)
    theta_cl_rad = np.arctan2(DECs - A2261_ctr_DEC, 
                              RAs  - A2261_ctr_RA)
    
    # Calculate tangential shear g+ by using Pranjal+22
    gts = -(g1s * np.cos(2*theta_cl_rad) + g2s * np.sin(2*theta_cl_rad))
    gxs =  (g1s * np.sin(2*theta_cl_rad) - g2s * np.cos(2*theta_cl_rad))
    
    # σ^​2(g+) ​= (cos2ϕ)^2 σ^2(g1) ​+ (sin2ϕ)^2 σ^2(g2)
    # σ^​2(gx) ​= (sin2ϕ)^2 σ^2(g1) ​+ (cos2ϕ)^2 σ^2(g2)
    gt_errs = (np.cos(2*theta_cl_rad)**2 * g1_errs**2 + 
               np.sin(2*theta_cl_rad)**2 * g2_errs**2 )**0.5
    gx_errs = (np.sin(2*theta_cl_rad)**2 * g1_errs**2 + 
               np.cos(2*theta_cl_rad)**2 * g2_errs**2 )**0.5
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4,6), gridspec_kw={'height_ratios': [3, 1]}, dpi=200)
    plt.subplots_adjust(hspace=0.3, wspace=0.6) # (height, width)
    ax[0].scatter(A2261_ctr_RA, A2261_ctr_DEC, s=80, marker='x', label='A2261')
    ax[0].scatter(RAs[0], DECs[0], s=40, marker='*', color='red', label=f'#1: {g_cat[1, 1]}')
    ax[0].scatter(RAs[1], DECs[1], s=40, marker='*', color='green', label=f'#2: {g_cat[2, 1]}')
    ax[0].scatter(RAs[2], DECs[2], s=40, marker='*', color='blue', label=f'#3: {g_cat[3, 1]}')
    ax[0].text(RAs[0], DECs[0], 
               r' $\theta_\mathrm{cluster} =$'+f'{int(theta_cl_rad[0]*57.3)}'+r'$^\circ$'+'\n'+
               r' (g$_1$, g$_2$) = '+f'({g1s[0]:.2f}, {g2s[0]:.2f})', 
               size=8, va='top')
    ax[0].text(RAs[1], DECs[1], 
               r' $\theta_\mathrm{cluster} =$'+f'{int(theta_cl_rad[1]*57.3)}'+r'$^\circ$'+'\n'+
               r' (g$_1$, g$_2$) = '+f'({g1s[1]:.2f}, {g2s[1]:.2f})', 
               size=8)
    ax[0].text(RAs[2], DECs[2], 
               r' $\theta_\mathrm{cluster} =$'+f'{int(theta_cl_rad[2]*57.3)}'+r'$^\circ$'+'\n'+
               r' (g$_1$, g$_2$) = '+f'({g1s[2]:.2f}, {g2s[2]:.2f})', 
               size=8, va='top', ha='right')
    ax[0].set_xlim(A2261_ctr_RA +np.max(Rs)/45, A2261_ctr_RA -np.max(Rs)/45)
    ax[0].set_ylim(A2261_ctr_DEC-np.max(Rs)/45, A2261_ctr_DEC+np.max(Rs)/45)
    ax[0].set_xlabel('<--- East            RA            West --->')
    ax[0].set_ylabel('DEC')
    ax[0].legend()
    ax[0].set_aspect('equal')
    ax[1].scatter([1, 2, 3], [0.208, 0.041, 0.144], 
                  c=['red', 'green', 'blue'], marker='s', s=30, label='Pranjal+24')
    ax[1].scatter([1, 2, 3], gts, 
                  c='orange', s=20, label='Jiyun\'s refit')
    ax[1].set_xlabel('Slit #')
    ax[1].set_ylabel('g_t')
    ax[1].legend()
    plt.savefig(f'survey_{run_dir_suffix}.png', dpi=200, bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    