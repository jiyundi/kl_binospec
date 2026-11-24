# import os
# os.chdir('../')

import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from adjustText import adjust_text
from astropy.io import fits
from astropy.visualization import make_lupton_rgb # Creating color RGB images
            
from data_prep.image.read_and_cutout import read_subaru_img_wcs, cutoutimg
# from data_prep.spec.read_and_cutout  import readinfodat
from core.fitting_result_utils import complete_fit_params
from core.plot_corner  import read_post

from summary.survey_not_overlaps import find_not_overlaps
from summary.survey_bad_g1g2 import check_g1g2_post

from klm.safe_plot import setup; setup() # must before plt
plt.style.use('default')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})



def posterior_to_cat(g_cat, post_dir, pkl_folder, spec1d2dfolder02, config_filepath, ztable, slits_not_finished=None):
    """
        Generate a shear catalog.
    """
    slits_not_finished = set(slits_not_finished or [])
    slitnums_, _, not_overlaps = find_not_overlaps(pkl_folder)
    overlap_by_slit = {}
    for slit, accepted in zip(slitnums_, not_overlaps):
        overlap_by_slit.setdefault(slit, []).append(accepted)

    # Keep the measurement columns in their existing order.
    with open(config_filepath, encoding='utf-8') as config_file:
        shared_params = yaml.safe_load(config_file)['shared_params']
    parameter_columns = {}
    for par in shared_params:
        start = g_cat.shape[1]
        labels = [par, f'{par}_median', f'{par}_err-', f'{par}_err+']
        columns = np.full((g_cat.shape[0], 4), np.nan, dtype=object)
        columns[1] = labels
        g_cat = np.concatenate((g_cat, columns), axis=1)
        parameter_columns[par] = start
    status_columns = {}
    for label in ('skyline_status', 'g1_status', 'g2_status', 'fit_status', 'quality_note'):
        status_columns[label] = g_cat.shape[1]
        column = np.full((g_cat.shape[0], 1), None, dtype=object)
        column[1, 0] = label
        g_cat = np.concatenate((g_cat, column), axis=1)

    def set_status(row, label, value):
        g_cat[row, status_columns[label]] = value
    
    for slit_num in range(1, 143):
        row = slit_num + 1
        for label in ('g1_status', 'g2_status'):
            set_status(row, label, 'unknown')
        set_status(row, 'fit_status', 'pending')
        goods_or_bads = overlap_by_slit.get(slit_num, [])
        if goods_or_bads:
            skyline_status = ('ok' if np.sum(goods_or_bads) >= 0.5 * len(goods_or_bads)
                              else 'bad')
        else:
            skyline_status = 'unknown'
        set_status(row, 'skyline_status', skyline_status)
        z_spec = ztable[slit_num, 1]
        
        # Match RA/DEC
        infdatfilename = f'info.829.{slit_num:03d}.{slit_num+100305:06d}.dat'
        dat_dict   = {}
        with open(spec1d2dfolder02 + infdatfilename) as infile_dat:
            for sen in infile_dat:
                if sen[:2] != 'ID':
                    dat_dict[sen.split('       ')[0]] = float(sen.split('       ')[1][:-1])
        g_cat[slit_num+1, 15:17] = dat_dict['RA'], dat_dict['DEC']
        
        if np.isnan(z_spec)==True:
            set_status(row, 'fit_status', 'no_redshift')
            set_status(row, 'quality_note', 'No secure redshift')
            print( "\033[43m" + 'WARNING: ' + "\033[0m " + 
                  f'Slit {slit_num} skipped because no secure redshift found.\n')
            continue
    
        elif np.isnan(z_spec)==False:
            # Load pkl
            try:
                with open(Path(pkl_folder) / 'pkl' / f'slit_{slit_num:03d}.pkl', "rb") as f:
                    data_info = joblib.load(f)
            except FileNotFoundError: 
                set_status(row, 'fit_status', 'no_pkl')
                set_status(row, 'quality_note', 'No pkl file')
                print( "\033[43m" + 'WARNING: ' + "\033[0m " + 
                      f'Slit {slit_num} skipped because no pkl found.\n')
                continue
            
            # Check if already solved M_stellar
            log10_Mstar = data_info['galaxy']['log10_Mstar']
            log10_Mstar_err = data_info['galaxy']['log10_Mstar_err']
            g_cat[slit_num+1, 17:19] = log10_Mstar, log10_Mstar_err
            del log10_Mstar, log10_Mstar_err
            
            if slit_num in slits_not_finished:
                set_status(row, 'fit_status', 'unfinished')
                set_status(row, 'quality_note', 'Fitting marked unfinished')
                print( "\033[43m" + 'WARNING: ' + "\033[0m " + 
                      f'Slit {slit_num} skipped since you specified the fitting is not finished.\n')
                continue
            
            full_run_dir_1 = f'{post_dir}Slit_{slit_num:03d}/'
            if Path(full_run_dir_1).exists() is False:
                set_status(row, 'fit_status', 'no_run')
                set_status(row, 'quality_note', 'Run directory missing')
                print(f'Slit {slit_num} does not exist.\n')
                continue
            
            post_path1 = f'{full_run_dir_1}post.txt'
            best_path1 = f'{full_run_dir_1}best_fit.json'
            
            if not Path(post_path1).is_file():
                set_status(row, 'fit_status', 'no_posterior')
                set_status(row, 'quality_note', 'Posterior file missing')
                print( "\033[45m" + 'WARNING: ' + "\033[0m " + 
                      f"Slit {slit_num} skipped because {post_path1} not found. \nPerhaps fitting was not completed for this slit? \n")
                continue
            
            run_samples1, par_names1, weights1, \
            loglikes1, samples1, mask1, no_plot1 = read_post(post_path1)
            samples1   = run_samples1[1:, 2:].astype(float)
            
            # Check bad g1/g2, pass posterior txt --> dict
            alllinespecies = []
            for spec in data_info['spec']:
                alllinespecies.append(spec['meta']['line_species'])
            percentile = analyze_percentile(samples1, 
                                            config_filepath, 
                                            alllinespecies)
            
            if not isinstance(percentile, dict): 
                set_status(row, 'fit_status', 'config_mismatch')
                set_status(row, 'quality_note', 'Posterior and configuration mismatch')
                print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                      f'Slit {slit_num} hint: your run does not match your current pkl or config yaml. Skipped.\n')
                continue
            
            # Check if specs are overlapped with any sky line masks
            if skyline_status == 'bad':
                print(f'Slit {slit_num:03d}: major specs are covered by sky lines.')
            
            # Add best params into percentile array
            try:
                with open(best_path1, "r") as f:
                    best_par = json.load(f)['maximum_likelihood']['point']
            except FileNotFoundError:
                set_status(row, 'fit_status', 'no_best_fit')
                set_status(row, 'quality_note', 'Best-fit file missing')
                continue
            if not all(key in best_par for key in percentile):
                set_status(row, 'fit_status', 'config_mismatch')
                set_status(row, 'quality_note', 'Best-fit parameters and posterior mismatch')
                continue
            for key, vals in percentile.items():
                percentile[key]['best'] = best_par[key]
            for par in ('g1', 'g2'):
                result = percentile.get(f'shared_params-{par}')
                if result is not None:
                    set_status(row, f'{par}_status', result['status'])
            
            # Add more columns to contain posteriors
            for key, values in percentile.items():
                if not key.startswith('shared_params-'):
                    continue
                par = key.split('-', 1)[1]
                if par in parameter_columns:
                    col = parameter_columns[par]
                    g_cat[slit_num+1, col:col+4] = (
                        values['best'], values['median'],
                        values['err_lo'], values['err_hi']
                    )
            set_status(row, 'fit_status', 'ok')
            notes = []
            if skyline_status == 'bad':
                notes.append('Major spectra affected by sky lines')
            for par in ('g1', 'g2'):
                if g_cat[row, status_columns[f'{par}_status']] == 'bad':
                    notes.append(f'{par} posterior rejected')
            set_status(row, 'quality_note', '; '.join(notes) or None)
    
            print("\033[42m" + 'INFO:    ' + "\033[0m " + 
                  f'Slit {slit_num} recorded. 👍\n')
    
    return g_cat


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
    
    if nparams != len(config_param_names):
        return 'ERROR'
    for j in range(nparams):
        if j >= len(config_param_names):
            return 'ERROR'
        samp_points = samples[:,j]
        x123 = np.percentile(samp_points, [16, 50, 84])
        err_lo, median, err_hi = x123[0]-x123[1], x123[1], x123[2]-x123[1]
        status = None
        if config_param_names[j] in ('shared_params-g1', 'shared_params-g2'):
            is_g_bad = check_g1g2_post(
                samp_points, 
                pmin=config_par[config_param_names[j]]['prior']['min'], 
                pmax=config_par[config_param_names[j]]['prior']['max']
            )
            status = 'bad' if is_g_bad else 'ok'
            if is_g_bad:
                print(f'Bad shear detected: {config_param_names[j]}')
        
        arr[j] = np.around([median, err_lo, err_hi], decimals=4)
        
        dic_percent[config_param_names[j]] = {
            'median': arr[j, 0], 'err_lo': arr[j, 1],
            'err_hi': arr[j, 2], 'status': status}

    return dic_percent


def binospec_fov(ax, ctr_pt,rotatn):
    ra0, de0 = ctr_pt[0], ctr_pt[1]
    cnrA1r = ra0 + ((-480-96)/3600)*np.cos(rotatn) - ((+450)/3600)*np.sin(rotatn)
    cnrA1d = de0 + ((-480-96)/3600)*np.sin(rotatn) + ((+450)/3600)*np.cos(rotatn)
    cnrA2r = ra0 + ((-000-96)/3600)*np.cos(rotatn) - ((+450)/3600)*np.sin(rotatn)
    cnrA2d = de0 + ((-000-96)/3600)*np.sin(rotatn) + ((+450)/3600)*np.cos(rotatn)
    cnrA3r = ra0 + ((-000-96)/3600)*np.cos(rotatn) - ((-450)/3600)*np.sin(rotatn)
    cnrA3d = de0 + ((-000-96)/3600)*np.sin(rotatn) + ((-450)/3600)*np.cos(rotatn)
    cnrA4r = ra0 + ((-480-96)/3600)*np.cos(rotatn) - ((-450)/3600)*np.sin(rotatn)
    cnrA4d = de0 + ((-480-96)/3600)*np.sin(rotatn) + ((-450)/3600)*np.cos(rotatn)
    Ax_values = [cnrA1r, cnrA2r, cnrA3r, cnrA4r, cnrA1r]
    Ay_values = [cnrA1d, cnrA2d, cnrA3d, cnrA4d, cnrA1d]
    cnrB1r = ra0 + ((+480+96)/3600)*np.cos(rotatn) - ((+450)/3600)*np.sin(rotatn)
    cnrB1d = de0 + ((+480+96)/3600)*np.sin(rotatn) + ((+450)/3600)*np.cos(rotatn)
    cnrB2r = ra0 + ((+000+96)/3600)*np.cos(rotatn) - ((+450)/3600)*np.sin(rotatn)
    cnrB2d = de0 + ((+000+96)/3600)*np.sin(rotatn) + ((+450)/3600)*np.cos(rotatn)
    cnrB3r = ra0 + ((+000+96)/3600)*np.cos(rotatn) - ((-450)/3600)*np.sin(rotatn)
    cnrB3d = de0 + ((+000+96)/3600)*np.sin(rotatn) + ((-450)/3600)*np.cos(rotatn)
    cnrB4r = ra0 + ((+480+96)/3600)*np.cos(rotatn) - ((-450)/3600)*np.sin(rotatn)
    cnrB4d = de0 + ((+480+96)/3600)*np.sin(rotatn) + ((-450)/3600)*np.cos(rotatn)
    Bx_values = [cnrB1r, cnrB2r, cnrB3r, cnrB4r, cnrB1r]
    By_values = [cnrB1d, cnrB2d, cnrB3d, cnrB4d, cnrB1d]
    ax.plot(Ax_values, Ay_values, color='#2176ff', # BinoSpec FoV-1A
            transform=ax.get_transform('world'), zorder=2)
    ax.plot(Bx_values, By_values, color='#2176ff', # BinoSpec FoV-1B
            transform=ax.get_transform('world'), zorder=2)
    return 


def scale_calc(z, H0, Omega_M, Omega_Lam):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0, Om0=Omega_M) # a selected cosmological model
    d_A = cosmo.kpc_proper_per_arcmin(z)
    d_A = d_A.value/60 # kpc/arcmin --> kpc/arcsec
    return d_A


def draw_r500_r200(plt, ax, r_500, r_200, color='greenyellow'):
    circ_500 = plt.Circle(( (( 2)+(48)/60+( 3.3)/3600)*15,
                           -(( 3)+(31)/60+(46.4)/3600)   ), 
                          r_500/3600, 
                          color=color, fill=False, linestyle='--', # 'dashdot'
                          linewidth=1.2, zorder=1, 
                          transform=ax.get_transform('world'))
    circ_200 = plt.Circle(( (( 2)+(48)/60+( 3.3)/3600)*15,
                           -(( 3)+(31)/60+(46.4)/3600)   ), 
                          r_200/3600, 
                          color=color, fill=False, linestyle=':', 
                          zorder=1, 
                          transform=ax.get_transform('world'))
    ax.text( (( 2)+(48)/60+( 3.3)/3600)*15,
             -(( 3)+(31)/60+(46.4)/3600) - r_500/3600,
             r'$r_{\rm 500}$', color=color, size=18, 
             verticalalignment='top', 
             transform=ax.get_transform('world'))
    ax.text( (( 2)+(48)/60+( 3.3)/3600)*15,
             -(( 3)+(31)/60+(46.4)/3600) - r_200/3600,
             r'$r_{\rm 200}$', color=color, size=18, 
             verticalalignment='top', 
             transform=ax.get_transform('world'))
    ax.add_patch(circ_500)
    ax.add_patch(circ_200)
    return circ_500, circ_200


def slit_design_plot(g_cat, maskCenterRA, maskCenterDEC, imgR, imgG, imgB, wcs, 
                     slits_not_finished=None):
    assert ~np.any(np.isnan(g_cat[2:, 15:17].astype(float))), \
        'Some slits have no RA/DEC. Maybe slits were not iterated completed?'
    
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(8,8), dpi=450)
    plt.subplots_adjust(hspace=0, wspace=0) # h=height
    gs = fig.add_gridspec(1, 1,
                          height_ratios=[1],
                          width_ratios=[1])
    ax1 = fig.add_subplot(gs[0, 0], projection=wcs)
    
    g_cat_return = {}
    
    # (1) Slits without secure z and M*
    z_arr = np.array(g_cat[2:,  1], dtype=float) # Must use float
    Ms_ar = np.array(g_cat[2:, 17], dtype=float) # Must use float
    no_z  = np.isnan(z_arr)
    no_Ms = np.isnan(Ms_ar)
    have_z_and_Ms = ~no_z & ~no_Ms
    g_cat_secure = g_cat[np.r_[True, True,  have_z_and_Ms]]
    g_cat_no_z   = g_cat[np.r_[True, True, ~have_z_and_Ms]]
    g_cat_return['g_cat_no_z'] = g_cat_no_z
    print(f'No secure z ({len(g_cat_no_z[2:, 0])}): {g_cat_no_z[2:, 0]}')
    
    # (2) Slits whose major spectra are affected by sky lines
    headers = {name: col for col, name in enumerate(g_cat[1]) if isinstance(name, str)}
    skyline_status = g_cat_secure[2:, headers['skyline_status']]
    g_cat_sky = g_cat_secure[np.r_[True, True, skyline_status == 'bad']]
    g_cat_OK = g_cat_secure[np.r_[True, True, skyline_status != 'bad']]
    g_cat_return['g_cat_fail'] = g_cat_sky
    print(f'Sky lines ({len(g_cat_sky[2:, 0])}): {g_cat_sky[2:, 0]}')
    
    # (3) Separate unfinished fits from evaluated shear posteriors.
    fit_status = g_cat_OK[2:, headers['fit_status']]
    g_cat_tbd = g_cat_OK[np.r_[True, True, fit_status != 'ok']]
    g_cat_solved = g_cat_OK[np.r_[True, True, fit_status == 'ok']]
    g_cat_return['g_cat_tbd'] = g_cat_tbd
    print(f'Not finished ({len(g_cat_tbd[2:, 0])}): {g_cat_tbd[2:, 0]}')

    # (4) Classify evaluated g1/g2 directly from their quality flags.
    mask_good_g1g2 = ((g_cat_solved[2:, headers['g1_status']] == 'ok') &
                      (g_cat_solved[2:, headers['g2_status']] == 'ok'))
    g_cat_good_g1g2 = g_cat_solved[np.r_[True, True, mask_good_g1g2]]
    g_cat_bad_g1g2 = g_cat_solved[np.r_[True, True, ~mask_good_g1g2]]
    g_cat_return['g_cat_good_g1g2'] = g_cat_good_g1g2
    g_cat_return['g_cat_bad_g1g2' ] = g_cat_bad_g1g2
    print(f'Bad g1g2 ({len(g_cat_bad_g1g2[2:, 0])}): {g_cat_bad_g1g2[2:, 0]}')
    print(f'Good g1g2 ({len(g_cat_good_g1g2[2:, 0])}): {g_cat_good_g1g2[2:, 0]}')
    
    ax1.scatter(g_cat_no_z[2:, 15], 
                g_cat_no_z[2:, 16], 
                label=f'No redshift or M* ({len(g_cat_no_z[2:, 15])})', 
                marker='x',
                s=30, color='gray', 
                transform=ax1.get_transform('world'), zorder=4)
    ax1.scatter(g_cat_sky[2:, 15], 
                g_cat_sky[2:, 16], 
                label=r'Signals covered by sky lines '+
                f'({len(g_cat_sky[2:, 15])})', 
                marker='o',
                s=20, facecolors='gray', edgecolors='purple', 
                transform=ax1.get_transform('world'), zorder=4)
    ax1.scatter(g_cat_tbd[2:, 15],
                g_cat_tbd[2:, 16],
                label=f'Fitting incomplete ({len(g_cat_tbd[2:, 15])})',
                marker='s', s=25, facecolors='none', edgecolors='orange',
                transform=ax1.get_transform('world'), zorder=4)
    ax1.scatter(g_cat_bad_g1g2[2:, 15], 
                g_cat_bad_g1g2[2:, 16], 
                label=r'$g_1$'+' or '+r'$g_2$'+' rejected '+
                f'({len(g_cat_bad_g1g2[2:, 15])})', 
                marker='^',
                s=40, facecolors='violet', edgecolors='black', 
                transform=ax1.get_transform('world'), zorder=4)
    ax1.scatter(g_cat_good_g1g2[2:, 15], 
                g_cat_good_g1g2[2:, 16], 
                label=r'$g_1$'+' and '+r'$g_2$'+' accepted '+
                f'({len(g_cat_good_g1g2[2:, 15])})', 
                marker='o',
                s=40, facecolors='cyan', edgecolors='black', # '#0000ff'
                transform=ax1.get_transform('world'), zorder=4)
    
    # Annotate slits & Avoid overlapping
    texts = []
    # Not reliable fit
    for x, y, slit_id in zip(g_cat_sky[2:, 15]-5/3600, #  x
                             g_cat_sky[2:, 16]+5/3600, #  y
                             g_cat_sky[2:,  0]): # slit ID
        texts.append(
            ax1.text(x, y, str(slit_id), color='purple', 
                     transform=ax1.get_transform('world'), fontsize=4, zorder=3)
                     )
    # Extreme g1 or g2
    for x, y, slit_id in zip(g_cat_bad_g1g2[2:, 15]-3/3600, #  x
                             g_cat_bad_g1g2[2:, 16]+3/3600, #  y
                             g_cat_bad_g1g2[2:,  0]): # slit ID
        texts.append(
            ax1.text(x, y, str(slit_id), color='mediumorchid', 
                     transform=ax1.get_transform('world'), fontsize=6, zorder=4)
                     )
    # Solved
    for x, y, slit_id in zip(g_cat_good_g1g2[2:, 15]-3/3600, #  x
                             g_cat_good_g1g2[2:, 16]+3/3600, #  y
                             g_cat_good_g1g2[2:,  0]): # slit ID
        texts.append(
            ax1.text(x, y, str(slit_id), color='cyan', # '#0000ff'
                     transform=ax1.get_transform('world'), fontsize=8, zorder=5)
                     )
    adjust_text(
        texts,
        ax=ax1,
        expand_points=(1.2, 1.2),
        expand_text=(1.2, 1.2),
        arrowprops=dict(
            arrowstyle='-',
            lw=0.5,
            shrinkA=5, 
            shrinkB=5), 
        verbose=True
    )

    # BinoSpec FoV
    maskCenter = [maskCenterRA, maskCenterDEC]
    maskPA_rad = -25 / 57.3 # left-->west right-->east
    binospec_fov(ax1, maskCenter, maskPA_rad)
    
    # R_500, R_200(=R_500/0.65)
    R_500 = 944 # kpc, Vikhlinin et al. (2006) arXiv:astro-ph/0507092, Table 4
    d_A   = scale_calc(0.1883, 70, 0.3, 0.7) # kpc/arcsec
    r_500 = R_500 / d_A
    r_200 = r_500 / 0.65
    c500, c200 = draw_r500_r200(plt, ax1, r_500, r_200)
    
    # 1-arcsec-scale
    ax1.arrow(x=(41.95), y=(-3.66), dx=-(5)/60, dy=0, 
              head_width=0, head_length=0, 
              fc='white', ec='white', width=10/3600, 
              transform=ax1.get_transform('world'))
    ax1.text((41.95)-(2.5)/60, (-3.66), 
             '5'+'\'', size=18,
             horizontalalignment='center', verticalalignment='bottom', 
             color='white', 
             transform=ax1.get_transform('world'))
    
    # Astropy RGB solution
    ax1.imshow(make_lupton_rgb(imgR, imgG, imgB, 
                               # stretch=0.07, Q=5
                               ), 
               origin='lower', zorder=-1, alpha=1)
    
    # ax1.set_xlim(ax1.get_xlim()[1], ax1.get_xlim()[0]) # flip
    ax1.set_xlabel('RA (deg)', fontsize=15)
    ax1.set_ylabel('DEC (deg)', fontsize=15)
    ax1.minorticks_on()
    ax1.legend(prop={'size': 12}, facecolor='dimgray')
    ax1.set_aspect(1)
    plt.savefig("slit_distribution.jpg", dpi=200, bbox_inches='tight')
    
    return g_cat_return







if __name__ == '__main__':
    z_table_filename = "../scripts/redshift_table.xlsx"
    real_pkl_folder  = '../scripts/binospec_pkl_new/'
    Ms_folder        = '../../bagpipes-KL/'
    spec1d2dfolder02 = '../../RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles_counts/'
    new_filename     = "shear_table_20261120.xlsx"
    base_dir         = '../scripts/' # f'../../RSCH3/HPC_database/runs_{date_of_run}/'
    config_filepath  = '../config/binospec_fitting_params.yaml'
    
    # If any, specify slits that were not finished
    slits_not_finished = []
    
    # Redshifts
    df     = pd.read_excel(z_table_filename, header=None, engine='openpyxl')
    array  = df.to_numpy()
    ztable = array[1:, 0:2] # remove 1st line header, keep 1 remaining
    ltable = array[1:, [0,6,9,12]]
    # bltable = array[1:, [0,14]]
    
    # Add some columns
    g_cat = np.append(array, 
                      np.array([[np.nan, 'slit_RA' ]+[np.nan]*(len(array)-2), # line 15
                                [np.nan, 'slit_DEC']+[np.nan]*(len(array)-2), # line 16
                                [np.nan, 'M_s'     ]+[np.nan]*(len(array)-2), # line 17
                                [np.nan, 'M_s_err' ]+[np.nan]*(len(array)-2), # line 18
                                ]).T, 
                      axis=1)
    
    
    # date_of_run = str(input('Please enter the data of run: '))
    # date_of_run = '20260601'
    g_cat = posterior_to_cat(g_cat, base_dir, real_pkl_folder, spec1d2dfolder02, config_filepath, ztable, 
                             slits_not_finished)
    
    # Save shear catalog
    from openpyxl import load_workbook
    wb = load_workbook(z_table_filename)
    ws = wb.active # or, wb["Sheet1"]
    n_old_cols, n_rows = ws.max_column, ws.max_row
    new_cols = g_cat[:, n_old_cols:]
    for i in range(min(n_rows, new_cols.shape[0])):
        for j in range(new_cols.shape[1]):
            value = new_cols[i, j]
            if pd.isna(value):
                value = None
            elif isinstance(value, np.generic):
                value = value.item()
            ws.cell(row=i+1, column=n_old_cols + j + 1, value=value)
    wb.save(new_filename)
    
    # print('Finished reading shear catalog. Plotting...')
    
    # img_dir  = '../../RSCH3/HSC_img_A383/'
    # sci_fn_R = 'hlsp_clash_subaru_suprimecam_a383_z_2010-2002-v20110405_drz.fits'
    # sci_fn_G = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz.fits'
    # # sci_fn_G = 'hlsp_clash_subaru_suprimecam_a383_ic_2002-v20110405_drz.fits'
    # sci_fn_B = 'hlsp_clash_subaru_suprimecam_a383_v_2002-2005-2008-v20110405_drz.fits'
    # # sci_fn_B = 'hlsp_clash_subaru_suprimecam_a383_rc_2002-2007-2008-v20110405_drz.fits'
    
    # img_data_R = read_subaru_img_wcs(img_dir+sci_fn_R, None)
    # img_data_G = read_subaru_img_wcs(img_dir+sci_fn_G, None)
    # img_data_B = read_subaru_img_wcs(img_dir+sci_fn_B, None)
    # imgR, wcsR = img_data_R['science_data'], img_data_R['science_wcs' ]
    # imgG, wcsG = img_data_G['science_data'], img_data_G['science_wcs' ]
    # imgB, wcsB = img_data_B['science_data'], img_data_B['science_wcs' ]
    
    # hdulist01 = fits.open(spec1d2dfolder02+'../obj_abs_slits_extr.fits')
    # slitsInfo01 = np.append(hdulist01[4].data, hdulist01[5].data, axis=0)
    # maskCenterRA, maskCenterDEC = slitsInfo01['MASK_RA'][0], slitsInfo01['MASK_DEC'][0]
    
    # scale = 0.2 # arcsec/pix: 0.2 for HSC image
    # imgRc, RAlim, DEClim,       = cutoutimg(
    #     imgR, wcsR, maskCenterRA, maskCenterDEC, 
    #     img_width=(25)*60/scale, img_height=(25)*60/scale)
    # imgGc, RAlim, DEClim, wcsGc = cutoutimg(
    #     imgG, wcsG, maskCenterRA, maskCenterDEC, 
    #     img_width=(25)*60/scale, img_height=(25)*60/scale, outputWCS=True)
    # imgBc, RAlim, DEClim,       = cutoutimg(
    #     imgB, wcsB, maskCenterRA, maskCenterDEC, 
    #     img_width=(25)*60/scale, img_height=(25)*60/scale)
    
    # g_cats = slit_design_plot(g_cat, 
    #                           maskCenterRA, maskCenterDEC, 
    #                           imgRc*0.01, imgGc*0.01, imgBc*0.004, wcsGc, 
    #                           slits_not_finished)
    print('Done.')
