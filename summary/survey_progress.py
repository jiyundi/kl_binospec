import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from astropy.io import fits
from astropy.visualization import make_lupton_rgb # Creating color RGB images
from pathlib import Path
# from tqdm import tqdm
            
from klm.safe_plot import setup; setup() # must before plt
from data_prep.image_utils import cutoffimg
from data_prep.read_save_utils import read_hsc_img_wcs, readinfodat
from binospec_plot_corner_new  import get_max_num_subdir, read_post
from post_fitting import analyze_percentile

plt.style.use('default')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})



def posterior_to_cat(g_cat, post_dir, ztable):
    """
        Generate a shear catalog.
    """
    for slit_num in range(1, 143):
        z_spec = ztable[slit_num, 1]
        
        # Match RA/DEC
        infdatfilename =   f'info.829.{slit_num:03d}.{slit_num+100305:06d}.dat'
        dat_dict       = readinfodat(spec1d2dfolder02 + infdatfilename)
        g_cat[slit_num+1, 15:17] = dat_dict['RA'], dat_dict['DEC']
        
        if np.isnan(z_spec)==True:
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Slit {slit_num} skipped because no secure redshift found.\n')
            continue
    
        elif np.isnan(z_spec)==False:
            # Load pkl
            try:
                with open(f'{real_pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
                    data_info = joblib.load(f)
            except FileNotFoundError: 
                print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                      f'Slit {slit_num} skipped because no pkl found.\n')
                continue
            
            # Check if already solved M_stellar
            if data_info['galaxy']['log10_Mstar'] is None:
                # Read M_stellar
                dg = pd.read_csv(f"{Ms_folder}Mstellar_table.txt", 
                                 sep=r"\s+", header=0, 
                                 names=["slit","median","std","err_lo","mean","err_hi"])
                dg = dg.drop_duplicates(subset="slit", keep="last")
                Ms = dg.sort_values(by='slit').to_numpy()
                for row in Ms: 
                    if row[0]==slit_num: 
                        log10_Mstar = row[1]
                        log10_Mstar_err = row[2]
                        break
            else:
                log10_Mstar = data_info['galaxy']['log10_Mstar']
                log10_Mstar_err = data_info['galaxy']['log10_Mstar_err']
            g_cat[slit_num+1, 17:19] = log10_Mstar, log10_Mstar_err
            del log10_Mstar, log10_Mstar_err
            
            # Find the most latest run
            base_dir = '../../../RSCH3/kl_github/'
            full_run_dir_1 = f'{base_dir}runs_nautilus/Slit_{slit_num:03d}/'
            if Path(full_run_dir_1).exists() is False:
                print(f'Slit {slit_num} does not exist.')
                continue
            
            date_of_run1 = get_max_num_subdir(full_run_dir_1) + '/'
            post_path1 = f'{full_run_dir_1}{date_of_run1}post.txt'
            best_path1 = f'{full_run_dir_1}{date_of_run1}best_fit.json'
            
            run_samples1, par_names1, weights1, \
            loglikes1, samples1, mask1, no_plot1 = read_post(
                post_path1, percentile=95)
            # par_names1 = run_samples1[0,  2:]
            samples1   = run_samples1[1:, 2:].astype(float)
            
            # Posterior txt --> dict
            alllinespecies = []
            for spec in data_info['spec']:
                alllinespecies.append(spec['par_meta']['line_species'])
            percentile = analyze_percentile(samples1, 
                                            "config/binospec_fitting_params.yaml", 
                                            alllinespecies)
            if not isinstance(percentile, dict): 
                print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                      f'Slit {slit_num} hint: your run does not match your current pkl or config yaml. Skipped.\n')
                continue
            
            # Add best params into percentile array
            with open(best_path1, "r") as f:
                best_par = json.load(f)['maximum_likelihood']['point']
            for key, vals in percentile.items():
                percentile[key]['best'] = best_par[key]
            
            # Add more columns to contain posteriors
            n_rows, n_cols = g_cat.shape
            for key in list(percentile.keys()):
                # We only do once in the slit for loop!
                if n_cols <= 19: 
                    if key.split('-')[0] == 'shared_params':
                        par = key.split('-')[1]
                        g_cat = np.append(
                            g_cat, 
                            np.array([
                                [np.nan, f'{par}'     ]+[np.nan]*(n_rows-2), # col 19
                                [np.nan, f'{par}_mean']+[np.nan]*(n_rows-2), # col 20
                                [np.nan, f'{par}_err-']+[np.nan]*(n_rows-2), # col 21
                                [np.nan, f'{par}_err+']+[np.nan]*(n_rows-2), # col 22
                                ]).T, 
                            axis=1)
                
                # Assign values from posterior
                par = key.split('-')[1]
                for j in range(len(g_cat[1])):
                    head_key = g_cat[1, j]
                    if par == head_key:
                        g_cat[slit_num+1, j:j+4] = list([
                            percentile[key]['best'], 
                            percentile[key]['median'], # or mean
                            percentile[key]['err_lo'], 
                            percentile[key]['err_hi'], 
                            ])
    
            print(f'INFO:    Slit {slit_num} recorded. 👍')
    
    return g_cat


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


def draw_r500_r200(plt, ax, r_500, r_200):
    circ_500 = plt.Circle(( (( 2)+(48)/60+( 3.3)/3600)*15,
                           -(( 3)+(31)/60+(46.4)/3600)   ), 
                          r_500/3600, 
                          color='yellow', fill=False, linestyle='dashdot', 
                          zorder=1, 
                          transform=ax.get_transform('world'))
    circ_200 = plt.Circle(( (( 2)+(48)/60+( 3.3)/3600)*15,
                           -(( 3)+(31)/60+(46.4)/3600)   ), 
                          r_200/3600, 
                          color='yellow', fill=False, linestyle=':', 
                          zorder=1, 
                          transform=ax.get_transform('world'))
    ax.text( (( 2)+(48)/60+( 3.3)/3600)*15,
             -(( 3)+(31)/60+(46.4)/3600) - r_500/3600,
             r'$r_{\rm 500}$', color='yellow', size=18, 
             verticalalignment='top', 
             transform=ax.get_transform('world'))
    ax.text( (( 2)+(48)/60+( 3.3)/3600)*15,
             -(( 3)+(31)/60+(46.4)/3600) - r_200/3600,
             r'$r_{\rm 200}$', color='yellow', size=18, 
             verticalalignment='top', 
             transform=ax.get_transform('world'))
    ax.add_patch(circ_500)
    ax.add_patch(circ_200)
    return circ_500, circ_200


def slit_design_plot(g_cat, maskCenterRA, maskCenterDEC, imgR, imgG, imgB, wcs):
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
    print(f'len = ({len(g_cat_no_z[2:, 0])}): {g_cat_no_z[2:, 0]}')
    
    # (2) Slits with z but not yet solve posterior
    g1_arr = np.array(g_cat_secure[2:, 19], dtype=float)
    g_cat_solved = g_cat_secure[np.r_[True, True, ~np.isnan(g1_arr)]]
    g_cat_tbd    = g_cat_secure[np.r_[True, True,  np.isnan(g1_arr)]]
    g_cat_return['g_cat_tbd'] = g_cat_tbd
    print(f'len = ({len(g_cat_tbd[2:, 0])}): {g_cat_tbd[2:, 0]}')
    
    # (3) Slits with z & posterior solved but not valid (FAILED)
    g1_errs = np.array(g_cat_solved[2:, 20], dtype=float)
    g_cat_fail  = g_cat_solved[np.r_[True, True,  np.isnan(g1_errs)]]
    g_cat_valid = g_cat_solved[np.r_[True, True, ~np.isnan(g1_errs)]]
    g_cat_return['g_cat_fail'] = g_cat_fail
    print(f'len = ({len(g_cat_fail[2:, 0])}): {g_cat_fail[2:, 0]}')
    
    # (4) Slits with z & posterior solved & valid but extreme g1/g2
    g1 = np.array(g_cat_valid[2:, 19], dtype=float)
    g2 = np.array(g_cat_valid[2:, 23], dtype=float)
    g1err1 = np.array(g_cat_valid[2:, 21], dtype=float)
    g1err2 = np.array(g_cat_valid[2:, 22], dtype=float)
    g2err1 = np.array(g_cat_valid[2:, 25], dtype=float)
    g2err2 = np.array(g_cat_valid[2:, 26], dtype=float)
    
    mask_good_g1    = (g1 > -0.13) & (g1 < 0.13) & \
                      (g1 + g1err1 > -0.14) & (g1 + g1err2 < 0.14)
    mask_good_g2    = (g2 > -0.13) & (g2 < 0.13) & \
                      (g2 + g2err1 > -0.14) & (g2 + g2err2 < 0.14)
    mask_good_g1g2  = mask_good_g1 & mask_good_g2
    g_cat_good_g1g2 = g_cat_valid[np.r_[True, True,  mask_good_g1g2]]
    g_cat_bad_g1g2  = g_cat_valid[np.r_[True, True, ~mask_good_g1g2]]
    g_cat_return['g_cat_good_g1g2'] = g_cat_good_g1g2
    g_cat_return['g_cat_bad_g1g2' ] = g_cat_bad_g1g2
    print(f'len = ({len(g_cat_bad_g1g2[2:, 0])}): {g_cat_bad_g1g2[2:, 0]}')
    print(f'len = ({len(g_cat_good_g1g2[2:, 0])}): {g_cat_good_g1g2[2:, 0]}')
    
    ax1.scatter(g_cat_no_z[2:, 15], 
                g_cat_no_z[2:, 16], 
                label=f'No redshift or M* ({len(g_cat_no_z[2:, 15])})', 
                marker='x',
                s=30, color='gray', 
                transform=ax1.get_transform('world'), zorder=4)
    ax1.scatter(g_cat_tbd[2:, 15], 
                g_cat_tbd[2:, 16], 
                label=f'Running on HPC ({len(g_cat_tbd[2:, 15])})', 
                marker='o',
                s=30, facecolors='none', edgecolors='black', 
                transform=ax1.get_transform('world'), zorder=4)
    ax1.scatter(g_cat_bad_g1g2[2:, 15], 
                g_cat_bad_g1g2[2:, 16], 
                label=r'$|g_1|$'+' or '+r'$|g_2|$'+' > 0.13 '+
                f'({len(g_cat_bad_g1g2[2:, 15])})', 
                marker='^',
                s=40, facecolors='gold', edgecolors='black', 
                transform=ax1.get_transform('world'), zorder=4)
    ax1.scatter(g_cat_good_g1g2[2:, 15], 
                g_cat_good_g1g2[2:, 16], 
                label=f'Solved ({len(g_cat_good_g1g2[2:, 15])})', 
                marker='o',
                s=40, facecolors='#0000ff', edgecolors='black', 
                transform=ax1.get_transform('world'), zorder=4)
    
    # Annotate slits & Avoid overlapping
    texts = []
    # Not reliable fit
    for x, y, slit_id in zip(g_cat_fail[2:, 15]-3/3600, #  x
                             g_cat_fail[2:, 16]+3/3600, #  y
                             g_cat_fail[2:,  0]): # slit ID
        texts.append(
            ax1.text(x, y, str(slit_id), color='violet', zorder=3, 
                     transform=ax1.get_transform('world'), fontsize=8)
                     )
    # Extreme g1 or g2
    for x, y, slit_id in zip(g_cat_bad_g1g2[2:, 15]-3/3600, #  x
                             g_cat_bad_g1g2[2:, 16]+3/3600, #  y
                             g_cat_bad_g1g2[2:,  0]): # slit ID
        texts.append(
            ax1.text(x, y, str(slit_id), color='darkgoldenrod', zorder=3, 
                     transform=ax1.get_transform('world'), fontsize=8)
                     )
    # Solved
    for x, y, slit_id in zip(g_cat_good_g1g2[2:, 15]-3/3600, #  x
                             g_cat_good_g1g2[2:, 16]+3/3600, #  y
                             g_cat_good_g1g2[2:,  0]): # slit ID
        texts.append(
            ax1.text(x, y, str(slit_id), color='#0000ff', zorder=3, 
                     transform=ax1.get_transform('world'), fontsize=8)
                     )
    adjust_text(
        texts,
        ax=ax1,
        expand_points=(1.2, 1.2),
        expand_text=(1.2, 1.2),
        arrowprops=dict(
            arrowstyle='-',
            lw=0.5,
            shrinkA=5,   # 缓解 warning 提到的问题
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
    ax1.imshow(make_lupton_rgb(imgR, imgG, imgB, stretch=0.04, Q=5), 
               origin='lower', zorder=-1, alpha=0.3)
    
    # ax1.set_xlim(ax1.get_xlim()[1], ax1.get_xlim()[0]) # flip
    ax1.set_xlabel('RA (deg)', fontsize=15)
    ax1.set_ylabel('DEC (deg)', fontsize=15)
    ax1.minorticks_on()
    ax1.legend(prop={'size': 12}, facecolor='lightgray')
    ax1.set_aspect(1)
    plt.savefig("slit_distribution.jpg", dpi=450, bbox_inches='tight')
    
    return g_cat_return







if __name__ == '__main__':
    z_table_filename = "./redshift_table.xlsx"
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
    
    real_pkl_folder  = 'binospec_pkl/'
    Ms_folder        =  '../../bagpipes-KL/'
    post_dir         = '../../../RSCH3/kl_github/runs_nautilus/'
    spec1d2dfolder02 = '../../../RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles/'
    
    g_cat = posterior_to_cat(g_cat, post_dir, ztable)
    
    # Save shear catalog
    from openpyxl import load_workbook
    new_filename = "./redshift_table_with_shear.xlsx"
    wb = load_workbook(z_table_filename)
    ws = wb.active # or, wb["Sheet1"]
    n_old_cols, n_rows = ws.max_column, ws.max_row
    new_cols = g_cat[:, n_old_cols:]
    new_cols = np.where(new_cols == 'nan', None, new_cols)
    for i in range(n_rows):
        for j in range(new_cols.shape[1]):
            ws.cell(row=i+1, column=n_old_cols + j + 1,
                    value=new_cols[i, j])
    wb.save(new_filename)
    
    print('Finished reading shear catalog. Plotting...')
    
    hscimagefolder01 = '../../../RSCH3/HSC_img_A383/'
    hsc_R_filename   = 'hlsp_clash_subaru_suprimecam_a383_z_2010-2002-v20110405_drz.fits'
    hsc_G_filename   = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz.fits'
    hsc_B_filename   = 'hlsp_clash_subaru_suprimecam_a383_v_2002-2005-2008-v20110405_drz.fits'
    
    imgR, ____ = read_hsc_img_wcs(hscimagefolder01 + hsc_R_filename)
    imgG, wcsG = read_hsc_img_wcs(hscimagefolder01 + hsc_G_filename)
    imgB, ____ = read_hsc_img_wcs(hscimagefolder01 + hsc_B_filename)
    
    hdulist01 = fits.open(spec1d2dfolder02+'../obj_abs_slits_extr.fits')
    slitsInfo01 = np.append(hdulist01[4].data, hdulist01[5].data, axis=0)
    maskCenterRA, maskCenterDEC = slitsInfo01['MASK_RA'][0], slitsInfo01['MASK_DEC'][0]
    
    scale = 0.2 # arcsec/pix: 0.2 for HSC image
    imgRc, RAlim, DEClim, = cutoffimg(
        imgR, wcsG, maskCenterRA, maskCenterDEC, 
        img_width=(25)*60/scale, img_height=(25)*60/scale)
    imgGc, RAlim, DEClim, wcsGc = cutoffimg(
        imgG, wcsG, maskCenterRA, maskCenterDEC, 
        img_width=(25)*60/scale, img_height=(25)*60/scale, outputWCS=True)
    imgBc, RAlim, DEClim, = cutoffimg(
        imgB, wcsG, maskCenterRA, maskCenterDEC, 
        img_width=(25)*60/scale, img_height=(25)*60/scale)
    
    g_cats = slit_design_plot(g_cat, maskCenterRA, maskCenterDEC, 
                              imgRc, imgGc, imgBc, wcsGc)
    print('Done.')