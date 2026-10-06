import json
import yaml
import numpy as np
from pathlib import Path
            
from klm.nautilus_sampler import NautilusSampler
from klm.parameters       import Parameters
from scripts.main_fitting import load_mock
from core.make_config_dic import make_config_dic

from klm.safe_plot import setup; setup() # must before plt
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def find_chi2():
    """
        Generate a shear catalog.
    """
    shear_cat = {}
    for slit_num in range(1, 143):
        try:
            data_info = load_mock(pkl_folder, slit_num)
        except FileNotFoundError: 
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Slit {slit_num} skipped because no pkl found.')
            continue
        
        # Find the most latest run
        if Path(f'{base_dir}/Slit_{slit_num:03d}/').exists() is False:
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Slit {slit_num} does not exist.')
            continue
        
        # Load YAML file for config
        with open(fiducial_yaml, "r", encoding="utf-8") as file1:
            fid_params     = yaml.safe_load(file1)
        with open(fitting_yaml, "r", encoding="utf-8") as file2:
            fitting_params = yaml.safe_load(file2)
        
        linespecies = []
        for spec in data_info['spec']:
            linespecies.append(spec['meta']['line_species'])
        
        config_dic = make_config_dic(
            linespecies, fitting_params, fid_params, 
            log10_Mstar=data_info['galaxy']['log10_Mstar'], 
            log10_Mstar_err=data_info['galaxy']['log10_Mstar_err'],
            use_line_profile = 'extracted', # None, 'raw' 
            )
        
        with open(f'{base_dir}/Slit_{slit_num:03d}/best_fit.json', "r") as f:
            best_par = json.load(f)['maximum_likelihood']['point']
        
        inference = NautilusSampler(data_info, config_dic, verbose=False)
        inference.params.params['shared_params'].update({
            **inference.params.params['shared_params'], 
            **Parameters._unflatten(best_par)['shared_params']
            })
        
        image_obs  = inference.data_image
        image_msk  = inference.mask_image
        image_var  = inference.var_image
        SNR_image  = np.sum(image_obs[image_msk]) / np.sqrt(np.sum(image_obs[image_msk] + image_var[image_msk]))
        imgDOF     = image_obs.shape[0] * image_obs.shape[1] - len(best_par)
        image_fit  = inference.image_model.get_image(
            inference.params.params['shared_params']
            )
        chi2_image = inference.calc_image_loglike(
            inference.params.params
            )
        chi2_image_reduced = chi2_image / imgDOF
        Moran_idx_image = morans_I(image_obs - image_fit, image_msk)
        
        chi2_spec, chi2_spec_reduced, SNR_spec, Moran_idx_spec = [], [], [], []
        for i in range(len(inference.data_spec)):
            inference.spec_model[i]._init_observable(
                data_info['galaxy'], 
                data_info['spec'][i]['meta'])
            
            # Update by set. Assign I01 from I01_spec1/2/3 now
            line = linespecies[i//3]
            best_line_pars = {
                **inference.params.params[f'{line}_params'], 
                **Parameters._unflatten(best_par)[f'{line}_params']
                }
            
            # Use I02, I01 in calculation instead
            set_num = i % 3 + 1
            for k in ['I02', 'I01']: #, 'dx_vel', 'dy_vel', 'bkg_level']:
                best_line_pars[k] = None
                if k+f'_spec{set_num}' in best_line_pars.keys():
                    best_line_pars[k] = best_line_pars[k+f'_spec{set_num}']
            
            spec0_obs = inference.data_spec[i]
            spec0_msk = inference.mask_spec[i]
            spec0_var = inference.var_spec[i]
            spec0_fit = inference.spec_model[i].get_observable(
                {**inference.params.params[ 'shared_params'],
                 **best_line_pars }
                )
            spec0_SNR  = (
                np.sum(spec0_obs[spec0_msk]) /
                np.sqrt(np.sum(spec0_obs[spec0_msk] + spec0_var[spec0_msk]))
            )
            spec0_chi2 = inference._loglike_one_slit(
                data_spec=spec0_obs, 
                mask_spec=spec0_msk,
                var_spec=spec0_var, 
                model_spec=spec0_fit)
            specDOF = spec0_obs.shape[0] * spec0_obs.shape[1] - len(best_par)
            spec0_Moran_idx = morans_I(spec0_obs - spec0_fit, spec0_msk)
            
            chi2_spec.append(spec0_chi2)
            chi2_spec_reduced.append(spec0_chi2 / specDOF)
            SNR_spec.append(spec0_SNR)
            Moran_idx_spec.append(spec0_Moran_idx)
        
        # pack up the dict
        shear_cat[f'{slit_num:03d}'] = {
            'SNR_image':   float(np.round(SNR_image, 2)),
            'Moran_image': float(np.round(Moran_idx_image, 2)),
            'chi2_image':           int(np.round(chi2_image, 0)),
            'chi2_image_reduced': float(np.round(chi2_image_reduced, 2)),
            'SNR_spec':    np.round(SNR_spec, 2).tolist(), # only accept list
            'Moran_spec':  np.round(Moran_idx_spec, 2).tolist(),
            'chi2_spec':         np.round(chi2_spec).astype(int).tolist(),
            'chi2_spec_reduced': np.round(chi2_spec_reduced, 2).tolist(),
            }
        
        # PKL saving test
        if len(shear_cat) == 1:
            with open('shear_catalog_detelethis.yaml', "w", encoding="utf-8") as file1:
                yaml.safe_dump(shear_cat, file1)
            print("\033[42m" + 'INFO:   ' + "\033[0m " + 
                  'PKL successful.')
        
        print("\033[42m" + 'INFO:   ' + "\033[0m " + 
              f'Slit {slit_num} recorded. 👍')
    
    with open('shear_catalog.yaml', "w", encoding="utf-8") as file1:
        yaml.safe_dump(shear_cat, file1)
    
    return shear_cat


def morans_I(arr, mask=None):
    """
    Global Moran's I with Queen contiguity.

    Parameters
    ----------
    arr : 2D ndarray
        Spatial field, e.g. velocity residual map.
        Shape = (ny, nx)

    mask : 2D bool ndarray, optional
        True = valid pixel
        False = ignore

    Returns
    -------
    I : float
        Moran's I statistic
    """

    data = arr.copy()
    data[~mask] = np.nan
    # ny, nx = data.shape

    # flatten valid pixels
    v = data[mask].flatten()
    N = len(v)
    v_mean = np.mean(v)
    dv     = v - v_mean

    # ------------------------------------
    # build Queen spatial weight matrix
    # ------------------------------------
    coords = np.argwhere(mask)
    W = np.zeros((N, N))
    for i, (y1, x1) in enumerate(coords):
        for j, (y2, x2) in enumerate(coords):

            if i == j:
                continue

            # Queen neighborhood:
            # share edge OR vertex
            if max(abs(y1-y2), abs(x1-x2)) == 1:
                W[i,j] = 1

    # sum of weights
    W_sum = W.sum()

    # Moran index
    numerator = np.sum(
        W * np.outer(dv, dv)
    )
    denominator = np.sum(dv**2)
    I = (N / W_sum) * (numerator / denominator)

    return I


def plot_all_slits(shear_cat, 
                   only_plot_these=None, not_plot_these=None):
    if (only_plot_these is not None):
        fig = plt.figure(figsize=(6, 9), dpi=300)
    elif (not_plot_these is not None):
        fig = plt.figure(figsize=(9, 9), dpi=300)
    else:
        fig = plt.figure(figsize=(12, 9), dpi=300)
    plt.subplots_adjust(hspace=0.0, wspace=0) # h=height
    gs = fig.add_gridspec(5, 1,
                          height_ratios=[3,3,1,3,3],
                          width_ratios=[1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])
    
    for slit_num in range(1, 143):
        # if specify this is a slit NOT to plot, use continue to skip
        if not_plot_these is not None:
            suffix = 'removed_specified'
            if slit_num in not_plot_these:
                continue
        
        # if specify this is a slit to plot AND REMOVE other slits, 
        # use continue to skip other slits
        if only_plot_these is not None:
            suffix = 'included_selected'
            if slit_num not in only_plot_these:
                continue
        
        try:
            shear_cat_this_slit = shear_cat[f'{slit_num:03d}']
        except KeyError: 
            continue
        
        SNR_image  = shear_cat_this_slit['SNR_image']
        SNR_spec   = shear_cat_this_slit['SNR_spec']
        # chi2_image = shear_cat_this_slit['chi2_image']
        # chi2_spec  = shear_cat_this_slit['chi2_spec']
        chi2_image_reduced = shear_cat_this_slit['chi2_image_reduced']
        chi2_spec_reduced  = shear_cat_this_slit['chi2_spec_reduced']
        Moran_image = shear_cat_this_slit['Moran_image']
        Moran_spec  = shear_cat_this_slit['Moran_spec']
        
        # ax1: image SNRs
        ax1.scatter(slit_num, 
                    SNR_image, marker='s', 
                    facecolor='yellow', edgecolor='black', zorder=2)
        
        # ax2, 3: spec SNRs
        SNR_spec = np.array(SNR_spec)
        positive_SNRs = SNR_spec[SNR_spec >= 0]
        negative_SNRs = SNR_spec[SNR_spec <  0]
        ax2.scatter(slit_num * np.ones(len(positive_SNRs)), 
                    positive_SNRs, linestyle='-', marker='s', 
                    facecolor='cyan', edgecolor='midnightblue', zorder=2)
        ax3.scatter(slit_num * np.ones(len(negative_SNRs)), 
                    negative_SNRs, linestyle='-', marker='s', 
                    facecolor='magenta', edgecolor='midnightblue', zorder=2)
        if len(negative_SNRs) != 0:
            ax2.plot(slit_num * np.ones(len(positive_SNRs)+1), 
                     list(positive_SNRs)+[0], linestyle='-', color='cyan', zorder=1)
            ax3.plot(slit_num * np.ones(len(negative_SNRs)+1), 
                     list(negative_SNRs)+[0], linestyle='-', color='magenta', zorder=1)
        else:
            ax2.plot(slit_num * np.ones(len(positive_SNRs)), 
                     list(positive_SNRs), linestyle='-', color='cyan', zorder=1)
        
        # ax4: reduced chi2
        chi2_spec_reduced = np.array(chi2_spec_reduced)
        ax4.scatter(slit_num, 
                    chi2_image_reduced, marker='^', 
                    facecolor='yellow', edgecolor='black', s=60, 
                    zorder=2)
        ax4.scatter(slit_num * np.ones(len(chi2_spec_reduced)), 
                    chi2_spec_reduced, linestyle='-', marker='o', 
                    facecolor='cyan', edgecolor='midnightblue', 
                    zorder=2)
        
        # ax5: Moran index
        Moran_spec = np.array(Moran_spec)
        ax5.scatter(slit_num, 
                    Moran_image, marker='^', s=60, 
                    facecolor='yellow', edgecolor='black', zorder=2)
        ax5.scatter(slit_num * np.ones(len(Moran_spec)), 
                    Moran_spec, marker='o', 
                    facecolor='cyan', edgecolor='midnightblue', zorder=2)
        
        last_slit = slit_num
        last_chi2_image_reduced = chi2_image_reduced
        last_chi2_spec_reduced  = chi2_spec_reduced
        last_Moran_image = Moran_image
        last_Moran_spec  = Moran_spec
    
    # plot once for labels
    ax4.scatter(last_slit, 
                last_chi2_image_reduced, marker='^', 
                facecolor='yellow', edgecolor='black', s=60, 
                label='image', zorder=3)
    ax4.scatter(last_slit, 
                last_chi2_spec_reduced[0], marker='o', 
                facecolor='cyan', edgecolor='midnightblue', 
                label='spec', zorder=3)
    ax5.scatter(last_slit, 
                last_Moran_image, marker='^', 
                facecolor='yellow', edgecolor='black', s=60, 
                label='image', zorder=2)
    ax5.scatter(last_slit, 
                last_Moran_spec[0], marker='o', 
                facecolor='cyan', edgecolor='midnightblue', 
                label='spec', zorder=2)
    
    # reference lines
    ax4.axhline(y=1, linestyle='-', lw=2, color='lime', zorder=-1)
    ax5.axhline(y=0, linestyle='-', lw=2, color='lime', zorder=-1)
    
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax4.minorticks_on()
    ax5.minorticks_on()
    ax1.tick_params(axis='both', which='minor', direction='in',    length=5)
    ax2.tick_params(axis='both', which='minor', direction='inout', length=5)
    ax3.tick_params(axis='both', which='minor', direction='inout', length=5)
    ax4.tick_params(axis='both', which='minor', direction='inout', length=5)
    ax5.tick_params(axis='both', which='minor', direction='inout', length=5)
    ax1.tick_params(axis='both', which='major', direction='in',    length=9)
    ax2.tick_params(axis='both', which='major', direction='inout', length=9)
    ax3.tick_params(axis='both', which='major', direction='inout', length=9)
    ax4.tick_params(axis='both', which='major', direction='inout', length=9)
    ax5.tick_params(axis='both', which='major', direction='inout', length=9)
    ax1.tick_params(axis='y', right=True, labelright=True)
    ax2.tick_params(axis='y', right=True, labelright=True)
    ax3.tick_params(axis='y', right=True, labelright=True)
    ax4.tick_params(axis='y', right=True, labelright=True)
    ax5.tick_params(axis='y', right=True, labelright=True)
    ax1.tick_params(top=True, labelbottom=False)
    ax2.tick_params(top=True, labelbottom=False)
    ax3.tick_params(top=True, labelbottom=False)
    ax4.tick_params(top=True, labelbottom=False)
    ax5.tick_params(top=True, labelbottom=True)
    ax1.set_xlim(left=0, right=slit_num+1)
    ax2.set_xlim(left=0, right=slit_num+1)
    ax3.set_xlim(left=0, right=slit_num+1)
    ax4.set_xlim(left=0, right=slit_num+1)
    ax5.set_xlim(left=0, right=slit_num+1)
    ax2.set_ylim(bottom=0, top=None)
    ax3.set_ylim(top=0, bottom=None)
    ax4.set_yscale('log')
    ax4.legend(prop={'size': 9})
    ax5.legend(prop={'size': 9})
    ax1.set_ylabel('Image SNR', fontsize=15)
    ax2.set_ylabel('Spec SNR', fontsize=15)
    ax3.set_ylabel('Negative\nspec SNR')
    ax4.set_ylabel(r'$\chi^2_\nu$', fontsize=18)
    ax5.set_ylabel(r'Moran Index $I$', fontsize=15)
    ax1.grid(which='major', linestyle=':', color='black', alpha=0.5, zorder=-1)
    ax2.grid(which='major', linestyle=':', color='black', alpha=0.5, zorder=-1)
    ax3.grid(which='major', linestyle=':', color='black', alpha=0.5, zorder=-1)
    ax4.grid(which='major', linestyle=':', color='black', alpha=0.5, zorder=-1)
    ax5.grid(which='major', linestyle=':', color='black', alpha=0.5, zorder=-1)
    ax5.set_xlabel('Slit number', labelpad=1)
    
    if suffix is not None:
        plt.savefig(f"fitting_stats_{suffix}.jpg", dpi=300, bbox_inches='tight')
    else:
        plt.savefig("fitting_stats.jpg", dpi=300, bbox_inches='tight')
        
    return





pkl_folder    =  "./scripts/binospec_pkl/"
fiducial_yaml =  "./config/binospec_fid_params.yaml"
fitting_yaml  =  "./config/binospec_fitting_params.yaml"
date_of_run   = 20260601
base_dir      = f"../../RSCH3/HPC_database/runs_{date_of_run}/"
        

if __name__ == '__main__':
    assert Path(base_dir).exists(), \
    f'Fitting result folder {base_dir} does not exist.'
    
    # df     = pd.read_excel(shear_table_fnme, header=None, engine='openpyxl')
    # array  = df.to_numpy()
    # ztable = (df.iloc[2:, [5, 8, 11]]
    #             .apply(pd.to_numeric, errors='coerce') 
    #             .mean(axis=1, skipna=True)
    #             ).to_numpy()
    # array[2:, 1] = ztable.T
    
    # # Add some columns
    # columns = np.array(
    #     [['chi2', 'chi2_image'            ]+[np.nan]*(len(array)-2),
    #      [np.nan, 'chi2_image_reduced'    ]+[np.nan]*(len(array)-2),
    #      [np.nan, 'chi2_spec_sum'         ]+[np.nan]*(len(array)-2),
    #      [np.nan, 'chi2_spec_reduced_avg' ]+[np.nan]*(len(array)-2),
    #      [np.nan, 'chi2_joint'            ]+[np.nan]*(len(array)-2),
    #      [np.nan, 'chi2_joint_reduced_avg']+[np.nan]*(len(array)-2),
    #      ])
    # g_cat = np.append(array, columns.T, axis=1)
    try:
        with open('shear_catalog.yaml', "r", encoding="utf-8") as file1:
            shear_cat = yaml.safe_load(file1)
    except FileNotFoundError:
        shear_cat = find_chi2()
    
    arr_shears = np.genfromtxt('./summary/arr_shears.txt', skip_header=1)
    good_slits = arr_shears[:, 0].astype(int)
    
    plot_all_slits(shear_cat, not_plot_these=good_slits)
    
    # Save shear catalog
    # from openpyxl import load_workbook
    # new_filename = "./redshift_table_with_shear_chi2.xlsx"
    # wb = load_workbook(shear_table_fnme)
    # ws = wb.active # or, wb["Sheet1"]
    # n_old_cols, n_rows = ws.max_column, ws.max_row
    # new_cols = g_cat[:, n_old_cols:]
    # new_cols = np.where(new_cols == 'nan', None, new_cols)
    # for i in range(n_rows):
    #     for j in range(new_cols.shape[1]):
    #         ws.cell(row=i+1, column=n_old_cols + j + 1,
    #                 value=new_cols[i, j])
    # wb.save(new_filename)