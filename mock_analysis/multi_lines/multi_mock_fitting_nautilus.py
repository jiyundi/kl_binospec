import joblib
import yaml
import json
import galsim
import os
import argparse
import time
import numpy as np
import corner

from post_fitting         import plot_obs_fit_res, complete_flattened_fit_params, complete_fit_params 
from klm.parameters       import Parameters
from klm.nautilus_sampler import NautilusSampler
from klm.safe_plot import setup; setup() # must before plt

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def load_mock(pkl_folder='mock/', Ms_folder='./', slit_num=95, case_num=1, 
              rescale_image=False):
    with open(f'{pkl_folder}pkl/mock_{slit_num:03d}_{case_num:03d}.pkl', "rb") as f:
        data_info = joblib.load(f)
    
    assert data_info['galaxy']['log10_Mstar'] != None, \
        "Cannot find corresponing stallar mass M*"  # if not, error
    
    # Recover wcs(galsim.wcs) from ap_wcs
    ap_wcs  = data_info['image']['par_meta']['ap_wcs']
    data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    return data_info


def make_config_dic(linespecies, fitting_params, fid_params, 
                    log10_Mstar=9.30, log10_Mstar_err=0.05, 
                    use_line_profile=None):
    config_dic = {
        'galaxy_params': {
            'obs_type':     'slit', 
            'line_species':    linespecies, 
            'log10_Mstar':     log10_Mstar, 
            'log10_Mstar_err': log10_Mstar_err, 
            'line_profile_path': use_line_profile,
            }, 
        'likelihood': {
            'fit_image':  True, 
            'fit_spec':   True, 
            'set_non_analytic_prior': None,
            'fid_params': fid_params
            }, 
        'TFprior': {
            'use_TFprior': True, 
            'log10_vTF':   None,
            'sigmaTF':     None, 
            'a':            None, 
            'b':            None, 
            'sigmaTF_intr': None, 
            'relation':     None
            }, 
        'params': fitting_params,
        'truevalues': None
        }
    
    # True values: (1) expand by 3 sets
    mock_params  = data_info['par_fit']
    line_species = config_dic['galaxy_params']['line_species']
    lines        = list(dict.fromkeys(line_species))
    n_sets       = len(line_species) // len(lines)
    doublet_lines  = ['O2']
    for i in range(1, n_sets+1):
        for line in lines:
            suffix  = f'_spec{i}'
            if line != doublet_lines[0]:
                prm_add = {'I01': None, 'bkg_level': None}
            else:
                prm_add = {'I01': None, 'I02': None, 'bkg_level': None}
            for key, _ in prm_add.copy().items():
                if '_spec' not in key:
                    mock_params['line_params-'+key+suffix] = mock_params['line_params-'+key]
    
    # True values: (2) assign values
    true_params = np.zeros((1,3))
    for shared_or_lines, subdict in fitting_params.items():
        if subdict:
            for this_par_name, this_par_dict in subdict.items():
                for true_name, true_value in mock_params.items():
                    if this_par_name == true_name.split('-')[1]:
                        true_params = np.append(true_params,
                                                [[true_name, 
                                                  this_par_name, 
                                                  true_value]], 
                                                axis=0)
    true_params = np.delete(true_params, (0), axis=0) 
    
    # True values: (3) expand by all lines
    doub_pars      = ['v_0_2','dx_vel_2','dy_vel_2','I02','f2_1','f2_2']
    doub_pars_twin = ['v_0',  'dx_vel',  'dy_vel',  'I01','f1_1','f1_2']
    
    trueparams = np.zeros((1, len(true_params[0])))
    for i in range(len(true_params)):
        p = true_params[i, 0]
        if p.split('-')[0] == 'shared_params':
            trueparams = np.append(trueparams, 
                                   [true_params[i]], axis=0)
                                    
        elif p.split('-')[0] == 'line_params':
            p2name = p.split('-')[1] # v_0, bkg, ...
            for line in lines:
                line_p = f'{line}_params-{p2name}'
                trueparams = np.append(
                    trueparams, 
                    [[line_p, true_params[i,1], true_params[i,2]]], 
                    axis=0)
                
                if line in doublet_lines:
                    if 'spec' in p2name.split('_')[1]: # I01_spec
                        p2namenospec = p2name.split('_')[0]
                        i_q    = doub_pars_twin.index(p2namenospec)
                        q2name = doub_pars[i_q]
                        line_q = f'{line}_params-{q2name}_{p2name.split("_")[1]}'
                        trueparams = np.append(
                            trueparams, 
                            [[line_q, q2name+f'_{p2name.split("_")[1]}', true_params[i,-1]]], 
                            axis=0)
                    else: # v_0
                        i_q    = doub_pars_twin.index(p2name)
                        q2name = doub_pars[i_q]
                        line_q = f'{line}_params-{q2name}'
                        trueparams = np.append(
                            trueparams, 
                            [[line_q, q2name, true_params[i,-1]]], 
                            axis=0)
    trueparams = np.delete(trueparams, (0), axis=0) 
    
    config_dic['truevalues'] = trueparams[:, 2].astype(float)
    
    return config_dic

















if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--slitID', default=    3, type=int)
    parser.add_argument('--caseID', default=    1, type=int)
    parser.add_argument('--run',    default=    1, type=int)
    # Warning: ONLY input True if you want following two arguments 
    #          because of bool("non_empty_str") == True.
    parser.add_argument('--test',   default=False, type=bool)
    parser.add_argument('--contin', default=False, type=bool)
    slit_name = parser.parse_args().slitID
    case_num  = parser.parse_args().caseID
    run       = parser.parse_args().run
    if_test   = parser.parse_args().test
    if_continue_last_run = parser.parse_args().contin
    
    pkl_folder  = './binospec_multi_mock/'
    Ms_folder   =  '../../../../bagpipes-KL/'
    slit_folder = f'./Slit_{slit_name:03d}_{case_num:03d}/'
    fiduci_yaml =  "./config/binospec_fid_params.yaml"
    fittin_yaml =  "./config/binospec_fitting_params.yaml"
    save_path   = slit_folder
    
    # ------------- 1. Load observation data or mock ---------------- #
    try:
        data_info = load_mock(pkl_folder, Ms_folder, slit_name, case_num, 
                              # rescale_image=True
                              )
    except FileNotFoundError:
        print( "\033[43m" + 'WARNING:' + "\033[0m " + 
              f'Slit {slit_name} skipped because no PKL found.\n')
        os._exit(0)
    
    # ------------- 2. Load configuration --------------------------- #
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
        use_line_profile = None, #  'meta' 
        )
    
    nautilus_sampler = NautilusSampler(data_info, config_dic)
    for par, prior in nautilus_sampler.config.params.prior.items(): print(par, prior)
    
    # ----------------------- 3. Run inference ---------------------- #
    t_start = time.time()
    sampler = nautilus_sampler.run(
        output_dir=f'{slit_folder}run0.{run:02d}/', 
        test_run=if_test, run_num=run,
        )
    points, log_w, log_l = sampler.posterior()
    t_end = time.time()
    print('Total time: {:.1f}s'.format(t_end - t_start))
    
    # Save sample points
    header  = "weight logl " + " ".join(nautilus_sampler.config.params.names)
    weights = np.exp(log_w)
    data    = np.column_stack([weights, log_l, points])
    min_wgt = np.percentile(data[1:,0].astype(float), 95)
    mask  = [True] + list(data[1:,0].astype(float) > min_wgt)
    data_ = data[mask, :] # ONLY save top 5% weighted points
    np.savetxt(
        f'{slit_folder}post.txt',
        data_,
        header=header,
        comments=""
    )
    
    # Save best fit points
    best = points[np.argmax(log_l)]
    best_dict = dict(zip(nautilus_sampler.config.params.names, best))
    out = { 
        # Note: json does not support array.
        "fid_params":     fid_params, 
        "fitting_params": fitting_params, 
        "maximum_likelihood": {
            "point": best_dict,
            "log_likelihood": float(np.max(log_l))
        }
    }
    with open(slit_folder + "best_fit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    
    # Plot - best fit
    print('Plotting best fit comparison with observed spec/image...')
    fitting_params_flat = Parameters._flatten(fitting_params, level=1)
    fitting_par = complete_flattened_fit_params(
        fitting_params_flat, 
        line_species=nautilus_sampler.config.galaxy_params.line_species
        )
    best_fit_dict  = nautilus_sampler.params.gen_param_dict(fitting_par.keys(), 
                                                            best_dict.values())
    plot_obs_fit_res(data_info, 
                     nautilus_sampler, 
                     best_fit_dict, 
                     fitting_params, 
                     run, save_path=save_path)
    
    # Plot - corner
    # fitting_par = complete_fit_params(fitting_params, linespecies)
    par_names, label_latex = [], []
    for key, subdict in fitting_par.items():
        par_names.append(key.split('-')[1])
        label_latex.append(subdict['latex_name'])
    corner.corner(points, weights=np.exp(log_w), 
                  show_titles=True, 
                  title_kwargs={'size': 36},
                  labels=label_latex, 
                  label_kwargs={'size': 36},
                  color='black')
    plt.savefig(f'{save_path}corner.jpg', dpi=100, bbox_inches='tight')
    print('Plotting done.')