import yaml
import joblib
# import pickle
import numpy as np
import galsim
import os
import argparse

from   klm.ultranest_sampler import UltranestSampler
# from   post_fitting          import load_best_fit_json, plot_obs_fit_res

from   klm.safe_plot import setup; setup() # must before plt
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def load_mock(existed_mock_filepath='mock/a.pkl'):
    with open(existed_mock_filepath, "rb") as f:
        data_info = joblib.load(f)
        
    # correct spec flux < 0 to >= 0 by an offset
    for spec_idx in range(len(data_info['spec'])):
        spec0_obs = data_info['spec'][spec_idx]['data']
        min_flux  = np.min(spec0_obs)
        if min_flux < 0:
            mask = (spec0_obs < 0)
            corrected_value = 0
            spec0_obs[mask] = corrected_value
            spec0_obs_new   = spec0_obs
            data_info['spec'][spec_idx]['data'] = spec0_obs_new
            print(f'Bad flux<0 found: Correct {min_flux:.2f} ~ 0 to {corrected_value:.2g}.')
        
    # Recover wcs(galsim.wcs) from ap_wcs
    ap_wcs  = data_info['image']['par_meta']['ap_wcs']
    data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    return data_info

def make_config_dic(linespecies, fitting_params, fid_params, 
                    log10_Mstar=9.30, log10_Mstar_err=0.05):
    config_dic = {
        'galaxy_params': {
            'obs_type':     'slit', 
            'line_species':    linespecies, 
            'log10_Mstar':     log10_Mstar, 
            'log10_Mstar_err': log10_Mstar_err, 
            'line_profile_path': None,
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

def copy_to_cache(file_path, dest_path, cache_dir='./cache'):
    import os
    import shutil
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Note: Cache folder created: {cache_dir}")

    shutil.copy2(file_path, dest_path)
    print(f"Note: Cache folder created: {dest_path}")
    return

def overwrite_best_fit_params_in_dic(inference):
    # See parameters.py:128 for details
    bestfitdic = inference.params.gen_param_dict(inference.config.params.names, 
                                                 [0.6])
    return bestfitdic

# def complete_flattened_fit_params(fitting_params, line_species):
#     lines        = list(dict.fromkeys(line_species))
#     doublet_lines  = ['O2']
#     doub_pars      = ['v_0_2','dx_vel_2','dy_vel_2','I02','f2_1','f2_2']
#     doub_pars_twin = ['v_0',  'dx_vel',  'dy_vel',  'I01','f1_1','f1_2']
    
#     fitting_par = {}
#     for key, emptydic in fitting_params.items():
#         if key.split('-')[0] == 'shared_params':
#             fitting_par[key] = emptydic
        
#         elif key.split('-')[0] == 'line_params':
#             pname_w_spec = key.split('-')[1] # v_0, I01_spec1, ...
#             for line in lines:
#                 line_p = f'{line}_params-{pname_w_spec}'
#                 fitting_par[line_p] = emptydic
                
#                 if line in doublet_lines:
#                     if 'spec' in pname_w_spec.split('_')[1]: # I01_spec1
#                         pname_wo_spec = pname_w_spec.split('_')[0]
#                         i_q    = doub_pars_twin.index(pname_wo_spec)
#                         q2name = doub_pars[i_q]
#                         line_q = f'{line}_params-{q2name}_{pname_w_spec.split("_")[1]}'
#                         fitting_par[line_q] = emptydic
                        
#                     else: # v_0
#                         i_q    = doub_pars_twin.index(pname_w_spec)
#                         q2name = doub_pars[i_q]
#                         line_q = f'{line}_params-{q2name}'
#                         fitting_par[line_q] = emptydic
                    
#     return fitting_par

# def complete_fit_params(fitting_params, line_species):
#     lines        = list(dict.fromkeys(line_species))
#     doublet_lines  = ['O2']
#     doub_pars      = ['v_0_2','dx_vel_2','dy_vel_2','I02','f2_1','f2_2']
#     doub_pars_twin = ['v_0',  'dx_vel',  'dy_vel',  'I01','f1_1','f1_2']
    
#     fitting_par = {}
#     for key, emptydic in fitting_params.items():
#         if key == 'shared_params':
#             fitting_par[key] = emptydic
        
#         elif key == 'line_params':
#             fitting_par[key] = {}
#             for pname_w_spec, emptydict in emptydic.items():
#                 for line in lines:
#                     line_p = f'{pname_w_spec}'
#                     fitting_par[key][line_p] = emptydic
                    
#                     if line in doublet_lines:
#                         if 'spec' in pname_w_spec.split('_')[1]: # I01_spec1
#                             pname_wo_spec = pname_w_spec.split('_')[0]
#                             i_q    = doub_pars_twin.index(pname_wo_spec)
#                             q2name = doub_pars[i_q]
#                             line_q = f'{q2name}_{pname_w_spec.split("_")[1]}'
#                             fitting_par[key][line_q] = emptydic
                            
#                         else: # v_0
#                             i_q    = doub_pars_twin.index(pname_w_spec)
#                             q2name = doub_pars[i_q]
#                             line_q = f'{q2name}'
#                             fitting_par[key][line_q] = emptydic
                    
#     return fitting_par















if __name__ == '__main__':
    """
        Other external Python scripts may load functions above
        If so, "iter_num" or "run" variables defined wherelse may 
        be contaminated/destroyed!!!
        To protect them, only when directly execute this script,
        the variables below will then be executed.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=1, type=int)
    run = parser.parse_args().run
    
    pkl_folder = './binospec_multi_mock/'
    slit_name  = 3
    # run = 1
    if_continue_last_run = True
    
    # ------------- 1. Load observation data or mock ---------------- #
    data_info = load_mock(f'{pkl_folder}pkl/mock_{slit_name:03d}_{run:03d}.pkl')
    
    # ------------- 2. Load configuration --------------------------- #
    with open("config/binospec_fid_params.yaml", "r", encoding="utf-8") as file1:
        fid_params     = yaml.safe_load(file1)
    with open("config/binospec_fitting_params.yaml", "r", encoding="utf-8") as file2:
        fitting_params = yaml.safe_load(file2)
    
    # Note: Make cache fiducial+fitting files for future
    copy_to_cache(file_path= './config/binospec_fid_params.yaml', 
                  cache_dir= './config/cache/', 
                  dest_path=f'./config/cache/fid_params_slit{slit_name}_{run}.yaml')
    copy_to_cache(file_path= './config/binospec_fitting_params.yaml', 
                  cache_dir= './config/cache/',
                  dest_path=f'./config/cache/fitting_params_slit{slit_name}_{run}.yaml')
    
    linespecies = []
    for spec in data_info['spec']:
        linespecies.append(spec['par_meta']['line_species'])
    
    config_dic = make_config_dic(linespecies, 
                                 fitting_params, fid_params, 
                                 log10_Mstar=10.0, 
                                 log10_Mstar_err=0.05)
    
    # ----------------------- 3. Run inference ---------------------- #
    """ To see each iter/eval's param set in fitting, debug by setting 
        a breakpoint on Line 2628 in site-packages/ultranest/integrator.py
        or Line 120 in ./ultranest_sampler.py 
    """
    inference = UltranestSampler(data_info, config_dic)
    
    if if_continue_last_run:
        last_run_dir = f'./run0.{run:02d}/run{run}'
        prompt = f'''Please note: You are continuing previous chain run. 
             Starting with last run folder: {last_run_dir}.
             Your points were in {last_run_dir}/results/points.hdf5
             Please confirm or input a correct folder (Y/n): '''
        this_input = input(prompt)
        if this_input != 'n':
            if len(this_input) > 1:
                temp_dir = this_input
                if input(f'''Confirming: You specified the folder should be: {temp_dir}. 
            Points file at {temp_dir}/results/points.hdf5 ready? (Y/n): ''') != 'n':
                    pass
                else:
                    print(f'Note: Using {last_run_dir}. See code for questions.')
                    input('Press any key to continue...: ')
            sampler = inference.run(output_dir=last_run_dir, 
                                    if_continue_last_run='resume',
                                    test_run=False, run_num=run,
                                    )
    else:
        sampler = inference.run(output_dir=f'./run0.{run:02d}/', 
                                test_run=False, run_num=run,
                                )
    
    # ------------------- 4. Save sample results -------------------- #
    # save_path = './'
    # with open(f'{save_path}/run0.{run:02d}/ultranest_sampler_results.pkl', 'wb') as f:
    #     pickle.dump(sampler.results, f)
    
    # estimates = sampler.results['maximum_likelihood']['point']
    
    # best_fit_dict = inference.params.gen_param_dict(inference.config.params.names, 
    #                                                 estimates)
    # line = config_dic['galaxy_params']['line_species']
    # this_line_dict = {**best_fit_dict['shared_params'], 
    #                   **best_fit_dict[f'{line[0]}_params']}
    # best_fit_params = {'best_fit_dict': best_fit_dict,
    #                    'this_line_dict': this_line_dict}
    # with open(f'{save_path}/run0.{run:02d}/ultranest_best_fit_params.pkl', "wb") as f:
    #     joblib.dump(best_fit_params, f)
    
    # --------------------- 5. Main Plot Module --------------------- #
    sampler.print_results()
    sampler.plot()
    
    # --------------------- 6. Spec + Image Plot -------------------- #
    # json_filename = f'{save_path}run0.{run:02d}/run{run}/info/results.json'
    # estimates, best_fit_params, fitting_par = load_best_fit_json(
    #     inference, fitting_params, json_filename
    #     )
    
    # print('Plotting best fit comparison with observed spec/image...')
    
    # plot_obs_fit_res(data_info, 
    #                  inference, best_fit_params, fitting_par, 
    #                  run, save_path)

