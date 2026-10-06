import yaml
import joblib
import pickle
import galsim
import os
import argparse

from   klm.ultranest_sampler import UltranestSampler
from   post_fitting          import load_best_fit_json, plot_obs_fit_res

from klm.safe_plot import setup; setup() # must before plt
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def load_mock(pkl_folder='mock/', Ms_folder='./', slit_num=95, 
              rescale_image=False):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
        data_info = joblib.load(f)
    
    if rescale_image:
        import copy
        data_info = copy.deepcopy(data_info)
        data_info['image']['data'] = data_info['image']['data'] * 10
        data_info['image']['var' ] = data_info['image']['var' ] * 100
    
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
    parser.add_argument('--slitID', default=  3,   type=int)
    parser.add_argument('--run',    default=  1,   type=int)
    parser.add_argument('--test',   default=False, type=bool)
    parser.add_argument('--contin', default=False, type=bool)
    slit_name = parser.parse_args().slitID
    run       = parser.parse_args().run
    if_test   = parser.parse_args().test
    if_continue_last_run = parser.parse_args().contin
    # if_test = True
    
    pkl_folder  =  './binospec_mock_pkl/'
    Ms_folder   =  '../../bagpipes-KL/'
    slit_folder = f'./Slit_{slit_name:03d}_runs_/'
    fiduci_yaml =  "./config/binospec_fid_params.yaml"
    fittin_yaml =  "./config/binospec_fitting_params.yaml"
    
    # if_continue_last_run = False # True 
    
    # ------------- 1. Load observation data or mock ---------------- #
    data_info = load_mock(pkl_folder, Ms_folder, slit_name,
                          # rescale_image=True
                          )
    
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
    
    # ----------------------- 3. Run inference ---------------------- #
    """ To see each iter/eval's param set in fitting, debug by setting 
        a breakpoint on Line 2628 in site-packages/ultranest/integrator.py
        or Line 120 in ./ultranest_sampler.py 
    """
    inference = UltranestSampler(data_info, config_dic)
    
    if not os.path.exists(slit_folder):
        os.makedirs(slit_folder)
        print(f"Note: Cache folder created: {slit_folder}")
    
    if if_continue_last_run:
        last_run_dir = f'{slit_folder}run0.{run:02d}/run{run}'
        # prompt = f'''Please note: You are continuing previous chain run. 
        #      Starting with last run folder: {last_run_dir}.
        #      Your points were in {last_run_dir}/results/points.hdf5
        #      Please confirm or input a correct folder (Y/n): '''
        # this_input = input(prompt)
        # if this_input != 'n':
        #     if len(this_input) > 1:
        #         temp_dir = this_input
        #         if input(f'''Confirming: You specified the folder should be: {temp_dir}. 
        #     Points file at {temp_dir}/results/points.hdf5 ready? (Y/n): ''') != 'n':
        #             pass
        #         else:
        #             print(f'Note: Using {last_run_dir}. See code for questions.')
        #             input('Press any key to continue...: ')
        sampler = inference.run(
            output_dir=last_run_dir, 
            if_continue_last_run='resume',
            test_run=if_test, run_num=run,
            )
    else:
        sampler = inference.run(
            output_dir=f'{slit_folder}run0.{run:02d}/', 
            test_run=if_test, run_num=run,
            )
    
    # ------------------- 4. Save sample results -------------------- #
    save_path = slit_folder
    with open(f'{save_path}/run0.{run:02d}/ultranest_sampler_results.pkl', 'wb') as f:
        pickle.dump(sampler.results, f)
    
    estimates = sampler.results['maximum_likelihood']['point']
    
    best_fit_dict = inference.params.gen_param_dict(inference.config.params.names, 
                                                    estimates)
    line = config_dic['galaxy_params']['line_species']
    this_line_dict = {**best_fit_dict['shared_params'], 
                      **best_fit_dict[f'{line[0]}_params']}
    best_fit_params = {'best_fit_dict': best_fit_dict,
                       'this_line_dict': this_line_dict}
    with open(f'{save_path}/run0.{run:02d}/ultranest_best_fit_params.pkl', "wb") as f:
        joblib.dump(best_fit_params, f)
    
    # --------------------- 5. Main Plot Module --------------------- #
    sampler.print_results()
    sampler.plot()
    
    # --------------------- 6. Spec + Image Plot -------------------- #
    json_filename = f'{save_path}run0.{run:02d}/run{run}/info/results.json'
    estimates, best_fit_params, fitting_par = load_best_fit_json(
        inference, fitting_params, json_filename
        )
    
    print('Plotting best fit comparison with observed spec/image...')
    
    plot_obs_fit_res(data_info, 
                     inference, best_fit_params, fitting_params, 
                     run, save_path=save_path)
    
    print('Plotting done.')
