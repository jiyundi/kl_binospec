import yaml
# import joblib
import numpy as np

from   klm.ultranest_sampler import UltranestSampler
from   binospec_main_fitting import load_mock, make_config_dic
from   post_fitting          import load_best_fit_json

from klm.safe_plot import setup; setup() # must before plt
import matplotlib.pyplot as plt
plt.style.use('default')

# Useful functions
def rough_check_gamma_convergence(inference, best_param, 
                                  param_1Dgrid, 
                                  par_idx, par_name):
    log_like = []
    
    for i in range(len(param_1Dgrid)):
        param_grid          = best_param
        param_grid[par_idx] = param_1Dgrid[i]
        
        likeli = inference.calc_joint_loglike(param_grid)
        log_like.append(likeli)
    
    fig = plt.figure(figsize=(6,3))  # (length, height)
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0, 0])
    
    ax.plot(param_1Dgrid, np.array(log_like))
    
    ax.grid()
    plt.title(par_name)
    plt.show()
    return






if __name__ == '__main__':
    slit_name = 89
    run       = 1
    date      = 20260116
    save_path = f'../../../RSCH3/kl_github/runs_{date}/Slit_{slit_name:03d}_runs/'
    json_filename = f'{save_path}run0.{run:02d}/run{run}/info/results.json'
    
    # Load
    pkl_folder =  './binospec_data_pkl/'
    Ms_folder  =  '../../bagpipes-KL/'
    data_info  = load_mock(pkl_folder, Ms_folder, slit_name, 
                           # rescale_image=True,
                           )
    
    for i in range(len(data_info['spec'])):
        spec_obs = data_info['spec'][i]['data']
        spec_var = data_info['spec'][i]['var' ]
        spec_con = data_info['spec'][i]['cont_model']
        
        perc_val = np.percentile(spec_obs.flatten(), 95)
        spec_obs *= (50 / perc_val)
        spec_var *= (50 / perc_val)**2
        spec_con *= (50 / perc_val)
        data_info['spec'][i]['data'] = spec_obs
        data_info['spec'][i]['var' ] = spec_var
        data_info['spec'][i]['cont_model'] = spec_con
        
    # Load YAML file for config
    with open("./config/binospec_fid_params.yaml", "r", encoding="utf-8") as file1:
        fid_params     = yaml.safe_load(file1)
    with open("./config/binospec_fitting_params.yaml", "r", encoding="utf-8") as file2:
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
    
    inference = UltranestSampler(data_info, config_dic)
    
    # Make test params
    par_name = 'O3b_params-I01_spec1'
    for par_name in inference.config.params.names:
        try: 
            if 'spec' in par_name.split('_')[2]:
                par_min  =  20
                par_max  = 120
                par_idx  = inference.config.params.names.index(par_name)
                param_1Dgrid = np.array(np.linspace(par_min, par_max, 20))
                
                # slit_folder = f'./Slit_{slit_name:03d}_runs/'
                # with open(f'{slit_folder}run0.{run:02d}/ultranest_sampler_results.pkl', "rb") as f:
                #     fl = joblib.load(f)
                
                # best_param = fl['maximum_likelihood']['point']
                # par_names  = fl['paramnames']
                # assert len(best_param)==len(par_names)
                
                json_filename = f'{save_path}run0.{run:02d}/run{run}/info/results.json'
                estimates, best_fit_params, fitting_par = load_best_fit_json(
                    inference, fitting_params, json_filename
                    )
                
                rough_check_gamma_convergence(inference, estimates, 
                                              param_1Dgrid, par_idx, 
                                              par_name)
        except IndexError:
            pass
    
    # with open(f'{slit_folder}run0.{run_num:02d}/ultranest_best_fit_params.pkl', "rb") as f:
    #     best_param = joblib.load(f)
    # plot_obs_fit_res(data_info, 
    #                  inference, best_param['best_fit_dict'], 
    #                  fitting_params, #fit_core_params, 
    #                  run_num, save_path=None)