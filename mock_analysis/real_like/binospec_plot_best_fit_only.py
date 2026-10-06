import yaml

from   klm.ultranest_sampler import UltranestSampler
from   binospec_main_fitting import make_config_dic
from   binospec_main_fitting import load_mock
from   post_fitting          import load_best_fit_json, plot_obs_fit_res

def load_fit_params_config(iter_num, lines=['O2', 'O2', 'O2'], 
                           fid_fn=None, fitting_fn=None):
    # Load YAML for params + fiducial to make config_dic
    try:
        if fid_fn is None:
            fid_fn = f"../config/fid_cache/binospec_fid_params_{iter_num}.yaml"
        with open(fid_fn, "r", encoding="utf-8") as file1:
            fid_params     = yaml.safe_load(file1)
    except:
        if fid_fn is None:
            fid_fn =  "../config/binospec_fid_params.yaml"
        with open(fid_fn, "r", encoding="utf-8") as file1:
            fid_params     = yaml.safe_load(file1)
    print('Reading fid_params     from:', fid_fn)
    
    try:
        if fitting_fn is None:
            fitting_fn = f"../config/fitting_cache/binospec_fitting_params_{iter_num}.yaml"
        with open(fitting_fn, "r", encoding="utf-8") as file2:
            fitting_params = yaml.safe_load(file2)
    except:
        if fitting_fn is None:
            fitting_fn = f"../config/fitting_cache/binospec_fitting_params_{iter_num}.yaml"
        fitting_fn =  "../config/binospec_fitting_params.yaml"
        with open(fitting_fn, "r", encoding="utf-8") as file2:
            fitting_params = yaml.safe_load(file2)
    print('Reading fitting_params from:', fitting_fn)
    
    doublet_lines = ['O2']
    unique_lines  = list(dict.fromkeys(lines))
    linepars      = fitting_params['line_params']
    
    nspec = 0
    last_prefix = ''
    for k, _ in linepars.items():
        if '_spec' in k:
            prefix = k.split('_')[0]
            if prefix != last_prefix and nspec!=0:
                break
            nspec += 1
            last_prefix = prefix
    
    prm_add = {'I01':     None, 'bkg_level': None, 
               'dx_spec': None, 'dy_spec':   None}
    for line in unique_lines:
        if line in doublet_lines:
            prm_add['I02'] = None
    
    fit_k_changed = []
    for prm_add_k, _ in prm_add.copy().items():
        for fit_k, _ in linepars.copy().items():
            if fit_k.split('_')[0] in prm_add_k:
                for line in unique_lines:
                    linepars[line+'_params-'+fit_k] = linepars[fit_k]
                fit_k_changed.append(fit_k)
    for k in fit_k_changed: linepars.pop(k)
    
    for line in unique_lines:
        dic = {}
        for ky, ar in linepars.items():
            line_ = ky.split('-')[0].split('_')[0]
            par_  = ky.split('-')[1]
            if line_ == line:
                dic[par_] = ar
        fitting_params[line+'_params'] = dic
    fitting_params.pop('line_params')
    
    config_dic = make_config_dic(lines, 
                                 fitting_params, fid_params, 
                                 log10_Mstar=9.30, 
                                 log10_Mstar_err=0.05)
    
    return fid_params, fitting_params, config_dic






if __name__ == '__main__':
    """
        Other external Python scripts may load functions above
        If so, "iter_num" or "run" variables defined wherelse may 
        be contaminated/destroyed!!!
        To protect them, only when directly execute this script,
        the variables below will then be executed.
    """
    slit_name = 3
    run       = 1
    date      = 20260116
    # save_path = f'../../../RSCH3/kl_github/runs_{date}/Slit_{slit_name:03d}_runs/'
    # save_path = f'../../../RSCH3/kl_github/runs_survey/Slit_{slit_name:03d}_runs/runs_{date}/'
    save_path = f'Slit_{slit_name:03d}_runs/'
    mock_num  = None
    json_filename = f'{save_path}run0.{run:02d}/run{run}/info/results.json'
    print('Plotting best fit comparison with observed spec/image...')
    
    # Load
    pkl_folder =  './binospec_data_pkl/'
    Ms_folder  =  '../../bagpipes-KL/'
    data_info  = load_mock(pkl_folder, Ms_folder, slit_name, 
                           # rescale_image=True,
                           )
    
    # import numpy as np
    # for i in range(len(data_info['spec'])):
    #     spec_obs = data_info['spec'][i]['data']
    #     spec_var = data_info['spec'][i]['var' ]
    #     spec_con = data_info['spec'][i]['cont_model']
        
    #     perc_val = np.percentile(spec_obs.flatten(), 95)
    #     spec_obs *= (50 / perc_val)
    #     spec_var *= (50 / perc_val)**2
    #     spec_con *= (50 / perc_val)
    #     data_info['spec'][i]['data'] = spec_obs
    #     data_info['spec'][i]['var' ] = spec_var
    #     data_info['spec'][i]['cont_model'] = spec_con
    
    # fid_params, fit_params, config_dic = load_fit_params_config(
    #     run, 
    #     lines=[spec['par_meta']['line_species'] for spec in data_info['spec']],
    #     fitting_fn="./config/binospec_fid_params.yaml", 
    #     fid_fn="./config/binospec_fid_params.yaml"
    #     )
    
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
        log10_Mstar = data_info['galaxy']['log10_Mstar'], 
        log10_Mstar_err = data_info['galaxy']['log10_Mstar_err'],
        use_line_profile = None, #  'meta' 
        )
    
    inference = UltranestSampler(data_info, config_dic)
    
    # --------------------- 6. Spec + Image Plot -------------------- #
    json_filename = f'{save_path}run0.{run:02d}/run{run}/info/results.json'
    estimates, best_fit_params, fitting_par = load_best_fit_json(
        inference, fitting_params, json_filename
        )
    
    
    
    best_fit_params['shared_params']['g1'  ] = 0.0
    best_fit_params['shared_params']['g2'  ] = 0.0
    # best_fit_params['shared_params']['r_hl_disk'] = 0.5
    # best_fit_params['shared_params']['theta_int'] = 0/180*3.14
    # best_fit_params['shared_params']['dx_disk'] =  2.0
    # best_fit_params['shared_params']['dy_disk'] =  2.0
    # best_fit_params['shared_params']['dx_bulge'] = 0.0
    # best_fit_params['shared_params']['dy_bulge'] = 0.0
    # best_fit_params['shared_params']['cosi'] = 0.3
    # best_fit_params['shared_params']['vcirc'] = 200
    # best_fit_params['shared_params']['flux'] += 1
    
    # best_fit_params['Hg_params']['v_0'  ] = 80
    # best_fit_params['O2_params']['v_0_2'] = -160
    # best_fit_params['O2_params']['I01_spec1'] = 150
    # best_fit_params['O2_params']['I02_spec1'] = 200
    # best_fit_params['O2_params']['I01_spec2'] =  20
    # best_fit_params['O2_params']['I02_spec2'] =  30
    # best_fit_params['O2_params']['I01_spec3'] =  50
    # best_fit_params['O2_params']['I02_spec3'] =  50
    
    # best_fit_params['Hb_params']['v_0'] = 80
    # best_fit_params['Hb_params']['I01_spec1'] =  85
    # best_fit_params['Hb_params']['I01_spec2'] =  45
    # best_fit_params['Hb_params']['I01_spec3'] =  50
    
    # best_fit_params['O3a_params']['v_0'] = 80
    # best_fit_params['O3a_params']['I01_spec1'] = 80
    # best_fit_params['O3a_params']['I01_spec2'] = 45
    # best_fit_params['O3a_params']['I01_spec3'] = 55
    
    # best_fit_params['O3b_params']['I01_spec1'] = 85
    # best_fit_params['O3b_params']['I01_spec2'] = 45
    # best_fit_params['O3b_params']['I01_spec3'] = 55
    
    # best_fit_params['O3b_params']['v_0'] = 80
    # best_fit_params['Ha_params']['I01_spec1'] =  90
    # best_fit_params['Ha_params']['I01_spec2'] =  90
    # best_fit_params['Ha_params']['I01_spec3'] =  90
    
    # If change ΔRA to RA in plot,
    # best_fit_params['shared_params']['g2'] *= -1
    
    print('Plotting best fit comparison with observed spec/image...')
    
    plot_obs_fit_res(data_info, 
                     inference, best_fit_params, fitting_par, 
                     run, other_path_filename='best_fit.png')

    
    
    
    
    