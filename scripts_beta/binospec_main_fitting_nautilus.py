import joblib
import yaml
import json
import galsim
import os
import argparse
import time
import numpy as np

from core.make_config_dic import make_config_dic
from core.post_fitting import plot_obs_fit_res
from core.fitting_result_utils import complete_flattened_fit_params
from klm.parameters import Parameters
from klm.nautilus_sampler import NautilusSampler
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
    
    assert data_info['galaxy']['log10_Mstar'] != None, \
        "Cannot find corresponing stallar mass M*"  # if not, error
    
    # Recover wcs(galsim.wcs) from ap_wcs
    ap_wcs  = data_info['image']['par_meta']['ap_wcs']
    data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    return data_info

















if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--slitID', default=   95, type=int)
    parser.add_argument('--run',    default=    1, type=int)
    # Warning: ONLY input True if you want following two arguments 
    #          because of bool("non_empty_str") == True.
    parser.add_argument('--test',   default=False, type=bool)
    parser.add_argument('--contin', default=False, type=bool)
    slit_name = parser.parse_args().slitID
    run       = parser.parse_args().run
    if_test   = parser.parse_args().test
    if_continue_last_run = parser.parse_args().contin
    
    pkl_folder  =  './binospec_pkl/'
    Ms_folder   =  '../../bagpipes-KL/'
    slit_folder = f'./Slit_{slit_name:03d}/'
    fiduci_yaml =  "../config/binospec_fid_params.yaml"
    fittin_yaml =  "../config/binospec_fitting_params.yaml"
    save_path   = slit_folder
    
    # if_continue_last_run = False # True 
    
    # ------------- 1. Load observation data or mock ---------------- #
    try:
        data_info = load_mock(pkl_folder, Ms_folder, slit_name)
        
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
        use_line_profile = 'extracted', # None, 'raw' 
        )
    
    nautilus_sampler = NautilusSampler(data_info, config_dic)
    for par, prior in nautilus_sampler.config.params.prior.items(): print(par, prior)
    
    # ------------- 3. Start fitting --------------------------- #
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
                     slit_name, save_path=save_path)
    
    # Plot - corner
    # import corner
    # from core.fitting_result_utils import complete_fit_params
    # fitting_par = complete_fit_params(fitting_params, linespecies)
    # par_names, label_latex = [], []
    # for key, subdict in fitting_par.items():
    #     par_names.append(key.split('-')[1])
    #     label_latex.append(subdict['latex_name'])
    from core.plot_corner import plot_corner
    if if_test is False:
        # corner.corner(points, weights=np.exp(log_w), 
        #               show_titles=True, 
        #               title_kwargs={'size': 36},
        #               labels=label_latex, 
        #               label_kwargs={'size': 36},
        #               color='black')
        post_path_new  = f'{slit_folder}post.txt'
        plot_corner(
            [post_path_new],  
            [f'#{slit_name}'],  
            nautilus_color='dimgray', # deepskyblue
            # params_to_plot = params_to_plot,
            percentile=0, 
            change_to_equal_weights_in_case=True,
            corner_name=f'{slit_folder}corner_all.png',
            test=False,
            )
        print('Plotting done.')
    else:
        print('Corner plotting skipped because this is a test run.')