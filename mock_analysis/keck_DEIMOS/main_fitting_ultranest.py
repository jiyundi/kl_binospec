import joblib
import yaml

import galsim
import os
import argparse
import time
import numpy as np

from core.make_config_dic import make_config_dic
from core.post_fitting import plot_obs_fit_res
from core.fitting_result_utils import load_best_fit_json
from klm.ultranest_sampler import UltranestSampler

from klm.safe_plot import setup; setup() # must before plt

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def load_mock_Pranjal(pkl_folder='mock/', slit_num=95):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}_converted.pkl', "rb") as f:
        data_info_raw = joblib.load(f)
    
    data_info = data_info_raw.copy()
    assert data_info['galaxy']['log10_Mstar'] != None, \
        "Cannot find corresponing stallar mass M*"  # if not, error
    
    image_data = data_info['image']['data']
    image_mask = data_info['image']['mask']
    assert ~np.any(np.isnan(image_data[image_mask])), \
        "Image: contains NaN pixel values."
    
    for i in range(len(data_info['spec'])):
        spec_data = data_info['spec'][i]['data']
        spec_mask = data_info['spec'][i]['mask']
        assert ~np.any(np.isnan(spec_data[spec_mask])), \
            f"Spec {i}: contains NaN pixel values."
    
    # Recover wcs(galsim.wcs) from ap_wcs
    ap_wcs  = data_info['image']['meta']['ap_wcs']
    data_info['image']['meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    
    return data_info

















if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--slitID', default=  2, type=int)
    parser.add_argument('--run',    default=  1, type=int)
    # Warning: ONLY input True if you want following two arguments 
    #          because of bool("True"/"False") == True.
    parser.add_argument('--test',   default=False, type=bool)
    parser.add_argument('--contin', default=False, type=bool)
    slit_name = parser.parse_args().slitID
    run       = parser.parse_args().run
    if_test   = parser.parse_args().test
    if_continue_last_run = parser.parse_args().contin
    
    pkl_folder  =  './'
    # Ms_folder   =  '../../bagpipes-KL/'
    slit_folder = f'./Slit_{slit_name:03d}/'
    fiduci_yaml =  "../config/binospec_fid_params_Pranjal.yaml"
    fittin_yaml =  "../config/binospec_fitting_params_Pranjal_g05.yaml"
    save_path   = slit_folder
    
    # ------------- 1. Load observation data or mock ---------------- #
    print(f'\nFitting for Slit {slit_name:03d} .............................................................')
    try:
        data_info = load_mock_Pranjal(pkl_folder, slit_name)
        
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
        linespecies.append(spec['meta']['line_species'])
    
    config_dic = make_config_dic(
        linespecies, fitting_params, fid_params, 
        log10_Mstar=data_info['galaxy']['log10_Mstar'], 
        log10_Mstar_err=data_info['galaxy']['log10_Mstar_err'],
        use_line_profile = 'extracted', #  'meta' 
        )
        
    inference = UltranestSampler(data_info, config_dic)

    # ----------------------- 3. Run inference ---------------------- #
    """ To see each iter/eval's param set in fitting, debug by setting 
        a breakpoint on Line 2628 in site-packages/ultranest/integrator.py
        or Line 120 in ./ultranest_sampler.py 
    """
    t_start = time.time()
    if if_continue_last_run:
        last_run_dir = f'{slit_folder}run0.{run:02d}/run{run}'
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
    
    t_end = time.time()
    print('Total time: {:.1f}s'.format(t_end - t_start))

    # --------------------- 4. Main Plot Module --------------------- #
    sampler.print_results()
    sampler.plot()
    
    # --------------------- 5. Spec + Image Plot -------------------- #
    json_filename = f'{save_path}run0.{run:02d}/run{run}/info/results.json'
    import json
    with open(json_filename, 'r', encoding='utf-8') as f:
        json_estimates = json.load(f)
        
    estimates, best_fit_params, fitting_par = load_best_fit_json(
        inference, fitting_params, json_filename
        )
    
    print('Plotting best fit comparison with observed spec/image...')
    
    plot_obs_fit_res(data_info, 
                    inference, best_fit_params, fitting_params, 
                    slit_name, save_path=save_path)
    
    print('Plotting done.')
