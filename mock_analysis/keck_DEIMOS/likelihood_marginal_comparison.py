import joblib
import yaml
import galsim
import os
import argparse
import numpy as np
from tqdm import tqdm

from core.make_config_dic import make_config_dic
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


def load_mock_Pranjal(pkl_folder='mock/', slit_num=95):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
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
    parser.add_argument('--slitID', default=  4, type=int)
    parser.add_argument('--run',    default=  1, type=int)
    # Warning: ONLY input True if you want following two arguments 
    #          because of bool("True"/"False") == True.
    parser.add_argument('--test',   default=False, type=bool)
    parser.add_argument('--follow' ,default= True, type=bool)
    
    # for slit_name in np.arange(0, 11):
    slit_name = parser.parse_args().slitID
    run       = parser.parse_args().run
    if_test   = parser.parse_args().test
    if_continue_last_run = parser.parse_args().follow
    
    pkl_folder  =  './'
    slit_folder = f'./Slit_{slit_name:03d}/'
    fiduci_yaml =  "./binospec_fid_params_Pranjal.yaml"
    fittin_yaml =  "./binospec_fitting_params_Pranjal.yaml"
    save_path   = slit_folder
    
    special_idxs = ['0.0', '0.3', '0.7', 
                    '1.0', '1.4', '1.6', '1.7', 
                    '2.1', '2.4', '2.8',
                    '3.1' ]
    mock_folder = '008b_vary_thetaint_slitLPA_major/'
    
    # ------------- 1. Load observation data or mock ---------------- #
    print(f'\nFitting for Slit {slit_name:03d} .............................................................')
    try:
        with open(f'{mock_folder}/slit_002_{special_idxs[slit_name]}.pkl', "rb") as f:
            data_info = joblib.load(f)
        print('Mock setting: slit_LPA  =', data_info['spec'][0]['meta']['slitLPA'])
        print('Mock setting: theta_int =', data_info['fid_params']['shared_params']['theta_int'])
        print('Mock setting: cosi      =', data_info['fid_params']['shared_params']['cosi'])
                                
    except FileNotFoundError:
        print( "\033[43m" + 'WARNING:' + "\033[0m " + 
            f'Slit {slit_name} skipped because no PKL found.\n')
        os._exit(0)
    
    # ------------- 2. Load configuration --------------------------- #
    with open(fiduci_yaml, "r", encoding="utf-8") as file1:
        fiducial_params = yaml.safe_load(file1)
    with open(fittin_yaml, "r", encoding="utf-8") as file2:
        fitting_params  = yaml.safe_load(file2)
    
    linespecies = []
    for spec in data_info['spec']:
        linespecies.append(spec['meta']['line_species'])
    
    config_dict = make_config_dic(
        linespecies, fitting_params, fiducial_params, 
        log10_Mstar=data_info['galaxy']['log10_Mstar'], 
        log10_Mstar_err=data_info['galaxy']['log10_Mstar_err'],
        use_line_profile = 'extracted', # None, 'raw' 
        )
    
    nautilus_sampler = NautilusSampler(data_info, config_dict, (not if_test))
    fitting_params_flat = Parameters._flatten(fitting_params, level=1)
    fitting_par = complete_flattened_fit_params(
        fitting_params_flat, 
        line_species=nautilus_sampler.config.galaxy_params.line_species
        )
    
    # ------------- 3. Testing (g1, theta_int) --------------------------- #
    fid_pars = {
        'shared_params-g1':        data_info['fid_params']['shared_params']['g1'], # -0.04124161290994201,
        'shared_params-theta_int': data_info['fid_params']['shared_params']['theta_int'], # result: 1.3962634015954636
        'shared_params-vscale':       0.10314708567188158,
        'Hb_params-v_0':           -153.17706740192824,
        'Hb_params-I01_spec1':       20.07195794200264,
        'Hb_params-bkg_level_spec1': -2.866095297084968
        }
    fid_dict = nautilus_sampler.params.gen_param_dict(fitting_par.keys(), fid_pars.values())


    # ------------- 4. Compute image chi2 grid violently--------------------------- #
    # g1_grid = np.linspace(-0.40, 0.05, 301)
    # theta_grid = np.linspace(0.0, np.pi, 361)

    # chi2_grid = np.empty((len(g1_grid), len(theta_grid)))

    # base = fid_dict['shared_params'].copy()
    # data = data_info['image']['data']
    # var = data_info['image']['var']

    # print('Computing chi2 grid for g1 and theta_int ...')
    # for i, g1_value in enumerate(g1_grid):
    #     for j, theta_value in enumerate(theta_grid):
    #         pars = base.copy()
    #         pars['g1'] = g1_value
    #         pars['theta_int'] = theta_value

    #         model = nautilus_sampler.image_model.get_image(pars)

    #         chi2_grid[i, j] = np.sum(
    #             (data - model)**2 / var
    #         )
    # print('Chi2 grid computation completed.')

    # logl_grid = -0.5 * chi2_grid

    # from scipy.special import logsumexp

    # # 对每个 g1，取 theta 上的最大似然
    # profile_g1 = np.max(logl_grid, axis=1)

    # # 对 theta 积分，得到 g1 的边际分布
    # marginal_g1 = logsumexp(logl_grid, axis=1)

    # profile_g1 -= np.max(profile_g1)
    # marginal_g1 -= np.max(marginal_g1)

    # print("Starting to plot the profile and marginal distributions for g1 ...")

    # import matplotlib.pyplot as plt

    # plt.plot(g1_grid, profile_g1, label='Profile log-likelihood')
    # plt.plot(g1_grid, marginal_g1, label='Marginal log-posterior')
    # plt.axvline(-0.0412416, color='k', ls='--', label='Truth')
    # plt.xlabel('g1')
    # plt.legend()
    # plt.savefig('g1_marginal.png')


    # ------------- 4. Compute spec chi2 grid violently--------------------------- #
    g1_grid    = np.linspace(-0.50, 0.50, 25)
    theta_grid = np.linspace(0.0, np.pi, 25)
    chi2_grid  = np.empty((len(g1_grid), len(theta_grid)))

    try:
        chi2_grid = np.loadtxt("chi2_grid.txt")
    except FileNotFoundError:
        base_pars = {**fid_dict['shared_params'].copy(), **fid_dict['Hb_params'].copy()}
        for k in ('I01', 'I02'):
            spec_key = f"{k}_spec1"
            if spec_key in base_pars:
                base_pars[k] = base_pars[spec_key]
        fid_data  = data_info['spec'][0]['data']
        fid_var   = data_info['spec'][0]['var']

        from scipy.optimize import minimize

        def profile_chi2(g1, theta):
            def objective(x):
                v0, I0, bkg = x
                pars_one_level = {
                    **base_pars,
                    "g1": g1,
                    "theta_int": theta,
                    "v_0": v0,
                    "I01": I0,
                    "bkg_level": bkg,
                }
                model = nautilus_sampler.spec_model[0].get_observable(pars_one_level)
                return np.sum((fid_data - model)**2 / fid_var)

            result = minimize(
                objective,
                x0=[-153.177, 20.072, -2.866],
                method="Nelder-Mead",
            )
            return result.fun

        print('Computing chi2 grid for g1 and theta_int ...')
        for i, g1_value in tqdm(enumerate(g1_grid), desc="Processing g1 values"):
            for j, theta_value in tqdm(enumerate(theta_grid), desc="Processing theta values"):
                chi2_grid[i, j] = profile_chi2(g1_value, theta_value)
        print('Chi2 grid computation completed.')
        np.savetxt("chi2_grid.txt", chi2_grid)

    delta_chi2 = chi2_grid - np.min(chi2_grid)

    plt.imshow(
        delta_chi2.T,
        origin="lower",
        aspect="auto",
        extent=[
            g1_grid[0], g1_grid[-1],
            theta_grid[0], theta_grid[-1]
        ],
    )

    plt.colorbar(label=r"$\Delta\chi^2$")
    plt.scatter(-0.04, 1.4, marker="*", color="red", label="Truth")
    plt.xlabel(r"$g_1$")
    plt.ylabel(r"$\theta_{\rm int}$")
    plt.legend()
    plt.savefig("chi2_grid.png")