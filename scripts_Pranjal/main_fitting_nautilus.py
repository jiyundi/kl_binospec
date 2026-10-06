import joblib
import yaml
import json
import galsim
import os
import argparse
import time
import numpy as np
import astropy.units as u

from core.make_config_dic import make_config_dic
from core.post_fitting import plot_obs_fit_res
from core.fitting_result_utils import complete_flattened_fit_params
from klm.parameters import Parameters
from klm.nautilus_sampler import NautilusSampler
from klm.line_profile_extraction import extract_asymmetric_LPF, exam_line_profile
from klm.line_profile_extraction import extract_asymmetric_lpf_A, exam_line_profile_A
from klm.safe_plot import setup; setup() # must before plt

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--slitID', default=  0, type=int)
    parser.add_argument('--run',    default=  1, type=int)
    # Warning: ONLY input True if you want following two arguments 
    #          because of bool("True"/"False") == True.
    parser.add_argument('--test',    default=False, type=bool)
    parser.add_argument('--parallel',default=False, type=bool)
    slit_name   = parser.parse_args().slitID
    run         = parser.parse_args().run
    if_test     = parser.parse_args().test
    if_parallel = parser.parse_args().parallel
        
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
    current_time = time.ctime()
    print("Current time:", current_time)
    try:
        data_info_path = f'{mock_folder}/slit_002_{special_idxs[slit_name]}.pkl'
        with open(data_info_path, "rb") as f:
            data_info = joblib.load(f)
        print('Mock setting: slit_LPA  =', data_info['spec'][0]['meta']['slitLPA'])
        print('Mock setting: theta_int =', data_info['fid_params']['shared_params']['theta_int'])
        print('Mock setting: cosi      =', data_info['fid_params']['shared_params']['cosi'])
                                
    except FileNotFoundError:
        print( "\033[43m" + 'WARNING:' + "\033[0m " + 
              f'Slit {slit_name} skipped because no PKL found at: {data_info_path}\n')
        os._exit(0)
    
    # CAUTION
    # print('CAUTION: spec is masked out the central region.')
    # spec_mask = data_info['spec'][0]['mask']
    # new_mask  = np.ones(spec_mask.shape, dtype=bool)
    # new_mask[new_mask.shape[0]//2 - 5 : new_mask.shape[0]//2 + 5, :] = False
    # data_info['spec'][0]['mask'] = new_mask

    spec_data = data_info['spec'][0]['data']
    spec_mask = data_info['spec'][0]['mask']
    spec_wave = data_info['spec'][0]['meta']['lambda_grid']
    # lambda_scale = 0.33 # A/px
        
    # Delete my LPFs, overwrite with Pranjal's
    data_info['spec'][0]['meta'].pop('line_profile')

    # ------------- 1.1 Load LPF by Pranjal ---------------- #
    with open(f'{mock_folder}/spec_extract_a2261b_008.pkl', "rb") as f:
        line_profile = joblib.load(f)
        
    line_profile_Hb = line_profile['Hb']
    Amp1     = line_profile_Hb[0][:, 0]
    mu1      = line_profile_Hb[1][:, 0]
    sigma_1l = line_profile_Hb[2][:, 0] # First Gaussian, left  wing
    sigma_1r = line_profile_Hb[3][:, 0] # First Gaussian, right wing
    Amp2     = line_profile_Hb[0][:, 1]
    mu2      = line_profile_Hb[1][:, 1]
    sigma_2l = line_profile_Hb[2][:, 1] # Second Gaussian, left  wing
    sigma_2r = line_profile_Hb[3][:, 1] # Second Gaussian, right wing

    # Clear zero mu wavelengths
    zero_mask = (mu1 == 0) & (mu2 == 0)
    for arr in (Amp1, mu1, sigma_1l, sigma_1r, 
                Amp2, mu2, sigma_2l, sigma_2r):
        arr[zero_mask] = np.nan

    # Unit conversion: A -> px
    # align min spec_wavelength with x=0
    # E.g. mu = [8848 A, 8850 A, ...] --> [24 px, 25 px, ...]
    # first_col_wave = spec_wave[:, 0].value # unit is A
    # mu1 = (mu1.value - first_col_wave) / lambda_scale # A/px
    # if np.nansum(mu2.value) != 0:
    #     mu2 = (mu2.value - first_col_wave) / lambda_scale # A/px
    
    Amp = np.array([Amp1, Amp2]).T / np.nanmax([Amp1, Amp2]) * np.nanmax(spec_data[spec_mask])
    Mu  = np.array([ mu1,  mu2]).T
    sigma1 = np.array([sigma_1l, sigma_2l]).T
    sigma2 = np.array([sigma_1r, sigma_2r]).T

    Amp_JD, Mu_JD, sigma1_JD, sigma2_JD = extract_asymmetric_lpf_A(
        spec_data,
        spec_wave,
        line='Hb'
    )

    restored, residual = exam_line_profile_A(
        (Amp, Mu, sigma1, sigma2),
        spec_data,
        spec_mask,
        wavelength=spec_wave,
        line='Hb', 
        plot_with=(Amp_JD[:, 0], Mu_JD[:, 0], sigma1_JD[:, 0], sigma2_JD[:, 0]), 
        label_of_plot_with='JD extracted'
        )

    # restored, residual = exam_line_profile(
    #         (Amp, Mu, sigma1, sigma2),
    #         spec_data,
    #         spec_mask, 
    #         spec_var = data_info['spec'][0]['var'],
    #         lambda_scale = lambda_scale
    #     )

    # ------------- 1.2 Convert JD LPF into dict ---------------- #
    # lambda_scale = np.nanmedian(np.abs(np.diff(spec_wave.to(u.Angstrom).value, axis=1)), axis=1)
    # Amp_JD, Mu_JD, sigma1_JD, sigma2_JD = extract_asymmetric_lpf_A(
    #     spec_data,
    #     spec_wave,
    #     line='Hb'
    # )

    # LPFs_JD = (Amp_JD, Mu_JD, sigma1_JD, sigma2_JD)

    restored, residual = exam_line_profile_A(
        (Amp_JD, Mu_JD, sigma1_JD, sigma2_JD),
        spec_data,
        spec_mask, 
        wavelength=spec_wave,
        line='Hb',
        filename="test_LPF_A_JD.jpg",
        )

    Amp1     = Amp_JD[:, 0]
    mu1      = Mu_JD[ :, 0]
    sigma1_1 = sigma1_JD[:, 0] # First Gaussian, left  wing
    sigma1_2 = sigma2_JD[:, 0] # First Gaussian, right wing
    Amp2     = Amp_JD[:, 1]
    mu2      = Mu_JD[ :, 1]
    sigma2_1 = sigma1_JD[:, 1] # Second Gaussian, left  wing
    sigma2_2 = sigma2_JD[:, 1] # Second Gaussian, right wing
    
    Amp1 /= np.nanmax(Amp1)
    if np.any(Amp2 > 0): Amp2 /= np.nanmax(Amp2)

    # For JD:
    mask = ~np.isnan(Amp1) & ~np.isnan(Amp2)
    Amp1[~mask], Amp2[~mask] = 0, 0
    
    # save in dict
    LP_line1 = {
        'amp': Amp1,
        'mean': mu1, 
        'std_left':  sigma1_1,
        'std_right': sigma1_2,
        'reliability': mask,
        'bkg': np.zeros(Amp1.shape) # Pranjal assumed zero bkg
        }
    LP_line2 = {
        'amp': Amp2,
        'mean': mu2, 
        'std_left':  sigma2_1,
        'std_right': sigma2_2,
        'reliability': mask,
        'bkg': np.zeros(Amp2.shape) # Pranjal assumed zero bkg
        }

    data_info['spec'][0]['meta']['line_profile'] = (LP_line1, LP_line2)


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
    
    # if if_test is False:
    for par, prior in nautilus_sampler.config.params.prior.items():
        print(par, prior)
    print('slit PA:', data_info['spec'][0]['meta']['slitLPA'])





    # fid_params_flat = Parameters._flatten(
    #     data_info["fid_params"],
    #     level=1,
    # )
    # fid_params_flat['Hb_params-I01_spec1']       = data_info['fid_params']['Hb_params']['I01']
    # fid_params_flat['Hb_params-bkg_level_spec1'] = data_info['fid_params']['Hb_params']['bkg_level']
    # fid_params_flat['Hb_params-dx_vel_spec1']    = data_info['fid_params']['Hb_params']['dx_vel']
    # missing_params = [
    #     name
    #     for name in nautilus_sampler.config.params.names
    #     if name not in fid_params_flat
    # ]

    # if missing_params:
    #     print(
    #         "Cannot build test cube from fid_params. "
    #         "Missing parameters:"
    #     )
    #     for name in missing_params:
    #         print("  ", name)
    # else:
    #     test_cube = np.asarray(
    #         [
    #             fid_params_flat[name]
    #             for name in nautilus_sampler.config.params.names
    #         ],
    #         dtype=float,
    #     )

    #     print("Testing likelihood at fiducial parameters:")
    #     for name, value in zip(
    #         nautilus_sampler.config.params.names,
    #         test_cube,
    #     ):
    #         print(f"  {name}: {value}")

    #     nautilus_sampler.check_parallel_loglike(
    #         cube=test_cube,
    #         n_workers=None,  # 自动读取 SLURM_CPUS_PER_TASK
    #         n_repeat=16,
    #     )
    # time.sleep(10)
    # os._exit(0) 





    # ------------- 3. Start fitting --------------------------- #
    t_start = time.time()
    sampler = nautilus_sampler.run(
        output_dir=f'{slit_folder}run0.{run:02d}/', 
        test_run=if_test, run_num=run, parallel=if_parallel, 
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
    if if_test is False:
        best = points[np.argmax(log_l)]
        best_dict = dict(zip(nautilus_sampler.config.params.names, best))
        
        # Compute posterior weights
        weights /= weights.sum()
        
        # Median: weighted percentile for each parameter
        median_point = np.array([
            np.interp(0.5, np.cumsum(w := weights[np.argsort(points[:, i])]), 
                      points[np.argsort(points[:, i]), i])
            for i in range(points.shape[1])
        ])
        median_dict = dict(zip(nautilus_sampler.config.params.names, median_point))
        
        # Peak (Mode): weighted KDE for each parameter
        from scipy.stats import gaussian_kde
        mode_point = np.array([
            (lambda p, x: x[np.argmax(gaussian_kde(p, weights=weights, bw_method='scott')(x))])(
                points[:, i], np.linspace(points[:, i].min(), points[:, i].max(), 200)
            )
            for i in range(points.shape[1])
        ])
        mode_dict = dict(zip(nautilus_sampler.config.params.names, mode_point))
        
        out = { 
            # Note: json does not support array.
            "fid_params":     fiducial_params, 
            "fitting_params": fitting_params, 
            "maximum_likelihood": {
                "point": best_dict,
                "log_likelihood": float(np.max(log_l))
            },
            "posterior_median": {
                "point": {k: float(v) for k, v in median_dict.items()}
            },
            "posterior_mode": {
                "point": {k: float(v) for k, v in mode_dict.items()}
            }
        }
        with open(slit_folder + "best_fit.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)
    
    # Plot - best fit
    if if_test is False: 
        print('Plotting best fit comparison with observed spec/image...')
        fitting_params_flat = Parameters._flatten(fitting_params, level=1)
        fitting_par = complete_flattened_fit_params(
            fitting_params_flat, 
            line_species=nautilus_sampler.config.galaxy_params.line_species
            )
        best_fit_dict  = nautilus_sampler.params.gen_param_dict(fitting_par.keys(), 
                                                                mode_dict.values())
        
        plot_obs_fit_res(data_info, 
                         nautilus_sampler, 
                         best_fit_dict, 
                         fitting_params, 
                         slit_name, save_path=save_path)
        print('Plotting done.')
    
    else:
        print('Best-fit plotting skipped because this is a test run.')
    
    # Plot - corner
    if if_test is False:
        from core.plot_corner import plot_corner
        post_path_new  = f'{slit_folder}post.txt'
        
        # Calibrate v_0 due to different definitions of Hb
        c_kms = 299792.458
        Hb_wave_Pranjal = 486.1333 * u.nm
        Hb_wave_JD = 4862.683 * u.Angstrom
        wave_shift = Hb_wave_Pranjal - Hb_wave_JD
        v0_shift   = c_kms * wave_shift / Hb_wave_JD
        v0_Pranjal = data_info['fid_params']['Hb_params']['v_0']
        print('Corrected v_0 (km/s):', v0_Pranjal + v0_shift.decompose())
    
        true_values = [
            data_info['fid_params']['shared_params']['gamma_t'],   # g_t
            data_info['fid_params']['shared_params']['theta_int'], # theta_int
            data_info['fid_params']['shared_params']['vcirc'],     # vcirc
            data_info['fid_params']['shared_params']['cosi'],      # cosi
            data_info['fid_params']['shared_params']['r_hl_disk'], # r_hl_disk
            data_info['fid_params']['shared_params']['flux'],      # flux
            data_info['fid_params']['shared_params']['dx_disk'],   # dx_disk
            data_info['fid_params']['shared_params']['dy_disk'],   # dy_disk
            data_info['fid_params']['shared_params']['vscale'],    # vscale
            # data_info['fid_params']['shared_params']['flux_bulge'], # flux_bulge
            # data_info['fid_params']['shared_params']['r_hl_bulge'], # r_hl_bulge
            # data_info['fid_params']['shared_params']['dx_bulge'],   # dx_bulge
            # data_info['fid_params']['shared_params']['dy_bulge'],   # dy_bulge
            v0_Pranjal + v0_shift.decompose(), # convert units, result: -153.17706740192824
            data_info['fid_params']['Hb_params']['I01'], # I0
            data_info['fid_params']['Hb_params']['bkg_level'], # bkg_level
            data_info['fid_params']['Hb_params']['dx_vel'], # dx_vel
            1,
            1
        ]

        plot_corner(
            [post_path_new],  
            [f'#{slit_name}'],  
            contour_color='dimgray', # deepskyblue
            read_latex_from=fitting_params,
            true_values=true_values,
            corner_name=f'{slit_folder}corner_all.png',
            )
        print('Plotting done.')
    else:
        print('Corner   plotting skipped because this is a test run.')
        print('Test run finished.\n')
