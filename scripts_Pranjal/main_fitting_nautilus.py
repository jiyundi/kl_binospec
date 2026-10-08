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
# from klm.line_profile_extraction import extract_asymmetric_LPF, exam_line_profile
from klm.line_profile_extraction import extract_asymmetric_lpf_A, exam_line_profile_A
from klm.safe_plot import setup; setup() # must before plt

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def _plot_image(data, mask, increasing_RA=False, increasing_Dec=True, plotname='test.jpg'):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3)) # (width, height)
    im = ax.imshow(np.where(mask, data, np.nan), aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel('Increasing RA -->'  if increasing_RA  else 'Decreasing RA -->')
    ax.set_ylabel('Increasing Dec -->' if increasing_Dec else 'Decreasing Dec -->')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(plotname)
    return

def _plot_spec(data, mask, wave=None, plotname='test.jpg'):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3)) # (width, height)
    if wave is None:
        ax.set_xlabel('Wavelength pixels')
        im = ax.imshow(np.where(mask, data, np.nan), aspect='auto', cmap='viridis', origin='lower')
    else:
        ax.set_xlabel(f'Wavelength (N={wave.shape[1]} px)')
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad("white")
        im = ax.pcolormesh(
            wave.value if isinstance(wave, u.Quantity) else wave,       # shape (ny, nx)，单位 Å
            np.broadcast_to(np.arange(len(data))[:, None], wave.shape), # shape (ny, nx)
            np.ma.array(data, mask=(~np.isfinite(data)|~mask)),    # mask显示成白色
        )
    ax.set_ylabel('Slit spatial pixels')
    ax.grid(linestyle='--', color='silver', alpha=0.25)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(plotname)
    return

def circular_maskout(image_data, image_mask=None, 
                     n_circles=1, radius_range=(2, 16), avoid_center=5,
                     np_random_gen=None, plot=False):
    r_min, r_max = radius_range # px

    if image_mask is not None:
        circular_mask = ~image_mask.copy()
    else:
        circular_mask = np.zeros((ny, nx), dtype=bool)
    
    ny, nx = image_data.shape
    yy, xx = np.ogrid[:ny, :nx]
    if np_random_gen is None:
        np_random_gen = np.random.default_rng() # any random seed
    
    x0 = (nx - 1) / 2
    y0 = (ny - 1) / 2

    for _ in range(n_circles):
        r = np_random_gen.integers(r_min, r_max + 1)

        while True:
            cx = np_random_gen.integers(r, nx)
            cy = np_random_gen.integers(r, ny)

            distance_to_center = np.hypot(cx - x0, cy - y0)

            if distance_to_center >= r + avoid_center:
                break

        region = (xx - cx)**2 + (yy - cy)**2 <= r**2
        circular_mask |= region
    
    if plot: 
        RA_plus  = (True if data_info['image']['meta']['wcs'].cd[0, 0] > 0 else False)
        Dec_plus = (True if data_info['image']['meta']['wcs'].cd[1, 1] > 0 else False)
        _plot_image(image_data, ~circular_mask, RA_plus, Dec_plus)

    return ~circular_mask

def spec_add_noise( spec_noise, shape, 
                    sigma_spatial=5,   # in spatial pixels
                    sigma_wavepixel=2, # in wavelength pixels, keep small for less corr on wave
                    np_random_gen=None):
    if np_random_gen is None:
        np_random_gen = np.random.default_rng()

    white_noise = np_random_gen.normal(loc=0.0, scale=spec_noise, size=shape)
    corr_noise  = gaussian_filter(
        white_noise, mode="reflect", # Spatially-correlated noise
        sigma =(sigma_spatial, sigma_wavepixel) # keep small for less corr on wave
        )
    corr_noise -= corr_noise.mean() # unbias
    corr_noise *= spec_noise / corr_noise.std() # normalize

    return corr_noise



if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--slitID', default=  1, type=int)
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

    spec_data_no_noise = data_info['spec'][0]['data'].copy()
    
    # ------------- 1.0 Realistic obs conditions ---------------- #
    print('CAUTION: adding some realistics on raw data.')
    from scipy.ndimage import gaussian_filter

    np_random_gen_image = np.random.default_rng(seed = slit_name)
    np_random_gen_spec  = np.random.default_rng(seed = slit_name + 1)

    # Read image
    image_data = data_info['image']['data']
    image_mask = data_info['image']['mask']
    image_var  = data_info['image']['var' ]
    _plot_image(image_data, image_mask)

    # Add noise (image)
    image_noise = np.sqrt(image_var) # don't filter b/c all bkg px == 0
    image_data += np_random_gen_image.normal(
        loc=0.0, scale=image_noise, size=image_data.shape
        )
    _plot_image(image_data, image_mask)

    # Masking out 5 fake neighboring galaxies (image)
    for _ in range(5):
        image_mask[:] = circular_maskout( # [:] means overwrite dict array
            image_data, image_mask, np_random_gen=np_random_gen_image
            )
    
    # Mask out (flux + Gauss_noise + var) <= 0 (image)
    image_mask &= (image_data + image_var > 0)
    _plot_image(image_data, image_mask)

    # Read spec
    spec_data = data_info['spec'][0]['data']
    spec_mask = data_info['spec'][0]['mask']
    spec_var  = data_info['spec'][0]['var' ]
    spec_wave = data_info['spec'][0]['meta']['lambda_grid']
    _plot_spec(spec_data, spec_mask, spec_wave)

    # Add noise (spec):
    # Spec contains negative flux, the noise level needs to keep SNR same
    spec_noise  = np.sqrt(spec_var)
    spec_clean  = spec_data.copy()
    white_noise = np_random_gen_spec.normal(
        loc=0.0, scale=spec_noise, size=spec_clean.shape
        )
    spec_noisy = spec_clean + white_noise * 0.25
    
    # Check SNR
    from core.spec_snr_estimate import bkg_estimate
    bg = bkg_estimate(spec_noisy)[0] # = -2.9597
    spec_obs_zero_bg = np.where(spec_mask, spec_noisy - bg, np.nan)
    print('Spec SNR (after adding noise) =', 
        np.nansum(spec_obs_zero_bg) /
        np.sqrt(np.nansum(spec_var + spec_obs_zero_bg))
    )
    # os._exit(0)

    # If SNR is good, then write noisy in data
    data_info['spec'][0]['data'] = spec_noisy
    spec_data = data_info['spec'][0]['data']
    _plot_spec(spec_data, spec_mask, spec_wave)
    
    # Masking out fake line masks (spec)


    # Mask out (flux + Gauss_noise + var) <= 0 (spec)
    spec_mask &= (spec_data + spec_var > 0)
    _plot_spec(spec_data, spec_mask, spec_wave)

    # Delete my LPFs, overwrite with Pranjal's
    data_info['spec'][0]['meta'].pop('line_profile')

    # ------------- 1.1 Load LPF by Pranjal ---------------- #
    # with open(f'{mock_folder}/spec_extract_a2261b_008.pkl', "rb") as f:
    #     line_profile = joblib.load(f)

    # ------------- 1.2 Convert JD LPF into dict ---------------- #
    Amp_0, Mu_0, sigma1_0, sigma2_0 = extract_asymmetric_lpf_A(
        spec_data_no_noise,
        spec_wave,
        line='Hb'
        )
    
    Amp, Mu, sigma1, sigma2 = extract_asymmetric_lpf_A(
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
        filename=f"LPF_mock_{slit_name:03d}.jpg",
        plot_with=(Amp_0[:, 0], Mu_0[:, 0], sigma1_0[:, 0], sigma2_0[:, 0]), 
        label_of_plot_with='No noise'
        )

    Amp1     = Amp[:, 0]
    mu1      = Mu[ :, 0]
    sigma1_1 = sigma1[:, 0] # First Gaussian, left  wing
    sigma1_2 = sigma2[:, 0] # First Gaussian, right wing
    Amp2     = Amp[:, 1]
    mu2      = Mu[ :, 1]
    sigma2_1 = sigma1[:, 1] # Second Gaussian, left  wing
    sigma2_2 = sigma2[:, 1] # Second Gaussian, right wing
    
    # normalize
    Amp1 /= np.nanmax(Amp1)
    if np.any(Amp2 > 0): Amp2 /= np.nanmax(Amp2)

    # mask it to filter out some rows with no signal
    mask = ~np.isnan(Amp1) & ~np.isnan(Amp2)
    Amp1[~mask], Amp2[~mask] = 0, 0
    
    # save in dict
    LP_line1 = {
        'amp': Amp1,
        'mean': mu1, 
        'std_left':  sigma1_1,
        'std_right': sigma1_2,
        'reliability': mask,
        'bkg': np.zeros(Amp1.shape)
        }
    LP_line2 = {
        'amp': Amp2,
        'mean': mu2, 
        'std_left':  sigma2_1,
        'std_right': sigma2_2,
        'reliability': mask,
        'bkg': np.zeros(Amp2.shape)
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
