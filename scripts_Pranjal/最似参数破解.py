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
from klm.safe_plot import setup; setup() # must before plt

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


import numpy as np
from scipy.optimize import curve_fit


def two_half_gaussians(x, amp, mu, sigma_left, sigma_right):
    """单个非对称 Gaussian。"""
    sigma = np.where(x < mu, sigma_left, sigma_right)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def asymmetric_doublet(x,
                       amp1, mu1, sigma1_left, sigma1_right,
                       amp2, dmu, sigma2_left, sigma2_right):
    """
    非对称双峰模型。

    使用 dmu > 0 保证第二个峰位于第一个峰右侧。
    """
    mu2 = mu1 + dmu

    peak1 = two_half_gaussians(
        x, amp1, mu1, sigma1_left, sigma1_right
    )
    peak2 = two_half_gaussians(
        x, amp2, mu2, sigma2_left, sigma2_right
    )

    return peak1 + peak2


def extract_asymmetric_LPF(data, lambda_scale,
                           min_separation=2.0,
                           max_separation=15.0,
                           min_sigma=0.2,
                           max_sigma=10.0):
    """
    对每个空间位置取相邻三行的平均谱，并拟合非对称双峰。

    Parameters
    ----------
    data : ndarray, shape (ny, nx)
        二维光谱数据，允许包含 NaN。

    lambda_scale : float
        像素到波长的转换系数。

    Returns
    -------
    Amp : ndarray, shape (ny, 2)
        两个峰的振幅。

    Mu : ndarray, shape (ny, 2)
        两个峰的中心，单位为像素。

    sigma1 : ndarray, shape (ny, 2)
        两个峰的左翼 sigma，单位为波长。

    sigma2 : ndarray, shape (ny, 2)
        两个峰的右翼 sigma，单位为波长。
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data 必须是二维数组")

    ny, nx = data.shape
    x_all = np.arange(nx, dtype=float)

    Amp = np.full((ny, 2), np.nan)
    Mu = np.full((ny, 2), np.nan)
    sigma1 = np.full((ny, 2), np.nan)
    sigma2 = np.full((ny, 2), np.nan)

    for y in range(ny):

        # 相邻三行；边界处重复边界行
        if y == 0:
            rows = data[[0, 0, 1]]
        elif y == ny - 1:
            rows = data[[ny - 2, ny - 1, ny - 1]]
        else:
            rows = data[y - 1:y + 2]

        with np.errstate(invalid="ignore"):
            profile = np.nanmean(rows, axis=0)

        valid = np.isfinite(profile)
        x = x_all[valid]
        flux = profile[valid]

        if x.size < 10:
            continue

        # 估计并减去背景
        background = np.nanmedian(flux)
        flux = flux - background

        if not np.any(np.isfinite(flux)) or np.nanmax(flux) <= 0:
            continue

        # 最高峰
        peak1_idx = np.nanargmax(flux)
        peak1_x = x[peak1_idx]

        # 屏蔽最高峰附近，再寻找第二峰
        flux_masked = flux.copy()
        suppress = np.abs(x - peak1_x) < min_separation
        flux_masked[suppress] = -np.inf

        if not np.any(np.isfinite(flux_masked)):
            continue

        peak2_idx = np.argmax(flux_masked)
        peak2_x = x[peak2_idx]

        # 按波长位置区分左右峰
        if peak1_x <= peak2_x:
            mu1_0 = peak1_x
            mu2_0 = peak2_x
            amp1_0 = max(flux[peak1_idx], 1e-3)
            amp2_0 = max(flux[peak2_idx], 1e-3)
        else:
            mu1_0 = peak2_x
            mu2_0 = peak1_x
            amp1_0 = max(flux[peak2_idx], 1e-3)
            amp2_0 = max(flux[peak1_idx], 1e-3)

        dmu_0 = np.clip(
            mu2_0 - mu1_0,
            min_separation,
            max_separation
        )

        p0 = [
            amp1_0,
            mu1_0,
            2.0,
            2.0,
            amp2_0,
            dmu_0,
            2.0,
            2.0
        ]

        lower_bounds = [
            0.0,
            1.0,
            min_sigma,
            min_sigma,
            0.0,
            min_separation,
            min_sigma,
            min_sigma
        ]

        upper_bounds = [
            np.inf,
            nx - 2.0,
            max_sigma,
            max_sigma,
            np.inf,
            max_separation,
            max_sigma,
            max_sigma
        ]

        # 防止初值落在边界之外
        p0 = np.clip(
            p0,
            np.asarray(lower_bounds) + 1e-8,
            np.asarray(upper_bounds) - 1e-8
        )

        try:
            popt, _ = curve_fit(
                asymmetric_doublet,
                x,
                flux,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue

        # popt 的正确参数顺序
        (
            amp1,
            mu1,
            sigma1_left,
            sigma1_right,
            amp2,
            dmu,
            sigma2_left,
            sigma2_right
        ) = popt

        mu2 = mu1 + dmu

        # 避免第二峰中心超出图像
        if mu2 >= nx:
            continue

        Amp[y] = [amp1, amp2]
        Mu[y] = [mu1, mu2]

        # sigma 从像素转换为波长单位
        sigma1[y] = np.array(
            [sigma1_left, sigma2_left]
        ) * lambda_scale

        sigma2[y] = np.array(
            [sigma1_right, sigma2_right]
        ) * lambda_scale

    return Amp, Mu, sigma1, sigma2

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter


def exam_line_profile(LPFs, spec_data, spec_mask, lambda_scale,
                      patch_size=3, filename="test_LPF.jpg"):
    """
    LPFs = (Amp, Mu, sigma_left, sigma_right)

    Amp、Mu、sigma_left、sigma_right: shape (ny, 2)
    Mu 为像素单位；sigma 为波长单位。
    """
    Amp, Mu, sigma_l, sigma_r = [
        np.asarray(v, dtype=float) for v in LPFs
    ]

    arr = np.where(spec_mask, spec_data, np.nan)
    ny, nx = arr.shape

    if Amp.shape != (ny, 2):
        raise ValueError("LPF参数的形状必须为 (ny, 2)")

    # sigma：波长单位转换回像素，用于重建谱线 [1]
    sigma_l_pix = sigma_l / lambda_scale
    sigma_r_pix = sigma_r / lambda_scale

    def half_gaussian(x, amp, mu, sig_l, sig_r):
        sigma = np.where(x < mu, sig_l, sig_r)
        sigma = np.maximum(sigma, 1e-8)
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    x = np.arange(nx, dtype=float)
    restored = np.full((ny, nx), np.nan)

    for y in range(ny):
        if not np.all(np.isfinite(
            [*Amp[y], *Mu[y], *sigma_l_pix[y], *sigma_r_pix[y]]
        )):
            continue

        restored[y] = (
            half_gaussian(
                x, Amp[y, 0], Mu[y, 0],
                sigma_l_pix[y, 0], sigma_r_pix[y, 0]
            )
            + half_gaussian(
                x, Amp[y, 1], Mu[y, 1],
                sigma_l_pix[y, 1], sigma_r_pix[y, 1]
            )
        )

    restored = np.where(spec_mask, restored, np.nan)

    # 忽略 NaN 的局部中值滤波
    smoothed = generic_filter(
        arr, np.nanmedian, size=patch_size, mode="nearest"
    )
    smoothed = np.where(spec_mask, smoothed, np.nan)
    residual = smoothed - restored

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

    panels = [
        (arr, "Data", "viridis"),
        (smoothed, f"Median filtered ({patch_size}×{patch_size})", "viridis"),
        (restored, "Asymmetric double-peak restored", "viridis"),
        (residual, "Filtered - restored", "coolwarm"),
    ]

    for ax, (image, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(
            image, origin="lower", aspect="auto", cmap=cmap
        )
        ax.set_title(title)
        ax.set_xlabel("Spectral pixel")
        ax.set_ylabel("Spatial row")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return restored, residual








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
        # data_info = load_mock_Pranjal(pkl_folder, slit_name)

        with open(f'{mock_folder}/slit_002_{special_idxs[slit_name]}.pkl', "rb") as f:
            data_info = joblib.load(f)
        print('Mock setting: slit_LPA  =', data_info['spec'][0]['meta']['slitLPA'])
        print('Mock setting: theta_int =', data_info['fid_params']['shared_params']['theta_int'])
        print('Mock setting: cosi      =', data_info['fid_params']['shared_params']['cosi'])
                                
    except FileNotFoundError:
        print( "\033[43m" + 'WARNING:' + "\033[0m " + 
            f'Slit {slit_name} skipped because no PKL found.\n')
        os._exit(0)
    
    
    # ------------- 1.2 Convert JD LPF into dict ---------------- #
    # line_profile_JD = data_info['spec'][0]['meta']['line_profile']
    # Amp1     = line_profile_JD[0][1:, 2].astype(float)
    # mu1      = line_profile_JD[0][1:, 0].astype(float)
    # sigma1_1 = line_profile_JD[0][1:, 1].astype(float) * u.Angstrom # First Gaussian, left  wing
    # sigma1_2 = line_profile_JD[0][1:, 1].astype(float) * u.Angstrom # First Gaussian, right wing
    # Amp2     = line_profile_JD[1][1:, 2].astype(float)
    # mu2      = line_profile_JD[1][1:, 0].astype(float)
    # sigma2_1 = line_profile_JD[1][1:, 1].astype(float) * u.Angstrom # Second Gaussian, left  wing
    # sigma2_2 = line_profile_JD[1][1:, 1].astype(float) * u.Angstrom # Second Gaussian, right wing
    
    spec_data = data_info['spec'][0]['data']
    spec_mask = data_info['spec'][0]['mask']

    Amp, Mu, sigma1, sigma2 = extract_asymmetric_LPF(
        spec_data,
        lambda_scale=0.33 # A/px
    )

    LPFs = (Amp, Mu, sigma1, sigma2)

    restored, residual = exam_line_profile(
        (Amp, Mu, sigma1, sigma2),
        spec_data,
        spec_mask,
        lambda_scale = 0.33 # A/px
    )

    Amp1     = Amp[:, 0]
    mu1      = Mu[ :, 0]
    sigma1_1 = sigma1[:, 0] * u.Angstrom # First Gaussian, left  wing
    sigma1_2 = sigma1[:, 1] * u.Angstrom # First Gaussian, right wing
    Amp2     = Amp[:, 1]
    mu2      = Mu[ :, 1]
    sigma2_1 = sigma2[:, 0] * u.Angstrom # Second Gaussian, left  wing
    sigma2_2 = sigma2[:, 1] * u.Angstrom # Second Gaussian, right wing
    
    Amp1 /= np.nanmax(Amp1)
    if np.any(Amp2 > 0): Amp2 /= np.nanmax(Amp2)

    # Check if sigma is badly-defined value. Make a mask.
    # mask = np.ones(len(sigma1_1), dtype=bool)
    # checklist = (sigma1_1, sigma1_2, sigma2_1, sigma2_2)
    # if np.all(sigma2_1.value == 0): 
    #     checklist = (sigma1_1, sigma1_2)
    #     sigma2_1[:], sigma2_2[:] = 999 * u.Angstrom, 999 * u.Angstrom
    # for sigma in checklist:
    #     for i in range(len(sigma)):
    #         if sigma[i].value <= 0.35:
    #             sigma[i] = 999 * u.Angstrom
    #             mask[i]  = False
    
    # save in dict
    LP_line1 = {
        'amp': Amp1,
        'mean': mu1, 
        'std_left':  sigma1_1,
        'std_right': sigma1_2,
        'reliability': ~np.isnan(Amp1),
        'bkg': np.zeros(Amp1.shape) # Pranjal assumed zero bkg
        }
    LP_line2 = {
        'amp': Amp2,
        'mean': mu2, 
        'std_left':  sigma2_1,
        'std_right': sigma2_2,
        'reliability': ~np.isnan(Amp2),
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
    
    # ------------- Prepare fitting parameters --------------------------- #
    print('Plotting best fit comparison with observed spec/image...')
    fitting_params_flat = Parameters._flatten(fitting_params, level=1)
    fitting_par = complete_flattened_fit_params(
        fitting_params_flat, 
        line_species=nautilus_sampler.config.galaxy_params.line_species
        )
    
    # ------------- 3. Bi-modality of two sets of (g1, theta_int) --------------------------- #
    # Calibrate v_0 due to different definitions of Hb
    c_kms = 299792.458
    Hb_wave_Pranjal = 486.1333 * u.nm
    Hb_wave_JD = 4862.683 * u.Angstrom
    wave_shift = Hb_wave_Pranjal - Hb_wave_JD
    v0_shift   = c_kms * wave_shift / Hb_wave_JD
    v0_Pranjal = data_info['fid_params']['Hb_params']['v_0']
    fid_pars = {
        'shared_params-g1':        data_info['fid_params']['shared_params']['gamma_t'], # -0.04124161290994201,
        'shared_params-theta_int': data_info['fid_params']['shared_params']['theta_int'], # result: 1.3962634015954636
        'shared_params-vcirc':     data_info['fid_params']['shared_params']['vcirc'], # result: 
        'shared_params-cosi':      data_info['fid_params']['shared_params']['cosi'], # result:   0.733767712971195,
        'shared_params-r_hl_disk': data_info['fid_params']['shared_params']['r_hl_disk'], # result: 
        'shared_params-vscale':    data_info['fid_params']['shared_params']['vscale'], # result:    0.10314708567188158,
        'Hb_params-v_0':           v0_Pranjal + v0_shift.decompose(), # convert units, result: -153.17706740192824,
        'Hb_params-I01_spec1':     data_info['fid_params']['Hb_params']['I01'], # result:   20.07195794200264,
        'Hb_params-bkg_level_spec1': data_info['fid_params']['Hb_params']['bkg_level'], # result: -2.866095297084968
        'Hb_params-f1_1_spec1': 1, # result: 1
        'Hb_params-f1_2_spec1': 1, # result: 1
        }
    fid_dict = nautilus_sampler.params.gen_param_dict(fitting_par.keys(), fid_pars.values())
    
    # IMAGE PART ----------------------------------------------------------------------------------
    # fid_data = data_info['image']['data']
    # fid_var  = data_info['image']['var']

    # m_true   = nautilus_sampler.image_model.get_image(fid_dict['shared_params'])

    # try_dict1 = fid_dict.copy()
    # try_dict1['shared_params'] = fid_dict['shared_params'].copy() # Since .copy() only works for one level
    # try_dict1['shared_params']['g1'] = -0.04
    # try_dict1['shared_params']['theta_int'] = 1.4
    # m_peak1 = nautilus_sampler.image_model.get_image(try_dict1['shared_params'])

    # try_dict2 = fid_dict.copy()
    # try_dict2['shared_params'] = fid_dict['shared_params'].copy() # Since .copy() only works for one level
    # try_dict2['shared_params']['g1'] = -0.30
    # try_dict2['shared_params']['theta_int'] = 0.0
    # m_peak2 = nautilus_sampler.image_model.get_image(try_dict2['shared_params'])

    # SPEC PART -----------------------------------------------------------------------------------------------
    fid_data = data_info['spec'][0]['data']
    fid_var  = data_info['spec'][0]['var']
    best_one_level = {**fid_dict['shared_params'], **fid_dict['Hb_params']}
    for k in ('I01', 'I02', 'bkg_level', 'f1_1', 'f1_2', 'f2_1', 'f2_2'):
        spec_key = f"{k}_spec1"
        if spec_key in best_one_level:
            best_one_level[k] = best_one_level[spec_key]
    m_true = nautilus_sampler.spec_model[0].get_observable(best_one_level)

    # try_dict1 = fid_dict.copy()
    # try_dict1['shared_params'] = fid_dict['shared_params'].copy() # Since .copy() only works for one level
    # try_dict1['Hb_params'    ] = fid_dict['Hb_params'    ].copy() # Since .copy() only works for one level
    # try_dict1['shared_params']['g1'] = -0.04
    # try_dict1['shared_params']['theta_int'] = 1.4
    # try_dict1_one_level = {**try_dict1['shared_params'], **try_dict1['Hb_params']}
    # for k in ('I01', 'I02'):
    #     spec_key = f"{k}_spec1"
    #     if spec_key in try_dict1_one_level:
    #         try_dict1_one_level[k] = try_dict1_one_level[spec_key]
    # m_peak1 = nautilus_sampler.spec_model[0].get_observable(try_dict1_one_level)

    # try_dict2 = fid_dict.copy()
    # try_dict2['shared_params'] = fid_dict['shared_params'].copy() # Since .copy() only works for one level
    # try_dict2['Hb_params'    ] = fid_dict['Hb_params'    ].copy() # Since .copy() only works for one level
    # try_dict2['shared_params']['g1'] = -0.30
    # try_dict2['shared_params']['theta_int'] = 0.0
    # try_dict2_one_level = {**try_dict2['shared_params'], **try_dict2['Hb_params']}
    # for k in ('I01', 'I02'):
    #     spec_key = f"{k}_spec1"
    #     if spec_key in try_dict2_one_level:
    #         try_dict2_one_level[k] = try_dict2_one_level[spec_key]
    # m_peak2 = nautilus_sampler.spec_model[0].get_observable(try_dict2_one_level)

    # MODEL COMPARE --------------------------
    d_true = np.sum((fid_data - m_true)**2 / fid_var)
    # d_1t   = np.sum((m_peak1 - m_true )**2 / fid_var)
    # d_2t   = np.sum((m_peak2 - m_true )**2 / fid_var)
    # d_12   = np.sum((m_peak1 - m_peak2)**2 / fid_var)
    
    print("chi2 at truth =", d_true)
    # print("chi2 between peak1 and truth =", d_1t)
    # print("chi2 between peak2 and truth =", d_2t)
    # print("chi2 separation between models =", d_12)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5)) # (width, height)
    im = ax.imshow(fid_data, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('test_fid_data.jpg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5)) # (width, height)
    im = ax.imshow(m_true, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('test_m_true.jpg')

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5)) # (width, height)
    # im = ax.imshow(m_peak1, aspect='auto', cmap='viridis', origin='lower')
    # plt.colorbar(im, ax=ax)
    # plt.tight_layout()
    # plt.savefig('test_m_peak1.jpg')

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5)) # (width, height)
    # im = ax.imshow(m_peak2, aspect='auto', cmap='viridis', origin='lower')
    # plt.colorbar(im, ax=ax)
    # plt.tight_layout()
    # plt.savefig('test_m_peak2.jpg')

print('Done.')

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,2)) # (width, height)
# ax.plot(slitPAs, spec_chi2s, label=f"theta_int = {data_info['fid_params']['shared_params']['theta_int']:.2f}")
# plt.minorticks_on()
# plt.legend()
# plt.grid()
# plt.show()