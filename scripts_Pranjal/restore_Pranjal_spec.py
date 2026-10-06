import joblib
import yaml
import numpy as np
import astropy.units as u

from core.make_config_dic import make_config_dic
from klm.nautilus_sampler import NautilusSampler


pkl_folder  =  './'
mock_folder = '008b_vary_thetaint_slitLPA_major/'
keys = ['Amp1', 'mu1', 'sigma1_1', 'sigma1_2', 
        'Amp2', 'mu2', 'sigma2_1', 'sigma2_2']


# ------------- 1. Load LPF ---------------- #
with open(f'{mock_folder}/spec_extract_a2261b_008.pkl', "rb") as f:
    line_profile = joblib.load(f)

line_profile_Hb = line_profile['Hb']


# ------------- 3. Convert two-half  LPF ---------------- #
with open(f'{mock_folder}/slit_002_1.4.pkl', "rb") as f:
    data_info = joblib.load(f)
# lambda_grid = data_info['spec'][0]['meta']['lambda_grid']

# Delete my LPFs, overwrite with Pranjal's
data_info['spec'][0]['meta'].pop('line_profile')

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
    
# save in dict
LP_line1 = {
    'amp': Amp1,
    'mean': mu1, 
    'std_left':  sigma_1l,
    'std_right': sigma_1r,
    'bkg': np.zeros(Amp1.shape) # Pranjal assumed zero bkg
    }
LP_line2 = {
    'amp': Amp2,
    'mean': mu2, 
    'std_left':  sigma_2l,
    'std_right': sigma_2r,
    'bkg': np.zeros(Amp2.shape) # Pranjal assumed zero bkg
    }

data_info['spec'][0]['meta']['line_profile'] = (LP_line1, LP_line2)


# -- 5. Read mock params and generate velocity model ------ #
fid_data = data_info['spec'][0]['data']
fid_var  = data_info['spec'][0]['var']
best_one_level = {**data_info['fid_params']['shared_params'], 
                  **data_info['fid_params']['Hb_params']}
best_one_level['v_outer'] = 0 # klm legacy issue
for k in ('I01', 'I02'):
    spec_key = f"{k}_spec1"
    if spec_key in best_one_level:
        best_one_level[k] = best_one_level[spec_key]
        
# Calibrate v_0 due to different definitions of Hb
# c_kms = 299792.458
# Hb_wave_Pranjal = 486.1333 * u.nm
# Hb_wave_JD = 4862.683 * u.Angstrom
# wave_shift = Hb_wave_Pranjal - Hb_wave_JD
# v0_shift   = c_kms * wave_shift / Hb_wave_JD
# v0_Pranjal = data_info['fid_params']['Hb_params']['v_0']
# best_one_level['v_0'] = v0_Pranjal + v0_shift.decompose() # convert units

fiduci_yaml =  "./binospec_fid_params_Pranjal.yaml"
fittin_yaml =  "./binospec_fitting_params_Pranjal.yaml"
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

# Note: we did NOT use any fitting results. NautilusSampler is just for completing best_one_level.
nautilus_sampler = NautilusSampler(data_info, config_dict)

# spec_restored_cosi = nautilus_sampler.spec_model[0].get_observable(best_one_level)
RC_restored_cosi = nautilus_sampler.spec_model[0].get_observable(
    best_one_level, 
    return_type='rotation_curve', wave_only=True
    )

spec_data = data_info['spec'][0]['data']
spec_mask = data_info['spec'][0]['mask']
spec_wave = data_info['spec'][0]['meta']['lambda_grid']

Amp = np.array([Amp1, Amp2]).T / np.nanmax([Amp1, Amp2]) * np.nanmax(spec_data[spec_mask])
Mu  = np.array([ mu1,  mu2]).T
sigma1 = np.array([sigma_1l, sigma_2l]).T
sigma2 = np.array([sigma_1r, sigma_2r]).T

from klm.line_profile_extraction import exam_line_profile_A
restored, residual = exam_line_profile_A(
    (Amp, Mu, sigma1, sigma2),
    spec_data,
    spec_mask,
    wavelength=spec_wave,
    line='Hb', 
    plot_with=RC_restored_cosi, label_of_plot_with='from cosi'
    )

    

# ------------- 5. Compare ---------------- #
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13,4)) # (width, height)

# spec_true = data_info['spec'][0]['data']
# im0 = ax[0].imshow(spec_true, aspect='auto', cmap='viridis', origin='lower')
# ax[0].set_title("True (Pranjal's)")

# im1 = ax[1].imshow(spec_restored, aspect='auto', cmap='viridis', origin='lower')
# ax[1].set_title("Restored")

# # 使用 np.errstate 静默警告
# im2 = ax[2].imshow(spec_true - spec_restored, aspect='auto', cmap='coolwarm', origin='lower')
# ax[2].set_title("True - Restored")

# ax[0].set_ylabel('Spatial pixels')
# ax[1].set_ylabel('Spatial pixels')
# ax[2].set_ylabel('Spatial pixels')
# ax[0].set_xlabel('wavelength pixels')
# ax[1].set_xlabel('wavelength pixels')
# ax[2].set_xlabel('wavelength pixels')
# plt.colorbar(im0, ax=ax[0])
# plt.colorbar(im1, ax=ax[1])
# plt.colorbar(im2, ax=ax[2])
# plt.tight_layout()
# plt.savefig('test_spec_restored.jpg')
# print('Done.')
# print('Difference^2 =', np.sum((spec_true - spec_restored)**2))