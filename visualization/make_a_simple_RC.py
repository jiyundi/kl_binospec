import yaml
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from klm             import utils
from klm.spec_model  import SlitModel
from klm.parameters  import Parameters

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Inter",
    "font.serif": "Inter",
})


line = 'Ha'
redshift   =   0
slit_width =   1
slit_RA    =   0.00000*u.deg
slit_Dec   =   0.00000*u.deg
slit_LPA   =  90
lamb_disp       = 0.61 * u.Angstrom # A/px
spec_pix_scale  = 0.24      # arcsec/pix by Binospec
spec_shape      = [32, 30]  # (spatial pixels, N wavelength points)

with open("about_binospec_mock_analysis/multi_lines/binospec_mock_params.yaml", "r", encoding="utf-8") as file:
    mock_params = yaml.safe_load(file)
# Replace 'line' with 'O2_params', 'Ha_params', etc
mock_params_keys, mock_params_deletedkeys = [], []
for k in mock_params.keys(): 
    if k.split('_')[0] == 'line':
        mock_params_keys.append(line + '_' + '_'.join(k.split('_')[1:]))
    else:
        mock_params_keys.append(k)
    
    if (('_2' in k) or ('I02' in k)) and line != 'O2':
        mock_params_keys = mock_params_keys[:-1]
        mock_params_deletedkeys.append(k)
for del_key in mock_params_deletedkeys:
    mock_params.pop(del_key)

all_emilines_support = {'Ha':  np.array([6564.608])* u.Angstrom}
lamb_avg = np.mean(all_emilines_support[line])
lamb_0   = (1 + redshift) * lamb_avg
LAMBDA_1D    = utils.build_1d_grid(spec_shape[1], lamb_disp) + lamb_0
lambda_grid  = np.repeat([LAMBDA_1D], spec_shape[0], axis=0) * LAMBDA_1D.unit
meta_spec = {
    'line_species':   line, 
    'ngrid':          spec_shape,
    'lambda_grid':    lambda_grid, # nm, or *u.Angstrom,
    'pixScale':       spec_pix_scale,  # arcsec/px
    'rhl':            0.6,
    'slitRA':    slit_RA,
    'slitDec':   slit_Dec,
    'slitWidth': slit_width,
    'slitLen':   spec_pix_scale * spec_shape[0],
    'slitLPA':   (90 - slit_LPA) * u.deg,
    'slitWPA':   (90 - slit_LPA) * u.deg + 90*u.deg  # Assume rectangular slit
    }

meta_gal = {
    'RA':      slit_RA,
    'Dec':     slit_Dec,
    'beta':    0*u.deg,
    'redshift': redshift,
    'log10_Mstar':     10.00, 
    'log10_Mstar_err':  0.05,
    }

spec_model = SlitModel(obj_param=meta_gal, 
                       meta_param=meta_spec)
params     = Parameters(line_species=[line])
updated_dict = params.gen_param_dict(mock_params_keys, 
                                     mock_params.values())
this_line_dict  = {**updated_dict['shared_params'], 
                   **updated_dict[f'{line}_params']} # merge dict
# this_line_dict['sersic_spec'] = 0.1
this_line_dict.pop('sersic_spec')
spec_data       = spec_model.get_observable(this_line_dict)


fig = plt.figure(figsize=(6, 6))  # (length, height)
gs = fig.add_gridspec(nrows=1, ncols=1)
ax1 = fig.add_subplot(gs[0, 0])
noise     = np.std(spec_data)
imshows1  = ax1.imshow(spec_data, origin='lower', 
                       cmap='pink', aspect='equal', 
                       vmin=0, vmax=0 + 5*noise)
ax1.text(2,2, f'{slit_LPA}'+r'$^\circ$', 
         color='white', fontsize=8, ha='center', va='center')
def rot_rectangle(ax, x0, y0, dx, dy, rotation, color, ls):
    xUL = x0 + (-dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
    yUL = y0 + (-dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
    xUR = x0 + (+dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
    yUR = y0 + (+dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
    xLL = x0 + (-dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
    yLL = y0 + (-dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
    xLR = x0 + (+dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
    yLR = y0 + (+dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
    ax.plot([xUL, xUR, xLR, xLL, xUL], 
            [yUL, yUR, yLR, yLL, yUL], color=color, linestyle=ls)
    return 
rot_rectangle(ax1, 2, 2, 1, 4, (slit_LPA)/57.3, 'white', '-')
# ax1.set_title(f'Spectrum (Sersic index = {this_line_dict["sersic_spec"]})', fontsize=16)