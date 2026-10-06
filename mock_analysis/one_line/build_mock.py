import joblib
import yaml
import numpy as np
import astropy.units as u
import galsim
from astropy import wcs

from klm             import utils
from klm.parameters  import Parameters
from klm.spec_model  import SlitModel
from klm.image_model import ImageModel
from klm.mock        import Mock
from build_mock_plot import make_exam_plots

from klm.safe_plot   import setup; setup() # must before plt
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})
    
def prepare_an_image(RA_obj, Dec_obj):
    """
        Make a dic for image data
    """
    image_shape     = (30, 30) # nRA, nDEC
    image_pix_scale = 0.2      # arcsec/pix: 0.2 for HSC image
    psfFWHM         = 0.6      # arcsec
    ap_wcs           = wcs.WCS(naxis=2) # Create WCS
    ap_wcs.wcs.crpix = np.array([image_shape[0]/2+0.5,
                                 image_shape[0]/2+0.5]) # Cntrl ref pix (0.5-based)
    ap_wcs.wcs.crval = [RA_obj.value, Dec_obj.value] # RA/Dec (deg) central pixel
    ap_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN'] 
    ap_wcs.wcs.cdelt = [1, 1]
    ap_wcs.wcs.pc    = np.array([[-image_pix_scale/3600, 0], [0, image_pix_scale/3600]]) # deg/px
    galsim_wcs       = galsim.AstropyWCS(wcs=ap_wcs)
    meta_image = {
        'ngrid':    image_shape,
        'pixScale': image_pix_scale,
        'psfFWHM':  psfFWHM,
        'wcs':      galsim_wcs,
        'ap_wcs':   ap_wcs,
        'RA':       RA_obj.value,
        'Dec':      Dec_obj.value}
    
    return meta_image

def prepare_a_slitspec(RA_obj, Dec_obj, Set, emi_line_lst, meta_gal):
    """
        Make a dic for slit data
    """
    slit_len   =  8
    slit_width =  1
    slit_LPA   =  25*u.deg #+ 90*u.deg
    # slit_LPA   = 90*u.deg - slit_LPA  # degrees east of north. Want w.r.t. east.
    spec_pix_scale  = 0.24      # arcsec/pix by Binospec
    spec_shape      = [54, 38]  # (spatial pixels, N wavelength points)
    
    # Only select one from these 3 RA-Dec settings
    if   Set == 'C':
        # (1) Set C
        slit_RA    = RA_obj
        slit_Dec   = Dec_obj
    elif Set == 'A':
        # (2) Set A ONLY: 1 arcsec offset (from Set C slit)
        slit_RA    = RA_obj  - slit_width*u.arcsec * np.cos(slit_LPA)
        slit_Dec   = Dec_obj + slit_width*u.arcsec * np.sin(slit_LPA)
    elif Set == 'B':
        # (3) Set B ONLY: -1 arcsec offset
        slit_RA    = RA_obj  + slit_width*u.arcsec * np.cos(slit_LPA)
        slit_Dec   = Dec_obj - slit_width*u.arcsec * np.sin(slit_LPA)
    elif Set == 'P':
        # (4) Crossed slit: 90 degree to original slit PA
        slit_RA    = RA_obj
        slit_Dec   = Dec_obj
        slit_LPA  += 90*u.deg
    
    if   emi_line_lst[0] == 'O2': 
        lamb_0 = (1 + meta_gal['redshift']) * 372.7000 * 10
    elif emi_line_lst[0] == 'O3a': 
        lamb_0 = (1 + meta_gal['redshift']) * 436.3210 * 10
    elif emi_line_lst[0] == 'O3b': 
        lamb_0 = (1 + meta_gal['redshift']) * 495.8911 * 10
    elif emi_line_lst[0] == 'O3c': 
        lamb_0 = (1 + meta_gal['redshift']) * 500.6843 * 10
    elif emi_line_lst[0] == 'Ha': 
        lamb_0 = (1 + meta_gal['redshift']) * 656.2819 * 10
    elif emi_line_lst[0] == 'Hb':
        lamb_0 = (1 + meta_gal['redshift']) * 486.1333 * 10
    lamb_mean    = lamb_0       # A
    lamb_disp    =    0.61      # A/px
    LAMBDA_1D    = utils.build_1d_grid(spec_shape[1], lamb_disp/10) + lamb_mean/10 # nm
    lambda_grid  = np.repeat([LAMBDA_1D], spec_shape[0], axis=0)*u.nm
    # lamb_min     = lamb_mean - lamb_disp *  int(spec_shape[1]/2)
    # lamb_max     = lamb_min  + lamb_disp * (int(spec_shape[1]) - 1) # N-1 segments
    # lamb_one_row = np.linspace(lamb_min, lamb_max, int(spec_shape[1]))
    # lambda_grid  = np.tile(lamb_one_row, (int(spec_shape[0]), 1))
    meta_spec = {
        'line_species':   emi_line_lst[0], 
        'ngrid':          spec_shape,
        'lambda_grid':    lambda_grid, # nm, or *u.Angstrom,
        'pixScale':       spec_pix_scale,  # arcsec/px
        'rhl':            0.95,
        'slitRA':    slit_RA,
        'slitDec':   slit_Dec,
        'slitWidth': slit_width,
        'slitLen':   slit_len,
        'slitLPA':   slit_LPA,
        'slitWPA':   slit_LPA + 90*u.deg  # Assume rectangular slit
    }
    return meta_spec

def make_mock_data(emi_line_lst, meta_gal, mock_params, 
                   Set='C', if_add_noise=True):
    """
        Start generating data by using dic we created
    """
    meta_imag  = prepare_an_image(  meta_gal['RA'], meta_gal['Dec'])
    meta_spec  = prepare_a_slitspec(meta_gal['RA'], meta_gal['Dec'], 
                                    Set, emi_line_lst, meta_gal)
    
    params       = Parameters(line_species=emi_line_lst)
    updated_dict = params.gen_param_dict(mock_params.keys(), 
                                         mock_params.values())
    spec_snr  = mock_params['shared_params-spec_snr']
    image_snr = mock_params['shared_params-image_snr']
    
    # 1. Spec
    # Note: SlitModel requires a line_profile_path but now I don't have one.
    #       I assume a rhl = average of rhl_disk and rhl_bulge.
    # meta_spec['rhl'] = (mock_params['shared_params-r_hl_disk'])#+
                        # mock_params['shared_params-r_hl_bulge'])/2
    # spec_flux = mock_params[f'{emi_line_lst[0]}_params-I01']
    
    spec_model      = SlitModel(obj_param=meta_gal, 
                                meta_param=meta_spec)
    
    this_line_dict  = {**updated_dict['shared_params'], 
                       **updated_dict[f'{emi_line_lst[0]}_params']} # merge dict
    spec_data       = spec_model.get_observable(this_line_dict)
    cont_model_spec = np.ones(spec_data.shape) * 0
    spec_var        = np.ones(spec_data.shape) * 100
    spec_var        = Mock._set_snr(spec_data, spec_var, # Set var to match SNR
                                    spec_snr, 'spec', verbose=False)
    if if_add_noise==True:
        # noise_std   = np.max(spec_data) * (14/200) # MMT: 20 noise in 500 flux counts
        noise_std   = np.sqrt(spec_var)
        # noise_std   = np.sqrt(spec_var) * 2
        spec_noise  = noise_std * np.random.randn(spec_data.shape[0],
                                                  spec_data.shape[1]) # if add noise
        spec_data   = spec_data + spec_noise # spec_noise
        print(f'Added a noise level: {np.min(spec_noise):.1f} - {np.max(spec_noise):.1f}')
    else:
        spec_data   = spec_data + 0 # no spec_noise
    
    # 2. Image
    image_model     = ImageModel(meta_image=meta_imag)
    image_data      = image_model.get_image(updated_dict['shared_params'])
    image_sky_var   = np.ones(image_data.shape) * 100
    image_var       = image_data + image_sky_var
    image_var       = Mock._set_snr(image_data, image_var, # Set var to match SNR
                                    image_snr, 'image', verbose=False)
    if if_add_noise==True:
        image_noise = np.sqrt(image_var)*np.random.randn(image_var.shape[0],
                                                         image_var.shape[1]) # if adding noise
        image_data  = image_data + image_noise # image_noise
    else:
        image_data  = image_data + 0 # no image_noise
    
    print('Final image SNR:', round(Mock._calculate_snr(image_data, image_var, 'image')))
    print('Final spec  SNR:', round(Mock._calculate_snr(spec_data,  spec_var,  'spec')))
    print('Final spec estimated  bkg_level:', f'{np.median(spec_data):.1f}')
    print('Final spec estimated  bkg noise:', f'{np.std(   spec_data):.1f}')
    # mask_highs = (spec_data >= np.median(spec_data) + 2 * np.std(spec_data))
    # mask_lows  = (spec_data <= np.median(spec_data) - 2 * np.std(spec_data))
    # mask_valid = (~mask_highs) & (~mask_lows)
    # print('Final spec masked-out bkg noise:', f'{np.std(spec_data[mask_valid]):.1f}')
    print('Final spec min/max:', f'{np.min(spec_data):.1f} / {np.max(spec_data):.1f}\n')
    
    return {'meta_spec':  meta_spec,  'spec_data':  spec_data,  'spec_var':  spec_var,
            'meta_image': meta_imag, 'image_data': image_data, 'image_var': image_var, 
            'cont_model_spec': cont_model_spec }

def save_dic_and_pkl(slit_name, mock_data_all_sets, 
                     mock_params, meta_gal, iter_num, mock_folder='./'):
    """
        Make a dic and pkl to save all generated data
    """
    specs_data_info = []
    for data_each_set in mock_data_all_sets:
        spec_data_Set0 = {
            'data':       data_each_set['spec_data'],
            'var':        data_each_set['spec_var'],
            'cont_model': data_each_set['cont_model_spec'],
            'par_meta':   data_each_set['meta_spec']
            }
        specs_data_info.append(spec_data_Set0)
    
    image_data_info = {
        'data':     mock_data_all_sets[0]['image_data'],
        'var':      mock_data_all_sets[0]['image_var'],
        'par_meta': mock_data_all_sets[0]['meta_image']
        }
    
    mock_data_info = {
        'spec':    specs_data_info,
        'image':   image_data_info,
        'galaxy':  meta_gal
        }
    
    if mock_params is not None:
        mock_data_info['par_fit'] = mock_params
    
    # Unfortunately, my galsim.wcs objects cannot be packed in PKL files.
    # To pack galsim.wcs, DELETE it before packing in PKL.
    # To read, always regenerate by using ap_wcs. (by JD)
    mock_data_info['image']['par_meta'].pop('wcs') # delete it!
    
    # Save mocks
    with open(f'{mock_folder}pkl/mock_{slit_name}_{iter_num}.pkl', "wb") as f:
        joblib.dump(mock_data_info, f)

    return mock_data_info

def make_a_mock_3slits(update_dict=None, iter_num=999, if_add_noise=True, mock_folder='./'):
    # Now get object data from object catalog
    slit_name = '095'
    RA_obj     = 42.00292 *u.deg
    Dec_obj    = -3.404363*u.deg
    redshift   =  0.94
    emi_line_lst = ['O2']
    meta_gal = {}
    meta_gal = {
                'redshift': redshift,
                'RA':      RA_obj,
                'Dec':     Dec_obj,
                'beta':    0*u.deg,
                'log10_Mstar': 9.30, 
                # None, by log10(vcirc) = (log10_Mstar - 1.718) / 3.869
                'log10_Mstar_err': 0.05,
                }
    
    with open("./binospec_mock_params.yaml", "r", encoding="utf-8") as file:
        mock_params = yaml.safe_load(file)
    
    if update_dict != None:
        mock_params.update(update_dict)
    
    mock_data_SetA = make_mock_data(emi_line_lst, meta_gal, mock_params, 
                                    Set='A', if_add_noise=if_add_noise)
    mock_data_SetC = make_mock_data(emi_line_lst, meta_gal, mock_params, 
                                    Set='C', if_add_noise=if_add_noise)
    mock_data_SetB = make_mock_data(emi_line_lst, meta_gal, mock_params, 
                                    Set='B', if_add_noise=if_add_noise)
    mock_data_all_sets = [mock_data_SetA,
                          mock_data_SetC,
                          mock_data_SetB]
    mock_data_info = save_dic_and_pkl(slit_name, mock_data_all_sets, 
                                      mock_params, meta_gal, iter_num, 
                                      mock_folder=mock_folder)
    make_exam_plots(mock_data_info, slit_name, iter_num, mock_folder)    
    return mock_data_info

# def make_a_mock_crossed_slits(update_dict=None, iter_num=999, 
#                               if_add_noise=True, mock_folder='./'):
#     # Now get object data from object catalog
#     slit_name = '095'
#     RA_obj     = 42.00292 *u.deg
#     Dec_obj    = -3.404363*u.deg
#     redshift   =  0.94
#     emi_line_lst = ['Halpha']
#     meta_gal = {}
#     meta_gal = {
#                 'redshift': redshift,
#                 'RA':      RA_obj,
#                 'Dec':     Dec_obj,
#                 'beta':    0*u.deg,
#                 'log10_Mstar': np.log10(142) * 3.869 + 1.718, 
#                 # None, by log10(vcirc) = (log10_Mstar - 1.718) / 3.869
#                 'log10_Mstar_err': 0.0,
#                 }
    
#     with open("../config/binospec_mock_params.yaml", "r", encoding="utf-8") as file:
#         mock_params = yaml.safe_load(file)
    
#     if update_dict != None:
#         mock_params.update(update_dict)
    
#     mock_data_SetC = make_mock_data(emi_line_lst, meta_gal, mock_params, 
#                                     Set='C', if_add_noise=if_add_noise)
#     mock_data_SetP = make_mock_data(emi_line_lst, meta_gal, mock_params, 
#                                     Set='P', if_add_noise=if_add_noise)
    
#     mock_data_all_sets = [mock_data_SetC,
#                           mock_data_SetP]
#     mock_data_info = save_dic_and_pkl(slit_name, mock_data_all_sets, 
#                                       mock_params, meta_gal, iter_num, 
#                                       mock_folder=mock_folmake_a_mock_3slits(update_dict=None, iter_num=999, if_add_noise=True, mock_folder='./')der)
#     make_exam_plots(mock_data_info, slit_name, iter_num, mock_folder) 
    
#     return mock_data_info

# def make_a_mock_only_one_slit(update_dict=None, iter_num=999, 
#                               if_add_noise=True, mock_folder='./'):
#     # Now get object data from object catalog
#     slit_name = '095'
#     RA_obj     = 42.00292 *u.deg
#     Dec_obj    = -3.404363*u.deg
#     redshift   =  0.94
#     emi_line_lst = ['Halpha']
#     meta_gal = {}
#     meta_gal = {
#                 'redshift': redshift,
#                 'RA':      RA_obj,
#                 'Dec':     Dec_obj,
#                 'beta':    0*u.deg,
#                 'log10_Mstar': np.log10(142) * 3.869 + 1.718, 
#                 # None, by log10(vcirc) = (log10_Mstar - 1.718) / 3.869
#                 'log10_Mstar_err': 0.1,
#                 }
    
#     with open("../config/binospec_mock_params.yaml", "r", encoding="utf-8") as file:
#         mock_params = yaml.safe_load(file)
    
#     if update_dict != None:
#         mock_params.update(update_dict)
    
#     mock_data_SetC = make_mock_data(emi_line_lst, meta_gal, mock_params, 
#                                     Set='C', if_add_noise=if_add_noise)
#     mock_data_all_sets = [mock_data_SetC]
#     mock_data_info = save_dic_and_pkl(slit_name, mock_data_all_sets, 
#                                       mock_params, meta_gal, iter_num, 
#                                       mock_folder='mock/')
#     make_exam_plots(mock_data_info, slit_name, iter_num, mock_folder)    
#     return mock_data_info






if __name__ == '__main__':
    """
        Other external Python scripts may load functions above
        If so, "iter_num" or "run" variables defined wherelse may 
        be contaminated/destroyed!!!
        To protect them, only when directly execute this script,
        the variables below will then be executed.
    """
    # mock_folder = 'binospec_mock_theta_sini_noisy_SNR20/'
    # iternum = 1
    # sini_list  = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # cosi_list  = np.sqrt(1 - sini_list**2)
    # theta_list = np.linspace(0, np.pi, 6, endpoint=False)
    # for addnoise in [False]:
    #     for spec_snr in [20]:
    #         for theta in theta_list:
    #             for cosi in cosi_list:
    #                 print(f'iternum {iternum}: addnoise = {addnoise} spec_snr = {spec_snr} theta = {theta:.2f} sini = {np.sqrt(1-cosi**2):.2f}')
    #                 update_dict = {
    #                     'shared_params-spec_snr': spec_snr,
    #                     'shared_params-cosi': cosi,
    #                     'shared_params-theta_int': theta,
    #                 }
    #                 make_a_mock_3slits(update_dict=update_dict, iter_num=iternum, 
    #                                    if_add_noise=True, mock_folder=mock_folder)
    #                 iternum += 1
    #             iternum += 5
    
    mock_folder = 'binospec_mock/'
    iternum = 1
    make_a_mock_3slits(update_dict=None, iter_num=iternum, 
                       if_add_noise=True, mock_folder=mock_folder)
                    