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
from core.find_snr   import snr_estimate
from core.var_snr_match import find_background_variance
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

def prepare_a_slitspec(RA_obj, Dec_obj, Set, line, meta_gal, idx_line=0):
    """
        Make a dic for slit data
    """
    # slit_len   =  8
    slit_width =  1
    slit_LPA   =  25 #+ 90*u.deg
    spec_pix_scale  = 0.24     # arcsec/pix by Binospec
    spec_shape      = [20, 40] # (spatial pixels, N wavelength points)
    
    # Only select one from these 3 RA-Dec settings
    if   Set == 'C':
        # (1) Set C
        slit_RA    = RA_obj
        slit_Dec   = Dec_obj
        idx_lpf    = idx_line * 3 + 1
    elif Set == 'A':
        # (2) Set A ONLY: 1 arcsec offset (from Set C slit)
        slit_RA    = RA_obj  - slit_width*u.arcsec * np.cos(slit_LPA*u.deg)
        slit_Dec   = Dec_obj + slit_width*u.arcsec * np.sin(slit_LPA*u.deg)
        idx_lpf    = idx_line * 3 + 0
    elif Set == 'B':
        # (3) Set B ONLY: -1 arcsec offset
        slit_RA    = RA_obj  + slit_width*u.arcsec * np.cos(slit_LPA*u.deg)
        slit_Dec   = Dec_obj - slit_width*u.arcsec * np.sin(slit_LPA*u.deg)
        idx_lpf    = idx_line * 3 + 2
    elif Set == 'P':
        # (4) Crossed slit: 90 degree to original slit PA
        slit_RA    = RA_obj
        slit_Dec   = Dec_obj
        slit_LPA  += 90
    
    all_emilines_support = {'O2':  np.array([3727.092, 3729.875])* u.Angstrom,
                            'Ha':  np.array([6564.608])* u.Angstrom,
                            'Hb':  np.array([4862.683])* u.Angstrom, 
                            'Hg':  np.array([4341.684])* u.Angstrom, 
                            'O3a': np.array([4960.295])* u.Angstrom, 
                            'O3b': np.array([5008.240])* u.Angstrom, 
                            'N2a': np.array([6549.86 ])* u.Angstrom, 
                            'N2b': np.array([6585.27 ])* u.Angstrom,
                            }
    if line == 'O2':
        lamb_avg = np.mean(all_emilines_support[line])
        lamb_0   = (1 + meta_gal['redshift']) * lamb_avg
    else:
        lamb_0   = (1 + meta_gal['redshift']) * all_emilines_support[line][0]
    lamb_disp    = 0.61 * u.Angstrom # A/px
    LAMBDA_1D    = utils.build_1d_grid(spec_shape[1], lamb_disp) + lamb_0
    lambda_grid  = np.repeat([LAMBDA_1D], spec_shape[0], axis=0) * LAMBDA_1D.unit

    # Optional: 
    lpf_filenames = [
        '../../scripts/binospec_pkl/LPF_slit_109_3.pkl', 
        '../../scripts/binospec_pkl/LPF_slit_109_4.pkl', 
        '../../scripts/binospec_pkl/LPF_slit_109_5.pkl', 
        '../../scripts/binospec_pkl/LPF_slit_128_3.pkl', 
        '../../scripts/binospec_pkl/LPF_slit_128_4.pkl', 
        '../../scripts/binospec_pkl/LPF_slit_128_5.pkl', 
    ]
    with open(lpf_filenames[idx_lpf], "rb") as f:
        line_profile_1, line_profile_2 = joblib.load(f)

    meta_spec = {
        'line_species':   line, 
        'line_profile':   (line_profile_1, line_profile_2),
        'ngrid':          spec_shape,
        'lambda_grid':    lambda_grid, # nm, or *u.Angstrom,
        'pixScale':       spec_pix_scale,  # arcsec/px
        'rhl':            None, # must be None if LPF provided
        'slitRA':    slit_RA,
        'slitDec':   slit_Dec,
        'slitWidth': slit_width,
        'slitLen':   spec_pix_scale * spec_shape[0],
        'slitLPA':   (90 - slit_LPA) * u.deg,
        'slitWPA':   (90 - slit_LPA) * u.deg + 90*u.deg  # Assume rectangular slit
    }
    return meta_spec

def make_mock_data(emi_line_lst, meta_gal, mockparams, 
                   Set='C', if_add_noise=True):
    """
        Start generating data by using dic we created
    """
    mock_params = mockparams.copy()
    
    meta_imag  = prepare_an_image(  meta_gal['RA'], meta_gal['Dec'])
    params     = Parameters(line_species=emi_line_lst)
    
    # 1. Spec
    spec_this_set = {}
    
    for i, line in enumerate(emi_line_lst):
        meta_spec = prepare_a_slitspec(meta_gal['RA'], 
                                       meta_gal['Dec'], 
                                       Set, line, meta_gal, idx_line=i)
        
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
        
        spec_model   = SlitModel(obj_param=meta_gal, 
                                 meta_param=meta_spec, 
                                 use_what_line_proflie='meta')
        
        updated_dict = params.gen_param_dict(mock_params_keys, 
                                             mock_params.values())
        this_line_dict  = {**updated_dict['shared_params'], 
                           **updated_dict[f'{line}_params']} # merge dict
        spec_data       = spec_model.get_observable(this_line_dict)
        # cont_vertical   = np.exp(
        #     -np.abs(np.arange(spec_data.shape[0]) - spec_data.shape[0]/2)**2 / 10
        #     ) * np.std(spec_data)
        # cont_model_spec = np.tile(np.array([cont_vertical]).T, 
        #                           (1, spec_data.shape[1])
        #                           # (spec_data.shape[1], 1)
        #                           )
        # cont_model_spec = np.zeros(spec_data.shape)
        spec_var = np.ones(spec_data.shape) * find_background_variance(
            spec_data,
            bkg_is_subtracted=False,
            target_snr=mock_params['shared_params-spec_snr']
        )
        if if_add_noise==True:
            # noise_level = np.sqrt(spec_var)
            spec_noise  = np.sqrt(
                np.clip(spec_var, a_min=0.0, a_max=None)
            )

            white_noise = np.random.normal(
                    loc=0.0,
                    scale=spec_noise,
                    size=spec_data.shape,
                )
            
            spec_data   = spec_data + white_noise # spec_noise
        else:
            spec_data   = spec_data + 0 # no spec_noise

        # Create the spectral mask.
        spec_mask = np.ones(spec_data.shape, dtype=bool)
        spec_mask &= spec_data + spec_var > 0

        spec_this_line_this_set = {'meta_spec':  meta_spec,  
                                   'spec_data':  spec_data,  
                                   'spec_var':   spec_var, 
                                   'spec_mask':  spec_mask
                                   }
        
        spec_this_set[line] = spec_this_line_this_set

        spec_snr = snr_estimate(
            np.where(spec_mask, spec_data, np.nan),
            bkg_is_subtracted=False, 
            data_is_noisy=True, noise_var='extract'
        )
        print('Final spec  SNR:', round(spec_snr, 1))
        # print('Final spec estimated  bkg_level:', f'{np.median(spec_data):.1f}')
        # print('Final spec estimated  bkg noise:', f'{np.std(   spec_data):.1f}')
        # print('Final spec min/max:', f'{np.min(spec_data):.1f} / {np.max(spec_data):.1f}\n')
    
    # 2. Image
    image_model     = ImageModel(meta_image=meta_imag)
    image_data      = image_model.get_image(updated_dict['shared_params'])
    image_var       = np.ones(image_data.shape) * find_background_variance(
        image_data,
        bkg_is_subtracted=False,
        target_snr=mock_params['shared_params-image_snr']
    )
    if if_add_noise==True:
        image_noise = np.sqrt(
            np.clip(image_var, a_min=0.0, a_max=None)
        )
    
        white_noise = np.random.normal(
            loc=0.0,
            scale=image_noise,
            size=image_data.shape,
        )

        image_data  = image_data + white_noise # image_noise
    else:
        image_data  = image_data + 0 # no image_noise

    # Create the spectral mask.
    image_mask = np.ones(image_data.shape, dtype=bool)
    image_mask &= image_data + image_var > 0

    image_snr = snr_estimate(
        np.where(image_mask, image_data, np.nan), 
        bkg_is_subtracted=False, data_is_noisy=True, noise_var='extract'
    )
    print('Final image SNR:', round(image_snr, 1))
    
    image_this_set = {'meta_image': meta_imag, 
                      'image_data': image_data, 
                      'image_mask': image_mask,
                      'image_var':  image_var 
                      }
    
    return spec_this_set, image_this_set

def save_dic_and_pkl(slit_name, all_specs, all_image, 
                     mock_params, meta_gal, iter_num, mock_folder='./'):
    """
        Make a dic and pkl to save all generated data
    """
    specs_data_info = []
    for spec_this_line_this_set in all_specs:
        specs_data_info.append(
            {'data':       spec_this_line_this_set['spec_data'],
             'mask':       spec_this_line_this_set['spec_mask'],
             'var':        spec_this_line_this_set['spec_var'],
             'meta':   spec_this_line_this_set['meta_spec'] }
            )
    
    image_data_info = {
        'data': all_image['image_data'],
        'mask': all_image['image_mask'],
        'var':  all_image['image_var'],
        'meta': all_image['meta_image']
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
    mock_data_info['image']['meta'].pop('wcs') # delete it!
    
    # Save mocks
    with open(f'{mock_folder}pkl/mock_{slit_name}_{iter_num:03d}.pkl', "wb") as f:
        joblib.dump(mock_data_info, f)

    return mock_data_info

def make_mock_3slits_multi(emi_line_lst, update_dict=None, iter_num=999, 
                           if_add_noise=True, mock_folder='./', slitname=95):
    # Now get object data from object catalog
    slit_name = '{:03d}'.format(slitname)
    RA_obj     = 10/3600 *u.deg
    Dec_obj    = 10/3600 *u.deg
    redshift   = 0.5
    meta_gal = {}
    meta_gal = {
                'redshift': redshift,
                'RA':      RA_obj,
                'Dec':     Dec_obj,
                'beta':    0*u.deg,
                'log10_Mstar': 10.00, 
                # None, by log10(vcirc) = (log10_Mstar - 1.718) / 3.869
                'log10_Mstar_err': 0.05,
                }
    
    with open("./binospec_mock_params.yaml", "r", encoding="utf-8") as file:
        mock_params = yaml.safe_load(file)
    
    if update_dict != None or len(update_dict) != 0:
        mock_params.update(update_dict)
    
    spec_this_setA, image_A = make_mock_data(
        emi_line_lst, meta_gal, mock_params, 
        Set='A', if_add_noise=if_add_noise
        )
    spec_this_setC, image_C = make_mock_data(
        emi_line_lst, meta_gal, mock_params, 
        Set='C', if_add_noise=if_add_noise
        )
    spec_this_setB, image_B = make_mock_data(
        emi_line_lst, meta_gal, mock_params, 
        Set='B', if_add_noise=if_add_noise
        )
    
    all_image = image_C
    all_specs = []
    for line in emi_line_lst:
        all_specs.append(spec_this_setA[line])
        all_specs.append(spec_this_setC[line])
        all_specs.append(spec_this_setB[line])
        
    mock_data_info = save_dic_and_pkl(slit_name, all_specs, all_image,
                                      mock_params, meta_gal, iter_num, 
                                      mock_folder=mock_folder)
    make_exam_plots(mock_data_info, slit_name, iter_num, mock_folder)    

    print('------------------------------------------------\n')
    
    return mock_data_info








if __name__ == '__main__':
    """
        Other external Python scripts may load functions above
        If so, "iter_num" or "run" variables defined wherelse may 
        be contaminated/destroyed!!!
        To protect them, only when directly execute this script,
        the variables below will then be executed.
    """
    mock_folder = 'mock_pkl/'
    slitname = 0
    lines = ['Hb','Ha']
    
    thetalist = np.array([0, 30, 60, 90, 120, 150])/(180/np.pi)
    cosi_list = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    iternums  = np.tile(range(len(cosi_list)), (len(thetalist), 1)) + 1
    for i in range(len(iternums)): 
        iternums[i] = iternums[i] + 10 * i
    
    update_dict = {}
    for i in range(len(thetalist)):
        for j in range(len(cosi_list)):
            iternum = iternums[i, j]

            # Update params. SNR settings are also in binospec_mock_params.yaml
            update_dict['shared_params-theta_int'] = thetalist[i]
            update_dict['shared_params-cosi']      = cosi_list[j]
            
            make_mock_3slits_multi(
                emi_line_lst=lines, slitname=slitname, iter_num=iternum, 
                mock_folder=mock_folder, if_add_noise=True,
                update_dict=update_dict, 
                )
                    