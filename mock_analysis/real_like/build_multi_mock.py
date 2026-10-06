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
    image_shape     = (22, 22) # nRA, nDEC
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

def prepare_a_slitspec(RA_obj, Dec_obj, Set, line, meta_gal):
    """
        Make a dic for slit data
    """
    # slit_len   =  8
    slit_width =  1
    slit_LPA   =  (90 - 25)*u.deg #+ 90*u.deg
    spec_pix_scale  = 0.24      # arcsec/pix by Binospec
    spec_shape      = [22, 20]  # (spatial pixels, N wavelength points)
    
    # Only select one from these 3 RA-Dec settings
    if   Set == 'C':
        # (1) Set C
        slit_RA    = RA_obj
        slit_Dec   = Dec_obj
    elif Set == 'A':
        # (2) Set A ONLY: 1 arcsec offset (from Set C slit)
        slit_RA    = RA_obj  - slit_width*u.arcsec * np.cos(90*u.deg-slit_LPA)
        slit_Dec   = Dec_obj + slit_width*u.arcsec * np.sin(90*u.deg-slit_LPA)
    elif Set == 'B':
        # (3) Set B ONLY: -1 arcsec offset
        slit_RA    = RA_obj  + slit_width*u.arcsec * np.cos(90*u.deg-slit_LPA)
        slit_Dec   = Dec_obj - slit_width*u.arcsec * np.sin(90*u.deg-slit_LPA)
    elif Set == 'P':
        # (4) Crossed slit: 90 degree to original slit PA
        slit_RA    = RA_obj
        slit_Dec   = Dec_obj
        slit_LPA  += 90*u.deg
    
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
    meta_spec = {
        'line_species':   line, 
        'ngrid':          spec_shape,
        'lambda_grid':    lambda_grid, # nm, or *u.Angstrom,
        'pixScale':       spec_pix_scale,  # arcsec/px
        'rhl':            0.57,
        'slitRA':    slit_RA,
        'slitDec':   slit_Dec,
        'slitWidth': slit_width,
        'slitLen':   spec_pix_scale * spec_shape[0],
        'slitLPA':   slit_LPA,
        'slitWPA':   slit_LPA + 90*u.deg  # Assume rectangular slit
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
                                       Set, line, meta_gal)
        
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
                                 meta_param=meta_spec)
        
        updated_dict = params.gen_param_dict(mock_params_keys, 
                                             mock_params.values())
        this_line_dict  = {**updated_dict['shared_params'], 
                           **updated_dict[f'{line}_params']} # merge dict
        spec_data       = spec_model.get_observable(this_line_dict)
        cont_model_spec = np.ones(spec_data.shape) * 0
        spec_var        = np.ones(spec_data.shape) * 100
        spec_var        = Mock._set_snr(spec_data, spec_var, # Set var to match SNR
                                        mock_params['shared_params-spec_snr'], 
                                        'spec', verbose=False)
        if if_add_noise==True:
            noise_level = np.sqrt(spec_var)
            spec_noise  = noise_level * np.random.randn(spec_data.shape[0],
                                                        spec_data.shape[1]) # if add noise
            spec_data   = spec_data + spec_noise # spec_noise
            print(f'Added a noise level: {np.min(spec_noise):.1f} - {np.max(spec_noise):.1f}')
        else:
            spec_data   = spec_data + 0 # no spec_noise
        
        spec_this_line_this_set = {'meta_spec':  meta_spec,  
                                   'spec_data':  spec_data,  
                                   'spec_var':   spec_var, 
                                   'cont_model_spec': cont_model_spec }
        
        spec_this_set[line] = spec_this_line_this_set
        print('Final spec  SNR:', round(Mock._calculate_snr(spec_data,  spec_var,  'spec')))
        print('Final spec estimated  bkg_level:', f'{np.median(spec_data):.1f}')
        print('Final spec estimated  bkg noise:', f'{np.std(   spec_data):.1f}')
        print('Final spec min/max:', f'{np.min(spec_data):.1f} / {np.max(spec_data):.1f}\n')
    
    # 2. Image
    image_model     = ImageModel(meta_image=meta_imag)
    image_data      = image_model.get_image(updated_dict['shared_params'])
    image_sky_var   = np.ones(image_data.shape) * 100
    image_var       = image_data + image_sky_var
    image_var       = Mock._set_snr(image_data, image_var, # Set var to match SNR
                                    mock_params['shared_params-image_snr'], 
                                    'image', verbose=False)
    if if_add_noise==True:
        noise_level_im = np.sqrt(image_var)/4
        image_noise = noise_level_im * np.random.randn(image_var.shape[0],
                                                       image_var.shape[1]) # if adding noise
        image_data  = image_data + image_noise - 25 # image_noise
    else:
        image_data  = image_data + 0 # no image_noise
    
    print('Final image SNR:', round(Mock._calculate_snr(image_data, image_var, 'image')))
    print()
    
    image_this_set = {'meta_image': meta_imag, 
                      'image_data': image_data, 
                      'image_var':  image_var }
    
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
             'var':        spec_this_line_this_set['spec_var'],
             'cont_model': spec_this_line_this_set['cont_model_spec'],
             'par_meta':   spec_this_line_this_set['meta_spec'] }
            )
    
    image_data_info = {
        'data':     all_image['image_data'],
        'var':      all_image['image_var'],
        'par_meta': all_image['meta_image']
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
    with open(f'{mock_folder}pkl/slit_{slit_name:03d}.pkl', "wb") as f:
        joblib.dump(mock_data_info, f)

    return mock_data_info

def make_mock_3slits_multi(emi_line_lst=['OII'], update_dict=None, iter_num=1, 
                           if_add_noise=True, mock_folder='./', slitname=8):
    # Now get object data from object catalog
    slit_name = '{:03d}'.format(slitname)
    RA_obj     = 42.1796   *u.deg
    Dec_obj    = -3.45204  *u.deg
    redshift   =  0.773
    meta_gal = {
        'redshift': redshift,
        'RA':       RA_obj,
        'Dec':      Dec_obj,
        'beta':     0*u.deg,
        'log10_Mstar':     9.986, 
        'log10_Mstar_err': 0.042,
        }
    
    with open("./binospec_mock_params.yaml", "r", encoding="utf-8") as file:
        mock_params = yaml.safe_load(file)
    
    if update_dict != None:
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
        
    mock_data_info = save_dic_and_pkl(slitname, all_specs, all_image,
                                      mock_params, meta_gal, iter_num, 
                                      mock_folder=mock_folder)
    make_exam_plots(mock_data_info, slit_name, iter_num, mock_folder)    
    return mock_data_info








if __name__ == '__main__':
    """
        Other external Python scripts may load functions above
        If so, "iter_num" or "run" variables defined wherelse may 
        be contaminated/destroyed!!!
        To protect them, only when directly execute this script,
        the variables below will then be executed.
    """
    mock_folder = 'binospec_data_pkl/'
    slitname = 0
    lines = ['O3b']
            
    make_mock_3slits_multi(
        emi_line_lst=lines, slitname=slitname, 
        mock_folder=mock_folder, if_add_noise=True,
        # update_dict=update_dict, 
        )
                    