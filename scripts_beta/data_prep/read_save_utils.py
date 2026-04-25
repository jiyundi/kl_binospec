import joblib
import numpy as np
from   astropy.io  import fits
from   astropy.wcs import WCS
import astropy.units as u


def read_hsc_img_wcs(hscimgfilepath):
    hdulist = fits.open(hscimgfilepath)
    arr = hdulist[0].data
    wcs_loaded = WCS(hscimgfilepath)
    return arr, wcs_loaded


def read_spec2d(spec2dfilepath, err_spec_path, exten_num):
    hdul = fits.open(spec2dfilepath)
    
    fdata = hdul[1].data['FLUX']
    wdata = hdul[1].data['LAMBDA']
    
    farr = fdata[0,:,:]
    warr = wdata[0,:,:]
    
    farr = np.nan_to_num(farr, nan=-1000) # replace nan with -1000
    warr = np.nan_to_num(warr, nan=0)
    
    # This means we DELETE SpecPro-reduced iarr and 
    # and USE raw-reduced error spectrum instead
    hdul = fits.open(err_spec_path)
    err  = hdul[exten_num].data
    var  = err**2
    
    return {'flux': farr, 'wave': warr, 'var': var}


def readinfodat(infdatfilepath):
    infile_dat = open(infdatfilepath)
    dat_dict   = {}
    for sen in infile_dat:
        if sen[:2] != 'ID':
            dat_dict[sen.split('       ')[0]] = float(sen.split('       ')[1][:-1])
    return dat_dict


def cont(spec, margin = 5): # px
    """
    Construct proper spec continuum
    """
    lvl_lf = np.mean(spec[:, 0:margin], axis=1)
    lvl_rt = np.mean(spec[:, -margin:], axis=1)
    conlvl = np.mean([lvl_lf, lvl_rt],  axis=0)
    return np.tile(np.array([conlvl]).T, (1, spec.shape[1]))


def save_dic_and_pkl(slit_num, real_data_all_lines_sets, meta_gal, Ms, rr, 
                     pkl_folder):
    """
        Make a dic and pkl to save all packed data
    """
    specs_data_info = []
    for linename, specABC_this_line in real_data_all_lines_sets.items():
        for data_each_set in specABC_this_line:
            spec_data_Set0 = {
                'data':       data_each_set['spec_data'],
                'var':        data_each_set['spec_var'],
                'cont_model': data_each_set['cont_model_spec'],
                'par_meta':   data_each_set['meta_spec'], 
                'mask':       data_each_set['spec_mask'],
                'gauss_back': data_each_set['spec_gauss_back'],
                }
            specs_data_info.append(spec_data_Set0)
    
    image_data_info = {
        'data':     real_data_all_lines_sets[linename][0]['image_data'],
        'var':      real_data_all_lines_sets[linename][0]['image_var'],
        'varr':     real_data_all_lines_sets[linename][0]['image_varr'],
        'mask':     real_data_all_lines_sets[linename][0]['image_mask'],
        'par_meta': real_data_all_lines_sets[linename][0]['meta_image']
        }
    
    real_data_info = {
        'spec':    specs_data_info,
        'image':   image_data_info,
        'galaxy':  meta_gal
        }
    
    data_info = {}
    
    for key, dic in real_data_info.items():
        if key == 'galaxy':
            data_info['galaxy'] = dic.copy()
            
            for row in Ms:
                if row[0] == slit_num:
                    data_info['galaxy']['log10_Mstar'] = row[1]
                    data_info['galaxy']['log10_Mstar_err'] = row[2]
                    print(f'\nUpdated: log10_Mstar = {data_info["galaxy"]["log10_Mstar"]}, err = {data_info["galaxy"]["log10_Mstar_err"]}')
                    break
        
        elif key == 'spec':
            data_info['spec'] = dic.copy()
            
            for row in rr:
                if row[0] == slit_num:
                    r_hl_disk = row[1]
                    for i in range(len(real_data_info['spec'])):
                        data_info['spec'][i]['par_meta']['rhl'] = r_hl_disk
                    print(f"\nUpdated: r_hl_disk = {data_info['spec'][i]['par_meta']['rhl']}")
                    break
        
        else:
            data_info[key] = dic
    
    # Unfortunately, my galsim.wcs objects cannot be packed in PKL files.
    # To pack galsim.wcs, DELETE it before packing in PKL.
    # To read, always regenerate by using ap_wcs. (by JD)
    data_info['image']['par_meta'].pop('wcs') # delete it!
    
    # Save mocks
    slit_name = f'{slit_num:03d}'
    with open(f'{pkl_folder}pkl/slit_{slit_name}.pkl', "wb") as f:
        joblib.dump(data_info, f)

    return data_info


def real_data_pack(spec2dA, spec2dC, spec2dB,
                   meta_image, image_data, image_var, image_varr, 
                   meta_spec_A, meta_spec_C, meta_spec_B
                   ):
    # Separately update spec wavelengths & spec_shape
    meta_spec_A['lambda_grid'] = spec2dA['wave'] * u.Angstrom
    meta_spec_B['lambda_grid'] = spec2dB['wave'] * u.Angstrom
    meta_spec_C['lambda_grid'] = spec2dC['wave'] * u.Angstrom
    meta_spec_A['ngrid'] = spec2dA['wave'].shape
    meta_spec_B['ngrid'] = spec2dB['wave'].shape
    meta_spec_C['ngrid'] = spec2dC['wave'].shape
    # Cutting finished. 
    
    image_mask = np.ones(image_data.shape, dtype=bool)
    
    # Pack them.
    real_data_SetA = {'meta_spec':  meta_spec_A.copy(),  
                      'spec_data':  spec2dA['flux'],  
                      'spec_var' :  spec2dA['var'],
                      # 'spec_sky' :  spec2dA['sky'],
                      'spec_mask':  spec2dA['mask'],
                      'meta_image': meta_image, 
                      'image_data': image_data, 
                      'image_var' : image_var, 
                      'image_varr': image_varr, 
                      'image_mask': image_mask,
                      'cont_model_spec': cont(spec2dA['flux'])  }
    real_data_SetC = {'meta_spec':  meta_spec_C.copy(),  
                      'spec_data':  spec2dC['flux'],  
                      'spec_var' :  spec2dC['var'],
                      # 'spec_sky' :  spec2dC['sky'],
                      'spec_mask':  spec2dC['mask'],
                      'meta_image': meta_image, 
                      'image_data': image_data, 
                      'image_var' : image_var, 
                      'image_varr': image_varr, 
                      'image_mask': image_mask,
                      'cont_model_spec': cont(spec2dC['flux'])  }
    real_data_SetB = {'meta_spec':  meta_spec_B.copy(),  
                      'spec_data':  spec2dB['flux'],  
                      'spec_var' :  spec2dB['var'],
                      # 'spec_sky' :  spec2dB['sky'],
                      'spec_mask':  spec2dB['mask'],
                      'meta_image': meta_image, 
                      'image_data': image_data, 
                      'image_var' : image_var, 
                      'image_varr': image_varr, 
                      'image_mask': image_mask,
                      'cont_model_spec': cont(spec2dB['flux'])  }

    real_data_this_line_all_sets = [real_data_SetA,
                                    real_data_SetC,
                                    real_data_SetB]
    
    return real_data_this_line_all_sets

