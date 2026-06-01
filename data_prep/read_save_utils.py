import joblib
import numpy as np


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




