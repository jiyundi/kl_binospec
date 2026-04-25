# Ignore this syntax check since it's an iPython command:
# %matplotlib qt
import os
import argparse
import pandas as pd
import numpy as np
import astropy.units as u
from   astropy.io  import fits
from   astropy.wcs import WCS

from   image_utils     import cutoffimg, Meta_image, half_light_radius_exp
from   line_processing import process_single_line, find_line_sigma
from   meta_utils      import meta_spec_ABC
from   plot            import make_exam_plots
from   read_save_utils import save_dic_and_pkl, read_spec2d, readinfodat
from   spec_utils      import stack_spec2d

# from   klm.safe_plot import setup; setup() # must before plt
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def main_cut_off(specA, specB, specC, redshift, emilines, 
                 meta_gal, 
                 meta_image, image_data, image_var, image_varr, 
                 meta_spec_A, meta_spec_B, meta_spec_C,
                 spec_width = 25, spec_height = 20): # px
    real_data_all_lines_sets = {}
    bad_lines = []
    how_cut = {}

    # For each emission line...
    for linename, this_emis_waves in emilines.items():
        print(f'\nCutting for emission line: {linename}...')
        real_data, how_cut0 = process_single_line(
            specA, specC, specB, 
            redshift, bad_lines, how_cut,
            real_data_all_lines_sets,
            linename, this_emis_waves,
            meta_gal, meta_image, image_data, image_var, image_varr, 
            meta_spec_A, meta_spec_C, meta_spec_B,
            spec_width, spec_height
        )
        
        # Optional: Update sigma across wavelen axis 
        # (replace line_profile_path in config)
        real_data[0]['meta_spec']['line_sig_amps'] = find_line_sigma(
            real_data[0], how_cut0['Set0'], linename
            )
        real_data[1]['meta_spec']['line_sig_amps'] = find_line_sigma(
            real_data[1], how_cut0['Set1'], linename
            )
        real_data[2]['meta_spec']['line_sig_amps'] = find_line_sigma(
            real_data[2], how_cut0['Set2'], linename
            )
        
        # Update slit len 
        spec_pix_scale = 0.24 # 0.24 arcsec/pix by Binospec
        real_data[0]['meta_spec']['slitLen'] = real_data[0]['spec_data'].shape[0] * spec_pix_scale
        real_data[1]['meta_spec']['slitLen'] = real_data[1]['spec_data'].shape[0] * spec_pix_scale
        real_data[2]['meta_spec']['slitLen'] = real_data[2]['spec_data'].shape[0] * spec_pix_scale
        
        real_data_all_lines_sets[linename] = real_data
        how_cut[linename] = how_cut0
        print(f'Cutting for emission line: {linename} finished. OK.')

    # Check if any one line cut was solved...
    if len(how_cut)==0: 
        raise KeyError("There is no available well-cut line in how_cut dict. "+ 
                       "Cutting for emission line: Failed. "+
                       "Go to line_processing.py > resolve_bad_lines() for details.")
    
    return real_data_all_lines_sets, how_cut





if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--slitID', default=8, type=int)
    parser.add_argument('--spec_width',  default=30, type=int)
    parser.add_argument('--spec_height', default=20, type=int)
    parser.add_argument('--img_width',   default=20, type=int)
    parser.add_argument('--img_height',  default=20, type=int)
    slit_num    = parser.parse_args().slitID
    spec_width  = parser.parse_args().spec_width
    spec_height = parser.parse_args().spec_height
    img_width   = parser.parse_args().img_width
    img_height  = parser.parse_args().img_height
    
    real_pkl_folder = '../binospec_pkl/'
    Ms_folder = '../../../bagpipes-KL/'
        
    # READ IMAGING DATA
    hscimagefolder01 = '../../../../RSCH3/HSC_img_A383/'
    hsc_filename = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz.fits'
    hsc_wght_fnm = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz-weight.fits'
    science_hdul = fits.open(hscimagefolder01 + hsc_filename)
    weights_hdul = fits.open(hscimagefolder01 + hsc_wght_fnm)
    
    # Pixel data
    science_raw = science_hdul[0].data
    weights_raw = weights_hdul[0].data
    
    # WCS mapping
    science_wcs = WCS(hscimagefolder01 + hsc_filename)
    weights_wcs = WCS(hscimagefolder01 + hsc_wght_fnm)
    science_hdr0_raw = science_hdul[0].header
    weights_hdr0_raw = weights_hdul[0].header
    sci_GAIN = science_hdr0_raw['GAIN']
    wei_GAIN = weights_hdr0_raw['GAIN']
    
    # Unit conversion
    # Science = science_raw (ADU/s)     * sci_GAIN    (e/ADU)
    # Weights = weights_raw (s^2/ADU^2) * wei_GAIN^-2 (e/ADU)^-2
    science_data = science_raw * sci_GAIN
    weights_data = weights_raw * wei_GAIN**(-2)
    
    # Spec files
    spec1d2dfolder01a = '../../../../RSCH3/UAO-S156-23B-A383/psf_m1/231023/1d2dspecfiles_counts/'
    spec1d2dfolder01b = '../../../../RSCH3/UAO-S156-23B-A383/psf_m1/231218/1d2dspecfiles_counts/'
    spec1d2dfolder01c = '../../../../RSCH3/UAO-S156-23B-A383/psf_m1/240115/1d2dspecfiles_counts/'
    spec1d2dfolder02  = '../../../../RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles_counts/'
    spec1d2dfolder03  = '../../../../RSCH3/UAO-S156-23B-A383/psf_p1/231019/1d2dspecfiles_counts/'
    spec2dfilename_temp = 'spec2d.829.{:03d}.{:06d}.fits'
    spec2d_err_filename = '../obj_counts_err_slits_lin.fits'
    infdatfilename_temp = 'info.829.{:03d}.{:06d}.dat'
    
    # Use the same setting in klm/parameters. From: 
    # https://sdss-mangadap.readthedocs.io/en/latest/emissionlines.html
    all_emilines_supported = {'O2':  [3727.092, 3729.875],
                              'Ha':  [6564.608],
                              'Hb':  [4862.683], 
                              'Hg':  [4341.684], 
                              'O3a': [4960.295], 
                              'O3b': [5008.240], 
                              'N2a': [6549.86 ], 
                              'N2b': [6585.27 ],
                              }
    z_table_filename = "../redshift_table.xlsx"
    df     = pd.read_excel(z_table_filename, header=None, engine='openpyxl')
    array  = df.to_numpy()
    ztable = array[1:, 0:2]
    ltable = array[1:, [0,6,9,12]]
    bltable = array[1:, [0,14]]
    
    print(f'Cutting for Slit {slit_num}...')
    
    infdatfilename = infdatfilename_temp.format(slit_num, slit_num+100305)
    dat_dict       = readinfodat(spec1d2dfolder03 + infdatfilename)
    objRA, objDEC  = dat_dict['RA'], dat_dict['DEC']
    
    z_spec = ztable[slit_num, 1]
    # z_spec = 0.5
    
    if np.isnan(z_spec)==False:
        # Now get object data from object catalog
        RA_obj     = objRA  * u.deg
        Dec_obj    = objDEC * u.deg
        redshift   = z_spec
        meta_gal = {
            'redshift': redshift,
            'RA':       RA_obj,
            'Dec':      Dec_obj,
            'beta':     0*u.deg, # Just give any number, only for g_t
            'log10_Mstar':     None, 
            'log10_Mstar_err': None,
            }
        
        # Image
        image_data, RAlim, DEClim = cutoffimg(science_data, 
                                              science_wcs, 
                                              objRA, objDEC, 
                                              img_width, img_height)
        image_ivar, RAlim, DEClim = cutoffimg(weights_data, 
                                              weights_wcs, 
                                              objRA, objDEC, 
                                              img_width, img_height)
        image_var  = image_ivar**(-1)
        meta_image = Meta_image(image_data.copy(), 
                                objRA, objDEC)
        
        # Rescale: ESTIMATE A REASONABLE LEVEL OF IMAGE NOISE (AND VARIANCE)
        noise_high  = np.median(image_data)
        noise_level = np.std(image_data[image_data < noise_high])
        var_high    = np.percentile(image_var, 95)
        image_varr  = image_var / var_high * (noise_level)**2
        
        # Spectra
        spec2dfilename = spec2dfilename_temp.format(slit_num, slit_num+100305)
        spec2d_A1 = read_spec2d(spec1d2dfolder01a + spec2dfilename, 
                                spec1d2dfolder01a + spec2d_err_filename, slit_num)
        spec2d_A2 = read_spec2d(spec1d2dfolder01b + spec2dfilename, 
                                spec1d2dfolder01b + spec2d_err_filename, slit_num)
        spec2d_A3 = read_spec2d(spec1d2dfolder01c + spec2dfilename, 
                                spec1d2dfolder01c + spec2d_err_filename, slit_num)
        spec2d_Cr = read_spec2d(spec1d2dfolder02  + spec2dfilename, 
                                spec1d2dfolder02  + spec2d_err_filename, slit_num)
        spec2d_Br = read_spec2d(spec1d2dfolder03  + spec2dfilename, 
                                spec1d2dfolder03  + spec2d_err_filename, slit_num)
        max_rows  = max(spec2d_A1['flux'].shape[0], 
                        spec2d_A2['flux'].shape[0], 
                        spec2d_A3['flux'].shape[0])
        spec2d_Ar = {'wave': np.tile(spec2d_A1['wave'][0, :], (max_rows, 1)),
                     'flux': stack_spec2d(spec2d_A1['flux'], 
                                          spec2d_A2['flux'], 
                                          spec2d_A3['flux'],
                                          spec2d_A1['var']**-1, 
                                          spec2d_A2['var']**-1, 
                                          spec2d_A3['var']**-1),
                     'var': stack_spec2d(spec2d_A1['var']**-1, 
                                         spec2d_A2['var']**-1, 
                                         spec2d_A3['var']**-1)**-1,
                     }
        
        # Due to MMT/Binospec reads spec from detector's top to bottom
        # we need to convert to bottom-->top for kl measurement
        need_to_flip_slit_spatial = True
        if need_to_flip_slit_spatial:
            for k in spec2d_Ar.copy().keys():
                spec2d_Ar[k] = np.flip(spec2d_Ar[k], axis=0)
                spec2d_Cr[k] = np.flip(spec2d_Cr[k], axis=0)
                spec2d_Br[k] = np.flip(spec2d_Br[k], axis=0)
        
        spec2d_Ar['mask'] = np.ones(spec2d_Ar['flux'].shape, dtype=bool)
        spec2d_Cr['mask'] = np.ones(spec2d_Cr['flux'].shape, dtype=bool)
        spec2d_Br['mask'] = np.ones(spec2d_Br['flux'].shape, dtype=bool)
        
        # Look up emission line wavelengths
        emilines = set()
        for idxcol in range(1,len(ltable[0])):
            linenames = ltable[slit_num,idxcol]
            if pd.notna(linenames):
                emilines.update(ltable[slit_num,idxcol].split(","))
        emilines = {k: all_emilines_supported[k] for k in emilines}
        emilines = dict(sorted(emilines.items(), key=lambda x: x[1][0]))
        
        # Remove those lines not good for KL
        remove_str = bltable[slit_num, 1]
        remove_lines = []
        if pd.notna(remove_str):
            remove_lines = [x.strip() for x in remove_str.split(",")]
        for r in remove_lines:
            emilines.pop(r, None)
        
        if len(emilines) == 0: 
            print('Process stopped because no line provided for cutting.')
            import sys; sys.exit(0)
        
        # Make meta_spec, but only one spec from Set C is required
        meta_spec_A, meta_spec_B, meta_spec_C = meta_spec_ABC(spec2d_Cr, dat_dict, 
                                                              max_rows)
        
        # For this slit...
        real_data_all_lines_sets, how_cut = main_cut_off(
            spec2d_Ar, spec2d_Br, spec2d_Cr, 
            redshift, emilines, 
            meta_gal, meta_image, 
            image_data, image_var, image_varr, 
            meta_spec_A, meta_spec_B, meta_spec_C,
            spec_width, spec_height
            )
        
        # Rescale: spec var - Final step of spectrum cutting
        data_renewed = {}
        for line, olddata_list in real_data_all_lines_sets.items():
            list_renewed = []
            
            for olddata_list_setX in olddata_list:
                newdic_this_line_setX = olddata_list_setX
                
                data = olddata_list_setX['spec_data']
                mask = olddata_list_setX['spec_mask']
                
                # OPTION 1: median filter
                # data_masked = data[mask]
                # noise_high  = np.median(data_masked)
                # noise_level = np.std(data_masked[data_masked < noise_high])
                # data_high   = np.percentile(data_masked, 95)
                # shot_noise  = (data - np.min(data_masked)) / data_high * (noise_level)
                # newdic_this_line_setX['back_gaussian_noise'] = shot_noise
                
                # OPTION 2: avoid central emission line
                mask_sides  = np.ones(mask.shape, dtype=bool)
                mask_sides[:,mask.shape[1]//4:-mask.shape[1]//4] = False
                mask_bkgrnd = mask_sides & mask
                
                # at least one entry are True
                if mask_bkgrnd.any() == True: 
                    data_masked = data[mask_bkgrnd]
                # all entries are False
                elif mask_bkgrnd.any() == False: 
                    data_masked = data[mask]
                
                noise_level = np.std(data_masked)
                back_noise  = np.ones(data.shape) * noise_level
                newdic_this_line_setX['spec_gauss_back'] = back_noise
                
                list_renewed.append(newdic_this_line_setX)
                
            data_renewed[line] = list_renewed
            
        real_data_all_lines_sets = data_renewed
            
        # Check folder exists 
        if not os.path.exists(real_pkl_folder):
            os.makedirs(real_pkl_folder)
            if not os.path.exists(real_pkl_folder+'pkl/'):
                os.makedirs(real_pkl_folder+'pkl/')
        
        # Rough estimate r_hl_disk
        image_pix_scale = 0.2 # arcsec/pix: 0.2 for HSC image
        rhl_pixels = half_light_radius_exp(image_data)
        rhl_arcsec = rhl_pixels * image_pix_scale
        print(f"Sérsic (n=1) r_hl_disk = {rhl_pixels:.1f} pixels",
              f"= {rhl_arcsec:.2f} arcsec.")
        with open(real_pkl_folder+"r_hl_table.txt", "a+") as f:
            f.seek(0, 2)
            f.write(f'{slit_num:>4d} {rhl_arcsec:>5.2f}'+ "\n")
        
        # Read [slit#, r_hl_disk]
        df = pd.read_csv(f"{real_pkl_folder}r_hl_table.txt",
                         sep=r"\s+", header=None, names=["slit", "rhl"])
        df = df.drop_duplicates(subset="slit", keep="last")
        rr = df.sort_values(by='slit').to_numpy()
        
        # Read [slit#, M_stellar, err]
        dg = pd.read_csv(f"{Ms_folder}Mstellar_table.txt",
                         sep=r"\s+", header=0,
                         names=["slit", "median", "std", "err_lo", "mean", "err_hi"])
        dg = dg.drop_duplicates(subset="slit", keep="last")
        Ms = dg.sort_values(by='slit').to_numpy()
        
        # Save pkl
        real_data_info = save_dic_and_pkl(
            slit_num, real_data_all_lines_sets, 
            meta_gal, Ms, rr, 
            real_pkl_folder)
        
        # Plot
        make_exam_plots(real_data_info, f'{slit_num:03d}', 
                        pkl_folder=real_pkl_folder,
                        how_cut=how_cut)
            
    else:
        print(f'No secure redshift found. Skipped this Slit {slit_num}.')
    
    print('\nslitLPA:', real_data_info['spec'][0]['par_meta']['slitLPA'].value)
    print()    
    print(f'Cutting for Slit {slit_num}... Finished. ✅')