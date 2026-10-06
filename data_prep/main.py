import os
import joblib
import argparse
import numpy as np
import pandas as pd
import astropy.units as u
from   dataclasses import dataclass
from   dataclasses import asdict

from plot           import make_exam_plots
from meta_config    import meta_spec_ABC, meta_image
from data_structure import RealDataInfo, ImageData
from image.masking         import mask_neighbor_star, find_mask_pars
from image.rescale_var     import rescale_var
from image.read_and_cutout import read_subaru_img_wcs, cutoutimg
from spec.line_pipeline    import process_single_line
from spec.read_and_cutout  import read_spec2d, stack_spec2d, readinfodat

from   klm.safe_plot import setup; setup() # must before plt
import matplotlib; matplotlib.use('TkAgg') # Comment out if not using Windows
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


# Use the same setting in klm/parameters. From: 
# https://sdss-mangadap.readthedocs.io/en/latest/emissionlines.html
_all_emilines_supported = {
    'O2':  [3727.092, 3729.875],
    'Ha':  [6564.608],
    'Hb':  [4862.683], 
    'Hg':  [4341.684], 
    'O3a': [4960.295], 
    'O3b': [5008.240], 
    'N2a': [6549.86 ], 
    'N2b': [6585.27 ],
    }


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--slitID', default=7, type=int)
    parser.add_argument('--spec_width',  default=40, type=int)
    parser.add_argument('--spec_height', default=20, type=int)
    parser.add_argument('--img_width',   default=20, type=int)
    parser.add_argument('--img_height',  default=20, type=int)
    parser.add_argument('--clear_mode',  default='simple', type=str)
    parser.add_argument('--cont_scale',  default=[1.], type=int, nargs='+')
    parser.add_argument('--cont_y0s',    default=[10], type=int, nargs='+')
    slit_num    = parser.parse_args().slitID
    spec_width  = parser.parse_args().spec_width
    spec_height = parser.parse_args().spec_height
    img_width   = parser.parse_args().img_width
    img_height  = parser.parse_args().img_height
    clear_mode  = parser.parse_args().clear_mode
    cont_scale  = parser.parse_args().cont_scale
    cont_y0ss   = parser.parse_args().cont_y0s
    
    # 1. FOLDER NAMES AND FILENAMES
    img_dir = '../../../RSCH3/HSC_img_A383/'
    sci_fn  = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz.fits'
    wgt_fn  = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz-weight.fits'
    spec_dir_1a = '../../../RSCH3/UAO-S156-23B-A383/psf_m1/231023/1d2dspecfiles_counts/'
    spec_dir_1b = '../../../RSCH3/UAO-S156-23B-A383/psf_m1/231218/1d2dspecfiles_counts/'
    spec_dir_1c = '../../../RSCH3/UAO-S156-23B-A383/psf_m1/240115/1d2dspecfiles_counts/'
    spec_dir_2  = '../../../RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles_counts/'
    spec_dir_3  = '../../../RSCH3/UAO-S156-23B-A383/psf_p1/231019/1d2dspecfiles_counts/'
    spec_fn = f'spec2d.829.{slit_num:03d}.{slit_num+100305:06d}.fits'
    info_fn = f'info.829.{slit_num:03d}.{slit_num+100305:06d}.dat'
    z_table_fn = "../scripts/redshift_table.xlsx"
    Ms_folder  = '../../bagpipes-KL/'
    real_pkl_folder = '../scripts/binospec_pkl/'
    
    # 2. CHECK IF SECURE REDSHIFT IS PROVIDED
    df     = pd.read_excel(z_table_fn, header=None, engine='openpyxl')
    array  = df.to_numpy()
    ztable = array[1:, 0:2]
    ltable = array[1:, [0,6,9,12]]
    bltable = array[1:, [0,14]]
    z_spec  = ztable[slit_num, 1]
    if np.isnan(z_spec) == True: 
        print(f'No secure redshift found. Skipped this Slit {slit_num}.')
        import sys; sys.exit(0)
    
    # 3. CHECK IF EMISSION LINES ARE DESIGNATED
    # 3.1) add all lines
    emilines = set()
    for idxcol in range(1,len(ltable[0])):
        linenames = ltable[slit_num,idxcol]
        if pd.notna(linenames):
            emilines.update(ltable[slit_num,idxcol].split(","))
    
    # 3.2) look up wavelengths
    emilines = {k: _all_emilines_supported[k] for k in emilines} 
    emilines = dict(sorted(emilines.items(), key=lambda x: x[1][0]))
    
    # 3.3) remove some lines additionally by user
    remove_str = bltable[slit_num, 1]
    remove_lines = []
    if pd.notna(remove_str):
        remove_lines = [x.strip() for x in remove_str.split(",")]
    for r in remove_lines:
        emilines.pop(r, None)
    
    # 3.4) stop if all lines are rejected by user
    if len(emilines) == 0: 
        print('Process stopped because no line provided for cutting.')
        import sys; sys.exit(0)
    
    # 4. MAKE PKL FOLDER IF NOT EXISTS 
    if not os.path.exists(real_pkl_folder):
        os.makedirs(real_pkl_folder)
        if not os.path.exists(real_pkl_folder+'pkl/'):
            os.makedirs(real_pkl_folder+'pkl/')
    
    # 5. META INFORMATION 
    # 5.1) meta gal - this slit (galaxy)
    dat_dict      = readinfodat(spec_dir_3 + info_fn)
    objRA, objDEC = dat_dict['RA'], dat_dict['DEC']
    RA_obj     = objRA  * u.deg
    Dec_obj    = objDEC * u.deg
    redshift   = z_spec
    meta_gal = {
        'redshift': redshift,
        'RA':       RA_obj,
        'Dec':      Dec_obj,
        'beta':     0*u.deg, # Nothing to do with g1/g2 if are fitting
        'log10_Mstar':     None, 
        'log10_Mstar_err': None,
        }
    
    # 5.2) meta image - (ngrid, pixScale) - ONLY add slit RA/DEC
    meta_image_dic = meta_image(
        None, objRA, objDEC, add_ap_wcs=False
        )
    
    # 5.3) initialize meta spec by each line - ONLY add slit RA/DEC
    meta_spec_raws = meta_spec_ABC(
        dat_dict, spec_shape=None, slit_len=None
        )
    
    # ========================================================================
    # PRE-CHECK COMPLETE. NOW START CUTTING...
    print(f'Cutting for Slit {slit_num}...')
    
    # -------------------------------------------------------------
    # 6. READ AND CUTOUT IMAGING DATA
    # -------------------------------------------------------------
    
    # 6.1) Entire 30' x 30' image
    imaging_data = read_subaru_img_wcs(img_dir+sci_fn, img_dir+wgt_fn)
    science_data = imaging_data['science_data']
    science_wcs  = imaging_data['science_wcs' ]
    weights_data = imaging_data['weights_data']
    weights_wcs  = imaging_data['weights_wcs' ]
    
    # 6.2) Cutout image to 4'' x 4'' size
    image_data, RAlim, DEClim = cutoutimg(
        science_data, science_wcs, objRA, objDEC, img_width, img_height
        )
    image_ivar, RAlim, DEClim = cutoutimg(
        weights_data, weights_wcs, objRA, objDEC, img_width, img_height
        )
    image_var_raw = image_ivar**(-1)
    empty_mask    = np.ones(image_data.shape, dtype=bool)
    
    # 6.3) Mask out nearby galaxies and stars
    kws = find_mask_pars(slit_num)
    if kws is not None:
        image_mask = mask_neighbor_star(
            image_data, empty_mask, *kws
            )
    else:
        image_mask = empty_mask
    
    # 6.4) Rescale: ESTIMATE A REASONABLE LEVEL OF IMAGE VARIANCE
    image_var = rescale_var(image_data, image_var_raw, image_mask)
    
    # 6.5) Update meta image
    meta_image_dic = meta_image(
        image_data.shape, objRA, objDEC, add_ap_wcs=True
        )
    
    # 6.6) Pack image data
    image_data_info = ImageData(
        data     = image_data,
        mask     = image_mask,
        var      = image_var,
        var_raw  = image_var_raw,
        meta     = meta_image_dic
    )
    
    # -------------------------------------------------------------
    # 7. READ AND CUTOUT SPECTRA
    # -------------------------------------------------------------
    
    # 7.1) read raw specs
    spec_C = read_spec2d(spec_dir_2  + spec_fn)
    spec_B = read_spec2d(spec_dir_3  + spec_fn)
    A1     = read_spec2d(spec_dir_1a + spec_fn)
    A2     = read_spec2d(spec_dir_1b + spec_fn)
    A3     = read_spec2d(spec_dir_1c + spec_fn)
    nrowsA = np.max([len(A1['flux']), len(A2['flux']), len(A3['flux'])])
    spec_A = {
        'wave': np.tile(A1['wave'][0, :], (nrowsA, 1)),
        'flux': stack_spec2d(A1['flux'], A2['flux'], A3['flux']),
        }
    
    # 7.2) convert to bottom -> top 
    #      since MMT/Binospec reads spec from detector's top -> bottom
    for k in spec_A.copy().keys():
        spec_A[k] = np.flip(spec_A[k], axis=0)
        spec_C[k] = np.flip(spec_C[k], axis=0)
        spec_B[k] = np.flip(spec_B[k], axis=0)
    spec_A['mask'] = np.ones(spec_A['flux'].shape, dtype=bool)
    spec_C['mask'] = np.ones(spec_C['flux'].shape, dtype=bool)
    spec_B['mask'] = np.ones(spec_B['flux'].shape, dtype=bool)
    
    # 7.3) wrap up specs
    @dataclass
    class SpecSet:
        specA: dict; specB: dict; specC: dict
    raw_specs = SpecSet(specA=spec_A, specC=spec_C, specB=spec_B)
    
    # 7.4) cut out by each line
    specs_data_info, redo_zs, line_i = [], [], 0
    for linename, this_emis_wave in emilines.items():
        print(f'\nCutting for emission line {line_i+1}: {linename}...')
        
        # 7.4.1) fill info in dataclass
        @dataclass
        class LineInfo:
            name: str; wav: list; redshift: float
        line_info = LineInfo(
            name = linename,
            wav  = this_emis_wave,
            redshift = redshift
        )
        
        # 7.4.2) core function of cutting
        if cont_y0ss is not None:
            if len(cont_y0ss) > 1:
                cont_y0s = [ cont_y0ss[3*line_i    ], 
                             cont_y0ss[3*line_i + 1], 
                             cont_y0ss[3*line_i + 2] ]
            else:
                cont_y0s = None
        
        if len(cont_scale) > 1:
            cont_scales = [cont_scale[3*line_i    ], 
                           cont_scale[3*line_i + 1], 
                           cont_scale[3*line_i + 2] ]
        else:
            cont_scales = None
        
        data_this_line, line_info_updated = process_single_line(
            raw_specs, line_info, meta_spec_raws, 
            spec_width, spec_height, 
            clear_mode, 
            cont_scales=cont_scales, 
            cont_y0s=cont_y0s, 
            #manual=False
        )
        
        # 7.4.3) add to list
        redo_zs.append(line_info_updated.redshift)
        for data in data_this_line:
            specs_data_info.append(data)
        print(f'Cutting for emission line: {linename} finished. OK.')
        
        line_i += 1
    
    # -------------------------------------------------------------
    # 8. FINALLY UPDATE z and M* IN META GALAXY
    # -------------------------------------------------------------
    
    # 8.1) Read [slit#, M_stellar, err]
    dg = pd.read_csv(f"{Ms_folder}Mstellar_table.txt",
                     sep=r"\s+", header=0,
                     names=["slit", "median", "std", "err_lo", "mean", "err_hi"])
    dg = dg.drop_duplicates(subset="slit", keep="last")
    Ms = dg.sort_values(by='slit').to_numpy()
    for row in Ms:
        if row[0] == slit_num:
            log10_Mstar     = row[1]
            log10_Mstar_err = row[2]
            print(f'Updated: log10_Mstar = {log10_Mstar}, err = {log10_Mstar_err}')
            break
    
    # 8.2) From spec cutouts, take an average of precise redshifts
    redo_z = np.mean(redo_zs)
    
    # 8.3) wrap up
    meta_gal['redshift'] = redo_z
    meta_gal['log10_Mstar']     = log10_Mstar
    meta_gal['log10_Mstar_err'] = log10_Mstar_err
    
    # ========================================================================
    # 9. NOW SAVING PKL
    # ========================================================================
    
    data_info = RealDataInfo(spec   = specs_data_info,
                             image  = image_data_info,
                             galaxy = meta_gal)
    
    # Save mocks
    slit_name = f'{slit_num:03d}'
    with open(f'{real_pkl_folder}pkl/slit_{slit_name}.pkl', "wb") as f:
        joblib.dump(asdict(data_info), f)
    
    # Plot
    make_exam_plots(data_info, f'{slit_num:03d}', 
                    pkl_folder=real_pkl_folder)
    
    print()    
    print(f'Cutting for Slit {slit_num}... Finished. ✅\n')
