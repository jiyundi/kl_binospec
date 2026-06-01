import numpy as np
import astropy.units as u
from   scipy.ndimage  import median_filter

from  data_structure   import SpecData
from .build_var          import build_var 
from .continuum          import extract_2d_continuum
from .line_center_crop   import EmissionProcessor
from .line_manual_fix    import manual_correct
from .line_width_profile import find_line_sigma
from .masking            import mask_out_sky_lines, bad_pixel_mask
from .read_and_cutout    import cutoutspec

import matplotlib.pyplot as plt

_spec_idx_map    = ['A', 'C', 'B']
_spec_colors_map = ['orangered', 'cyan', 'gold']
        

def process_single_line(specs, lineinfo, meta_raws, 
                        spec_width, spec_height, 
                        clear_mode='simple', 
                        clear_cont_strength=None, manual=True):
    
    # 1. Rough cut: a 30 Angstrom window
    wave = np.mean(lineinfo.wav) * (1 + lineinfo.redshift)
    cutA = cutoutspec(specs.specA, wave, width=42) # A, 0.6 A/px
    cutB = cutoutspec(specs.specB, wave, width=42) # A, 0.6 A/px
    cutC = cutoutspec(specs.specC, wave, width=42) # A, 0.6 A/px
    
    # 2. Basic masks
    # 2.1) old NaN (-500/-200/-1000) values
    maskA_nonNaN = (cutA['flux'] > -100)
    maskB_nonNaN = (cutB['flux'] > -100)
    maskC_nonNaN = (cutC['flux'] > -100)
    
    # 2.2) vertical strip masks for sky lines
    maskA_nosky = mask_out_sky_lines(cutA.copy(), maskA_nonNaN)
    maskB_nosky = mask_out_sky_lines(cutB.copy(), maskB_nonNaN)
    maskC_nosky = mask_out_sky_lines(cutC.copy(), maskC_nonNaN)
    
    # 2.3) local masks for removing bad/overexposed pixels
    maskA_goodpx = ~bad_pixel_mask(np.where(maskA_nosky, cutA['flux'], 0))
    maskB_goodpx = ~bad_pixel_mask(np.where(maskB_nosky, cutB['flux'], 0))
    maskC_goodpx = ~bad_pixel_mask(np.where(maskC_nosky, cutC['flux'], 0))
    
    # 2.4) combine
    maskA = maskA_nosky & maskA_goodpx & maskA_nonNaN
    maskB = maskB_nosky & maskB_goodpx & maskB_nonNaN
    maskC = maskC_nosky & maskC_goodpx & maskC_nonNaN
    cutA['mask'] = maskA
    cutB['mask'] = maskB
    cutC['mask'] = maskC
    
    # ---------------------------------------------------------------------
    # 3. Precise cut out and centerization
    emission_before_cut = EmissionProcessor(
        cutA, cutC, cutB, 
        lineinfo.redshift
        )
    spec2dACB, how_cut0 = emission_before_cut.process(
        line = lineinfo.name, 
        this_line_waves = lineinfo.wav, 
        spec_width = spec_width, 
        spec_height = spec_height
        )
    
    # 4. Now, let users decide to accept or reject this cut
    try:
        fig, axs = plt.subplots(3, 1, figsize=(1.4, 2*3)) # (length, height)
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        for i in range(3):
            spec_data = np.where(spec2dACB[i]['mask'], 
                                 spec2dACB[i]['flux'], np.nan)
            
            xmin =  spec2dACB[i]['wave'][0, 0] # x_left
            xmax =  spec2dACB[i]['wave'][0,-1] # x_right
            ymin = -len(spec_data)/2 # y_bottom
            ymax = +len(spec_data)/2 # y_top
            
            ax1 = axs[i]
            im_spec = ax1.imshow(
                spec_data, 
                extent=[xmin, xmax, ymin, ymax],
                cmap='viridis', aspect='auto', origin='lower', 
                vmin=0 - np.nanstd(spec_data), 
                vmax=0 + 5*np.nanstd(spec_data)
                )
            fig.colorbar(im_spec, ax=ax1)
            ax1.xaxis.get_major_formatter().set_useOffset(False)
            ax1.set_ylabel(r'Spatial Position (px)')
            ax1.grid(linestyle=':', color='orangered', alpha=0.5)
            ax1.set_title(f'Set {_spec_idx_map[i]}', 
                          color=_spec_colors_map[i], size=15)
        plt.show(block=False)
        plt.pause(0.1)
        fail_pre = False
    except IndexError:
        fail_pre = True
    
    # 4.1) Now the user will make a decision...
    if manual:
        if fail_pre==True or input(
            '=========================================\n'+
            '----------- Reject this cut? ------------ \n'+
            '\u25A1 Yes - Press "Enter" to manually redo. (default)\n'+
            '\u25A1 No  - Type "ok" to skip manual redo and accept it. \n'+
            'Your decision: '
            )=='':
            plt.close()
            idx_xys = np.array([
                manual_correct(cutA.copy(), spec_width, spec_height, 
                               tag=f'{lineinfo.name}, A'), 
                manual_correct(cutC.copy(), spec_width, spec_height, 
                               tag=f'{lineinfo.name}, C'), 
                manual_correct(cutB.copy(), spec_width, spec_height, 
                               tag=f'{lineinfo.name}, B')
                ]) # [[ix, iy], ...]
            spec2dACB, how_cut0 = emission_before_cut.process(
                line = lineinfo.name, 
                this_line_waves = lineinfo.wav, 
                spec_width = spec_width, 
                spec_height = spec_height, 
                idx_xs=idx_xys[:,0], idx_ys=idx_xys[:,1], # Set A, C, B
                )
        
        else:
            plt.close()
    
    # 4.2) If require not to manually fix...
    else:
        idx_xys = np.array([
            [31.8, 22], [38.0, 26], [38.0, 25]
            ]) # [[ix, iy], ...]
        spec2dACB, how_cut0 = emission_before_cut.process(
            line = lineinfo.name, 
            this_line_waves = lineinfo.wav, 
            spec_width = spec_width, 
            spec_height = spec_height, 
            idx_xs=idx_xys[:,0], idx_ys=idx_xys[:,1], # Set A, C, B
            )
        plt.close()
    
    # Precise cutting ends here
    # ---------------------------------------------------------------------
    
    # 5. Update redshift in lineinfo
    redshift = redshift_update(
        spec2dACB, lineinfo.redshift, 
        how_cut0, lineinfo.wav,
        )
    lineinfo.redshift = redshift
    
    # 6. Non-cutting processing 
    # 6.1) Extract continuum AND SUBSTRACT
    print('Continuums are being extracted and substracting data...')
    contA = extract_2d_continuum(spec2dACB[0]['flux'], 
                                 spec2dACB[0]['mask'], 
                                 # clear_cont_strength, 
                                 mode=clear_mode, 
                                 verbose=True, 
                                 smooth=9)
    contC = extract_2d_continuum(spec2dACB[1]['flux'], 
                                 spec2dACB[1]['mask'], 
                                 # clear_cont_strength, 
                                 mode=clear_mode, 
                                 verbose=True, 
                                 smooth=9, 
                                 # cont_y0=len(spec2dACB[1]['flux'])//2-1
                                 )
    contB = extract_2d_continuum(spec2dACB[2]['flux'], 
                                 spec2dACB[2]['mask'],
                                 # clear_cont_strength, 
                                 mode=clear_mode, 
                                 verbose=True,
                                 smooth=9, 
                                 # cont_y0=0
                                 )
    fluxA = spec2dACB[0]['flux'] - contA
    fluxC = spec2dACB[1]['flux'] - contC
    fluxB = spec2dACB[2]['flux'] - contB
    
    # 6.2) Rescale: SPEC VAR - FINAL STEP OF SPECTRUM CUTTING
    varA, mask_addedA = build_var(fluxA, spec2dACB[0]['mask'], option=1)
    varC, mask_addedC = build_var(fluxC, spec2dACB[1]['mask'], option=1)
    varB, mask_addedB = build_var(fluxB, spec2dACB[2]['mask'], option=1)
    
    # 6.3) Line profile solution by each set
    line_profile_A = find_line_sigma(
        median_filter(fluxA, size=3), lineinfo.name, 'y0!=0'
        )
    line_profile_C = find_line_sigma(
        median_filter(fluxC, size=3), lineinfo.name, 'y0!=0'
        )
    line_profile_B = find_line_sigma(
        median_filter(fluxB, size=3), lineinfo.name, 'y0!=0'
        )
    
    # 7. Write meta spec info - linename, wave spec, ngrid...
    meta_specA = {}
    meta_specB = {}
    meta_specC = {}
    for key in meta_raws['A'].keys():
        if key == 'line_species':
            meta_specA['line_species'] = lineinfo.name
            meta_specB['line_species'] = lineinfo.name
            meta_specC['line_species'] = lineinfo.name
        elif key == 'line_profile':
            meta_specA['line_profile'] = line_profile_A
            meta_specB['line_profile'] = line_profile_B
            meta_specC['line_profile'] = line_profile_C
        elif key == 'lambda_grid':
            meta_specA['lambda_grid'] = spec2dACB[0]['wave'] * u.Angstrom
            meta_specB['lambda_grid'] = spec2dACB[2]['wave'] * u.Angstrom
            meta_specC['lambda_grid'] = spec2dACB[1]['wave'] * u.Angstrom
        elif key == 'slitLen':
            meta_specA['slitLen'] = len(spec2dACB[0]['wave']) * meta_raws['A']['pixScale'] 
            meta_specB['slitLen'] = len(spec2dACB[2]['wave']) * meta_raws['B']['pixScale'] 
            meta_specC['slitLen'] = len(spec2dACB[1]['wave']) * meta_raws['C']['pixScale'] 
        elif key == 'ngrid':
            meta_specA['ngrid'] = spec2dACB[0]['wave'].shape
            meta_specB['ngrid'] = spec2dACB[2]['wave'].shape
            meta_specC['ngrid'] = spec2dACB[1]['wave'].shape
        else:
            meta_specA[key] = meta_raws['A'][key]
            meta_specB[key] = meta_raws['B'][key]
            meta_specC[key] = meta_raws['C'][key]
    
    # 8. Pack all data
    SetA = SpecData(data = fluxA, 
                    var  = varA, 
                    cont = contA,
                    mask = mask_addedA,
                    meta = meta_specA
                    )
    SetC = SpecData(data = fluxC, 
                    var  = varC, 
                    cont = contC,
                    mask = mask_addedC,
                    meta = meta_specC
                    )
    SetB = SpecData(data = fluxB, 
                    var  = varB, 
                    cont = contB,
                    mask = mask_addedB,
                    meta = meta_specB
                    )
    spec_data_list = [SetA, SetC, SetB]
    
    return spec_data_list, lineinfo
    
    
# def check_x0y0_in_cropped(ls, howcut):
#     max_padding = 1 # px. Discard if too close to spec edge
    
#     # Spec in each set should be same-sized
#     # If not, we have a check after this block
#     specshapes, x0y0s = [], []
    
#     # For each set
#     for i, spec in enumerate(ls): 
#         spec_shape = spec['flux'].shape
#         specshapes.append(spec_shape)
        
#         # choose the "line" key
#         for key, val in howcut[f"Set{i}"].items():
#             if key[:4]=='line':
#                 x0 = howcut[f"Set{i}"][key]['x0']
#                 y0 = howcut[f"Set{i}"][key]['y0']
#                 x0y0s.append([x0, y0])
    
#     # check spec_shape
#     for i in range(1, len(specshapes)):
#         if specshapes[i] != specshapes[0]:
#             print(f'Found spec in {len(specshapes)} sets have different shapes.⚠️')
#             print(specshapes)
    
#     # check x0y0
#     bad_x0y0 = {}
#     for i in range(len(x0y0s)):
#         x0 = x0y0s[i][0]
#         y0 = x0y0s[i][1]
#         spec_shape = specshapes[0]
#         if (y0 >= spec_shape[0]-max_padding) or (x0 >= spec_shape[1]-max_padding):
#             bad_x0y0[f'set{i}'] = {'spec_shape': spec_shape, 
#                                    'bad_x0': x0, 'bad_y0': y0}

#     # if cutting is good...
#     assert len(bad_x0y0) == 0, \
#         'Cutting for emission line: Failed.' 
    
#     return bad_x0y0


def redshift_update(spec2dACB, redshift, 
                    how_cut, this_line_waves):
    # Correct redshift
    z0s = []
    for setname, setdict in how_cut.items():
        setidx = int(setname[-1])
        
        for j in range(len(this_line_waves)):
            x0 = setdict[f'line{j+1}']['x0']
            y0 = setdict[f'line{j+1}']['y0']
            x0_lf, y0_up = int(x0),   int(y0)
            x0_rt, y0_dn = int(x0)+1, int(y0)+1
            
            # left/right edge's wave on x0 pixel
            wave_lf  = spec2dACB[setidx]['wave'][y0_up, x0_lf]
            wave_rt  = spec2dACB[setidx]['wave'][y0_dn, x0_rt]
            thiswave = wave_lf + (x0 - x0_lf) * (wave_rt - wave_lf)
            print(f'Emi center at wavelength (A): {thiswave:.2f}')
            
            restwave = this_line_waves[j]
            z0 = thiswave / restwave - 1
            z0s.append(z0)
            j += 1

    new_z = np.mean(z0s)
    print(f'Redshift updated: {new_z:.5f} ({new_z-redshift:.5f})')
    
    return new_z


# def measure_v0(spec_ls, how_cut, this_line_waves):
#     z0s = []
#     for setname, setdict in how_cut.items():
#         setidx = int(setname[-1])
        
#         for j in range(len(this_line_waves)):
#             x0 = setdict[f'line{j+1}']['x0']
#             y0 = setdict[f'line{j+1}']['y0']
#             x0_lf, y0_up = int(x0),   int(y0)
#             x0_rt, y0_dn = int(x0)+1, int(y0)+1
            
#             # left/right edge's wave on x0 pixel
#             wave_lf  = spec_ls[setidx]['wave'][y0_up, x0_lf]
#             wave_rt  = spec_ls[setidx]['wave'][y0_dn, x0_rt]
#             thiswave = wave_lf + (x0 - x0_lf) * (wave_rt - wave_lf)
#             print(f'Emi center at wavelength (A): {thiswave:.2f}')
            
#             restwave = this_line_waves[j]
#             z0 = thiswave / restwave - 1
#             z0s.append(z0)
#             j += 1

#     return np.mean(z0s)

