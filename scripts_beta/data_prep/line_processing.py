import numpy as np

from   cut_emis_center import EmissionProcessor
from   manual_fix      import manual_correct
from   mask_utils      import mask_out_pixels, remove_bad_pixels
from   read_save_utils import real_data_pack
from   spec_utils      import cutoffspec
from   continuum       import build_2d_continuum

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def process_single_line(specA, specC, specB, 
                        redshift, bad_lines, how_cut,
                        real_data_all_lines_sets,
                        linename, this_emis_waves,
                        meta_gal, meta_image, 
                        image_data, image_var, image_varr, 
                        meta_spec_A, meta_spec_C, meta_spec_B,
                        spec_width, spec_height):
    """
    Returns:
        real_data_this_line
        how_cut
        bad_x0y0
    """
    # Non-spatial crop only
    wave = np.mean(this_emis_waves) * (1 + redshift)
    cutA, cutB, cutC, cutmask = cutoffspec(specA.copy(), 
                                           specB.copy(), 
                                           specC.copy(), wave, 
                                           width=30) #A, 0.6 A/px
    
    # Mask out sky lines' pixels
    sky_wav_min  = [8378,   8785, 8428, 8411, 8427, 8835,   8463, 8999, 8825, 8834]
    sky_wav_max  = [8382.5, 8789, 8432, 8416, 8432, 8837.5, 8467, 9003, 8828, 8838]
    
    sky_wav_min += [8834, 8824, 7339, 8884, 9304, 9311, 9321, 8833, 8765, 8757]
    sky_wav_max += [8838, 8829, 7342, 8888, 9308, 9315, 9326, 8838, 8770, 8763]
    
    sky_wav_min += [7327, 7314, 8883, 8917, 8941, 8865, 8341, 7716, 8955, 7850]
    sky_wav_max += [7330, 7318, 8887, 8921, 8944, 8869, 8345, 7722, 8959, 7854]
    
    sky_wav_min += [8984, 7910, 7919, 7791, 7819, 7273, 7748, 7758, 7990, 8060]
    sky_wav_max += [8988, 7915, 7923, 7795, 7823, 7277, 7752, 7761, 7994, 8063]
    
    sky_wav_min += [8296, 7806, 8023, 8277, 8900, 7366, 7887, 8350, 8308, 8847]
    sky_wav_max += [8300, 7809, 8027, 8281, 8904, 7370, 7891, 8353, 8311, 8851]
    
    sky_wav_min += [9035, 8309, 6862, 7977]
    sky_wav_max += [9039, 8312, 6865, 7980]
    
    cutA = mask_out_pixels(cutA.copy(), out_mask=[sky_wav_min, sky_wav_max])
    cutB = mask_out_pixels(cutB.copy(), out_mask=[sky_wav_min, sky_wav_max])
    cutC = mask_out_pixels(cutC.copy(), out_mask=[sky_wav_min, sky_wav_max])
    
    # Mask out bad pixels
    _, mask_bad_pxA = remove_bad_pixels(cutA['flux'])
    _, mask_bad_pxC = remove_bad_pixels(cutC['flux'])
    _, mask_bad_pxB = remove_bad_pixels(cutB['flux'])
    cutA['mask'] = (cutA['mask'] & ~mask_bad_pxA)
    cutC['mask'] = (cutC['mask'] & ~mask_bad_pxC)
    cutB['mask'] = (cutB['mask'] & ~mask_bad_pxB)
    
    # Precise cut off and centerization
    emission = EmissionProcessor(cutA.copy(), 
                                 cutC.copy(), 
                                 cutB.copy(), redshift)
    (spec2dA, spec2dC, spec2dB), how_cut0 = emission.process(
        line = linename, 
        this_line_waves = this_emis_waves, 
        spec_width = spec_width, 
        spec_height = spec_height
        )
    
    # Now, let users decide to accept or reject this cut
    try:
        specset = [spec2dA, spec2dC, spec2dB]
        setnames = ['A', 'C', 'B']
        colors   = ['orangered', 'cyan', 'gold']
        fig, axs = plt.subplots(3, 1, figsize=(4, 4*3)) # (length, height)
        for i in range(3):
            spec_data = specset[i]['flux']
            spec_data = np.where(specset[i]['mask'], spec_data, np.nan)
            
            noise = np.nanstd(spec_data)
            ny, nx = spec_data.shape
            xmin =  specset[i]['wave'][0, 0] # x_left
            xmax =  specset[i]['wave'][0,-1] # x_right
            ymin = -len(spec_data)/2  # y_bottom
            ymax = +len(spec_data)/2 # y_top
            
            ax1 = axs[i]
            im_spec = ax1.imshow(spec_data, 
                                 extent=[xmin, xmax, ymin, ymax],
                                 cmap='viridis', aspect='auto', origin='lower', 
                                 vmin=0-noise, vmax=0 + 5*noise
                                 )
            fig.colorbar(im_spec, ax=ax1)
            ax1.xaxis.get_major_formatter().set_useOffset(False)
            ax1.set_ylabel(r'Spatial Position (px)')
            ax1.grid(linestyle=':', color='orangered', alpha=0.5)
            ax1.set_title(f'Set {setnames[i]}', 
                          color=colors[i], size=15)
        plt.show(block=False)
        plt.pause(0.1)
        fail_pre = False
    except IndexError:
        fail_pre = True
    
    # After the user has made selection...
    if fail_pre==True or input('''=========================================
----------- Reject this cut? ------------ 
\u25A1 Yes - Press "Enter" to manually redo. (default)
\u25A1 No  - Type "ok" to skip manual redo and accept it. 
Your decision: ''')=='':
        plt.close()
        idx_xys = np.array([
            manual_correct(cutA.copy(), spec_width, spec_height, 
                           tag=f'{linename}, A'), 
            manual_correct(cutC.copy(), spec_width, spec_height, 
                           tag=f'{linename}, C'), 
            manual_correct(cutB.copy(), spec_width, spec_height, 
                           tag=f'{linename}, B')
            ]) # [[ix, iy], ...]
        
        # idx_xys = np.array([
        #     [18, 33], [18, 33], [18, 33]
        #     ]) # [[ix, iy], ...]
        
        emission_ = EmissionProcessor(cutA.copy(),
                                      cutC.copy(), 
                                      cutB.copy(), redshift)
        (spec2dA, spec2dC, spec2dB), how_cut0 = emission_.process(
            line = linename, 
            this_line_waves = this_emis_waves, 
            spec_width = spec_width, 
            spec_height = spec_height, 
            fac_max = 1.00, # mask_bulge_pixels param
            idx_xs=idx_xys[:,0], idx_ys=idx_xys[:,1], # Set A, C, B
            )
    
    else:
        plt.close()
    
    # if any (x0, y0) is bad, which is outside spec2dX.shape,
    # we withdraw cut_emi_center() result, only use cutoffspec() result.
    bad_x0y0 = check_x0y0_in_cropped([spec2dA.copy(), 
                                      spec2dC.copy(), 
                                      spec2dB.copy()], how_cut0.copy())
    
    spec2dA = build_2d_continuum(spec2dA, smooth=9, verbose=False)
    spec2dC = build_2d_continuum(spec2dC, smooth=9, verbose=False)
    spec2dB = build_2d_continuum(spec2dB, smooth=9, verbose=False)
    
    # if cutting is good...
    if len(bad_x0y0) == 0: 
        meta_gal = redshift_update(spec2dA.copy(), 
                                   spec2dC.copy(), 
                                   spec2dB.copy(), redshift, 
                                   how_cut0.copy(), 
                                   linename, 
                                   this_emis_waves,
                                   meta_gal.copy())
        
        # Change None to linename
        meta_spec_A['line_species'] = linename
        meta_spec_B['line_species'] = linename
        meta_spec_C['line_species'] = linename
            
        # Flip slit position axis (see spec_model.py)
        # spec2dA_flip, spec2dC_flip, spec2dB_flip = {}, {}, {}
        # for key in spec2dA.keys():
        #     spec2dA_flip[key] = np.flip(spec2dA[key], axis=0)
        #     spec2dC_flip[key] = np.flip(spec2dC[key], axis=0)
        #     spec2dB_flip[key] = np.flip(spec2dB[key], axis=0)
        
        # Flipped how_cut0
        # how_cut_line_f = {}
        # spec_raws = [specA,        specC,        specB]
        # spec_cuts = [spec2dA_flip, spec2dC_flip, spec2dB_flip]
        # for i in range(3):
        #     N_slit       = spec_raws[i]['flux'].shape[0] # raw N_slit from specA
        #     N_slit_after = spec_cuts[i]['flux'].shape[0]
        #     how_cut_line_f[f'Set{i}'] = {
        #         'UP': N_slit - how_cut0[f'Set{i}']['DN'], 
        #         'DN': N_slit - how_cut0[f'Set{i}']['UP'], 
        #         'LF':          how_cut0[f'Set{i}']['LF'], 
        #         'RT':          how_cut0[f'Set{i}']['RT'], 
        #         'up_flt': N_slit - how_cut0[f'Set{i}']['dn_flt'], 
        #         'dn_flt': N_slit - how_cut0[f'Set{i}']['up_flt'], 
        #         'lf_flt':          how_cut0[f'Set{i}']['lf_flt'], 
        #         'rt_flt':          how_cut0[f'Set{i}']['rt_flt'], 
        #         'line1': {
        #             'y0': N_slit_after - how_cut0[f'Set{i}']['line1']['y0'],
        #             'x0':                how_cut0[f'Set{i}']['line1']['x0'] 
        #             }, 
        #         'line2': {
        #             'y0': N_slit_after - how_cut0[f'Set{i}']['line2']['y0'],
        #             'x0':                how_cut0[f'Set{i}']['line2']['x0'] 
        #             }, 
        #         'fit_par': how_cut0[f'Set{i}']['fit_par'], 
        #         }
        
        # Positive flux mask
        # postiveA = np.where(spec2dA['flux']>=0, True, False)
        # postiveC = np.where(spec2dC['flux']>=0, True, False)
        # postiveB = np.where(spec2dB['flux']>=0, True, False)
        # spec2dA['mask'] = (spec2dA['mask'] & postiveA)
        # spec2dC['mask'] = (spec2dC['mask'] & postiveC)
        # spec2dB['mask'] = (spec2dB['mask'] & postiveB)
        
        real_data_this_line_all_sets = real_data_pack(
            spec2dA, spec2dC, spec2dB,
            meta_image, image_data, image_var, image_varr, 
            meta_spec_A, meta_spec_C, meta_spec_B
            )

    # if bad line, give a warning
    else: 
        bad_lines.append([linename, None])
        print(f'Cutting for emission line: {linename} Failed.')
    
    return real_data_this_line_all_sets, how_cut0
    
    
def check_x0y0_in_cropped(ls, howcut):
    """
        This case, each set's line center position, e.g. (x0=10, y0=10),
    doesn't agree with other set's so much (x0=4, y0=3) that their 
    cropped sizes (first: 0<x<20, 0<y<20; second: 0<x<8, 0<y<6) have 
    very big discrepancies!
        We say it failed to find emission line center and needs to be 
    corrected.
    """
    max_padding = 1 # px. Discard if too close to spec edge
    
    # Spec in each set should be same-sized
    # If not, we have a check after this block
    specshapes, x0y0s = [], []
    for i, spec in enumerate(ls): # For each set
        spec_shape = spec['flux'].shape
        specshapes.append(spec_shape)
        # choose the "line" key
        # singleline = None
        for key, val in howcut[f"Set{i}"].items():
            if key[:4]=='line':
                # if singleline is None:
                #     singleline = True
                # elif singleline: # This is line2
                #     singleline = False
                x0 = howcut[f"Set{i}"][key]['x0']
                y0 = howcut[f"Set{i}"][key]['y0']
                x0y0s.append([x0, y0])
    
    # check spec_shape
    for i in range(1, len(specshapes)):
        if specshapes[i] != specshapes[0]:
            print(f'Found spec in {len(specshapes)} sets have different shapes.⚠️')
            print(specshapes)
    
    # check x0y0
    bad_x0y0 = {}
    for i in range(len(x0y0s)):
        x0 = x0y0s[i][0]
        y0 = x0y0s[i][1]
        spec_shape = specshapes[0]
        if (y0 >= spec_shape[0]-max_padding) or (x0 >= spec_shape[1]-max_padding):
            bad_x0y0[f'set{i}'] = {'spec_shape': spec_shape, 
                                   'bad_x0': x0, 'bad_y0': y0}
    
    return bad_x0y0


def redshift_update(spec2dA, spec2dC, spec2dB, redshift, 
                    how_cut, linename, this_line_waves,
                    meta_gal):
    # Measure v0 and pass to fiducial params in spec fitting
    spec_ls = [spec2dA, spec2dC, spec2dB]
    v0s     = measure_v0(spec_ls, redshift, how_cut, linename, this_line_waves)
    
    # Correct redshift
    z = []
    for setname, setdict in v0s.items():
        for linename, linedict in setdict.items():
            z.append(linedict['z0'])
    redshift = np.mean(z)
    
    # Update v0s by new redshift
    v0s = measure_v0(spec_ls, redshift, how_cut, linename, this_line_waves)
    meta_gal['redshift'    ] = redshift
    meta_gal['line_centers'] = v0s
    
    return meta_gal


def measure_v0(spec_ls, redshift, how_cut, linename, this_line_waves):
    v0s, i = {}, 0
    for setname, setdict in how_cut.items():
        v0s[setname] = {}
        for key, linedict in setdict.items():
            for j in range(len(this_line_waves)):
                if key == f'line{j+1}':
                    v0s[setname][key] = {}
                    x0,    y0    = linedict['x0'], linedict['y0']
                    x0_lf, y0_up = int(x0),   int(y0)
                    x0_rt, y0_dn = int(x0)+1, int(y0)+1
                    
                    try:
                        wave_lf  = spec_ls[i]['wave'][y0_up, x0_lf]
                        wave_rt  = spec_ls[i]['wave'][y0_dn, x0_rt]
                    except IndexError:
                        fig = plt.figure(figsize=(6, 6))
                        ax  = fig.add_subplot()
                        f   = spec_ls[i]['flux']
                        im1 = ax.imshow(f)
                        fig.colorbar(im1)
                        ax.text(0, 0,         f'   {spec_ls[i]["wave"][0,0]:.1f}', ha='left',  va='top', color='red')
                        ax.text(len(f[0]), 0, f'{spec_ls[i]["wave"][0,-1]:.1f}  ', ha='right', va='top', color='red')
                        ax.scatter(0, 0,         marker='x', s=200, color='red', zorder=1)
                        ax.scatter(len(f[0]), 0, marker='x', s=200, color='red', zorder=1)
                        ax.scatter(y0_up, x0_lf, marker='x', s=200, color='red', zorder=1)
                        plt.title(f'Line {linename} in {setname}')
                        plt.grid()
                        plt.show()
                        raise IndexError(
    f"""Want to know exact central wavelength by 
    accessing the left/right edges of the wave axis pixel 
    at index = (x, y) = ({y0_up}, {x0_lf}) and ({y0_dn}, {x0_rt}). 
    However, the wave array size = (ny, nx) = {spec_ls[i]['wave'].shape} from {linename} {setname}.
    This line center may be TOO CLOSE to spec edges and 
    other set(s) have very different center(s) so this 
    center has been cropped. See plot for thi reason. """
    )
                    thiswave     = wave_lf + (x0 - x0_lf) * (wave_rt - wave_lf)
                    oldwave_rest = this_line_waves[j]
                    oldwave      = oldwave_rest * (1 + redshift)
                    z0 = thiswave / oldwave_rest - 1
                    v0 = (thiswave - oldwave) / thiswave * 3e5
                    v0s[setname][key]['x0'  ] = x0
                    v0s[setname][key]['y0'  ] = y0
                    v0s[setname][key]['wave_old'      ] = oldwave
                    v0s[setname][key]['wave_corrected'] = thiswave
                    v0s[setname][key]['z0'  ] = z0
                    v0s[setname][key]['v0'  ] = v0
                # end if
        i += 1
    return v0s


def resolve_bad_lines(bad_lines, 
                      real_data_all_lines_sets, how_cut, 
                      emilines, 
                      meta_gal, 
                      specA, specC, specB,
                      meta_spec_A, meta_spec_C, meta_spec_B,
                      meta_image, image_data, image_var,
                      ):
    # if bad line...
    for bad_line in bad_lines:
        # For this bad line
        linename, _ = bad_line
        print(f'Cutting for emission line: {linename} (Retry)...')
        
        # Look up indices and redshifts
        sets_y0 = {'UP': [], 'DN': [] }
        for _, anylinesets in how_cut.items():
            for setname, thissetcutparam in anylinesets.items():
                up_crop   = thissetcutparam['UP']
                down_crop = thissetcutparam['DN']
                sets_y0['UP'].append(up_crop)
                sets_y0['DN'].append(down_crop)
            break # only one line is enough
        sets_y0['UP'] = int(np.mean(sets_y0['UP']))
        sets_y0['DN'] = int(np.mean(sets_y0['DN']))
        
        # Again, do non-spatial crop but with smaller sizes
        this_emis_waves = emilines[linename]
        redshift = meta_gal['redshift']
        wave = np.mean(this_emis_waves) * (1 + redshift)
        cutA, cutB, cutC, cutmask = cutoffspec(specA.copy(), 
                                               specB.copy(), 
                                               specC.copy(), wave, 
                                               width=30) #A, 0.6 A/px
        
        all_cuts = [cutA, cutC, cutB]
        for i, cut in enumerate(all_cuts):
            for key, spec2d in cut.items():
                all_cuts[i][key] = spec2d[sets_y0['UP']: sets_y0['DN']]
        
        # Change None to linename
        meta_spec_A['line_species'] = linename
        meta_spec_B['line_species'] = linename
        meta_spec_C['line_species'] = linename
        
        # mask_min_max
        # all_cuts[0]['flux'], all_cuts[0]['ivar'] = mask_min_max_flux(
        #     all_cuts[0]['flux'].copy(), 
        #     all_cuts[0]['ivar'].copy(), 
        #     flux_min=0,
        #     flux_min_to=0,
        #     )
        # all_cuts[2]['flux'], all_cuts[2]['ivar'] = mask_min_max_flux(
        #     all_cuts[2]['flux'].copy(),                                    
        #     all_cuts[2]['ivar'].copy(), 
        #     flux_min=0,
        #     flux_min_to=0,
        #     )
        # all_cuts[1]['flux'], all_cuts[1]['ivar'] = mask_min_max_flux(
        #     all_cuts[1]['flux'].copy(), 
        #     all_cuts[1]['ivar'].copy(), 
        #     flux_min=0,
        #     flux_min_to=0,
        #     )
        
        real_data_this_line_all_sets = real_data_pack(
            all_cuts[0].copy(), all_cuts[1].copy(), all_cuts[2].copy(),
            meta_image.copy(),  image_data.copy(),  image_var.copy(),
            meta_spec_A.copy(), meta_spec_C.copy(), meta_spec_B.copy()
            )
        
        # Packaging for all emission lines
        real_data_all_lines_sets[linename] = real_data_this_line_all_sets
        print(f'Cutting for emission line: {linename} (Retry) finished. OK.')
        
    return real_data_all_lines_sets

# def find_line_sigma(spec2dX, howcut, line):
#     mask = spec2dX['spec_mask']
#     arr  = spec2dX['spec_data']
#     ny, nx = arr.shape
#     y0     = int(howcut['line1']['y0'])
#     mean1  = howcut['line1']['x0']
#     mean2  = howcut['line2']['x0']
#     best_std1 = np.ones(ny)
#     best_std2 = np.ones(ny)
#     best_amp1 = np.ones(ny)
#     best_amp2 = np.ones(ny)
    
#     # correct sky line region's flux by assigning a median
#     arr  = np.where(mask, arr, np.median(arr))
    
#     def _gaussian(xx, std, amp):
#         return amp * np.exp(-0.5 * ((xx - mean1) / std) ** 2)
    
#     def _double_gaussian(xx, std1, amp1, std2, amp2):
#         yy1 = amp1 * np.exp(-0.5 * ((xx - mean1) / std1) ** 2)
#         yy2 = amp2 * np.exp(-0.5 * ((xx - mean2) / std2) ** 2)
#         return yy1 + yy2
    
#     from scipy.optimize import curve_fit
#     fit_func = _double_gaussian if line == "O2" else _gaussian
#     bound1 = ((     1,      0), 
#               (np.inf, np.inf) )
#     bound2 = ((     1,      0,      1,      0), 
#               (np.inf, np.inf, np.inf, np.inf) )
#     bounds = bound2 if line == "O2" else bound1
    
#     # First fit center line over wavelength px
#     try: 
#         popt, _ = curve_fit(fit_func, 
#                             np.arange(nx), 
#                             arr[y0], 
#                             bounds=bounds, maxfev=100)
#         best_std1[y0] = popt[0]
#         best_amp1[y0] = popt[1]
#         best_std2[y0] = popt[2] if line == "O2" else popt[0]
#         best_amp2[y0] = popt[3] if line == "O2" else 0
#     except RuntimeError:
#         best_std1[y0] = mean1
#         best_amp1[y0] = np.max(arr)
#         best_std2[y0] = mean2       if line == "O2" else mean1
#         best_amp2[y0] = np.max(arr) if line == "O2" else 0
    
#     # Try other lines from center to edges
#     for y in np.hstack((np.arange(y0, ny), np.arange(0, y0)[::-1])):
#         try:
#             popt, _ = curve_fit(fit_func, 
#                                 np.arange(nx), 
#                                 arr[y], 
#                                 bounds=bounds, maxfev=100)
#             best_std1[y] = popt[0]
#             best_amp1[y] = popt[1]
#             best_std2[y] = popt[2] if line == "O2" else popt[0]
#             best_amp2[y] = popt[3] if line == "O2" else 0
#         except RuntimeError:
#             pass
    
#     return {'std1': best_std1, 'amp1': best_amp1, 
#             'std2': best_std2, 'amp2': best_amp2 }

