import numpy as np
import astropy.units as u
from   scipy.ndimage  import median_filter

from  data_structure   import SpecData
from .build_var          import build_var
from .continuum          import extract_2d_continuum
from .line_center_crop   import EmissionProcessor
from .line_manual_fix    import manual_correct
from klm.line_width_profile import find_line_sigma
from .crop_plan          import prepare_rough, make_plan, replay_plan, save_plan

import matplotlib.pyplot as plt

_spec_idx_map    = ['A', 'C', 'B']
_spec_colors_map = ['orangered', 'cyan', 'gold']


def process_single_line(specs, lineinfo, meta_raws,
                        spec_width, spec_height,
                        clear_mode='simple', cont_y0s=None,
                        cont_scales=None, manual=True, crop_plan=None, plan_path=None):

    sources = [specs.specA, specs.specC, specs.specB]
    cuts, rough_columns = prepare_rough(sources, lineinfo.wav, lineinfo.redshift)
    cutA, cutC, cutB = cuts
    initial_redshift = lineinfo.redshift
    manual_clicks = None

    if crop_plan is None:
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
                manual_clicks = idx_xys.tolist()
                spec2dACB, how_cut0 = emission_before_cut.process(
                    line = lineinfo.name,
                    this_line_waves = lineinfo.wav,
                    spec_width = spec_width,
                    spec_height = spec_height,
                    idx_xs=idx_xys[:,0], idx_ys=idx_xys[:,1], # Set A, C, B
                    )

            else:
                plt.close()

        # 4.2) Keep the automatic fit when manual review is disabled.
        else:
            plt.close()

        # Precise cutting ends here
        # ---------------------------------------------------------------------

        # 5. Update redshift in lineinfo
        redshift = redshift_update(
            spec2dACB, lineinfo.redshift,
            how_cut0, lineinfo.wav,
            )
        lineinfo.redshift = redshift
        if plan_path is not None:
            plan_record = make_plan(lineinfo.name, lineinfo.wav, initial_redshift,
                                    spec_width, spec_height, sources, rough_columns,
                                    how_cut0, redshift, clicks=manual_clicks)
            save_plan(plan_path, plan_record)

    else:
        if (crop_plan['name'] != lineinfo.name or
                not np.allclose(crop_plan['line_waves_angstrom'], lineinfo.wav) or
                not np.isclose(crop_plan['initial_redshift'], lineinfo.redshift)):
            raise ValueError('Saved crop plan does not match requested line or redshift')
        spec2dACB = replay_plan(sources, crop_plan, spec_width, spec_height)
        lineinfo.redshift = crop_plan['corrected_redshift']

    # 6. Non-cutting processing
    # 6.1) Extract continuum AND SUBSTRACT
    print('Continuums are being extracted and substracting data...')

    processed = []
    for i, cropped in enumerate(spec2dACB):
        coverage = cropped['coverage']
        rows, cols = np.where(coverage)
        if len(rows) == 0:
            raise ValueError(f'No observed pixels in Set {_spec_idx_map[i]}')
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        inner = np.s_[y0:y1, x0:x1]
        valid_flux = cropped['flux'][inner]
        base_mask = cropped['mask'][inner]
        cont = extract_2d_continuum(
            valid_flux, base_mask,
            cont_scale=(cont_scales[i] if cont_scales is not None else 1),
            cont_y0=(cont_y0s[i] - y0 if cont_y0s is not None else None),
            mode=clear_mode, verbose=False, smooth=9)
        residual = valid_flux - cont
        variance, final_mask = build_var(residual, base_mask, option=2)
        wave_step = np.mean(np.diff(cropped['wave'][y0, x0:x1]))
        profile = find_line_sigma(
            median_filter(residual, size=3), lineinfo.name,
            lambda_scale=wave_step)
        full_shape = cropped['flux'].shape
        full_cont = np.zeros(full_shape, dtype=cont.dtype)
        full_flux = np.zeros(full_shape, dtype=residual.dtype)
        full_var = np.zeros(full_shape, dtype=variance.dtype)
        full_mask = np.zeros(full_shape, dtype=bool)
        full_cont[inner], full_flux[inner] = cont, residual
        full_var[inner], full_mask[inner] = variance, final_mask
        if x0:
            for component in profile:
                component[1:, 0] = (component[1:, 0].astype(float) + x0).astype(str)
        if y0 or y1 != full_shape[0]:
            padded_profile = []
            for component in profile:
                out = np.zeros((full_shape[0] + 1, component.shape[1]),
                               dtype=component.dtype)
                out[0] = component[0]
                out[1 + y0:1 + y1] = component[1:]
                padded_profile.append(out)
            profile = tuple(padded_profile)
        processed.append((full_flux, full_cont, full_var, full_mask, profile))
    (fluxA, contA, varA, mask_addedA, line_profile_A) = processed[0]
    (fluxC, contC, varC, mask_addedC, line_profile_C) = processed[1]
    (fluxB, contB, varB, mask_addedB, line_profile_B) = processed[2]

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
                    meta = meta_specA,
                    mask_base = spec2dACB[0]['mask'],
                    coverage_mask = spec2dACB[0]['coverage']
                    )
    SetC = SpecData(data = fluxC,
                    var  = varC,
                    cont = contC,
                    mask = mask_addedC,
                    meta = meta_specC,
                    mask_base = spec2dACB[1]['mask'],
                    coverage_mask = spec2dACB[1]['coverage']
                    )
    SetB = SpecData(data = fluxB,
                    var  = varB,
                    cont = contB,
                    mask = mask_addedB,
                    meta = meta_specB,
                    mask_base = spec2dACB[2]['mask'],
                    coverage_mask = spec2dACB[2]['coverage']
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

