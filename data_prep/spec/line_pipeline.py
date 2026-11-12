import numpy as np
import astropy.units as u
from   scipy.ndimage import median_filter

from  data_structure import SpecData
from .build_var      import from_total_variance
from .continuum      import extract_2d_continuum
from .crop_plan      import replay_plan

from core.find_bkg import bkg_estimate
from core.find_snr import snr_estimate
from klm.line_width_profile import find_line_sigma

_spec_idx_map    = ['A', 'C', 'B']


def process_single_line(specs, lineinfo, meta_raws,
                        spec_width, spec_height,
                        clear_mode='simple', cont_y0s=None,
                        cont_scales=None, crop_plan=None):

    if crop_plan is None:
        raise ValueError(f'Manual crop plan is required for {lineinfo.name}')
    if (crop_plan['name'] != lineinfo.name or
            not np.allclose(crop_plan['line_waves_angstrom'], lineinfo.wav) or
            not np.isclose(crop_plan['initial_redshift'], lineinfo.redshift)):
        raise ValueError('Saved crop plan does not match requested line or redshift')

    sources = [specs.specA, specs.specC, specs.specB]
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
            mode=clear_mode, verbose=False, smooth=9
        )
        if np.isnan(cont).all():
            print(f'WARNING: Continuum is all NaN for Set {_spec_idx_map[i]}; skipping subtraction.')
            cont = np.zeros_like(valid_flux)
        residual = valid_flux - cont

        # Var estimated by the MMT Binospec pipeline
        variance, final_mask = from_total_variance(
            residual, base_mask, cropped['var_total'][inner]
        )
        
        # Raw counts (I) = Data (D) + Background (B0) if a good background (B0) can 
        # be estimated. Here, D := Emission line (F) + Continuum (C). A robust 
        # signal-to-noise ratio (SNR) requires an I or D, giving 
        # SNR = Sum(I or D) / SQRT(Sum(Var)). 

        # However, the MMT Binospec pipeline may subtract an inproperate background 
        # level (B0 + dB) and only D' = I - B0 - dB = D - dB is provided. One can find
        # the data has background pixels bkg_estimate(D') < 0 which hints the bias 
        # dB := -bkg_estimate(D') > 0. 

        # Hence, for a better-defined SNR, D' needs to be reverted to D:
        # D = D' + dB = D' - bkg_estimate(D')
        try: 
            bkg_est = bkg_estimate(residual)[0]
        except AssertionError: 
            print('No significant emission structure was detected. Set bkg to 0.')
            bkg_est = 0
        residual = residual - bkg_est
        spec_snr = snr_estimate(
            np.where(final_mask, residual, np.nan),
            bkg_is_subtracted=False, 
            var_given=variance
        )
        print(f'Spec SNR (Set {i}):', round(spec_snr, 1), )

        wave_step = np.mean(np.diff(cropped['wave'][y0, x0:x1]))
        profile = find_line_sigma(
            median_filter(residual, size=3), lineinfo.name,
            lambda_scale=wave_step
        )
        
        full_shape = cropped['flux'].shape
        full_cont  = np.zeros(full_shape, dtype=cont.dtype)
        full_flux  = np.zeros(full_shape, dtype=residual.dtype)
        full_var   = np.zeros(full_shape, dtype=variance.dtype)
        full_mask  = np.zeros(full_shape, dtype=bool)
        full_cont[inner], full_flux[inner] = cont, residual
        full_var[inner],  full_mask[inner] = variance, final_mask

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
                    coverage_mask = spec2dACB[0]['coverage'],
                    var_total = spec2dACB[0]['var_total']
                    )
    SetC = SpecData(data = fluxC,
                    var  = varC,
                    cont = contC,
                    mask = mask_addedC,
                    meta = meta_specC,
                    mask_base = spec2dACB[1]['mask'],
                    coverage_mask = spec2dACB[1]['coverage'],
                    var_total = spec2dACB[1]['var_total']
                    )
    SetB = SpecData(data = fluxB,
                    var  = varB,
                    cont = contB,
                    mask = mask_addedB,
                    meta = meta_specB,
                    mask_base = spec2dACB[2]['mask'],
                    coverage_mask = spec2dACB[2]['coverage'],
                    var_total = spec2dACB[2]['var_total']
                    )
    spec_data_list = [SetA, SetC, SetB]

    return spec_data_list, lineinfo


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
