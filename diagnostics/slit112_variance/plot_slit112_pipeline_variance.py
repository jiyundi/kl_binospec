"""Read-only diagnostic: replay saved crops and compare pipeline errors.

Run from repository root: python diagnostics/plot_slit112_pipeline_variance.py
No science products are modified. I is inferred from the saved pipeline formula,
not an independent measurement of raw detector counts.
"""
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_prep.spec.read_and_cutout import read_spec2d, stack_spec2d
from data_prep.spec.crop_plan import replay_plan


def main():
    raw = ROOT.parent / 'UAO-S156-23B-A383'
    out = ROOT / 'diagnostics' / 'slit112_variance'
    out.mkdir(exist_ok=True)
    plans = json.loads((ROOT / 'data_prep/slit_112_crops.json').read_text())
    saved = joblib.load(ROOT / 'scripts/binospec_pkl_new/pkl/slit_112.pkl')
    groups = {'A': ['psf_m1/231023', 'psf_m1/231218', 'psf_m1/240115'],
              'C': ['psf/231019'], 'B': ['psf_p1/231019']}
    sources, variances, counts, floors, checks = [], {}, {}, {}, []
    for name, folders in groups.items():
        inputs, vs, inferred, clipped = [], [], [], []
        for folder in folders:
            d = raw / folder
            split = d / '1d2dspecfiles_counts/spec2d.829.112.100417.fits'
            source = read_spec2d(split)
            with fits.open(split) as h:
                f = h[1].data['FLUX'][0]
            with fits.open(d / 'obj_counts_slits_lin.fits') as h:
                matches = [x for x in h[1:] if x.header.get('SLITTARG') == 100417]
                assert len(matches) == 1
                x = matches[0]
                extname = x.name
                coordinate_keys = ['SLITTARG', 'SIDE', 'SLITID', 'CRPIX1', 'CRVAL1',
                                   'CDELT1', 'CTYPE1', 'CUNIT1', 'CRPIX2', 'CRVAL2', 'CDELT2']
                coordinate_header = {k: x.header.get(k) for k in coordinate_keys}
                assert f.shape == x.data.shape
                # Split export clips extreme negative values to -200.
                assert np.array_equal(f, np.maximum(x.data, -200), equal_nan=True), folder
                clipped_negative_count = int(np.sum(np.isfinite(x.data) & (x.data < -200)))
            with fits.open(d / 'obj_counts_err_slits_lin.fits') as h:
                e = np.array(h[extname].data, dtype=float)
                assert h[extname].header['SLITTARG'] == 100417
                assert all(h[extname].header.get(k) == value
                           for k, value in coordinate_header.items())
                assert e.shape == f.shape
            v = e**2
            # The pipeline floors variance at 1; I cannot be inverted there.
            floor = (v <= 1.000001) | ~np.isfinite(v)
            ii = np.where(floor, np.nan, v - 16.)
            checks.append({'folder': folder, 'extension': extname,
                           'split_flux_identical_after_negative_200_clip': True,
                           'negative_200_clipped_pixels': clipped_negative_count,
                           'counts_err_coordinate_headers_match': True,
                           'shape': list(f.shape)})
            inputs.append(source); vs.append(v); inferred.append(ii); clipped.append(floor)
        if name == 'A':
            source = {'flux': stack_spec2d(*(s['flux'] for s in inputs)),
                      'wave': np.tile(inputs[0]['wave'][0], (55, 1))}
            assert all(s['flux'].shape == (55, 5632) for s in inputs)
            assert all(np.array_equal(s['wave'], inputs[0]['wave']) for s in inputs)
        else:
            source = inputs[0]
        n = len(inputs)
        # Independent exposures: variance of their arithmetic mean.
        variances[name] = np.flip(np.sum(vs, axis=0) / n**2, axis=0)
        counts[name] = np.flip(np.sum(inferred, axis=0) / n**2, axis=0)
        floors[name] = np.flip(np.any(clipped, axis=0), axis=0)
        source = {k: np.flip(v, axis=0) for k, v in source.items()}
        sources.append(source)
    stats, arrays = [], {}
    for line, plan in plans.items():
        crops = replay_plan(sources, plan)  # Checks all source SHA256 digests.
        fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True)
        for idx, name in enumerate(groups):
            crop = crops[idx]
            record = plan['sets'][name]
            b = record['bounds']; cols = record['rough_columns'][b['LF']:b['RT']]
            ys = slice(b['UP'], b['DN'])
            v = variances[name][ys][:, cols]
            ii = counts[name][ys][:, cols]
            floor = floors[name][ys][:, cols]
            spec = next(s for j, s in enumerate(saved['spec'])
                        if s['meta']['line_species'] == line and j % 3 == idx)
            # Verify PKL is the same crop before using its continuum and mask.
            assert np.allclose(spec['data'] + spec['cont'], crop['flux'], atol=1e-5)
            residual = spec['data']; mask = crop['mask'] & ~floor & np.isfinite(v)
            edge = mask.copy(); ny, nx = mask.shape
            edge[ny//4:ny-ny//4, nx//4:nx-nx//4] = False
            old = residual + spec['var']
            z = residual[edge] / np.sqrt(v[edge])
            row = {'line': line, 'set': name, 'valid_pixels': int(mask.sum()),
                   'floor_pixels': int(floor.sum()),
                   'median_I_effective': float(np.nanmedian(ii[mask])),
                   'median_pipeline_variance': float(np.median(v[mask])),
                   'median_pipeline_sigma': float(np.median(np.sqrt(v[mask]))),
                   'median_current_variance': float(np.median(old[mask])),
                   'edge_residual_std': float(np.std(residual[edge])),
                   'edge_normalized_mean': float(np.mean(z)),
                   'edge_normalized_std': float(np.std(z)),
                   'edge_pixels': int(edge.sum())}
            stats.append(row)
            panels = [(residual, 'Line data (continuum subtracted)', 'counts'),
                      (ii, 'I = err^2 - 16' if name != 'A' else 'Effective I = sum(I_j) / 9', 'count contribution'),
                      (v, 'Candidate total variance (err^2)', 'counts^2')]
            wave = crop['wave'][0]
            extent = [wave[0], wave[-1], -.5, ny-.5]
            for j, (arr, title, label) in enumerate(panels):
                valid = arr[mask]
                lo, hi = np.nanpercentile(valid, [2, 98])
                im = axes[idx, j].imshow(np.where(mask, arr, np.nan), origin='lower',
                    aspect='auto', extent=extent, vmin=lo, vmax=hi, cmap='viridis')
                axes[idx, j].set_title(f'Set {name}: {title}')
                axes[idx, j].set_xlabel('Wavelength (Angstrom; split-file grid)')
                axes[idx, j].set_ylabel('Crop row (flipped)')
                fig.colorbar(im, ax=axes[idx, j], label=label, shrink=.8)
            ax = axes[idx, 3]
            ax.plot(wave, np.nanmedian(np.where(mask, v, np.nan), axis=0), label='Pipeline total variance')
            ax.plot(wave, np.nanmedian(np.where(mask, old, np.nan), axis=0), label='Current data + var')
            ax.set_xlabel('Wavelength (Angstrom)'); ax.set_ylabel('Variance (counts^2)')
            ax.set_title(f'Medians; edge std(data/sigma)={np.std(z):.2f}', fontsize=10)
            ax.legend(fontsize=8); ax.grid(alpha=.2)
            for key, value in [('line_data', residual), ('I_effective', ii),
                               ('pipeline_variance', v), ('mask', mask), ('wave', wave)]:
                arrays[f'{line}_{name}_{key}'] = value
        fig.suptitle(f'Slit 112 / {line}: exact saved crops; A is a mean of 3 independent observations\n'
                     'I inferred using local pipeline formula; variance-floor and bad pixels excluded. '
                     'Image colors clipped to 2-98 percentiles.', fontsize=12)
        fig.savefig(out / f'{line}_comparison.png', dpi=160, bbox_inches='tight')
        fig.savefig(out / f'{line}_comparison.pdf', bbox_inches='tight')
        plt.close(fig)
    report = {'mapping_checks': checks, 'source_hashes_verified': True,
              'pkl_crop_verified': True, 'statistics': stats,
              'caveats': ['I is formula-inferred, not directly observed raw counts.',
                          'A uses sum(variance_j)/9, not mean(variance_j). Its inferred I is an effective variance contribution.',
                          'Edge pixels may contain line wings, continuum-fit residuals, and correlated resampling noise.',
                          'Local pipeline source may differ from historical execution version.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'crops.npz', **arrays)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
