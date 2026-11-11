import numpy as np
import pytest

from data_prep.reestimate_variance import reestimate
from data_prep.spec.crop_plan import (make_plan, prepare_rough, read_plans,
                                      replay_plan, save_plan)
from data_prep.spec.line_center_crop import EmissionLineFitter, SpectrumCropper


def _sources():
    wave = np.tile(np.arange(100., 200.), (20, 1))
    y, x = np.mgrid[:20, :100]
    flux = 20 * np.exp(-0.5 * ((x - 50) / 2) ** 2) * np.exp(-0.5 * ((y - 5) / 3) ** 2)
    return [{'flux': flux.copy(), 'wave': wave.copy(),
             'mask': np.ones(flux.shape, dtype=bool)} for _ in range(3)]


def test_fixed_shape_and_plan_replay(tmp_path):
    sources = _sources()
    cuts, columns = prepare_rough(sources, [150.], 0.)
    how_cut = {f'Set{i}': {'UP': -2, 'DN': 18, 'LF': -3, 'RT': 17,
                          'line1': {'x0': 10., 'y0': 7.}}
               for i in range(3)}
    direct = SpectrumCropper.apply_bounds(cuts, how_cut)
    assert all(cut['flux'].shape == (20, 20) for cut in direct)
    assert all(not cut['mask'][0].any() and not cut['mask'][:, 0].any()
               for cut in direct)
    assert all(np.all(cut['flux'][~cut['coverage']] == 0) for cut in direct)
    plan = make_plan('test', [150.], 0., 20, 20, sources, columns, how_cut, 0.1)
    path = tmp_path / 'crops.json'
    save_plan(path, plan)
    replay = replay_plan(sources, read_plans(path)['test'])
    for actual, expected in zip(replay, direct):
        for field in ('flux', 'wave', 'mask', 'coverage'):
            np.testing.assert_array_equal(actual[field], expected[field])
    resized = replay_plan(sources, plan, width=14, height=12)
    assert all(cut['flux'].shape == (12, 14) for cut in resized)
    assert all(cut['coverage'].any() for cut in resized)
    sources[0]['flux'][0, 0] += 1
    with pytest.raises(ValueError, match='source mismatch'):
        replay_plan(sources, plan)


def test_o2_uses_wavelength_step_for_automatic_and_manual_fit():
    rest = [3727.092, 3729.875]
    redshift, step = 0.2, 0.61
    separation = (rest[1] - rest[0]) * (1 + redshift) / step
    x = np.arange(70)
    row = 80 * np.exp(-0.5 * ((x - 30) / 2) ** 2)
    row += 70 * np.exp(-0.5 * ((x - 30 - separation) / 2) ** 2)
    flux = np.tile(row, (12, 1))
    fitter = EmissionLineFitter(redshift, 'O2', rest)
    _, automatic = fitter.fit_1d(flux, wave_per_pixel=step)
    _, manual = fitter.fit_1d(flux, wave_per_pixel=step,
                               x0_fixed=30 + separation / 2, y0_fixed=5)
    assert fitter.delta_pix == pytest.approx(separation)
    assert automatic[2] == pytest.approx(30, abs=0.5)
    assert manual[2] == pytest.approx(30, abs=0.5)


def test_variance_recalculation_uses_base_mask():
    data = np.ones((10, 12), dtype=float)
    coverage = np.ones(data.shape, dtype=bool)
    coverage[0] = False
    base = coverage.copy()
    base[4, 5] = False
    info = {'spec': [{'data': data, 'var': np.zeros_like(data),
                      'mask': np.zeros_like(base), 'mask_base': base,
                      'coverage_mask': coverage}],
            'image': {'data': np.arange(100.).reshape(10, 10),
                      'var_raw': np.ones((10, 10)),
                      'var': np.zeros((10, 10)),
                      'mask': np.ones((10, 10), dtype=bool)}}
    result = reestimate(info)
    assert not result['spec'][0]['mask'][0].any()
    assert not result['spec'][0]['mask'][4, 5]
    assert result['spec'][0]['mask'][4, 6]
