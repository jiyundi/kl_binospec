"""Portable, GalSim-free spectrum crop plans.

Coordinates refer to the stacked/flipped full spectrum used by data_prep.
The rough-cut column list maps them back to the full spectrum exactly.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .line_center_crop import SpectrumCropper
from .masking import bad_pixel_mask, mask_out_sky_lines

SETS = ('A', 'C', 'B')
PLAN_VERSION = 1


def prepare_rough(specs, line_waves, redshift):
    """Use the same 42-Angstrom window and masks for fitting and replay."""
    observed = np.mean(line_waves) * (1 + redshift)
    cuts, columns = [], []
    for source in specs:
        indices = np.flatnonzero((source['wave'][0] >= observed - 21) &
                                 (source['wave'][0] <= observed + 21))
        if len(indices) < 2:
            raise ValueError('Emission line has fewer than two wavelength pixels')
        cut = {key: value[:, indices].copy() for key, value in source.items()}
        valid = cut['flux'] > -100
        no_sky = mask_out_sky_lines(cut, valid)
        good_pixel = ~bad_pixel_mask(np.where(no_sky, cut['flux'], 0))
        cut['mask'] = no_sky & good_pixel & valid
        cuts.append(cut)
        columns.append(indices)
    return cuts, columns


def source_digest(source):
    """Detect a changed stacked/flipped input before replaying a plan."""
    digest = hashlib.sha256()
    for key in ('flux', 'wave'):
        arr = np.ascontiguousarray(source[key])
        digest.update(str(arr.shape).encode('ascii'))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_plan(name, line_waves, redshift, spec_width, spec_height,
              sources, columns, how_cut, corrected_redshift, clicks=None):
    sets = {}
    for i, set_name in enumerate(SETS):
        fit = how_cut[f'Set{i}']
        first_col = int(columns[i][0])
        lines = {}
        for j in range(len(line_waves)):
            local = fit[f'line{j+1}']
            lines[f'line{j+1}'] = {
                'full_x': float(local['x0'] + fit['LF'] + first_col),
                'full_y': float(local['y0'] + fit['UP']),
                'crop_x': float(local['x0']),
                'crop_y': float(local['y0']),
            }
        sets[set_name] = {
            'source_shape': list(sources[i]['flux'].shape),
            'source_sha256': source_digest(sources[i]),
            'rough_columns': columns[i].tolist(),
            'bounds': {key: int(fit[key]) for key in ('UP', 'DN', 'LF', 'RT')},
            'lines': lines,
            'fit_par': _plain(fit.get('fit_par')),
            'click_rough_xy': None if clicks is None else _plain(clicks[i]),
        }
    return {
        'version': PLAN_VERSION,
        'name': name,
        'line_waves_angstrom': list(map(float, line_waves)),
        'initial_redshift': float(redshift),
        'corrected_redshift': float(corrected_redshift),
        'requested_shape': [int(spec_height), int(spec_width)],
        'coordinate_frame': 'stacked/flipped full spectrum, numpy zero-based',
        'sets': sets,
    }


def redshift_from_crop(cropped, how_cut, line_waves):
    """Preserve the existing wavelength interpolation used by main.py."""
    measured = []
    for i in range(len(SETS)):
        for j, rest_wave in enumerate(line_waves):
            center = how_cut[f'Set{i}'][f'line{j+1}']
            x, y = center['x0'], center['y0']
            xi, yi = int(x), int(y)
            wave = cropped[i]['wave']
            if yi < 0 or xi < 0 or yi + 1 >= wave.shape[0] or xi + 1 >= wave.shape[1]:
                raise ValueError('Fitted center is too close to the crop edge for redshift')
            left = wave[yi, xi]
            right = wave[yi + 1, xi + 1]
            measured.append((left + (x - xi) * (right - left)) / rest_wave - 1)
    return float(np.mean(measured))


def replay_plan(sources, plan, width=None, height=None):
    if plan.get('version') != PLAN_VERSION:
        raise ValueError('Unsupported crop plan version')
    saved_height, saved_width = plan['requested_shape']
    width = saved_width if width is None else int(width)
    height = saved_height if height is None else int(height)
    if width <= 0 or height <= 0:
        raise ValueError('Crop dimensions must be positive')
    cuts, columns = prepare_rough(sources, plan['line_waves_angstrom'],
                                  plan['initial_redshift'])
    bounds = {}
    for i, set_name in enumerate(SETS):
        saved = plan['sets'][set_name]
        if (saved['source_shape'] != list(sources[i]['flux'].shape) or
                saved['source_sha256'] != source_digest(sources[i]) or
                saved['rough_columns'] != columns[i].tolist()):
            raise ValueError(f'Crop plan source mismatch for Set {set_name}')
        if (height, width) == (saved_height, saved_width):
            bounds[f'Set{i}'] = saved['bounds']
        else:
            center_y = saved['lines']['line1']['full_y']
            center_x = np.mean([line['full_x'] for line in saved['lines'].values()])
            cy = int(round(center_y))
            cx = int(round(center_x - int(columns[i][0])))
            up, lf = cy - height // 2, cx - width // 2
            bounds[f'Set{i}'] = {'UP': up, 'DN': up + height,
                                 'LF': lf, 'RT': lf + width}
    cropped = SpectrumCropper.apply_bounds(cuts, bounds)
    for i, set_name in enumerate(SETS):
        if list(cropped[i]['flux'].shape) != [height, width]:
            raise ValueError(f'Crop plan dimensions mismatch for Set {set_name}')
    return cropped


def read_plans(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as stream:
        plans = json.load(stream)
    if not isinstance(plans, dict):
        raise ValueError('Crop plan file must contain a line-name dictionary')
    return plans


def save_plan(path, plan):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plans = read_plans(path)
    plans[plan['name']] = _plain(plan)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(plans, stream, indent=2, allow_nan=False)
    temporary.replace(path)
