"""Saved elliptical masks for neighboring sources in imaging cutouts."""

import json
from pathlib import Path

import numpy as np


PLAN_VERSION = 1
# These slits had pixel-coordinate masks before plans were introduced. Require
# a review so an old mask cannot silently disappear when its plan is absent.
LEGACY_MASK_SLITS = {8, 13, 14, 27, 28, 35, 42, 55, 58, 64, 67, 69, 70,
                     75, 91, 99, 113, 128, 132, 133, 139}


def target_pixel(wcs, ra, dec):
    """Target position in a cutout, using zero-based NumPy pixel coordinates."""
    return tuple(map(float, wcs.world_to_pixel_values(ra, dec)))


def read_mask_plan(path, slit_num, allow_missing=False):
    """Read a plan; the editor may start with an empty unsaved plan."""
    path = Path(path)
    if not path.exists():
        if slit_num in LEGACY_MASK_SLITS and not allow_missing:
            raise FileNotFoundError(
                f'Slit {slit_num} previously had a hard-coded image mask. '
                f'Review it with manual_image_mask.ipynb and save {path}.')
        return []
    with path.open(encoding='utf-8') as stream:
        plan = json.load(stream)
    if plan.get('version') != PLAN_VERSION or plan.get('slit_num') != slit_num:
        raise ValueError(f'Invalid image mask plan for slit {slit_num}: {path}')
    ellipses = plan.get('ellipses')
    if not isinstance(ellipses, list):
        raise ValueError(f'Invalid ellipses in image mask plan: {path}')
    for ellipse in ellipses:
        if not isinstance(ellipse, dict) or not all(
                key in ellipse for key in ('offset_x', 'offset_y', 'radius_x',
                                            'radius_y', 'angle_deg')):
            raise ValueError(f'Invalid ellipse in image mask plan: {path}')
        values = [ellipse[key] for key in ('offset_x', 'offset_y', 'radius_x',
                                           'radius_y', 'angle_deg')]
        if not all(isinstance(value, (int, float)) and np.isfinite(value)
                   for value in values) or min(values[2:4]) <= 0:
            raise ValueError(f'Invalid ellipse dimensions in image mask plan: {path}')
    return ellipses


def save_mask_plan(path, slit_num, ellipses):
    """Save centers relative to the target, in image pixel units."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = {'version': PLAN_VERSION, 'slit_num': int(slit_num),
            'ellipses': ellipses}
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8')


def mask_neighbor_star(image_data, image_mask, ellipses, target_xy):
    """Apply any number of rotated ellipses; True pixels remain usable."""
    if image_data.shape != image_mask.shape:
        raise ValueError('Image data and mask shapes differ')
    y, x = np.indices(image_data.shape)
    result = np.asarray(image_mask, dtype=bool).copy()
    for ellipse in ellipses:
        cx = target_xy[0] + ellipse['offset_x']
        cy = target_xy[1] + ellipse['offset_y']
        angle = np.deg2rad(ellipse['angle_deg'])
        dx, dy = x - cx, y - cy
        xr = dx * np.cos(angle) + dy * np.sin(angle)
        yr = -dx * np.sin(angle) + dy * np.cos(angle)
        inside = ((xr / ellipse['radius_x']) ** 2 +
                  (yr / ellipse['radius_y']) ** 2) <= 1
        result &= ~inside
    return result
