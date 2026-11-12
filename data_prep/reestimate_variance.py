"""Re-estimate variance from saved crops without repeating the line fit."""

import argparse
from pathlib import Path

import joblib
import numpy as np

from .image.rescale_var import rescale_var
from .spec.build_var import build_var, from_total_variance


def reestimate(data_info):
    """Update variance and the variance-dependent mask in a saved data dict."""
    for spec in data_info['spec']:
        if 'mask_base' not in spec or 'coverage_mask' not in spec:
            raise ValueError('This pkl predates saved base/coverage masks; rerun crop processing once')
        coverage = spec['coverage_mask']
        rows, cols = np.where(coverage)
        if len(rows) == 0:
            raise ValueError('Spectrum has no observed pixels')
        region = np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1]
        if spec.get('var_total') is not None:
            variance, mask = from_total_variance(
                spec['data'][region], spec['mask_base'][region],
                spec['var_total'][region])
        else:
            variance, mask = build_var(spec['data'][region],
                                      spec['mask_base'][region], option=2)
        spec['var'] = np.zeros_like(spec['data'], dtype=float)
        spec['mask'] = np.zeros_like(spec['mask_base'], dtype=bool)
        spec['var'][region] = variance
        spec['mask'][region] = mask
    image = data_info['image']
    image['var'] = rescale_var(image['data'], image['var_raw'], image['mask'])
    return data_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Output must differ from input to preserve the original pkl')
    with args.input.open('rb') as stream:
        data_info = joblib.load(stream)
    reestimate(data_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        joblib.dump(data_info, stream)


if __name__ == '__main__':
    """
    Save a new pkl file without re-cutting, without re-fitting the line center, 
    and without re-composing the continuous spectrum:

    python -m data_prep.reestimate_variance input.pkl output.pkl

    Specific actions:

    - The new pkl has var_total: Use it to recalculate var = var_total - data 
      and the valid mask.

    - The old pkl does not have var_total: Still call build_var and estimate 
      the background variance based on the peripheral dispersion.

    - At the same time, rescale the variance of the image data.
    """
    main()
