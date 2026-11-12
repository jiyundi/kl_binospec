import numpy as np
import unittest
import tempfile
from pathlib import Path
from astropy.io import fits

from data_prep.spec.build_var import from_total_variance
from data_prep.spec.read_and_cutout import read_spec2d, stack_mean_variance
from data_prep.reestimate_variance import reestimate


class PipelineVarianceTests(unittest.TestCase):
    def test_total_variance_is_not_counted_twice_and_invalid_pixels_are_masked(self):
        data = np.array([[30., -12., 4., 5., np.nan]])
        total = np.array([[10., 25., 0., np.nan, 10.]])
        var, mask = from_total_variance(data, np.ones(data.shape, bool), total)
        np.testing.assert_array_equal(mask, [[True, True, False, False, False]])
        # Negative additive terms are allowed; it is total variance that must be >0.
        assert var[0, 0] == -20
        np.testing.assert_allclose((data + var)[mask], total[mask])


    def test_mean_variance_and_missing_rows(self):
        actual = stack_mean_variance(np.full((2, 3), 9.), np.full((2, 3), 18.),
                                     np.full((1, 3), 27.))
        np.testing.assert_allclose(actual[0], 6.)
        assert np.isnan(actual[1]).all()


    def test_loader_matches_target_and_rejects_misalignment(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        split = tmp_path / '1d2dspecfiles_counts'
        split.mkdir()
        counts = np.array([[2., -250., np.nan], [3., 4., 5.]], dtype=np.float32)
        flux = np.maximum(counts, -200)
        wave = np.tile(np.arange(3, dtype=np.float32), (2, 1))
        table = fits.BinTableHDU.from_columns([
            fits.Column(name=k, format='6E', dim='(3,2)', array=[v])
            for k, v in [('FLUX', flux), ('LAMBDA', wave)]])
        path = split / 'spec2d.829.112.100417.fits'
        fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
        header = fits.Header({'SLITTARG': 100417, 'SLITID': 39, 'SIDE': 'B', 'CRVAL1': 656.})
        for filename, data in [('obj_counts_slits_lin.fits', counts),
                               ('obj_counts_err_slits_lin.fits', np.array([[3., 1., 4.], [5., 6., 7.]]))]:
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, header=header, name='TARGET 112')]).writeto(tmp_path / filename)
        loaded = read_spec2d(path, with_variance=True)
        assert loaded['var_total'][0, 0] == 9.
        assert np.isnan(loaded['var_total'][0, 1])
        assert loaded['var_total'][1, 2] == 49.
        with fits.open(tmp_path / 'obj_counts_err_slits_lin.fits', mode='update') as h:
            h[1].header['CRVAL1'] = 657.
        with self.assertRaisesRegex(ValueError, 'coordinate mismatch'):
            read_spec2d(path, with_variance=True)


    def test_reestimate_preserves_pipeline_total_variance(self):
        data = np.full((10, 10), 30.)
        total = np.full_like(data, 12.)
        coverage = np.ones(data.shape, bool)
        coverage[0] = False
        info = {'spec': [{'data': data, 'var': np.zeros_like(data),
                          'var_total': total, 'mask': coverage.copy(),
                          'mask_base': coverage.copy(), 'coverage_mask': coverage}],
                'image': {'data': np.arange(100.).reshape(10, 10),
                          'var_raw': np.ones((10, 10)), 'mask': np.ones((10, 10), bool)}}
        spec = reestimate(info)['spec'][0]
        assert not spec['mask'][0].any()
        np.testing.assert_allclose((spec['data'] + spec['var'])[spec['mask']], 12.)
