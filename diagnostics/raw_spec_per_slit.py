import joblib
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

def read_cutout(hdul):
    slit_num = 109
    hdu = hdul[slit_num]
    spec2d = np.asarray(hdu.data)
    hdr = hdu.header.copy()
    wcs = WCS(hdr)
    
    # 将像素坐标转换为世界坐标
    ny, nx = spec2d.shape
    x = np.arange(nx, dtype=float)
    y = np.full(nx, (ny - 1) / 2.0)
    world = wcs.all_pix2world(x, y, 0)
    
    # 找到波长对应的 world axis
    spec_axis = wcs.wcs.spec
    if spec_axis < 0:
        physical_types = wcs.world_axis_physical_types
        candidates = [
            i for i, value in enumerate(physical_types)
            if value is not None and (
                "em.wl" in value or "spect" in value.lower()
            )
        ]
        if not candidates:
            raise RuntimeError(
                f"无法识别波长轴；CTYPE={wcs.wcs.ctype}"
            )
        spec_axis = candidates[0]
    
    # 选择 7310–7350 Angstrom
    wave_min = 7310.0 * u.AA
    wave_max = 7350.0 * u.AA
    wave_value = np.asarray(world[spec_axis])
    
    # 读取波长单位并转换为 Angstrom
    wave_unit_string = wcs.world_axis_units[spec_axis]
    wave_unit = u.Unit(wave_unit_string) if wave_unit_string else u.AA
    wave = (wave_value * wave_unit).to(u.AA)
    mask = (
        np.isfinite(wave.value)
        & (wave >= wave_min)
        & (wave <= wave_max)
    )
    
    if not np.any(mask):
        raise RuntimeError(
            f"数据波长范围为 {np.nanmin(wave):.2f} 到 "
            f"{np.nanmax(wave):.2f}，不包含 7310–7350 Angstrom"
        )
    
    indices = np.where(mask)[0]
    i0, i1 = indices.min(), indices.max() + 1
    
    sub_image = spec2d[:, i0:i1]
    
    # Add sky masks
    sub_image[:,  0:13] = np.nan
    sub_image[:, 29:33] = np.nan
    sub_image[:, 47:53] = np.nan
    
    return sub_image[4:-4]


hdul_f   = fits.open("D:\\_RschArchives\\RSCH3\\UAO-S156-23B-A383\\psf\\231019\\obj_counts_slits_lin.fits")
hdul_err = fits.open("D:\\_RschArchives\\RSCH3\\UAO-S156-23B-A383\\psf\\231019\\obj_counts_err_slits_lin.fits")
raw_flux = read_cutout(hdul_f)
raw_err  = read_cutout(hdul_err)

with open('D:\\_RschArchives\\RSCH3\\kl_binospec\\scripts\\binospec_pkl\\pkl\\slit_109.pkl', "rb") as file:
    data_info = joblib.load(file)
pkl_flux    = data_info['spec'][1]['data']
pkl_bkg_var = data_info['spec'][1]['var']
pkl_total_err = (pkl_flux + pkl_bkg_var)**0.5

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16,4)) # (width, height)
im1 = ax[1-1].imshow(raw_flux, aspect='auto', cmap='viridis')
im2 = ax[2-1].imshow(raw_err,  aspect='auto', cmap='viridis')
im3 = ax[3-1].imshow(raw_flux + raw_err**2, aspect='auto', cmap='viridis')
im4 = ax[4-1].imshow(pkl_total_err, aspect='auto', cmap='viridis', origin='lower')
ax[1-1].set_title('raw flux')
ax[2-1].set_title('raw error')
ax[3-1].set_title('Chi2 deno: flux + error^2')
ax[4-1].set_title('calculated error in PKL')
plt.colorbar(im1, ax=ax[1-1])
plt.colorbar(im2, ax=ax[2-1])
plt.colorbar(im3, ax=ax[3-1])
plt.colorbar(im4, ax=ax[4-1])