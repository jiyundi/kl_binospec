import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def another_load_mock(pkl_folder='mock/', slit_num=95):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
        data_info = joblib.load(f)
    
    return data_info

def gauss(y, A, y0, sigma, B):
    return A*np.exp(-(y-y0)**2/(2*sigma**2)) + B

def estimate_sigma_from_data(img, x1=0, x2=-1):

    # collapse wavelength region
    prof = np.nanmedian(img[:, x1:x2], axis=1)

    y = np.arange(len(prof))

    # initial guess
    A0 = prof.max() - np.median(prof)
    y0 = np.argmax(prof)
    sig0 = 2.5 # ~ r_hl_spec
    B0 = np.median(prof)

    popt, _ = curve_fit(
        gauss, y, prof,
        p0=[A0, y0, sig0, B0],
        bounds=([0,0,0.3,-np.inf],
                [np.inf,len(y),20,np.inf])
    )

    A, y0, sigma, B = popt

    return sigma

# def horne_extract(img, var, trace_y, aperture=6):

#     ny, nx = img.shape
#     flux = np.zeros(nx)
#     err  = np.zeros(nx)

#     ygrid = np.arange(ny)

#     for x in range(nx):

#         yc = trace_y[x]

#         # aperture mask
#         m = np.abs(ygrid - yc) <= aperture

#         y = ygrid[m]
#         d = img[m, x]
#         v = var[m, x]

#         # ----------------------------------------
#         # profile P(y): Gaussian estimate
#         # ----------------------------------------
#         sigma = aperture / 2.0
#         P = np.exp(-(y - yc)**2 / (2*sigma**2))
#         P /= P.sum()

#         # bad pixels
#         good = (v > 0) & np.isfinite(v)

#         if good.sum() < 3:
#             flux[x] = np.nan
#             err[x]  = np.nan
#             continue

#         Pg = P[good]
#         dg = d[good]
#         vg = v[good]

#         # ----------------------------------------
#         # Horne 1986 optimal extraction
#         # ----------------------------------------
#         num = np.sum(Pg * dg / vg)
#         den = np.sum(Pg**2 / vg)

#         if den <= 0:
#             flux[x] = np.nan
#             err[x]  = np.nan
#         else:
#             flux[x] = num / den
#             err[x]  = np.sqrt(1.0 / den)

#     return flux, err


# -------------------------------------------------
# 连续谱拟合
# -------------------------------------------------
def fit_continuum(flux, smooth=3, mask_lines=None):
    import warnings
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

    f_1d = np.nanmedian(flux, axis=0) # Suppress only this specific warning
    x_1d = np.arange(flux.shape[1])
    
    good = np.isfinite(f_1d)

    # emission line mask
    if mask_lines is not None:
        for a, b in mask_lines:
            good[a:b] = False

    # 初步平滑
    fs_1d = gaussian_filter1d(np.nan_to_num(f_1d), sigma=5)

    # spline continuum
    spl = UnivariateSpline(x_1d[good], fs_1d[good], s=smooth*good.sum())
    cont = spl(x_1d)

    return cont


def build_2d_continuum(single_spec_data, smooth=2, verbose=False):
    try:
        arr_flux = np.where(single_spec_data['mask'], 
                            single_spec_data['flux'], np.nan)
    except KeyError:
        arr_flux = np.where(single_spec_data['mask'], 
                            single_spec_data['data'], np.nan)
    
    ny, nx = arr_flux.shape # slit, wavelength
    
    cont = fit_continuum(arr_flux*2.5, 
                         mask_lines=[(ny//3, -ny//3)], smooth=smooth)
    sigma_profile = estimate_sigma_from_data(arr_flux)
    
    ygrid = np.arange(ny)
    P = np.zeros((ny, nx))

    # -----------------------------------
    # initial Gaussian spatial profile
    # -----------------------------------
    for x in range(nx):

        prof = np.exp(-(ygrid - ny // 2)**2 / (2*sigma_profile**2))

        P[:, x] = prof

    # -----------------------------------
    # smooth profile along wavelength
    # each row smooth in x direction
    # -----------------------------------
    for y in range(ny):
        P[y,:] = gaussian_filter1d(P[y,:], smooth)

    # -----------------------------------
    # build 2D continuum model
    # -----------------------------------
    model = P * cont[np.newaxis,:]
    
    try:
        single_spec_data['flux'] -= model
    except KeyError:
        single_spec_data['data'] -= model
            
    if verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        im0 = ax[0].imshow(arr_flux, aspect='auto', cmap='viridis')
        im1 = ax[1].imshow(model,    aspect='auto', cmap='viridis')
        im2 = ax[2].imshow(arr_flux - model, aspect='auto', cmap='viridis')
        plt.colorbar(im0, ax=ax[0])
        plt.colorbar(im1, ax=ax[1])
        plt.colorbar(im2, ax=ax[2])
        plt.tight_layout()
        plt.show()
        plt.close()

    return single_spec_data


if __name__ == '__main__':
    for slit_num in [7]:#, 8, 29, 35, 55, 56, 57]:
        data_info  = another_load_mock(pkl_folder='../../scripts_stable/binospec_pkl/', 
                                       slit_num=slit_num)
        
        for spec_idx in range(len(data_info['spec'])):
            single_spec_data = data_info['spec'][spec_idx]
            new_spec_data = build_2d_continuum(single_spec_data, smooth=9)

