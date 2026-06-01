import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


def extract_2d_continuum(flux, mask, 
                         cont_y0=None, mode='simple', 
                         smooth=2, verbose=False):
    arr_flux = np.where(mask, flux, np.nan)
    ny, nx = arr_flux.shape # slit, wavelength
    
    # if strength is None:
    #     empirical_factor = np.max([
    #         1.0, 
    #         np.nanmedian(arr_flux[arr_flux < 5 * np.nanstd(arr_flux)])
    #         ])
    #     print(f'[INFO] continuum: Empirical factor = {empirical_factor:.2f}')
    # elif strength == 0:
    #     return np.zeros(arr_flux.shape)
    # else:
    #     empirical_factor = strength
        
    if mode=='simple':
        flx = median_filter(arr_flux, size=3)
        spatial_profi = np.nanmean(
            np.concatenate((flx[:, :5], flx[:, -5:]), axis=1), 
            axis=1)
        spatial_profi = gaussian_filter1d(spatial_profi, sigma=1)
        spatial_profi = np.where(spatial_profi>0, spatial_profi, 0)
        model = np.ones(flux.shape) * spatial_profi[:,np.newaxis]
        
    elif mode=='no':
        model = np.zeros(flux.shape)
    
    else:
        cont = fit_continuum(arr_flux,  # make a uniform cont
                             mask_lines=[(ny//6, -ny//6)], smooth=80)
        cont = (cont - np.min(cont)) * np.max(cont) + 1
        
        sigma = estimate_sigma_from_data(arr_flux, 
                                         mask_lines=[(nx//3, -nx//3)])
        ygrid = np.arange(ny)
        P = np.zeros((ny, nx))
    
        # initial Gaussian spatial profile
        for x in range(nx):
            if cont_y0 is None:
                cont_y0 = ny // 2
            prof = np.exp(-(ygrid - cont_y0)**2 / (2*sigma**2))
            P[:, x] = prof
    
        # smooth profile along wavelength
        # each row smooth in x direction
        # for y in range(ny):
        #     P[y,:] = gaussian_filter1d(P[y,:], smooth)
        
        # build 2D continuum model
        model = P * cont[:,np.newaxis].T
            
    if verbose:
        vmin, vmax = np.nanmin(arr_flux), np.nanmax(arr_flux)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(8,3))
        im0 = ax[0].imshow(arr_flux, aspect='auto', cmap='viridis', origin='lower')
        im1 = ax[1].imshow(model,    aspect='auto', cmap='viridis', origin='lower')
        im2 = ax[2].imshow(arr_flux-model, aspect='auto', cmap='viridis', origin='lower',
                           vmin=vmin, vmax=vmax)
        plt.colorbar(im0, ax=ax[0])
        plt.colorbar(im1, ax=ax[1])
        plt.colorbar(im2, ax=ax[2])
        ax[0].set_title('Data')
        ax[1].set_title(f'Continuum (mode: {mode})')
        ax[2].set_title('Data - Continuum')
        plt.suptitle("Close window to continue...", fontsize=15, y=0.98)
        plt.tight_layout()
        # plt.subplots_adjust(top=0.98) # main title spacing
        plt.show()
        plt.close()

    return model


def fit_continuum(flux, smooth=3, mask_lines=None):
    # import warnings
    # warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    
    # f_1d = np.nanmedian(flux, axis=0) # Suppress only this specific warning
    x_1d = np.arange(flux.shape[1])
    
    # emission line mask
    good = np.ones(flux.shape, dtype=bool)
    if mask_lines is not None:
        for a, b in mask_lines:
            good[a:b,:] = False

    # firstly, smooth it - find mean along axis=0
    flux_masked = np.where(good, flux, 0)
    fs_1d = gaussian_filter1d(
        np.nanmean(flux_masked, axis=0), 
        sigma=5)

    # spline continuum
    spl  = UnivariateSpline(x_1d, fs_1d, s=smooth)
    cont = spl(x_1d)

    return cont


def estimate_sigma_from_data(img, mask_lines=None):

    def _gauss(y, A, y0, sigma, B):
        return A*np.exp(-(y-y0)**2/(2*sigma**2)) + B
    
    # emission line mask
    good = np.ones(img.shape, dtype=bool)
    if mask_lines is not None:
        for a, b in mask_lines:
            good[:,a:b] = False
    
    # collapse wavelength region
    prof = np.nanmean(np.where(good, img, np.nan), axis=1)

    y = np.arange(len(prof))

    # initial guess
    A0 = prof.max() - np.median(prof)
    y0 = np.argmax(prof)
    sig0 = 2.5 # ~ r_hl_spec
    B0 = np.median(prof)

    popt, _ = curve_fit(
        _gauss, y, prof,
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


# if __name__ == '__main__':
#     for slit_num in [7]:#, 8, 29, 35, 55, 56, 57]:
#         data_info  = another_load_mock(pkl_folder='../../scripts_stable/binospec_pkl/', 
#                                        slit_num=slit_num)
        
#         for spec_idx in range(len(data_info['spec'])):
#             single_spec_data = data_info['spec'][spec_idx]
#             new_spec_data = build_2d_continuum(single_spec_data, smooth=9)

