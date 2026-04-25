import numpy as np
from   astropy import wcs


def cutoffimg(arrimg, wcs_loaded, objRA, objDEC, 
              img_width=30, img_height=30, outputWCS=False):
    img_pix_scale = 0.2 # arcsec/pix: 0.2 for HSC image
    img_RAwidth   = img_width  * img_pix_scale
    img_DECheight = img_height * img_pix_scale
    RAmin  = objRA  - img_RAwidth  /2/3600  
    RAmax  = objRA  + img_RAwidth  /2/3600
    DECmin = objDEC - img_DECheight/2/3600
    DECmax = objDEC + img_DECheight/2/3600
    eastIndex, southIndex = wcs_loaded.wcs_world2pix(RAmax, DECmin, 0)
    westIndex, northIndex = wcs_loaded.wcs_world2pix(RAmin, DECmax, 0)
    arrimgcutoff =       arrimg[int(southIndex):int(northIndex)+1, :]
    arrimgcutoff = arrimgcutoff[:,   int(eastIndex):int(westIndex)+1]
    if outputWCS:
        wcs_cutoff = wcs_loaded[int(southIndex):int(northIndex)+1,
                                int( eastIndex):int( westIndex)+1]
        return arrimgcutoff, [RAmin, RAmax], [DECmin, DECmax], \
               wcs_cutoff
    else:
        return arrimgcutoff, [RAmin, RAmax], [DECmin, DECmax]


def Meta_image(arrimgcutoff, objRA, objDEC):
    """
        Make a dic for image data
    """
    import galsim
    image_shape     = arrimgcutoff.shape # nDEC, nRA
    image_pix_scale = 0.2      # arcsec/pix: 0.2 for HSC image
    psfFWHM         = 0.6      # arcsec
    ap_wcs           = wcs.WCS(naxis=2) # Create WCS
    ap_wcs.wcs.crpix = np.array([image_shape[0]/2+0.5,
                                 image_shape[0]/2+0.5]) # Cntrl ref pix (0.5-based)
    ap_wcs.wcs.crval = [objRA, objDEC] # RA/Dec (deg) central pixel
    ap_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN'] 
    ap_wcs.wcs.cdelt = [1, 1]
    ap_wcs.wcs.pc    = np.array([[-image_pix_scale/3600, 0], [0, image_pix_scale/3600]]) # deg/px
    galsim_wcs       = galsim.AstropyWCS(wcs=ap_wcs)
    meta_image = {
        'ngrid':    image_shape,
        'pixScale': image_pix_scale,
        'psfFWHM':  psfFWHM,
        'wcs':      galsim_wcs,
        'ap_wcs':   ap_wcs,
        'RA':       objRA,
        'Dec':      objDEC}
    return meta_image
        

def half_light_radius_exp(image, r_max=None, n_steps=100):
    """
    计算指数盘 (n=1 Sérsic) 圆形(face-on)星系的半光半径 (effective radius)
    
    Parameters
    ----------
    image : 2D numpy array
        星系图像
    x0, y0 : float
        星系中心坐标 (像素)
    r_max : float
        最大测量半径 (像素)。默认取图像的一半
    n_steps : int
        半径采样的步数

    Returns
    -------
    r_half : float
        半光半径 (像素单位)
    """
    from photutils.aperture import CircularAperture, aperture_photometry
    if r_max is None:
        r_max = min(image.shape) / 2.0
        print(f'\nEstimating r_hl_disk in range of 0 - {r_max:.1f}...')

    # 一系列圆形半径
    radii = np.linspace(1, r_max, n_steps)
    fluxes = []

    for r in radii:
        aper = CircularAperture((image.shape[0]/2, image.shape[1]/2), r=r)
        phot_table = aperture_photometry(image, aper)
        fluxes.append(phot_table['aperture_sum'][0])

    fluxes = np.array(fluxes)
    total_flux = fluxes[-1]
    half_flux = total_flux / 2.0

    # 插值找到半光半径
    r_half = np.interp(half_flux, fluxes, radii)
    return r_half


def mask_neighbor_gal(arrimg):
    img = None
    return img