import numpy as np
from   astropy.io  import fits
from   astropy.wcs import WCS
import warnings
from   astropy import log
from   astropy.wcs import FITSFixedWarning
from   astropy.io.fits.verify import VerifyWarning

def read_subaru_img_wcs(image_science_filename, 
                        image_weights_filename=None):
    """
    Read Subaru imaging science and weights data in units of 
    e/s and (e/s)**-2.
    
    Returns
    ----------
    {science_data, science_wcs, weights_data, weights_wcs}
    """
    # Clear warnings
    with warnings.catch_warnings():
        
        # Clear warnings
        warnings.simplefilter("ignore", VerifyWarning)
        warnings.simplefilter("ignore", FITSFixedWarning)
        old_level = log.level
        log.setLevel("ERROR")
    
        science_hdul = fits.open(image_science_filename)
        
        # Pixel data
        science_raw = science_hdul[0].data
        
        # WCS mapping
        science_wcs = WCS(image_science_filename)
        
        # Clear warnings
        log.setLevel(old_level)
        
        # Gain value
        science_hdr0_raw = science_hdul[0].header
        sci_GAIN = science_hdr0_raw['GAIN']
        sci_time = science_hdr0_raw['EXPTIME']
        
        # Unit conversion
        # Science [e/s]  =  science_raw [ADU/s]  *  sci_GAIN [e/ADU]
        science_data = (science_raw / sci_time) * sci_GAIN # e/s
    
        weights_data, weights_wcs = None, None
        if image_weights_filename is not None: 
            weights_hdul = fits.open(image_weights_filename)
            
            # Pixel data
            weights_raw  = weights_hdul[0].data
            
            # WCS mapping
            weights_wcs  = WCS(science_hdul[0].header)
            
            # Gain value
            weights_hdr0_raw = weights_hdul[0].header
            wei_GAIN = weights_hdr0_raw['GAIN']
            wei_time = sci_time # MUST FOLLOW SCI'S EXP TIME
            # wei_time = weights_hdr0_raw['EXPTIME']
            
            # Unit conversion
            # Weights [(s/e)^2] = weights_raw [ADU^-2 s^2] * wei_GAIN^-2 [(e/ADU)^-2]
            weights_data = (weights_raw * wei_time) * wei_GAIN**(-2) # [s^2 e^-2]
            
            # Clear warnings
            log.setLevel(old_level)
    
    return {'science_data': science_data, 
            'science_wcs':  science_wcs, 
            'weights_data': weights_data, 
            'weights_wcs':  weights_wcs}


def cutoutimg(arrimg, wcs_loaded, objRA, objDEC, 
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
    arrimgcutout =       arrimg[int(southIndex):int(northIndex)+1, :]
    arrimgcutout = arrimgcutout[:,   int(eastIndex):int(westIndex)+1]
    if outputWCS:
        wcs_cutout = wcs_loaded[int(southIndex):int(northIndex)+1,
                                int( eastIndex):int( westIndex)+1]
        return arrimgcutout, [RAmin, RAmax], [DECmin, DECmax], \
               wcs_cutout
    else:
        return arrimgcutout, [RAmin, RAmax], [DECmin, DECmax]


def rhl_measure(image, r_max=None, n_steps=100):
    """
    Calculate the effective radius (half-light radius) 
    of an elliptical (face-on) galaxy of the exponential 
    disk type (n = 1 Sérsic)
    
    Parameters
    ----------
    image : 2D numpy array
        galaxy image
    x0, y0 : float
        galaxy center (in pixels)
    r_max : float
        galaxy max redius (in pixels). Default: half size of image
    n_steps : int
        sampling steps

    Returns
    -------
    r_half : float
        half-light radius in pixels
    """
    from photutils.aperture import CircularAperture, aperture_photometry
    if r_max is None:
        r_max = min(image.shape) / 2.0
        print(f'\nEstimating r_hl_disk in range of 0 - {r_max:.1f}...')

    radii = np.linspace(1, r_max, n_steps)
    fluxes = []

    for r in radii:
        aper = CircularAperture((image.shape[0]/2, image.shape[1]/2), r=r)
        phot_table = aperture_photometry(image, aper)
        fluxes.append(phot_table['aperture_sum'][0])

    fluxes = np.array(fluxes)
    total_flux = fluxes[-1]
    half_flux = total_flux / 2.0

    r_half = np.interp(half_flux, fluxes, radii)
    return r_half

