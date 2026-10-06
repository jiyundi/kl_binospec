import numpy as np
import astropy.units as u
from   astropy import wcs


def meta_image(image_shape, objRA, objDEC, add_ap_wcs=False):
    """
        Make a dic for image data
    """
    image_pix_scale = 0.2      # arcsec/pix: 0.2 for HSC image
    psfFWHM         = 0.6      # arcsec
    meta_image_dic = {
        'ngrid':    image_shape,
        'pixScale': image_pix_scale,
        'psfFWHM':  psfFWHM,
        'RA':       objRA,
        'Dec':      objDEC,
        'ap_wcs':   None
        }
    
    if add_ap_wcs:
        ap_wcs           = wcs.WCS(naxis=2) # Create WCS
        ap_wcs.wcs.crpix = np.array([image_shape[0]/2+0.5,
                                     image_shape[0]/2+0.5]) # Cntrl ref pix (0.5-based)
        ap_wcs.wcs.crval = [objRA, objDEC] # RA/Dec (deg) central pixel
        ap_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN'] 
        ap_wcs.wcs.cdelt = [1, 1]
        ap_wcs.wcs.pc    = np.array([[-image_pix_scale/3600, 0], 
                                     [0, image_pix_scale/3600]]) # deg/px
        meta_image_dic['ap_wcs'] = ap_wcs
        
    return meta_image_dic
        

def meta_spec_ABC(dat_dict, spec_shape=None, slit_len=None):
    """
        Make a dic for slit data. 
        RA/DEC of the slit's center of the central set is REQUIRED.
    """
    spec_pix_scale = 0.24 # 0.24 arcsec/pix by Binospec
    dispersion_wav = 0.61 # 0.61 A/px by Binospec
    
    objRA      = dat_dict['RA']
    objDEC     = dat_dict['DEC']
    slit_width = dat_dict['slitwid']
    slit_PA    = dat_dict['slitPA']
    
    slit_setC_RA  = objRA *u.deg
    slit_setC_Dec = objDEC*u.deg
    slit_setA_RA  = objRA *u.deg - slit_width*u.arcsec * np.cos(slit_PA*u.deg)
    slit_setA_Dec = objDEC*u.deg + slit_width*u.arcsec * np.sin(slit_PA*u.deg)
    slit_setB_RA  = objRA *u.deg + slit_width*u.arcsec * np.cos(slit_PA*u.deg)
    slit_setB_Dec = objDEC*u.deg - slit_width*u.arcsec * np.sin(slit_PA*u.deg)
    
    meta_spec_C = {
        'ngrid':          spec_shape, # (spatial pixs, N wavelength pts)
        'pixScale':       spec_pix_scale,  # arcsec/px
        'dispersion':     dispersion_wav,  # A/px
        'slitRA':    slit_setC_RA,
        'slitDec':   slit_setC_Dec,
        'slitWidth': slit_width,
        'slitLen':   slit_len,
        'slitLPA':   (90 - slit_PA) * u.deg, # see Pranjal's paper
        'slitWPA':   (90 - slit_PA) * u.deg + 90*u.deg,  # Assume rectangular
        
        'line_species':   None, 
        'lambda_grid':    None,
        'rhl':            None,
        'line_profile':   None, # px, replace line_profile_path in config 
    }
    
    meta_spec_A = meta_spec_C.copy()
    meta_spec_B = meta_spec_C.copy()
    meta_spec_A['slitRA' ] = slit_setA_RA
    meta_spec_A['slitDec'] = slit_setA_Dec
    meta_spec_B['slitRA' ] = slit_setB_RA
    meta_spec_B['slitDec'] = slit_setB_Dec
    
    return {'A': meta_spec_A, 'B': meta_spec_B, 'C': meta_spec_C}