import numpy as np
import astropy.units as u


def meta_spec_ABC(spec2d_C, dat_dict, max_rows):
    """
        Make a dic for slit data
    """
    spec_pix_scale = 0.24 # 0.24 arcsec/pix by Binospec
    
    objRA      = dat_dict['RA']
    objDEC     = dat_dict['DEC']
    # slit_len   = dat_dict['slitlen']
    spec_shape = spec2d_C['flux'].shape # (spatial pixs, N wavelength pts)
    slit_len   = spec_pix_scale * np.min([max_rows, spec_shape[0]])
    slit_width = dat_dict['slitwid']
    slit_PA    = dat_dict['slitPA']
    
    slit_setC_RA  = objRA *u.deg
    slit_setC_Dec = objDEC*u.deg
    slit_setA_RA  = objRA *u.deg - slit_width*u.arcsec * np.cos(slit_PA*u.deg)
    slit_setA_Dec = objDEC*u.deg + slit_width*u.arcsec * np.sin(slit_PA*u.deg)
    slit_setB_RA  = objRA *u.deg + slit_width*u.arcsec * np.cos(slit_PA*u.deg)
    slit_setB_Dec = objDEC*u.deg - slit_width*u.arcsec * np.sin(slit_PA*u.deg)
    
    meta_spec_C = {
        'line_species':   None, 
        'ngrid':          spec_shape,
        'lambda_grid':    spec2d_C['wave']*u.Angstrom,
        'pixScale':       spec_pix_scale,  # arcsec/px
        'rhl':            1.0,
        'line_sig_amps':  None, # px, replace line_profile_path in config 
        'slitRA':    slit_setC_RA,
        'slitDec':   slit_setC_Dec,
        'slitWidth': slit_width,
        'slitLen':   slit_len,
        'slitLPA':   (90 - slit_PA) * u.deg, # see Pranjal's paper
        'slitWPA':   (90 - slit_PA) * u.deg + 90*u.deg  # Assume rectangular
    }
    meta_spec_A = meta_spec_C.copy()
    meta_spec_B = meta_spec_C.copy()
    meta_spec_A['slitRA' ] = slit_setA_RA
    meta_spec_A['slitDec'] = slit_setA_Dec
    meta_spec_B['slitRA' ] = slit_setB_RA
    meta_spec_B['slitDec'] = slit_setB_Dec
    
    return meta_spec_A, meta_spec_B, meta_spec_C


