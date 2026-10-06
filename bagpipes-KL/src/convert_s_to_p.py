import numpy as np

from src.load_spec  import load_spec


def convert_spec_to_eq_photo(slit_num_str, 
                             redshift, 
                             line_dic,
                             window=20):
    spectrum  = load_spec(slit_num_str, spec_scale=1, binfac=1)
    wave_full = spectrum[:, 0]   
    flux_full = spectrum[:, 1] 
    erro_full = spectrum[:, 2] 
    
    fluxes, fluxerrs, bonus_filters = [], [], []
    for _, wav in line_dic.items():
        wav = wav * (1+redshift)
        
        # Add your rectangle filter here 
        wl, tr = make_box_filter(center=wav, width=window, step=0.1, # A
                                 wave_axis=wave_full)
        
        # and solve for flux. 
        mask = (wave_full >= np.min(wl)) & (wave_full <= np.max(wl))
        y = flux_full[mask]
        y = np.where(np.isnan(y), 0, y)
        flux_area = np.trapz(y, wl)
        e = erro_full[mask]
        e = np.where(np.isnan(e), 0, e)
        flux_erro = np.sqrt(np.sum(e**2))
        
        # e-17 erg/s/cm^2 --> average --> e-17 erg/s/cm^2/A
        F_lambda_avg_flux = flux_area / window
        F_lambda_avg_erro = flux_erro / window
        
        # e-17 erg/s/cm^2/A --> e-17 erg/s/cm^2/Hz
        F_uJy = e_17_ergscm2A_to_uJy(F_lambda_avg_flux, wav)
        E_uJy = e_17_ergscm2A_to_uJy(F_lambda_avg_erro, wav)
        fluxes.append(  F_uJy)
        fluxerrs.append(E_uJy)
        
        # With filter to add before pipes.galaxy, 
        this_filter = np.c_[wl, tr]
        bonus_filters.append(this_filter)
        
    # Turn these into a 2D array and skipped AB mag conversion
    bonus_photometry = np.c_[fluxes, fluxerrs]

    return bonus_photometry, bonus_filters # e-17 erg/s/cm^2/Hz

def make_box_filter(center=7230.0, width=10, step=0.1, wave_axis=None):
    if wave_axis is not None:
        wl = wave_axis[(wave_axis >= center - width/2) & 
                       (wave_axis <= center + width/2)]
        tr = np.ones_like(wl)
    else:
        wl = np.arange(center - width/2, center + width/2 + step, step)
        tr = np.ones_like(wl)
    return wl, tr

def e_17_ergscm2A_to_uJy(F_lamb, effwav):
    # Constants
    c         = 2.998e10  # speed of light [cm/s]
    A_to_cm   = 1e-8
    to_micro  = 1e6
    Jy_to_cgs = 1e-23 # 1 Jy = 1e-23 erg/s/cm^2/Hz

    # e-17 erg/s/cm^2/A --> μJy (=e-17 erg/s/cm^2/Hz)
    effwav = effwav * A_to_cm # cm
    F_lamb = F_lamb / A_to_cm # e-17 erg/s/cm^2/cm
    F_lamb = F_lamb * 1e-17   #      erg/s/cm^2/cm
    F_nu   = (effwav**2 / c)  * F_lamb   # erg/s/cm^2/Hz
    F_nu   = F_nu / Jy_to_cgs * to_micro # micro-Jy
    return F_nu
    

