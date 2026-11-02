import numpy as np
import astropy.units as u

from klm.spec_model import SlitModel
from klm.transformations import Transformations
# from find_pkl_structures import another_load_mock
from build_multi_mock import prepare_a_slitspec

if __name__ == '__main__':
    redshift = 0.846
    slitLPA  = 0 * u.deg
    slitWPA  = slitLPA + 90 * u.deg
    
    meta_param = prepare_a_slitspec(
        RA_obj=0*u.deg, Dec_obj=0*u.deg, Set='C', 
        line='Hb', meta_gal={'redshift': redshift}, 
        spec_shape=[60, 20], slit_LPA=0
        )
    meta_param['slitLPA'] = slitLPA
    meta_param['slitWPA'] = slitWPA
    
    obj_param = {
        'redshift': redshift,
        'RA':       meta_param['slitRA'],
        'Dec':      meta_param['slitDec'],
        'beta':     0*u.deg, # Just give any number, only for g_t
        'log10_Mstar':     10, 
        'log10_Mstar_err':  0.1,
        }
    
    # params = Parameters(line_species=lines).params
    lines = [meta_param['line_species']] * 3
    params = {
        'shared_params': {
            'spec_snr': 20.0, 
            'image_snr': 80.0, 
            'g1': 0.0, 
            'g2': 0.0, 
            'cosi': 0.5, 
            'theta_int': 0.0, 
            'vscale': 0.5, 
            'r_hl_disk': 1.0, 
            'r_hl_bulge': 0.8, 
            'vcirc': 138.23079553462523, 
            'v_0': 0.0, 
            'v_outer': 0.0, 
            'dx_disk': 0.0, 
            'dy_disk': 0.0, 
            'dx_bulge': 0.0, 
            'dy_bulge': 0.0, 
            'flux': 10.0, 
            'flux_bulge': 0.0, 
            'aspect': 0.2, 
            'sersic_image': 1.0, 
            'beta': 0.0
            }, 
        f'{lines[0]}_params': {
            'v_0': 0.0, 
            'dx_vel': 0.0, 
            'dy_vel': 0.0, 
            'dx_spec': 0.0, 
            'dy_spec': 0.0, 
            'r_hl_spec': 1.0, 
            'f1_1': 1, 
            'f1_2': 1, 
            'I01': 100.0, 
            'bkg_level': 0.0, 
            'sersic_spec': 1, 
            'I01_spec1': 101.0, 
            'bkg_level_spec1': 1.0, 
            'dx_vel_spec1': 1.0, 
            'dy_vel_spec1': 1.0}
        }
    params = {**params['shared_params'],
              **params[f'{lines[0]}_params']}
    
    slit_model = SlitModel(obj_param, meta_param)
    disk_xx, disk_yy = Transformations.transform_frame(
        slit_model.obs_xx, slit_model.obs_yy, params=params, 
        start='obs', end='disk'
        )
    vfields = slit_model.build_line_vfield(params, disk_xx, disk_yy)
    
    
    
    
    cosis  = np.linspace(0.5, 1, 61)[::-1]
    thetas = np.linspace(0, 45,  61)/57.29583
    g1s    = np.linspace(0, 0.3, 61)
    g2s    = np.linspace(0, 0.3, 61)
    
    
    
    
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": "Inter",
        "font.serif": "Inter",
    })
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(np.rot90(vfields[0], k=-90/90), 
                   aspect='auto', cmap='RdBu_r', origin='lower',
                   vmin=-params['vcirc'], vmax=params['vcirc'])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label='Light-of-sight velocity (km/s)')
    # ax.set_ylabel('Position (px) along slit PA')
    # ax.set_xlabel('Position (px) perpendicular to slit PA')
    ax.set_ylabel('Position (px) along DEC')
    ax.set_xlabel('Position (px) along RA')
    ax.set_title('Velocity Map')
    
    def update(i):
        if i < len(cosis):
            params['cosi'] = cosis[i]
        elif i < len(cosis)+len(thetas):
            params['cosi']      = cosis[-1]
            params['theta_int'] = thetas[i-len(cosis)]
        elif i < len(cosis)+len(thetas)+len(g1s):
            params['cosi']      = cosis[ -1]
            params['theta_int'] = thetas[-1]
            params['g1']        = g1s[i-len(cosis)-len(thetas)]
        else:
            params['cosi']      = cosis[ -1]
            params['theta_int'] = thetas[-1]
            params['g1']        = g1s[   -1]
            params['g2']        = g2s[i-len(cosis)-len(thetas)-len(g1s)]
        ax.set_title("Velocity Map (PA=0)\n"+f"(i={int(np.arccos(params['cosi'])*57.29583)}"+
                     r'$^\circ$'+f", theta={int(params['theta_int']*57.29583)}"+
                     r'$^\circ$'+f", g1={params['g1']:.2f}, g2={params['g2']:.2f})", 
                     fontsize=16)
        
        disk_xx, disk_yy = Transformations.transform_frame(
            slit_model.obs_xx, slit_model.obs_yy, params=params, 
            start='obs', end='disk'
            )
        vfields = slit_model.build_line_vfield(params, disk_xx, disk_yy)
        im.set_data(np.rot90(vfields[0], k=-90/90))
        
    ani = FuncAnimation(fig, update, frames=list(range(len(cosis)+
                                                       len(thetas)+
                                                       len(g1s)+
                                                       len(g2s))))
    
    ani.save("slit_velocity_field.mp4", fps=60, 
             progress_callback=lambda i, n: print(f"{i}/{n}"))
    